//! 레이어 단위 부호화 (논문 15장; 하네스 CP-6)
//!
//! 층 = 뉴런 잠재 좌표 {z_i}(출력 M), {w_j}(입력 N) + 커널 장 원자 J개:
//!   W_ij = K(z_i, w_j) = sum_k A_k cos( lambda_k [B_{b_k}(z_i) - B_{b_k}(w_j)] + phi_k )
//!
//! 분리 항등식 (명제 15.2): cos(a - b + phi) = cos(a+phi)cos(b) + sin(a+phi)sin(b)
//! 이므로 원자당 랭크 2, 층 전체는 묶인 랭크 2J 인수분해 W = F_out F_in^T 이다.
//! 순전파는 W 를 실체화하지 않고 y = F_out (F_in^T x) 로 계산한다 (특징 캐시 금지 —
//! 캐시하면 압축이 무의미해지므로 매 호출 재생성, 논문 15.4절).
//!
//! 게이지 (15.7절, 부록 C.3): 뫼비우스 g 로 좌표와 경계점을 함께 변환하면
//! B_{g(b)}(g(z)) - B_{g(b)}(g(w)) = B_b(z) - B_b(w)  (코사이클 항 log|g'(b)| 상쇄)
//! 이므로 W 는 위상·진폭 보정 없이 정확히 불변이다.

use crate::core::math::busemann::{busemann_polar, Mobius};
use crate::core::math::phase_state::{PhaseState, PHASE_MASK};
use rayon::prelude::*;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug)]
pub struct LatentCoord {
    pub r: f64,
    pub theta: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct KernelAtom {
    pub theta_b: f64,
    pub lambda: f64,
    /// 위상 레지스터 (20비트) — 미분·양자화는 phase_state 규칙을 따른다
    pub phi_q: u32,
    pub log2_amp: f64,
}

impl KernelAtom {
    pub fn amp(&self) -> f64 {
        self.log2_amp.exp2()
    }
    pub fn phi(&self) -> f64 {
        PhaseState::phase_value(self.phi_q & PHASE_MASK)
    }
}

// ---------------------------------------------------------------------------
// 비트 직렬화 (하네스 CP-8 선행분)
//
// 원자 1개 = u64 하나:
//   [63:52] theta_b 격자 (12b, 2*pi*t/4096)
//   [51:42] lambda 격자 (10b, m/16 — 범위 [0, 63.94))
//   [41:22] 위상 레지스터 (20b)
//   [21:6]  로그 진폭 코드 (16b, atom::quantize_amp 규칙)
//   [5:0]   예약 (0)
// 좌표 1개 = u64 하나: [63:32] r (f32 비트), [31:0] theta (f32 비트)
//
// 격자값으로 양자화된 층은 pack -> unpack -> pack 이 비트 동일(멱등)해야 한다.
// ---------------------------------------------------------------------------

pub const THETA_B_BITS: u32 = 12;
pub const LAMBDA_BITS: u32 = 10;
/// lambda 격자 간격 = 1/16
pub const LAMBDA_STEP: f64 = 1.0 / 16.0;

pub fn pack_atom(a: &KernelAtom) -> u64 {
    let tb = ((a.theta_b / (2.0 * PI) * 4096.0).round() as i64).rem_euclid(4096) as u64;
    let lm = ((a.lambda / LAMBDA_STEP).round().clamp(0.0, 1023.0)) as u64;
    let amp = crate::core::math::atom::quantize_amp(a.log2_amp) as u64;
    (tb << 52) | (lm << 42) | ((a.phi_q as u64 & PHASE_MASK as u64) << 22) | (amp << 6)
}

pub fn unpack_atom(bits: u64) -> KernelAtom {
    let tb = (bits >> 52) & 0xFFF;
    let lm = (bits >> 42) & 0x3FF;
    let phi_q = ((bits >> 22) & PHASE_MASK as u64) as u32;
    let amp = ((bits >> 6) & 0xFFFF) as u16;
    KernelAtom {
        theta_b: 2.0 * PI * tb as f64 / 4096.0,
        lambda: lm as f64 * LAMBDA_STEP,
        phi_q,
        log2_amp: crate::core::math::atom::dequantize_amp(amp),
    }
}

pub fn pack_coord(c: &LatentCoord) -> u64 {
    (((c.r as f32).to_bits() as u64) << 32) | (c.theta as f32).to_bits() as u64
}

pub fn unpack_coord(bits: u64) -> LatentCoord {
    LatentCoord {
        r: f32::from_bits((bits >> 32) as u32) as f64,
        theta: f32::from_bits(bits as u32) as f64,
    }
}

/// 결정론적 연산량 계측 (하네스 L0: 벽시계가 아니라 카운터가 게이트)
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpCounter {
    pub flops: u64,
    pub bytes: u64,
}

/// 연산량 회계 상수 (docs/test/verification_harness.md CP-6; 변경 시 테스트 공식과 함께)
pub const FLOPS_BUSEMANN: u64 = 10;
/// 출력 특징: busemann(10) + 인자(2) + sin/cos(2) + 진폭 곱(2)
pub const FLOPS_FEAT_OUT: u64 = FLOPS_BUSEMANN + 6;
/// 입력 특징: busemann(10) + 인자(2) + sin/cos(2)
pub const FLOPS_FEAT_IN: u64 = FLOPS_BUSEMANN + 4;

#[derive(Clone)]
pub struct LayerCodec {
    pub rows: Vec<LatentCoord>,
    pub cols: Vec<LatentCoord>,
    pub atoms: Vec<KernelAtom>,
}

impl LayerCodec {
    /// 커널 직접 평가 (검증 기준 경로 — 프로덕션은 forward 의 인수분해 경로)
    pub fn eval_direct(&self, z: LatentCoord, w: LatentCoord) -> f64 {
        let mut sum = 0.0;
        for a in &self.atoms {
            let bz = busemann_polar(z.r, z.theta, a.theta_b);
            let bw = busemann_polar(w.r, w.theta, a.theta_b);
            sum += a.amp() * (a.lambda * (bz - bw) + a.phi()).cos();
        }
        sum
    }

    /// W 행렬 실체화 (검증용)
    pub fn materialize(&self) -> Vec<Vec<f64>> {
        self.rows
            .iter()
            .map(|&z| self.cols.iter().map(|&w| self.eval_direct(z, w)).collect())
            .collect()
    }

    /// 출력 특징 F_out (M x 2J): [A cos(lambda B + phi), A sin(lambda B + phi)]
    fn feature_out(&self, z: LatentCoord, counter: &mut OpCounter) -> Vec<f64> {
        let mut f = Vec::with_capacity(2 * self.atoms.len());
        for a in &self.atoms {
            let b = busemann_polar(z.r, z.theta, a.theta_b);
            let arg = a.lambda * b + a.phi();
            let amp = a.amp();
            f.push(amp * arg.cos());
            f.push(amp * arg.sin());
            counter.flops += FLOPS_FEAT_OUT;
        }
        f
    }

    /// 입력 특징 F_in (N x 2J): [cos(lambda B), sin(lambda B)]
    fn feature_in(&self, w: LatentCoord, counter: &mut OpCounter) -> Vec<f64> {
        let mut f = Vec::with_capacity(2 * self.atoms.len());
        for a in &self.atoms {
            let b = busemann_polar(w.r, w.theta, a.theta_b);
            let arg = a.lambda * b;
            f.push(arg.cos());
            f.push(arg.sin());
            counter.flops += FLOPS_FEAT_IN;
        }
        f
    }

    /// 순전파 y = F_out (F_in^T x). W 비실체화, 특징 비캐시 (논문 15.4절).
    pub fn forward(&self, x: &[f64], counter: &mut OpCounter) -> Vec<f64> {
        assert_eq!(x.len(), self.cols.len(), "입력 차원 불일치");
        let two_j = 2 * self.atoms.len();

        // 파라미터 읽기 회계: 좌표 (M+N) x 16B, 원자 J x 32B, 입출력 벡터
        counter.bytes += 16 * (self.rows.len() + self.cols.len()) as u64
            + 32 * self.atoms.len() as u64
            + 8 * (x.len() + self.rows.len()) as u64;

        // c = F_in^T x  (2J 누적기, FLOP = 2 * 2J * N)
        let mut c = vec![0.0f64; two_j];
        for (j, &xj) in x.iter().enumerate() {
            let f = self.feature_in(self.cols[j], counter);
            for (l, &fl) in f.iter().enumerate() {
                c[l] += fl * xj;
            }
            counter.flops += 2 * two_j as u64;
        }

        // y_i = F_out[i,:] . c  (FLOP = 2 * 2J * M)
        let mut y = Vec::with_capacity(self.rows.len());
        for &z in &self.rows {
            let f = self.feature_out(z, counter);
            let mut acc = 0.0;
            for (l, &fl) in f.iter().enumerate() {
                acc += fl * c[l];
            }
            counter.flops += 2 * two_j as u64;
            y.push(acc);
        }
        y
    }

    /// 순전파 FLOP 의 닫힌형 (논문 15.4절) — L0 카운터 일치 테스트의 독립 공식
    pub fn forward_flops_formula(m: usize, n: usize, j: usize) -> u64 {
        (m as u64) * (j as u64) * FLOPS_FEAT_OUT
            + (n as u64) * (j as u64) * FLOPS_FEAT_IN
            + 2 * (2 * j as u64) * (n as u64)
            + 2 * (2 * j as u64) * (m as u64)
    }

    pub fn forward_bytes_formula(m: usize, n: usize, j: usize) -> u64 {
        16 * (m + n) as u64 + 32 * j as u64 + 8 * (n + m) as u64
    }

    /// 층 부호 직렬화: [행 좌표 M] [열 좌표 N] [원자 J] 순서의 u64 벡터
    pub fn to_bits(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.rows.len() + self.cols.len() + self.atoms.len());
        out.extend(self.rows.iter().map(pack_coord));
        out.extend(self.cols.iter().map(pack_coord));
        out.extend(self.atoms.iter().map(pack_atom));
        out
    }

    pub fn from_bits(m: usize, n: usize, j: usize, bits: &[u64]) -> LayerCodec {
        assert_eq!(bits.len(), m + n + j, "부호 길이 불일치");
        LayerCodec {
            rows: bits[..m].iter().map(|&b| unpack_coord(b)).collect(),
            cols: bits[m..m + n].iter().map(|&b| unpack_coord(b)).collect(),
            atoms: bits[m + n..].iter().map(|&b| unpack_atom(b)).collect(),
        }
    }

    /// 층 부호 비트 수의 닫힌형 (논문 15.5절): 64 * (M + N + J)
    pub fn code_bits_formula(m: usize, n: usize, j: usize) -> u64 {
        64 * (m + n + j) as u64
    }

    /// f32 밀집 대비 압축률 (논문 15.5절)
    pub fn compression_ratio_vs_f32(m: usize, n: usize, j: usize) -> f64 {
        (32 * m * n) as f64 / Self::code_bits_formula(m, n, j) as f64
    }

    /// 매칭 퍼슈트로 적합된 층의 원자 부호가 재현하는 커널 행렬 (행 우선 평탄 벡터)
    pub fn materialize_flat(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.rows.len() * self.cols.len());
        for &z in &self.rows {
            for &w in &self.cols {
                out.push(self.eval_direct(z, w));
            }
        }
        out
    }

    /// 뫼비우스 게이지 변환: 좌표 z -> g(z), 경계점 b -> g(b).
    /// 부록 C.3 에 의해 상대 Busemann 차가 불변이므로 phi, A 는 무보정 (15.7절).
    pub fn gauge_transform(&self, g: &Mobius) -> LayerCodec {
        let map_coord = |c: &LatentCoord| {
            let (x, y) = g.apply(c.r * c.theta.cos(), c.r * c.theta.sin());
            LatentCoord {
                r: (x * x + y * y).sqrt(),
                theta: y.atan2(x),
            }
        };
        let map_atom = |a: &KernelAtom| {
            let (bx, by) = g.apply_boundary(a.theta_b.cos(), a.theta_b.sin());
            KernelAtom {
                theta_b: by.atan2(bx),
                ..*a
            }
        };
        LayerCodec {
            rows: self.rows.iter().map(&map_coord).collect(),
            cols: self.cols.iter().map(&map_coord).collect(),
            atoms: self.atoms.iter().map(&map_atom).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// 매칭 퍼슈트 원자 적합기 (논문 15.6절 (b) 단계; 하네스 CP-8)
//
// 좌표는 고정 결정론 배치(황금각 나선)다 — 좌표 학습(15.6절 (a) 리만 SGD)은
// 후속 작업이며, 여기서 측정되는 c 는 좌표 학습 이전의 하한이다.
//
// 정직성 구조: 탐색 격자는 직렬화 격자(2pi/4096, LAMBDA_STEP)의 부분집합이고,
// 선택된 원자의 계수 (A, phi) 는 pack/unpack 왕복으로 양자화한 뒤 그 "디코딩된
// 값"으로 잔차를 갱신한다. 따라서 c = 1 - ||R||^2/||W||^2 는 실제 부호가
// 재현하는 값 기준의 실측이다 (부호와 무관한 수치로 c 를 부풀릴 수 없다).
//
// 후보 선택은 Busemann 차 히스토그램 근사(위상 오차 <= SELECT_PHASE_ERR)로
// 가속한다. 근사는 탐욕 선택의 순서에만 영향을 주며, 선택 후 투영 계수는
// 정확한 최소자승으로 다시 풀기 때문에 부호·잔차·c 실측에는 개입하지 않는다.
// ---------------------------------------------------------------------------

/// 좌표 학습 이전의 초기 반지름 상한. Busemann 값은 |B| <= 2 artanh r 로
/// 경계에서 발산하므로, 격자 lambda 와의 곱이 과도한 진동을 만들지 않는 수준.
pub const FIT_COORD_R_MAX: f64 = 0.9;

/// 후보 선택 히스토그램의 위상 허용오차 (rad). 선택 휴리스틱 전용 (위 주석).
const SELECT_PHASE_ERR: f64 = 0.05;

#[derive(Clone, Copy, Debug)]
pub struct PursuitConfig {
    /// theta_b 후보 수 (4096 직렬화 격자의 등간격 부분집합)
    pub n_theta: usize,
    /// lambda 후보 수: m * LAMBDA_STEP, m = 1..=n_lambda
    pub n_lambda: usize,
    /// 원자 수 J
    pub n_atoms: usize,
}

pub struct PursuitResult {
    pub codec: LayerCodec,
    /// 원자 k개 시점의 에너지 포획률 c_k = 1 - ||R_k||^2 / ||W||^2
    pub c_curve: Vec<f64>,
    /// 최종 잔차 W - K (행 우선)
    pub residual: Vec<f64>,
}

/// 황금각 나선 배치 (결정론). 좌표는 직렬화(f32)에 먼저 스냅해
/// 적합에 쓰인 값과 부호가 재현하는 값이 비트 동일하도록 한다.
pub fn spiral_coords(count: usize) -> Vec<LatentCoord> {
    let golden = PI * (3.0 - 5.0f64.sqrt());
    (0..count)
        .map(|i| {
            let r = FIT_COORD_R_MAX * ((i as f64 + 0.5) / count as f64).sqrt();
            let theta = (golden * i as f64).rem_euclid(2.0 * PI);
            LatentCoord {
                r: r as f32 as f64,
                theta: theta as f32 as f64,
            }
        })
        .collect()
}

/// Gram 행렬 닫힌형: 원자 기저 c_ij=cos(l*d_ij), s_ij=sin(l*d_ij) (d=bz_i-bw_j) 에서
/// <c,c>, <c,s>, <s,s> 는 배각 합의 랭크-1 구조로 O(M+N) 에 정확히 계산된다.
fn gram_closed_form(bz: &[f64], bw: &[f64], lambda: f64) -> (f64, f64, f64) {
    let sum2 = |v: &[f64]| {
        v.iter().fold((0.0, 0.0), |(c, s), &b| {
            let (sn, cs) = (2.0 * lambda * b).sin_cos();
            (c + cs, s + sn)
        })
    };
    let (cz, sz) = sum2(bz);
    let (cw, sw) = sum2(bw);
    let mn = (bz.len() * bw.len()) as f64;
    let c2 = cz * cw + sz * sw;
    let s2 = sz * cw - cz * sw;
    ((mn + c2) / 2.0, s2 / 2.0, (mn - c2) / 2.0)
}

/// 2x2 정규방정식 풀이. 퇴화(det ~ 0, 예: lambda*d 범위가 극소)면 cos 성분만 쓴다.
fn solve_normal_2x2(gcc: f64, gcs: f64, gss: f64, pc: f64, ps: f64) -> (f64, f64) {
    let det = gcc * gss - gcs * gcs;
    if det <= f64::EPSILON.sqrt() * gcc * gss || !det.is_finite() {
        return if gcc > 0.0 { (pc / gcc, 0.0) } else { (0.0, 0.0) };
    }
    ((gss * pc - gcs * ps) / det, (gcc * ps - gcs * pc) / det)
}

/// 한 theta_b 에서의 최선 lambda 후보: 잔차 가중 Busemann 차 히스토그램으로
/// 투영 <R,c>, <R,s> 를 근사하고, Gram(정확)과 함께 에너지 감소량을 점수화한다.
fn best_lambda_for_theta(
    res: &[f64],
    bz: &[f64],
    bw: &[f64],
    lambdas: &[f64],
) -> (f64, usize) {
    let n = bw.len();
    let (bz_min, bz_max) = min_max(bz);
    let (bw_min, bw_max) = min_max(bw);
    let (d_min, d_max) = (bz_min - bw_max, bz_max - bw_min);
    let eps = 2.0 * SELECT_PHASE_ERR / lambdas.last().copied().unwrap_or(1.0);
    let nbins = (((d_max - d_min) / eps).ceil() as usize).max(1) + 1;
    let inv_eps = (nbins - 1) as f64 / (d_max - d_min).max(f64::MIN_POSITIVE);

    let mut hist = vec![0.0f64; nbins];
    for (i, &bzi) in bz.iter().enumerate() {
        let row = &res[i * n..(i + 1) * n];
        for (j, &bwj) in bw.iter().enumerate() {
            let idx = ((bzi - bwj - d_min) * inv_eps) as usize;
            hist[idx.min(nbins - 1)] += row[j];
        }
    }

    let mut best = (0.0f64, 0usize);
    for (li, &lambda) in lambdas.iter().enumerate() {
        let (mut pc, mut ps) = (0.0, 0.0);
        for (b, &h) in hist.iter().enumerate() {
            let d = d_min + b as f64 / inv_eps;
            let (sn, cs) = (lambda * d).sin_cos();
            pc += h * cs;
            ps += h * sn;
        }
        let (gcc, gcs, gss) = gram_closed_form(bz, bw, lambda);
        let (alpha, beta) = solve_normal_2x2(gcc, gcs, gss, pc, ps);
        let score = alpha * pc + beta * ps;
        if score > best.0 {
            best = (score, li);
        }
    }
    best
}

fn min_max(v: &[f64]) -> (f64, f64) {
    v.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        })
}

/// 선택된 (theta_b, lambda) 의 정확한 투영 <R,c>, <R,s> — O(MN) 직접 계산
fn exact_projection(res: &[f64], bz: &[f64], bw: &[f64], lambda: f64) -> (f64, f64) {
    let n = bw.len();
    let (cw, sw): (Vec<f64>, Vec<f64>) = bw
        .iter()
        .map(|&b| {
            let (s, c) = (lambda * b).sin_cos();
            (c, s)
        })
        .unzip();
    let (mut pc, mut ps) = (0.0, 0.0);
    for (i, &bzi) in bz.iter().enumerate() {
        let row = &res[i * n..(i + 1) * n];
        let (mut ac, mut asum) = (0.0, 0.0);
        for j in 0..n {
            ac += row[j] * cw[j];
            asum += row[j] * sw[j];
        }
        let (szi, czi) = (lambda * bzi).sin_cos();
        pc += czi * ac + szi * asum;
        ps += szi * ac - czi * asum;
    }
    (pc, ps)
}

/// 양자화-디코딩된 원자를 잔차에서 빼고 새 에너지 ||R||^2 를 반환.
/// A cos(l d + phi) = [A cos(l bz + phi)] cos(l bw) + [A sin(l bz + phi)] sin(l bw)
fn subtract_atom(res: &mut [f64], bz: &[f64], bw: &[f64], atom: &KernelAtom) -> f64 {
    let n = bw.len();
    let (amp, phi, lambda) = (atom.amp(), atom.phi(), atom.lambda);
    let (cw, sw): (Vec<f64>, Vec<f64>) = bw
        .iter()
        .map(|&b| {
            let (s, c) = (lambda * b).sin_cos();
            (c, s)
        })
        .unzip();
    let mut energy = 0.0;
    for (i, &bzi) in bz.iter().enumerate() {
        let (s, c) = (lambda * bzi + phi).sin_cos();
        let (ac, asn) = (amp * c, amp * s);
        let row = &mut res[i * n..(i + 1) * n];
        for j in 0..n {
            row[j] -= ac * cw[j] + asn * sw[j];
            energy += row[j] * row[j];
        }
    }
    energy
}

/// 정확한 최소자승 계수를 원자 파라미터로 변환해 직렬화 격자에 스냅.
/// alpha c + beta s = A cos(l d + phi), A = hypot, phi = -atan2(beta, alpha)
fn snapped_atom(theta_b: f64, lambda: f64, alpha: f64, beta: f64) -> Option<KernelAtom> {
    let amp = alpha.hypot(beta);
    if !(amp.is_finite() && amp > 0.0) {
        return None;
    }
    let atom = KernelAtom {
        theta_b,
        lambda,
        phi_q: PhaseState::quantize_phase((-beta).atan2(alpha)),
        log2_amp: amp.log2(),
    };
    let snapped = unpack_atom(pack_atom(&atom));
    (snapped.amp() > 0.0).then_some(snapped)
}

/// 매칭 퍼슈트 적합 (논문 15.6절 (b)). 잔차 에너지가 감소하는 동안만 원자를
/// 추가한다 (양자화 후 에너지가 늘면 그 원자를 되돌리고 종료 — c 단조성 보장).
pub fn fit_matching_pursuit(w: &[f64], m: usize, n: usize, cfg: &PursuitConfig) -> PursuitResult {
    fit_matching_pursuit_with_coords(w, m, n, cfg, spiral_coords(m), spiral_coords(n))
}

/// 좌표를 외부에서 주입하는 매칭 퍼슈트 (좌표 학습 루프가 사용)
pub fn fit_matching_pursuit_with_coords(
    w: &[f64],
    m: usize,
    n: usize,
    cfg: &PursuitConfig,
    rows: Vec<LatentCoord>,
    cols: Vec<LatentCoord>,
) -> PursuitResult {
    assert_eq!(w.len(), m * n, "행렬 크기 불일치");
    let thetas: Vec<f64> = (0..cfg.n_theta)
        .map(|k| 2.0 * PI * (k * 4096 / cfg.n_theta) as f64 / 4096.0)
        .collect();
    let lambdas: Vec<f64> = (1..=cfg.n_lambda).map(|k| k as f64 * LAMBDA_STEP).collect();
    let bz: Vec<Vec<f64>> = thetas
        .iter()
        .map(|&t| rows.iter().map(|c| busemann_polar(c.r, c.theta, t)).collect())
        .collect();
    let bw: Vec<Vec<f64>> = thetas
        .iter()
        .map(|&t| cols.iter().map(|c| busemann_polar(c.r, c.theta, t)).collect())
        .collect();

    let e0: f64 = w.iter().map(|x| x * x).sum();
    let mut res = w.to_vec();
    let mut energy = e0;
    let mut atoms = Vec::new();
    let mut c_curve = Vec::new();

    while atoms.len() < cfg.n_atoms {
        let (ti, li) = match select_candidate(&res, &bz, &bw, &lambdas) {
            Some(best) => best,
            None => break,
        };
        let (pc, ps) = exact_projection(&res, &bz[ti], &bw[ti], lambdas[li]);
        let (gcc, gcs, gss) = gram_closed_form(&bz[ti], &bw[ti], lambdas[li]);
        let (alpha, beta) = solve_normal_2x2(gcc, gcs, gss, pc, ps);
        let atom = match snapped_atom(thetas[ti], lambdas[li], alpha, beta) {
            Some(a) => a,
            None => break,
        };
        let new_energy = subtract_atom(&mut res, &bz[ti], &bw[ti], &atom);
        if new_energy >= energy {
            let inverse = KernelAtom {
                log2_amp: atom.log2_amp,
                phi_q: PhaseState::quantize_phase(atom.phi() + PI),
                ..atom
            };
            subtract_atom(&mut res, &bz[ti], &bw[ti], &inverse);
            break;
        }
        energy = new_energy;
        atoms.push(atom);
        c_curve.push(1.0 - energy / e0);
    }

    PursuitResult {
        codec: LayerCodec { rows, cols, atoms },
        c_curve,
        residual: res,
    }
}

// ---------------------------------------------------------------------------
// 좌표 학습 (논문 15.6절 (a); 하네스 CP-8 돌파구)
//
// 인코딩은 교대 최적화다: (라운드) 원자 매칭 퍼슈트 -> 좌표 리만 SGD -> 재적합.
// - 좌표 초기화: 상위 2 특이벡터(멱반복, 결정론적 초기 벡터)를 원판에 사상 —
//   지배 구조가 Busemann 특징에 노출되도록 하는 데이터 기반 초기 배치.
// - 좌표 그래디언트: dK/dB 는 정리 13.2 의 위상 이동(cos -> sin)이고,
//   dB/dz 는 부록 B.2 닫힌형. 리만 갱신은 계량 역행렬 (1-|z|^2)^2/4 배 (5.4절)
//   — 경계 감쇠가 좌표를 원판 내부에 유지하며, 안전을 위해 r <= FIT_COORD_R_MAX
//   재투영을 추가한다.
// - 정직성: 학습은 인코딩 시점 1회다 (규칙: 인코딩은 최초 압축 1회만).
//   c 는 항상 "재적합 후 양자화된 부호" 기준으로 측정하며, keep-best 라서
//   라운드 0(학습 없음)보다 나빠질 수 없다.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct LearnConfig {
    pub pursuit: PursuitConfig,
    /// 교대 라운드 수 (0 이면 초기 좌표의 퍼슈트만 — 기준선)
    pub rounds: usize,
    /// 라운드당 좌표 SGD 스텝 (행/열 교대)
    pub sgd_steps: usize,
    /// 반대편 차원 미니배치 크기
    pub batch: usize,
    /// 리만 SGD 학습률
    pub lr: f64,
    pub seed: u64,
}

/// 멱반복으로 상위 2 특이쌍을 구해 좌표로 사상 (결정론: 고정 초기 벡터)
pub fn svd_init_coords(w: &[f64], m: usize, n: usize) -> (Vec<LatentCoord>, Vec<LatentCoord>) {
    let (u1, v1, s1) = top_singular_pair(w, m, n, &[]);
    let d1 = [(u1.clone(), v1.clone(), s1)];
    let (u2, v2, _) = top_singular_pair(w, m, n, &d1);
    (project_to_disk(&u1, &u2), project_to_disk(&v1, &v2))
}

/// 상위 r 특이값 에너지 비율 (멱반복 + 순차 수축, 결정론) — 프로토콜 7 저랭크 대결의
/// "동일 비트 자유 저랭크 상한 c" 추정기. c 사다리(17.2절) 배분 판단에 사용.
pub fn svd_energy_ceiling(w: &[f64], m: usize, n: usize, r: usize) -> f64 {
    let total: f64 = w.iter().map(|x| x * x).sum();
    if total == 0.0 {
        return 0.0;
    }
    let mut deflated: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new();
    let mut captured = 0.0;
    for _ in 0..r {
        let (u, v, s) = top_singular_pair(w, m, n, &deflated);
        captured += s * s;
        deflated.push((u, v, s));
    }
    (captured / total).min(1.0)
}

/// 멱반복 (W^T W 30회). deflate 의 각 (u, v, sigma) 를 암시적으로 차감.
fn top_singular_pair(
    w: &[f64],
    m: usize,
    n: usize,
    deflate: &[(Vec<f64>, Vec<f64>, f64)],
) -> (Vec<f64>, Vec<f64>, f64) {
    let mut v: Vec<f64> = (0..n).map(|j| (j as f64 * 0.7391 + 0.31).sin()).collect();
    let mut u = vec![0.0; m];
    for _ in 0..30 {
        matvec(w, m, n, &v, &mut u, deflate, false);
        normalize(&mut u);
        matvec(w, m, n, &u, &mut v, deflate, true);
        normalize(&mut v);
    }
    matvec(w, m, n, &v, &mut u, deflate, false);
    let sigma = u.iter().map(|x| x * x).sum::<f64>().sqrt();
    normalize(&mut u);
    (u, v, sigma)
}

fn matvec(
    w: &[f64],
    m: usize,
    n: usize,
    x: &[f64],
    out: &mut [f64],
    deflate: &[(Vec<f64>, Vec<f64>, f64)],
    transpose: bool,
) {
    if transpose {
        out.iter_mut().for_each(|o| *o = 0.0);
        for i in 0..m {
            let xi = x[i];
            let row = &w[i * n..(i + 1) * n];
            for j in 0..n {
                out[j] += row[j] * xi;
            }
        }
    } else {
        for i in 0..m {
            out[i] = w[i * n..(i + 1) * n].iter().zip(x).map(|(a, b)| a * b).sum();
        }
    }
    for (u1, v1, s1) in deflate {
        if transpose {
            let d: f64 = u1.iter().zip(x).map(|(a, b)| a * b).sum();
            out.iter_mut().zip(v1).for_each(|(o, &v)| *o -= s1 * d * v);
        } else {
            let d: f64 = v1.iter().zip(x).map(|(a, b)| a * b).sum();
            out.iter_mut().zip(u1).for_each(|(o, &u)| *o -= s1 * d * u);
        }
    }
}

fn normalize(v: &mut [f64]) {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

/// 특이벡터 쌍 (u1_i, u2_i) 을 원판 좌표로: 성분별 최대절대값 정규화 후
/// 반지름 FIT_COORD_R_MAX 내 정사각형에 배치. f32 스냅 (직렬화 일치).
fn project_to_disk(a: &[f64], b: &[f64]) -> Vec<LatentCoord> {
    let scale = |v: &[f64]| {
        let mx = v.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
        if mx > 0.0 {
            FIT_COORD_R_MAX / (mx * 2.0f64.sqrt())
        } else {
            0.0
        }
    };
    let (sa, sb) = (scale(a), scale(b));
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let (px, py) = (x * sa, y * sb);
            LatentCoord {
                r: (px * px + py * py).sqrt() as f32 as f64,
                theta: py.atan2(px) as f32 as f64,
            }
        })
        .collect()
}

/// dB/dz 닫힌형 (부록 B.2): grad B = -2z/(1-|z|^2) - 2(z-b)/|z-b|^2 (유클리드)
fn busemann_grad_xy(x: f64, y: f64, theta_b: f64) -> (f64, f64) {
    let r = (x * x + y * y).sqrt();
    let omr2 = (1.0 - r) * (1.0 + r);
    let (bx, by) = (theta_b.cos(), theta_b.sin());
    let (dx, dy) = (x - bx, y - by);
    let d2 = dx * dx + dy * dy;
    (-2.0 * x / omr2 - 2.0 * dx / d2, -2.0 * y / omr2 - 2.0 * dy / d2)
}

/// 리만 갱신: z <- z - lr * ((1-|z|^2)^2/4) * egrad, 이후 r <= FIT_COORD_R_MAX 재투영
fn riemannian_step(c: LatentCoord, gx: f64, gy: f64, lr: f64) -> LatentCoord {
    let (x, y) = (c.r * c.theta.cos(), c.r * c.theta.sin());
    let omr2 = (1.0 - c.r) * (1.0 + c.r);
    let scale = lr * omr2 * omr2 / 4.0;
    let (nx, ny) = (x - scale * gx, y - scale * gy);
    let mut r = (nx * nx + ny * ny).sqrt();
    let theta = ny.atan2(nx);
    if r > FIT_COORD_R_MAX {
        r = FIT_COORD_R_MAX;
    }
    LatentCoord {
        r: r as f32 as f64,
        theta: theta as f32 as f64,
    }
}

/// 활성 좌표(갱신 대상) 한쪽의 SGD 1스텝. `rows_active` 가 참이면 행 좌표를
/// 갱신하고 열에서 미니배치를 뽑는다 (거짓이면 대칭).
/// 반환: 갱신된 활성 좌표. 원자·반대편 좌표는 불변.
fn coord_sgd_step(
    w: &[f64],
    m: usize,
    n: usize,
    codec: &LayerCodec,
    rows_active: bool,
    batch_idx: &[usize],
    lr: f64,
) -> Vec<LatentCoord> {
    let atoms = &codec.atoms;
    let jn = atoms.len();
    let (active, fixed): (&[LatentCoord], &[LatentCoord]) = if rows_active {
        (&codec.rows, &codec.cols)
    } else {
        (&codec.cols, &codec.rows)
    };

    // 고정측 배치의 원자별 각도 성분 (행 활성: b = lambda*B(w); 열 활성: a = lambda*B(z)+phi)
    let fixed_trig: Vec<Vec<(f64, f64)>> = batch_idx
        .iter()
        .map(|&j| {
            atoms
                .iter()
                .map(|at| {
                    let b = busemann_polar(fixed[j].r, fixed[j].theta, at.theta_b);
                    let arg = if rows_active {
                        at.lambda * b
                    } else {
                        at.lambda * b + at.phi()
                    };
                    let (s, c) = arg.sin_cos();
                    (c, s)
                })
                .collect()
        })
        .collect();

    active
        .par_iter()
        .enumerate()
        .map(|(i, &zi)| {
            // 활성측 원자별 각도 성분
            let self_trig: Vec<(f64, f64)> = atoms
                .iter()
                .map(|at| {
                    let b = busemann_polar(zi.r, zi.theta, at.theta_b);
                    let arg = if rows_active {
                        at.lambda * b + at.phi()
                    } else {
                        at.lambda * b
                    };
                    let (s, c) = arg.sin_cos();
                    (c, s)
                })
                .collect();

            // 배치 잔차와 원자별 잔차 가중 삼각합 C_k, S_k
            let mut cs = vec![(0.0f64, 0.0f64); jn];
            for (bi, &j) in batch_idx.iter().enumerate() {
                let wij = if rows_active {
                    w[i * n + j]
                } else {
                    w[j * n + i]
                };
                let ft = &fixed_trig[bi];
                // K_ij = sum_k A [cos(a)cos(b) + sin(a)sin(b)]  (a: phi 포함 측)
                let mut kij = 0.0;
                for k in 0..jn {
                    let (ca, sa) = self_trig[k];
                    let (cb, sb) = ft[k];
                    kij += atoms[k].amp() * (ca * cb + sa * sb);
                }
                let r_ij = wij - kij;
                for k in 0..jn {
                    let (cb, sb) = ft[k];
                    cs[k].0 += r_ij * cb;
                    cs[k].1 += r_ij * sb;
                }
            }

            // dL/dB(활성)_k 를 모아 dB/dz 로 사상.
            // 행 활성: dL/dBz_k = 2 A lam [sin(a) C_k - cos(a) S_k]
            // 열 활성: dL/dBw_k = -2 A lam [cos(b) S'_k - sin(b) C'_k]  (부호 반대)
            let (x, y) = (zi.r * zi.theta.cos(), zi.r * zi.theta.sin());
            let (mut gx, mut gy) = (0.0, 0.0);
            let inv_b = 1.0 / batch_idx.len() as f64;
            for k in 0..jn {
                let (ca, sa) = self_trig[k];
                let (ck, sk) = cs[k];
                let coef = 2.0 * atoms[k].amp() * atoms[k].lambda * inv_b;
                let d = if rows_active {
                    coef * (sa * ck - ca * sk)
                } else {
                    -coef * (ca * sk - sa * ck)
                };
                let (bx, by) = busemann_grad_xy(x, y, atoms[k].theta_b);
                gx += d * bx;
                gy += d * by;
            }
            riemannian_step(zi, gx, gy, lr)
        })
        .collect()
}

/// 교대 최적화 (15.6절): 퍼슈트 -> 좌표 SGD -> 재적합, keep-best.
/// 라운드 0 은 SVD 초기 좌표의 퍼슈트 (학습 없음 기준선보다 항상 같거나 좋음).
pub fn fit_alternating(w: &[f64], m: usize, n: usize, cfg: &LearnConfig) -> PursuitResult {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    let (rows0, cols0) = svd_init_coords(w, m, n);
    let mut best = fit_matching_pursuit_with_coords(w, m, n, &cfg.pursuit, rows0, cols0);
    let mut codec = best.codec.clone();

    for _ in 0..cfg.rounds {
        for step in 0..cfg.sgd_steps {
            let rows_active = step % 2 == 0;
            let pool = if rows_active { n } else { m };
            let bsz = cfg.batch.min(pool);
            let batch: Vec<usize> = (0..bsz).map(|_| rng.gen_range(0..pool)).collect();
            let updated = coord_sgd_step(w, m, n, &codec, rows_active, &batch, cfg.lr);
            if rows_active {
                codec.rows = updated;
            } else {
                codec.cols = updated;
            }
        }
        let refit = fit_matching_pursuit_with_coords(
            w,
            m,
            n,
            &cfg.pursuit,
            codec.rows.clone(),
            codec.cols.clone(),
        );
        let c_new = refit.c_curve.last().copied().unwrap_or(0.0);
        let c_best = best.c_curve.last().copied().unwrap_or(0.0);
        codec = refit.codec.clone();
        if c_new > c_best {
            best = refit;
        }
    }
    best
}

/// 전 theta_b 병렬 스캔으로 전역 최선 후보 선택 (점수 0 이하면 None)
fn select_candidate(
    res: &[f64],
    bz: &[Vec<f64>],
    bw: &[Vec<f64>],
    lambdas: &[f64],
) -> Option<(usize, usize)> {
    let best = (0..bz.len())
        .into_par_iter()
        .map(|ti| {
            let (score, li) = best_lambda_for_theta(res, &bz[ti], &bw[ti], lambdas);
            (score, ti, li)
        })
        .reduce(|| (0.0, usize::MAX, 0), |a, b| if b.0 > a.0 { b } else { a });
    (best.1 != usize::MAX && best.0 > 0.0).then_some((best.1, best.2))
}

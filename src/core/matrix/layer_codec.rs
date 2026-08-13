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

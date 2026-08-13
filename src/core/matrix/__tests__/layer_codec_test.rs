//! CP-6: 레이어 코덱 검증 (하네스 2·4절)
//! L2: 분리 인수분해 동치 (Higham + cos 조건수 유도 상계), 게이지 불변성
//! L0: FLOP/바이트 카운터의 닫힌형 공식 일치 (정수 assert_eq)

use crate::core::math::busemann::{busemann_polar, Mobius};
use crate::core::math::verification::bounds;
use crate::core::matrix::layer_codec::{pack_atom, unpack_atom};
use crate::core::matrix::layer_codec::{KernelAtom, LatentCoord, LayerCodec, OpCounter};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

const R_MAX: f64 = 0.9;
const LAMBDA_MAX: f64 = 20.0;

fn random_layer(rng: &mut StdRng, m: usize, n: usize, j: usize) -> LayerCodec {
    let coord = |rng: &mut StdRng| LatentCoord {
        r: rng.gen_range(0.0..R_MAX),
        theta: rng.gen_range(0.0..2.0 * PI),
    };
    LayerCodec {
        rows: (0..m).map(|_| coord(rng)).collect(),
        cols: (0..n).map(|_| coord(rng)).collect(),
        atoms: (0..j)
            .map(|_| KernelAtom {
                theta_b: rng.gen_range(0.0..2.0 * PI),
                lambda: rng.gen_range(0.0..LAMBDA_MAX),
                phi_q: rng.r#gen::<u32>() & ((1 << 20) - 1),
                log2_amp: rng.gen_range(-2.0..1.0f64),
            })
            .collect(),
    }
}

/// 커널 항목별 유도 상계: 직접 경로와 인수분해 경로의 반올림 차.
/// 각 원자에서 cos 인자 크기 lambda*(|B_z|+|B_w|) 의 조건수(u*|arg|)가 지배하고 (CP-5 발견),
/// 인수분해 내적에는 Higham gamma_{2J} 가 |항| 합에 상대적으로 붙는다.
fn entry_bound(layer: &LayerCodec, z: LatentCoord, w: LatentCoord) -> f64 {
    let mut per_atom = 0.0;
    let mut sum_abs = 0.0;
    for a in &layer.atoms {
        let bz = busemann_polar(z.r, z.theta, a.theta_b).abs();
        let bw = busemann_polar(w.r, w.theta, a.theta_b).abs();
        let arg = a.lambda * (bz + bw) + 2.0 * PI;
        per_atom += a.amp() * (arg + 24.0) * 2.0 * bounds::U64;
        sum_abs += a.amp() * 2.0; // |cos cos| + |sin sin| <= 2
    }
    per_atom
        + bounds::dot_product(2 * layer.atoms.len()).max(bounds::U64) * sum_abs
        + sum_abs * bounds::f64_chain(4)
}

#[test]
fn 분리인수분해_직접평가_동치() {
    // 명제 15.2: W_ij 직접 평가 == (F_out F_in^T)_ij
    let mut rng = StdRng::seed_from_u64(0x5242_4561);
    let (m, n, j) = (24, 16, 8);
    let layer = random_layer(&mut rng, m, n, j);
    let w_direct = layer.materialize();

    // 인수분해 경로: 단위 벡터 입력으로 W 의 열을 추출
    for jj in 0..n {
        let mut x = vec![0.0; n];
        x[jj] = 1.0;
        let y = layer.forward(&x, &mut OpCounter::default());
        for ii in 0..m {
            let bound = entry_bound(&layer, layer.rows[ii], layer.cols[jj]);
            let diff = (y[ii] - w_direct[ii][jj]).abs();
            assert!(
                diff <= bound,
                "entry ({},{}): diff={:e} bound={:e} ratio={:.2}",
                ii,
                jj,
                diff,
                bound,
                diff / bound
            );
        }
    }
}

#[test]
fn 게이지_불변성() {
    // 15.7절 + 부록 C.3: 좌표·경계점을 뫼비우스로 함께 변환하면 W 불변
    // (코사이클 log|g'(b)| 가 상대 좌표차에서 정확히 상쇄 — 위상·진폭 무보정).
    // 상계: 경계점 재정규화/좌표 변환의 반올림이 B 의 theta_b 민감도 2/(1-r)^2 로 증폭.
    let mut rng = StdRng::seed_from_u64(0x5242_4562);
    let (m, n, j) = (12, 10, 6);
    let layer = random_layer(&mut rng, m, n, j);

    for _ in 0..20 {
        let ar = rng.gen_range(0.0..0.8f64);
        let at = rng.gen_range(0.0..2.0 * PI);
        let g = Mobius::new(ar * at.cos(), ar * at.sin(), rng.gen_range(0.0..2.0 * PI));
        let transformed = layer.gauge_transform(&g);

        let w0 = layer.materialize();
        let w1 = transformed.materialize();
        for ii in 0..m {
            for jj in 0..n {
                // 변환 후 좌표 반지름 (증폭 인자 산출용)
                let rz = transformed.rows[ii].r.max(transformed.cols[jj].r);
                let sens = 2.0 / ((1.0 - rz) * (1.0 - rz));
                let mut bound = 0.0;
                for a in &layer.atoms {
                    // 좌표 변환(16연산) + 경계 재정규화(8연산)의 오차가 B 민감도와 lambda 를 타고 전파
                    bound += a.amp() * a.lambda.max(1.0) * (24.0 * bounds::U64) * (sens + 8.0);
                }
                let diff = (w1[ii][jj] - w0[ii][jj]).abs();
                assert!(
                    diff <= bound,
                    "게이지 ({},{}): diff={:e} bound={:e} ratio={:.2}",
                    ii,
                    jj,
                    diff,
                    bound,
                    diff / bound
                );
            }
        }
    }
}

#[test]
fn 연산량_카운터_공식일치() {
    // L0: 결정론적 카운터가 논문 15.4절 닫힌형과 정수로 일치 (벽시계 게이트 금지)
    let mut rng = StdRng::seed_from_u64(0x5242_4563);
    for (m, n, j) in [(24, 16, 8), (7, 5, 3), (64, 32, 16), (1, 1, 1)] {
        let layer = random_layer(&mut rng, m, n, j);
        let x = vec![1.0; n];
        let mut counter = OpCounter::default();
        let _ = layer.forward(&x, &mut counter);
        assert_eq!(counter.flops, LayerCodec::forward_flops_formula(m, n, j));
        assert_eq!(counter.bytes, LayerCodec::forward_bytes_formula(m, n, j));
    }
}

#[test]
fn 순전파_직접행렬_동치() {
    // y = F_out(F_in^T x) == W_direct * x (일반 입력)
    let mut rng = StdRng::seed_from_u64(0x5242_4564);
    let (m, n, j) = (24, 16, 8);
    let layer = random_layer(&mut rng, m, n, j);
    let w = layer.materialize();
    let x: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0f64)).collect();

    let y = layer.forward(&x, &mut OpCounter::default());
    for ii in 0..m {
        let direct: f64 = (0..n).map(|jj| w[ii][jj] * x[jj]).sum();
        // 항목 상계를 |x_j| 로 가중 합산 + 두 경로의 내적 누적 오차
        let mut bound = 0.0;
        let mut sum_abs = 0.0;
        for jj in 0..n {
            bound += entry_bound(&layer, layer.rows[ii], layer.cols[jj]) * x[jj].abs();
            sum_abs += w[ii][jj].abs() * x[jj].abs();
        }
        bound += (bounds::dot_product(n) + bounds::dot_product(2 * j)) * sum_abs;
        let diff = (y[ii] - direct).abs();
        assert!(
            diff <= bound,
            "행 {}: diff={:e} bound={:e} ratio={:.2}",
            ii,
            diff,
            bound,
            diff / bound
        );
    }
}

#[test]
fn 직렬화_왕복_멱등성_비트동일성() {
    // L0 (CP-8 선행): 한 번 양자화된 층은 pack -> unpack -> pack 이 비트 동일해야 한다.
    // 좌표는 f32 그대로, 원자는 격자 재양자화가 항등이 되는지의 검증이다.
    let mut rng = StdRng::seed_from_u64(0x5242_4581);
    let (m, n, j) = (16, 12, 8);
    let layer = random_layer(&mut rng, m, n, j);

    let bits1 = layer.to_bits();
    assert_eq!(
        bits1.len() as u64 * 64,
        LayerCodec::code_bits_formula(m, n, j)
    );

    let layer_q = LayerCodec::from_bits(m, n, j, &bits1);
    let bits2 = layer_q.to_bits();
    assert_eq!(bits1, bits2, "직렬화 멱등성 위반");

    // 양자화된 층의 W 는 결정론적으로 재현되어야 한다 (f64 비트 동일)
    let w1 = layer_q.materialize();
    let w2 = LayerCodec::from_bits(m, n, j, &bits2).materialize();
    for (r1, r2) in w1.iter().zip(&w2) {
        for (a, b) in r1.iter().zip(r2) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

#[test]
fn 원자_격자값_왕복_정확성() {
    // L0: 격자 위의 원자 파라미터는 pack/unpack 후 f64 비트까지 동일해야 한다.
    let mut rng = StdRng::seed_from_u64(0x5242_4582);
    for _ in 0..10_000 {
        let grid_atom = KernelAtom {
            theta_b: 2.0 * PI * rng.gen_range(0u64..4096) as f64 / 4096.0,
            lambda: rng.gen_range(0u64..1024) as f64 / 16.0,
            phi_q: rng.r#gen::<u32>() & ((1 << 20) - 1),
            log2_amp: (rng.gen_range(0u64..65536) as f64 - 32768.0) / 2048.0,
        };
        let unpacked = unpack_atom(pack_atom(&grid_atom));
        assert_eq!(grid_atom.theta_b.to_bits(), unpacked.theta_b.to_bits());
        assert_eq!(grid_atom.lambda.to_bits(), unpacked.lambda.to_bits());
        assert_eq!(grid_atom.phi_q, unpacked.phi_q);
        assert_eq!(grid_atom.log2_amp.to_bits(), unpacked.log2_amp.to_bits());
    }
}

#[test]
fn 부호길이_압축률_공식일치() {
    // L0: 논문 15.5절 산수 — kogpt2 FFN (3072x768, J=512) 사례의 비트 회계
    let bits = LayerCodec::code_bits_formula(3072, 768, 512);
    assert_eq!(bits, 64 * (3072 + 768 + 512));
    // 압축률 = 32*M*N / bits (참값 대조: 분자 75,497,472 / 분모 278,528)
    let ratio = LayerCodec::compression_ratio_vs_f32(3072, 768, 512);
    assert_eq!(
        ratio.to_bits(),
        ((32u64 * 3072 * 768) as f64 / (64u64 * (3072 + 768 + 512)) as f64).to_bits()
    );
}

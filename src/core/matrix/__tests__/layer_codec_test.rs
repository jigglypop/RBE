//! CP-6: 레이어 코덱 검증 (하네스 2·4절)
//! L2: 분리 인수분해 동치 (Higham + cos 조건수 유도 상계), 게이지 불변성
//! L0: FLOP/바이트 카운터의 닫힌형 공식 일치 (정수 assert_eq)

use crate::core::math::busemann::{busemann_polar, Mobius};
use crate::core::math::phase_state::PhaseState;
use crate::core::math::verification::{bounds, check};
use crate::core::matrix::layer_codec::{
    fit_alternating, fit_matching_pursuit, fit_matching_pursuit_with_coords, spiral_coords,
    svd_init_coords, LearnConfig, PursuitConfig,
};
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
fn 적합기_다양체위_왕복_양자화바닥일치() {
    // 하네스 6절 L4-1 (합성 게이트): 격자 위 (theta_b, lambda) + 격자 밖 (phi, A) 의
    // 진짜 파라미터로 합성한 장을, 같은 파라미터의 양자화(위상 20b + 로그진폭 16b)로
    // 부호화해 디코딩하면 오차는 7.3절 양자화 바닥 이내여야 한다.
    // 인코더 탐색의 전역성 문제와 분리된, 코덱 자체의 정합성 게이트다.
    let mut rng = StdRng::seed_from_u64(0x5242_4583);
    let (m, n, j) = (48, 32, 12);
    let rows = spiral_coords(m);
    let cols = spiral_coords(n);

    let params: Vec<(f64, f64, f64, f64)> = (0..j)
        .map(|_| {
            (
                2.0 * PI * rng.gen_range(0u64..4096) as f64 / 4096.0,
                rng.gen_range(1u64..=128) as f64 / 16.0,
                rng.gen_range(0.0..2.0 * PI),      // phi: 격자 밖
                rng.gen_range(-2.0..1.0f64).exp2(), // A: 격자 밖
            )
        })
        .collect();

    let atoms: Vec<KernelAtom> = params
        .iter()
        .map(|&(tb, lm, phi, amp)| {
            let a = KernelAtom {
                theta_b: tb,
                lambda: lm,
                phi_q: PhaseState::quantize_phase(phi),
                log2_amp: amp.log2(),
            };
            unpack_atom(pack_atom(&a))
        })
        .collect();
    let codec = LayerCodec {
        rows: rows.clone(),
        cols: cols.clone(),
        atoms,
    };

    // 바닥 상계 (7.3절 합성): 원자별 A * [진폭 상대 반스텝 e^amp_quant - 1 + 위상 반스텝]
    let floor: f64 = params
        .iter()
        .map(|&(_, _, _, amp)| {
            amp * (bounds::amp_quant(11).exp_m1() + bounds::phase_quant(20) + bounds::f64_chain(8))
        })
        .sum();

    for (i, &z) in rows.iter().enumerate() {
        for (jj, &w) in cols.iter().enumerate() {
            let truth: f64 = params
                .iter()
                .map(|&(tb, lm, phi, amp)| {
                    let d = busemann_polar(z.r, z.theta, tb) - busemann_polar(w.r, w.theta, tb);
                    amp * (lm * d + phi).cos()
                })
                .sum();
            let decoded = codec.eval_direct(z, w);
            check(
                &format!("다양체 왕복 바닥 ({},{})", i, jj),
                (decoded - truth).abs(),
                floor,
            );
        }
    }
}

#[test]
fn 적합기_수렴_잔차부기_정합성() {
    // 매칭 퍼슈트 불변식 검증 (L4-2 의 수렴 RMSE 는 비게이트 — 보고만):
    // (1) c 곡선 단조 비감소 (에너지 증가 원자는 되돌리는 구현 규약)
    // (2) 적합된 원자는 전부 직렬화 격자 위 (pack -> unpack -> pack 비트 동일)
    // (3) 잔차 부기 정합: W - K(직접 평가) == 유지된 잔차 (f64 전파 상계 이내)
    let mut rng = StdRng::seed_from_u64(0x5242_4584);
    let (m, n, true_j) = (48, 32, 6);
    let rows = spiral_coords(m);
    let cols = spiral_coords(n);
    let cfg = PursuitConfig {
        n_theta: 32,
        n_lambda: 64,
        n_atoms: 16,
    };

    // 참 장: 탐색 격자 부분집합 위의 원자들 (적합 가능성 보장)
    let true_atoms: Vec<KernelAtom> = (0..true_j)
        .map(|_| {
            let a = KernelAtom {
                theta_b: 2.0 * PI * ((rng.gen_range(0usize..cfg.n_theta) * 4096 / cfg.n_theta) as f64)
                    / 4096.0,
                lambda: rng.gen_range(1u64..=cfg.n_lambda as u64) as f64 / 16.0,
                phi_q: rng.r#gen::<u32>() & ((1 << 20) - 1),
                log2_amp: rng.gen_range(-1.0..0.5f64),
            };
            unpack_atom(pack_atom(&a))
        })
        .collect();
    let truth = LayerCodec {
        rows: rows.clone(),
        cols: cols.clone(),
        atoms: true_atoms,
    };
    let w = truth.materialize_flat();

    let fit = fit_matching_pursuit(&w, m, n, &cfg);

    for pair in fit.c_curve.windows(2) {
        assert!(pair[1] >= pair[0], "c 곡선 단조성 위반: {:?}", pair);
    }
    for a in &fit.codec.atoms {
        let bits = pack_atom(a);
        assert_eq!(bits, pack_atom(&unpack_atom(bits)), "격자 이탈 원자");
    }

    let k = fit.codec.materialize_flat();
    for i in 0..m {
        for jj in 0..n {
            let idx = i * n + jj;
            let bound = entry_bound(&fit.codec, rows[i], cols[jj])
                + entry_bound(&truth, rows[i], cols[jj]);
            check(
                "잔차 부기 정합",
                (w[idx] - k[idx] - fit.residual[idx]).abs(),
                bound.max(bounds::U64 * fit.codec.atoms.len() as f64),
            );
        }
    }

    let e0: f64 = w.iter().map(|x| x * x).sum();
    let ef: f64 = fit.residual.iter().map(|x| x * x).sum();
    println!(
        "[보고] 적합 수렴 (비게이트): 원자 {}개, c = {:.4}, 잔차/전체 에너지 = {:.3e}/{:.3e}",
        fit.codec.atoms.len(),
        1.0 - ef / e0,
        ef,
        e0
    );
}

#[test]
fn 좌표학습_기준선보존_개선_보고() {
    // 15.6절 (a) 교대 최적화의 불변식 검증:
    // (1) keep-best: fit_alternating 의 c 는 라운드 0 (SVD 초기 좌표 퍼슈트) 의 c 보다
    //     항상 크거나 같다 (결정론이므로 정확 비교 가능)
    // (2) 개선 실측 보고 (비게이트): 고정 나선 기준선 대비 c 상승 폭
    // 참 장은 임의 좌표(나선 아님)의 격자 원자 합성 — 좌표 학습이 필요한 상황을 만든다.
    let mut rng = StdRng::seed_from_u64(0x5242_4585);
    let (m, n, true_j) = (48, 32, 6);
    let coord = |rng: &mut StdRng| LatentCoord {
        r: (rng.gen_range(0.0..0.85f64) as f32) as f64,
        theta: (rng.gen_range(0.0..2.0 * PI) as f32) as f64,
    };
    let truth = LayerCodec {
        rows: (0..m).map(|_| coord(&mut rng)).collect(),
        cols: (0..n).map(|_| coord(&mut rng)).collect(),
        atoms: (0..true_j)
            .map(|_| {
                let a = KernelAtom {
                    theta_b: 2.0 * PI * ((rng.gen_range(0usize..32) * 4096 / 32) as f64) / 4096.0,
                    lambda: rng.gen_range(1u64..=64) as f64 / 16.0,
                    phi_q: rng.r#gen::<u32>() & ((1 << 20) - 1),
                    log2_amp: rng.gen_range(-1.0..0.5f64),
                };
                unpack_atom(pack_atom(&a))
            })
            .collect(),
    };
    let w = truth.materialize_flat();

    let pursuit = PursuitConfig {
        n_theta: 32,
        n_lambda: 64,
        n_atoms: 12,
    };
    let cfg = LearnConfig {
        pursuit,
        rounds: 3,
        sgd_steps: 60,
        batch: 16,
        lr: 0.05,
        seed: 0x5242_4586,
    };

    let c_of = |r: &crate::core::matrix::layer_codec::PursuitResult| {
        r.c_curve.last().copied().unwrap_or(0.0)
    };
    let base_spiral = fit_matching_pursuit(&w, m, n, &pursuit);
    let (r0, c0) = svd_init_coords(&w, m, n);
    let base_svd = fit_matching_pursuit_with_coords(&w, m, n, &pursuit, r0, c0);
    let learned = fit_alternating(&w, m, n, &cfg);

    assert!(
        c_of(&learned) >= c_of(&base_svd),
        "keep-best 위반: 학습 c {} < 라운드0 c {}",
        c_of(&learned),
        c_of(&base_svd)
    );
    println!(
        "[보고] 좌표학습 (비게이트): 나선 c = {:.4}, SVD초기 c = {:.4}, 교대학습 c = {:.4}",
        c_of(&base_spiral),
        c_of(&base_svd),
        c_of(&learned)
    );
}

#[test]
#[ignore = "모델 파일 필요 (하네스 8절 L4: 명시 실행)"] // lint-allow: L4 는 모델 존재 시 명시 실행이 규약
fn 실층_좌표학습_c실측_wpe_cfc() {
    // 돌파구 실측 (L4-3, 보고 목적): SVD 진단이 가리킨 두 지점 —
    // (1) wpe (진짜 장: 동일비트 저랭크 상한 c ~ 0.90) 에서 좌표 학습의 c 도달치
    // (2) c_fc (상한 c ~ 0.09) 에서 고정 좌표 대비 갭 축소 폭
    let data = std::fs::read("models/skt-kogpt2-base-v2/model.safetensors").expect("모델 필요");
    let st = safetensors::SafeTensors::deserialize(&data).expect("파싱 실패");
    let load = |name: &str| -> (Vec<f64>, usize, usize) {
        let t = st.tensor(name).expect("텐서 없음");
        let sh = t.shape().to_vec();
        let v = t
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
            .collect();
        (v, sh[0], sh[1])
    };

    let pursuit = PursuitConfig {
        n_theta: 64,
        n_lambda: 128,
        n_atoms: 512,
    };
    let cfg = LearnConfig {
        pursuit,
        rounds: 6,
        sgd_steps: 300,
        batch: 96,
        lr: 0.05,
        seed: 0x5242_4587,
    };

    for name in ["transformer.wpe.weight", "transformer.h.0.mlp.c_fc.weight"] {
        let (w, m, n) = load(name);
        let t0 = std::time::Instant::now();
        let base = fit_matching_pursuit(&w, m, n, &pursuit);
        let learned = fit_alternating(&w, m, n, &cfg);
        let c_b = base.c_curve.last().copied().unwrap_or(0.0);
        let c_l = learned.c_curve.last().copied().unwrap_or(0.0);
        println!(
            "[보고] {} ({}x{}): 나선 c = {:.5} -> 좌표학습 c = {:.5} ({:.1}배, {:.0}s)",
            name,
            m,
            n,
            c_b,
            c_l,
            if c_b > 0.0 { c_l / c_b } else { f64::INFINITY },
            t0.elapsed().as_secs_f64()
        );
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

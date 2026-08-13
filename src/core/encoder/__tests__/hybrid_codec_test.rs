//! CP-7: L3 통계·정보이론 무결성 가드 (하네스 5절)
//! 원칙: 양측 판정 — 이론보다 나빠도 실패, 좋아도 실패.
//! "이론보다 좋은" 결과는 축하가 아니라 측정/구현 버그의 증거다 (샤논 하한 가드).

use crate::core::encoder::hybrid_codec::{energy_capture, hybrid_roundtrip, LloydMaxQuantizer};
use crate::core::math::verification::bounds;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// 정규 표본 (Box-Muller, 고정 시드 전용)
fn gaussian_sample(rng: &mut StdRng, n: usize) -> Vec<f64> {
    (0..n)
        .map(|_| {
            let u1: f64 = rng.gen_range(f64::EPSILON..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect()
}

/// 왜곡 추정치의 상대 신뢰 밴드 (양측):
/// 표본당 e^2 의 분산 <= 15 * D^2 (가우시안 4차 모멘트 3D^2 에 셀 꼬리 여유 5배),
/// 5-시그마 정책 -> band = 5 * sqrt(15 / n)
fn distortion_ci_rel(n: usize) -> f64 {
    5.0 * (15.0 / n as f64).sqrt()
}

#[test]
fn 양자화기_왜곡_이론일치_양측() {
    // Lloyd-Max 해석 왜곡(D = 1 - sum p l^2) vs 몬테카를로 실측 — 양측 판정
    let mut rng = StdRng::seed_from_u64(0x5242_4571);
    let n = 4_000_000;
    for bits in 1..=4u32 {
        let q = LloydMaxQuantizer::new_gaussian(bits);
        let sample = gaussian_sample(&mut rng, n);
        let mse: f64 = sample
            .iter()
            .map(|&x| {
                let e = x - q.index_to_value(q.quantize(x));
                e * e
            })
            .sum::<f64>()
            / n as f64;
        let rel = (mse - q.distortion_rel).abs() / q.distortion_rel;
        let band = distortion_ci_rel(n);
        assert!(
            rel <= band,
            "b={} 왜곡 불일치(양측): 실측={:.6} 이론={:.6} rel={:.4} band={:.4}",
            bits,
            mse,
            q.distortion_rel,
            rel,
            band
        );
    }
}

#[test]
fn 양자화기_문헌값_대조() {
    // Max(1960) 표의 가우시안 Lloyd-Max 왜곡 (유효 4자리). 밴드는 표의 반올림 여유 + 수렴 잔차.
    let published = [(1u32, 0.3634), (2, 0.1175), (3, 0.03454), (4, 0.009497)];
    for (bits, d_pub) in published {
        let q = LloydMaxQuantizer::new_gaussian(bits);
        let rel = (q.distortion_rel - d_pub).abs() / d_pub;
        assert!(
            rel <= 5e-3,
            "b={} 문헌값 대조 실패: 계산={:.6} 문헌={:.6} rel={:.4}",
            bits,
            q.distortion_rel,
            d_pub,
            rel
        );
    }
}

#[test]
fn 샤논하한_가드_준수() {
    // 부록 D.1: b비트 양자화 실측 RMSE 는 샤논 하한 이상이어야 한다.
    // (하한 미만 = 정보이론 위반 = 측정/구현 버그 판정)
    let mut rng = StdRng::seed_from_u64(0x5242_4572);
    let n = 100_000;
    let w = gaussian_sample(&mut rng, n);
    let sigma = (w.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
    let zeros = vec![0.0; n];
    for bits in 1..=4u32 {
        let q = LloydMaxQuantizer::new_gaussian(bits);
        let (_, rmse) = hybrid_roundtrip(&w, &zeros, &q);
        let floor = bounds::shannon_floor(sigma, bits as f64, n);
        assert!(
            rmse >= floor,
            "샤논 하한 위반 검출: b={} rmse={:.5} floor={:.5} — 인코더/측정 버그",
            bits,
            rmse,
            floor
        );
    }
}

#[test]
#[should_panic(expected = "샤논 하한 위반")]
fn 무결성가드_조작검출() {
    // 가드의 검출력 자체를 검증: "2 bpw 로 무손실 재현" 이라는 조작된 결과는
    // 하한 미만이므로 반드시 실패해야 한다.
    let sigma = 1.0;
    let fabricated_rmse = 0.0; // 조작: 원본을 그대로 돌려주며 2 bpw 라 주장
    let floor = bounds::shannon_floor(sigma, 2.0, 100_000);
    assert!(
        fabricated_rmse >= floor,
        "샤논 하한 위반 검출: rmse={:.5} floor={:.5}",
        fabricated_rmse,
        floor
    );
}

#[test]
fn 분산축소_손익분기_이론식일치() {
    // 논문 17.3절: W = sqrt(c) S + sqrt(1-c) E 에서 구조부 K = sqrt(c) S 를 알 때
    // 하이브리드 RMSE == q(b) * sigma * sqrt(1-c)  (이론식 자체가 오라클, 양측)
    let mut rng = StdRng::seed_from_u64(0x5242_4573);
    let n = 200_000;
    // 구조부: 결정론적 단위 에너지 패턴 (저주파 코사인)
    let s: Vec<f64> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * 3.0 * i as f64 / n as f64).cos())
        .collect();
    let s_norm = (s.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
    let s: Vec<f64> = s.iter().map(|x| x / s_norm).collect();
    let e = gaussian_sample(&mut rng, n);

    for c in [0.0f64, 0.5, 0.9] {
        for bits in [2u32, 3] {
            let q = LloydMaxQuantizer::new_gaussian(bits);
            let w: Vec<f64> = s
                .iter()
                .zip(&e)
                .map(|(si, ei)| c.sqrt() * si + (1.0 - c).sqrt() * ei)
                .collect();
            let k: Vec<f64> = s.iter().map(|si| c.sqrt() * si).collect();

            // c 추정기 보정
            let c_hat = energy_capture(&w, &k);
            let c_band = 5.0 * (2.0 * (c * (1.0 - c)).sqrt() + (1.0 - c)) / (n as f64).sqrt();
            assert!(
                (c_hat - c).abs() <= c_band.max(5.0 / (n as f64).sqrt()),
                "c 추정 편차: c={} c_hat={:.5} band={:.5}",
                c,
                c_hat,
                c_band
            );

            // 하이브리드 RMSE 의 이론 예측 (17.3절) 과 양측 대조
            let sigma_w = (w.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
            let (_, rmse) = hybrid_roundtrip(&w, &k, &q);
            let predicted = q.distortion_rel.sqrt() * sigma_w * (1.0 - c).sqrt();
            if c < 1.0 {
                let rel = (rmse - predicted).abs() / predicted.max(f64::MIN_POSITIVE);
                let band = distortion_ci_rel(n);
                assert!(
                    rel <= band,
                    "손익분기 이론식 불일치(양측): c={} b={} rmse={:.5} 예측={:.5} rel={:.4} band={:.4}",
                    c,
                    bits,
                    rmse,
                    predicted,
                    rel,
                    band
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// L4 종단 (하네스 6절, CP-8 본체): skt-kogpt2 실가중치.
// 모델 파일이 있어야 하므로 #[ignore] + 명시 실행 (하네스 8절 규약):
//   cargo test --release --lib -- --ignored 실층
// ---------------------------------------------------------------------------

/// kogpt2 층0 FFN c_fc 가중치 로드 (768 x 3072, f32). 실가중치만 허용 (nlp-verify).
fn kogpt2_ffn_weight() -> (Vec<f64>, Vec<f32>, usize, usize) {
    let path = std::path::Path::new("models/skt-kogpt2-base-v2/model.safetensors");
    assert!(
        path.exists(),
        "L4 는 실가중치 필수: {} 가 없다 (다운로드 후 재실행)",
        path.display()
    );
    let data = std::fs::read(path).expect("safetensors 읽기 실패");
    let st = safetensors::SafeTensors::deserialize(&data).expect("safetensors 파싱 실패");
    let t = st
        .tensor("transformer.h.0.mlp.c_fc.weight")
        .expect("c_fc 텐서 없음");
    assert_eq!(t.dtype(), safetensors::Dtype::F32);
    let shape = t.shape().to_vec();
    assert_eq!(shape, vec![768, 3072]);
    let w32: Vec<f32> = t
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let w64 = w32.iter().map(|&x| x as f64).collect();
    (w64, w32, shape[0], shape[1])
}

#[test]
#[ignore = "모델 파일 필요 (하네스 8절 L4: 명시 실행)"] // lint-allow: L4 는 모델 존재 시 명시 실행이 규약
fn 실층_c측정_하이브리드예측_candle_순전파_대조() {
    use crate::core::math::verification::{bounds, check};
    use crate::core::matrix::layer_codec::{fit_matching_pursuit, LayerCodec, PursuitConfig};
    use candle_core::{Device, Tensor};

    let (w, w32, m, n) = kogpt2_ffn_weight();
    let mn = m * n;
    let sigma = (w.iter().map(|x| x * x).sum::<f64>() / mn as f64).sqrt();

    // (1) 원자 적합과 c 실측 (L4-3: 측정이 목적, 게이트 없음)
    let cfg = PursuitConfig {
        n_theta: 64,
        n_lambda: 128,
        n_atoms: 512,
    };
    let fit = fit_matching_pursuit(&w, m, n, &cfg);
    let j = fit.codec.atoms.len();
    let c = fit.c_curve.last().copied().unwrap_or(0.0);
    for &jj in &[64usize, 128, 256] {
        if jj <= j {
            println!("[보고] c(J={}) = {:.6}", jj, fit.c_curve[jj - 1]);
        }
    }
    println!(
        "[보고] 실층 c 실측: J={}, c = {:.6}, 원자부 압축률 = {:.1}:1, sigma = {:.5e}",
        j,
        c,
        LayerCodec::compression_ratio_vs_f32(m, n, j),
        sigma
    );

    // (2) 하이브리드 R2a (원자 + 잔차 2비트): 17.3절 예측식과 양측 대조 (L4-4)
    let k: Vec<f64> = w.iter().zip(&fit.residual).map(|(a, r)| a - r).collect();
    let q = LloydMaxQuantizer::new_gaussian(2);
    let (recon, rmse) = hybrid_roundtrip(&w, &k, &q);
    let predicted = q.distortion_rel.sqrt() * sigma * (1.0 - c).sqrt();
    let band = bounds::rmse_ci_rel(mn);
    let bpw = (LayerCodec::code_bits_formula(m, n, j) as f64 + 2.0 * mn as f64 + 64.0) / mn as f64;
    println!(
        "[보고] 하이브리드 2bpw: 실측 RMSE = {:.6e}, 예측(17.3절) = {:.6e}, 비 = {:.4}, 밴드 = {:.4}, bpw = {:.3}",
        rmse,
        predicted,
        rmse / predicted,
        band,
        bpw
    );
    check("하이브리드 실측 <= 예측 상단", rmse, predicted * (1.0 + band));
    check("하이브리드 실측 >= 예측 하단", predicted * (1.0 - band), rmse);

    // (3) candle 순전파 대조 (L4-5): 상계는 |dy| <= |dW|_F |x| 전파식에서 유도
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0x5242_4585);
    let x: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0f64)).collect();
    let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();

    let y_ref: Vec<f64> = (0..m)
        .map(|i| (0..n).map(|jj| w[i * n + jj] * x[jj]).sum())
        .collect();

    let dev = Device::Cpu;
    let wt = Tensor::from_slice(&w32, (m, n), &dev).unwrap();
    let xt = Tensor::from_slice(&x32, (n, 1), &dev).unwrap();
    let y_candle: Vec<f32> = wt
        .matmul(&xt)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for i in 0..m {
        let sum_abs: f64 = (0..n).map(|jj| (w[i * n + jj] * x[jj]).abs()).sum();
        check(
            "candle f32 순전파 == f64 기준 (Higham)",
            (y_candle[i] as f64 - y_ref[i]).abs(),
            bounds::dot_product(n) * sum_abs + bounds::U32 * y_ref[i].abs(),
        );
    }

    let y_rbe: Vec<f64> = (0..m)
        .map(|i| (0..n).map(|jj| recon[i * n + jj] * x[jj]).sum())
        .collect();
    let dw_frob = (w
        .iter()
        .zip(&recon)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>())
    .sqrt();
    let x_norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    let dy_norm = y_ref
        .iter()
        .zip(&y_rbe)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    println!(
        "[보고] 압축 순전파: ||dy|| = {:.5e}, 상계 ||dW||_F ||x|| = {:.5e}, 상대 출력 오차 = {:.4}",
        dy_norm,
        dw_frob * x_norm,
        dy_norm / y_ref.iter().map(|v| v * v).sum::<f64>().sqrt()
    );
    check(
        "압축 순전파 오차 전파 상계",
        dy_norm,
        dw_frob * x_norm * (1.0 + bounds::f64_chain(n as u32)),
    );
}

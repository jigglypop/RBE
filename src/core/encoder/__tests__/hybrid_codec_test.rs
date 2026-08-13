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

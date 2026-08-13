//! CP-1: 검증 기반 모듈 자기검증 (docs/test/verification_harness.md 10절)
//! bounds 함수들이 문서 조견표(9.4절)와 일치하고 수학적 성질을 만족하는지 확인한다.
//! 조견표 대조는 동일 식 재실행(동어반복)을 피하기 위해 대수적으로 다른 경로로 계산한다.

use crate::core::math::verification::{bounds, check, oracle};

#[test]
fn 상수_정확값_비트동일성() {
    // u32 = 2^-24, u64 = 2^-53 은 f64 로 정확히 표현되므로 비트 동일 판정
    assert_eq!(bounds::U32, 2f64.powi(-24));
    assert_eq!(bounds::U64, 2f64.powi(-53));
}

#[test]
fn 조견표_대체경로_재계산_일치() {
    use std::f64::consts::PI;
    // lut_interp(12) = (pi/2^13)^2/8 = pi^2 / 2^29 (대수 동치, 부동소수점 경로는 다름)
    let alt = PI * PI / 2f64.powi(29);
    let v = bounds::lut_interp(12);
    check(
        "lut_interp(12) 대체경로",
        (v - alt).abs() / alt,
        bounds::f64_chain(4),
    );
    // sterbenz = 2u + u^2 = (2 + u) * u : 두 경로 모두 정확 표현 가능하므로 비트 동일
    assert_eq!(
        bounds::sterbenz_product(),
        (2.0 + bounds::U32) * bounds::U32
    );
    // phase_quant(20) = pi/2^20 = 2pi / 2^21
    let alt = 2.0 * PI / 2f64.powi(21);
    let v = bounds::phase_quant(20);
    check(
        "phase_quant(20) 대체경로",
        (v - alt).abs() / alt,
        bounds::f64_chain(4),
    );
}

#[test]
fn 상계함수_단조성() {
    // LUT 상계는 k 증가에 단조 감소, 내적 상계는 n 증가에 단조 증가
    for k in 4..20 {
        assert!(bounds::lut_interp(k + 1) < bounds::lut_interp(k));
    }
    for n in [8usize, 64, 512, 4096, 65536] {
        assert!(bounds::dot_product(n * 2) > bounds::dot_product(n));
    }
    for n in [100usize, 10_000, 1_000_000] {
        assert!(bounds::rmse_ci_rel(n * 10) < bounds::rmse_ci_rel(n));
    }
}

#[test]
fn 중심차분_최적스텝_최적성() {
    // h* 가 상계를 국소 최소화하는지 이웃 스텝과 비교 (리터럴 허용오차 없음)
    for (m0, m3) in [(1.0, 1.0), (1.0, 100.0), (10.0, 0.5)] {
        let h = bounds::central_diff_h_opt(m0, m3);
        let e = bounds::central_diff(h, m0, m3);
        assert!(e <= bounds::central_diff(h * 2.0, m0, m3));
        assert!(e <= bounds::central_diff(h * 0.5, m0, m3));
    }
}

#[test]
fn 샤논가드_성질() {
    // floor 는 양수이며 sigma 미만, bpw 증가에 단조 감소
    let n = 4096;
    let f1 = bounds::shannon_floor(1.0, 1.0, n);
    let f2 = bounds::shannon_floor(1.0, 2.0, n);
    assert!(f1 > 0.0 && f1 < 1.0);
    assert!(f2 < f1);
}

#[test]
fn 오라클_부제만_라디얼_쌍곡거리_일치() {
    // 부록 F.3: b 방향 반지름 위에서 B_b(r*b) = 2*artanh(r)
    // 허용오차: 감산 상쇄 증폭을 포함해 유도된 r별 상계 (bounds::busemann_radial_oracle).
    // 오라클의 소박한 1-|z|^2 는 경계 근처에서 상쇄가 일어나므로(7.2절) 상계가 r 의존이다.
    for i in 1..1000 {
        let r = i as f64 / 1000.0 * 0.999;
        for theta_b in [0.0, 1.0, 2.5, -2.0] {
            let z = (r * f64::cos(theta_b), r * f64::sin(theta_b));
            let b_val = oracle::busemann(z, theta_b);
            let exact = 2.0 * r.atanh();
            let denom = exact.abs().max(1.0);
            check(
                "부제만 라디얼 일치",
                (b_val - exact).abs() / denom,
                bounds::busemann_radial_oracle(r),
            );
        }
    }
}

#[test]
#[should_panic(expected = "하네스 위반")]
fn 판정헬퍼_상계초과시_실패() {
    check("의도된 실패", 2.0, 1.0);
}

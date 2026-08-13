//! CP-2: 위상-비트 상태 L0 테스트 (비트 동일성, 허용오차 0)
//! docs/test/verification_harness.md 2절. 이론이 "정확"을 보장하는 연산은
//! 근사 비교를 쓰지 않고 assert_eq! 비트 비교만 사용한다.

use crate::core::math::phase_state::{
    one_minus_r_sq, PhaseState, PHASE_BITS, PHASE_MASK, PHASE_QUARTER,
};
use crate::core::math::verification::{bounds, check};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::FRAC_PI_2;

/// 경계 케이스 위상값 (사분면 경계와 랩어라운드)
const EDGE_PHASES: [u32; 8] = [
    0,
    1,
    PHASE_QUARTER - 1,
    PHASE_QUARTER,
    2 * PHASE_QUARTER,
    3 * PHASE_QUARTER,
    PHASE_MASK - 1,
    PHASE_MASK,
];

#[test]
fn 위상미분_정수덧셈_양자화가환_비트동일성() {
    // 정리 3.2: 전체 2^20 위상 전수에 대해 "연속 미분 후 양자화" == "양자화 후 정수 덧셈"
    for q in 0..=PHASE_MASK {
        let phi = PhaseState::phase_value(q);
        let lhs = PhaseState::quantize_phase(phi + FRAC_PI_2);
        let rhs = (q + PHASE_QUARTER) & PHASE_MASK;
        assert_eq!(lhs, rhs, "q={}", q);
    }
}

#[test]
fn 위상미분_4회_원상복귀_비트동일성() {
    // 따름정리 3.2.1: D^4 는 위상 성분에서 항등 (hi 전체 비트 보존)
    let mut rng = StdRng::seed_from_u64(0x5242_4531);
    let mut states: Vec<u64> = (0..10_000).map(|_| rng.r#gen()).collect();
    for &q in EDGE_PHASES.iter() {
        let mut s = PhaseState::new(0);
        s.set_phase(q);
        states.push(s.hi);
    }
    for hi in states {
        let original = PhaseState::new(hi);
        let mut s = original;
        for _ in 0..4 {
            s.differentiate_circular();
        }
        assert_eq!(s, original);
    }
}

#[test]
fn 쌍곡라벨_2회_플립_원상복귀_비트동일성() {
    // 정리 3.3: 쌍곡 트랙 미분은 Z/2 순환. 다른 필드는 불변.
    let mut rng = StdRng::seed_from_u64(0x5242_4532);
    for _ in 0..10_000 {
        let original = PhaseState::new(rng.r#gen());
        let mut s = original;
        s.differentiate_hyperbolic();
        assert_ne!(s.s_h(), original.s_h());
        assert_eq!(s.phase(), original.phase());
        assert_eq!(s.amp_code(), original.amp_code());
        s.differentiate_hyperbolic();
        assert_eq!(s, original);
    }
}

#[test]
fn 사분면_카운터_증가_하위비트보존() {
    // 따름정리 3.2.1: 미분은 상위 2비트를 mod-4 증가시키고 하위 18비트를 보존
    let mut rng = StdRng::seed_from_u64(0x5242_4533);
    let low_mask = (1u32 << (PHASE_BITS - 2)) - 1;
    for _ in 0..10_000 {
        let mut s = PhaseState::new(rng.r#gen());
        let q0 = s.quadrant();
        let low0 = s.phase() & low_mask;
        s.differentiate_circular();
        assert_eq!(s.quadrant(), (q0 + 1) % 4);
        assert_eq!(s.phase() & low_mask, low0);
    }
}

#[test]
fn 델타격자_n계미분_단일덧셈_동일성() {
    // 부록 A.3: q + n*dq (단일 덧셈) == dq 를 n 번 더한 결과 (mod 2^20)
    let mut rng = StdRng::seed_from_u64(0x5242_4534);
    for _ in 0..2_000 {
        let hi: u64 = rng.r#gen();
        let dq: u32 = rng.r#gen::<u32>() & PHASE_MASK;
        let n: u32 = rng.gen_range(0..64);
        let mut single = PhaseState::new(hi);
        single.advance_phase_n(dq, n);
        let mut repeated = PhaseState::new(hi);
        for _ in 0..n {
            repeated.advance_phase(dq);
        }
        assert_eq!(single, repeated, "dq={} n={}", dq, n);
    }
}

#[test]
fn 적분_미분_역원_비트동일성() {
    // 3.1절: 적분은 미분의 정확한 역원 (양방향)
    let mut rng = StdRng::seed_from_u64(0x5242_4535);
    for _ in 0..10_000 {
        let original = PhaseState::new(rng.r#gen());
        let mut s = original;
        s.differentiate_circular();
        s.integrate_circular();
        assert_eq!(s, original);
        s.integrate_circular();
        s.differentiate_circular();
        assert_eq!(s, original);
    }
}

#[test]
fn 스털벤츠_감산_정확성_f32_전수() {
    // 부록 E.1: r in [0.5, 1] 의 모든 f32 (2^23 + 1 개)에 대해 fl(1-r) 이 정확
    let start = 0.5f32.to_bits();
    let end = 1.0f32.to_bits();
    for bits in start..=end {
        let r = f32::from_bits(bits);
        let computed = (1.0f32 - r) as f64;
        let exact = 1.0f64 - r as f64;
        assert_eq!(computed, exact, "r={}", r);
    }
}

#[test]
fn 경계감산_상대오차_균일상계() {
    // 부록 E.2: (1-r)(1+r) 의 상대오차는 r 에 무관하게 2u + u^2 이하 (L1)
    let bound = bounds::sterbenz_product();
    let mut rng = StdRng::seed_from_u64(0x5242_4536);
    let mut worst = 0f64;
    // 경계 격자 r = 1 - 2^-j
    let mut cases: Vec<f32> = (1..=23).map(|j| 1.0 - 0.5f32.powi(j)).collect();
    cases.extend((0..100_000).map(|_| rng.gen_range(0.5f32..1.0f32)));
    for r in cases {
        let computed = one_minus_r_sq(r) as f64;
        let rd = r as f64;
        let exact = (1.0 - rd) * (1.0 + rd);
        if exact > 0.0 {
            worst = worst.max((computed - exact).abs() / exact);
        }
    }
    check("경계감산 상대오차", worst, bound);
}

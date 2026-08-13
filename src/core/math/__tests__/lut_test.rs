//! CP-3: 사분파 LUT 평가기 검증 (하네스 10절)
//! L1: 보간 오차 전수 스윕 (상계 = bounds::lut_interp, 부록 E.3 유도)
//! L0: 대칭성 정확성 (반주기 부호, 거울 대칭 — 비트 수준 동치)

use crate::core::math::lut::{cos_phase_f64, sin_phase, sin_phase_f64};
use crate::core::math::phase_state::{PhaseState, PHASE_MASK};
use crate::core::math::verification::{bounds, check};

#[test]
fn 사분파LUT_보간오차_전수_상계준수() {
    // 위상 2^20 전수를 f64 sin 오라클과 대조. 상계는 부록 E.3 유도값만 사용.
    let bound = bounds::lut_interp(12);
    let mut worst = 0f64;
    let mut worst_q = 0u32;
    for q in 0..=PHASE_MASK {
        let exact = PhaseState::phase_value(q).sin();
        let err = (sin_phase_f64(q) - exact).abs();
        if err > worst {
            worst = err;
            worst_q = q;
        }
    }
    println!(
        "[CP-3] sin LUT 전수 스윕: max|err|={:e} bound={:e} ratio={:.3} (q={})",
        worst,
        bound,
        worst / bound,
        worst_q
    );
    check("LUT 보간오차 (sin, 전수)", worst, bound);
}

#[test]
fn 코사인_위상이동_전수_상계준수() {
    // cos = sin(위상 + pi/2) 경로가 f64 cos 오라클과 같은 상계를 만족하는지
    let bound = bounds::lut_interp(12);
    let mut worst = 0f64;
    for q in 0..=PHASE_MASK {
        let exact = PhaseState::phase_value(q).cos();
        let err = (cos_phase_f64(q) - exact).abs();
        worst = worst.max(err);
    }
    check("LUT 보간오차 (cos, 전수)", worst, bound);
}

#[test]
fn f32출력_합성상계_전수() {
    // f32 래퍼: 보간 오차 + 출력 반올림 반 ulp 의 합성 상계 (bounds::lut_eval_f32)
    let bound = bounds::lut_eval_f32(12);
    let mut worst = 0f64;
    for q in 0..=PHASE_MASK {
        let exact = PhaseState::phase_value(q).sin();
        let err = (sin_phase(q) as f64 - exact).abs();
        worst = worst.max(err);
    }
    check("LUT f32 출력 합성오차 (전수)", worst, bound);
}

#[test]
fn 반주기_부호대칭_비트동일성() {
    // L0: sin(phi + pi) == -sin(phi) 는 사분면 비트 매핑상 정확 (같은 테이블 경로 + 부호)
    let half = 1u32 << 19;
    for q in 0..=PHASE_MASK {
        let a = sin_phase_f64((q + half) & PHASE_MASK);
        let b = -sin_phase_f64(q);
        assert_eq!(a.to_bits(), b.to_bits(), "q={}", q);
    }
}

#[test]
fn 거울대칭_비트동일성() {
    // L0: sin(pi - phi) == sin(phi) — 1사분면 미러링이 0사분면과 같은 테이블 경로를 타므로 정확
    let half = 1u32 << 19;
    for q in 1..(1u32 << 18) {
        let a = sin_phase_f64(half - q);
        let b = sin_phase_f64(q);
        assert_eq!(a.to_bits(), b.to_bits(), "q={}", q);
    }
}

#[test]
fn 사분면경계_연속성() {
    // 사분면 경계(0, 2^18, 2^19, 3*2^18)에서 값이 정확한 격자점 (0, ±1) 과 일치
    let quarter = 1u32 << 18;
    assert_eq!(sin_phase_f64(0), 0.0);
    assert_eq!(sin_phase_f64(quarter), 1.0);
    assert_eq!(sin_phase_f64(2 * quarter), 0.0);
    assert_eq!(sin_phase_f64(3 * quarter), -1.0);
}

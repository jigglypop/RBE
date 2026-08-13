//! 사분파(quarter-wave) sin LUT 평가기 (논문 6.1절, 하네스 CP-3)
//!
//! 위상 레지스터 q (20비트) 로 직접 구동된다:
//!   [19:18] 사분면 (미분 카운터와 동일 비트 — 따름정리 3.2.1)
//!   [17:6]  테이블 인덱스 (k = 12, 4096 구간)
//!   [5:0]   선형 보간 분수 (6비트)
//!
//! 수학적 보간 오차 상계: (1/8) (pi/2^13)^2 = 1.84e-8  [부록 E.3]
//! 코어는 f64 로 평가해 보간 오차만 남기고, f32 출력 반올림은 별도 상계
//! (bounds::lut_interp + U32/2)로 다룬다.

use super::phase_state::{PHASE_BITS, PHASE_MASK, PHASE_QUARTER};
use std::f64::consts::FRAC_PI_2;
use std::sync::OnceLock;

/// 테이블 해상도 k: 2^k 구간, [0, pi/2]
pub const LUT_BITS: u32 = 12;
/// 사분면 내 위치 비트 수 = 18
const QUAD_BITS: u32 = PHASE_BITS - 2;
/// 보간 분수 비트 수 = 6
const FRAC_BITS: u32 = QUAD_BITS - LUT_BITS;
/// 엔트리 수: 2^k + 1 (끝점 포함) + 1 (미러 경계 여유, 값은 끝점 복제)
const TABLE_LEN: usize = (1 << LUT_BITS) + 2;

fn table() -> &'static [f64; TABLE_LEN] {
    static TABLE: OnceLock<[f64; TABLE_LEN]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0.0f64; TABLE_LEN];
        for (i, item) in t.iter_mut().enumerate().take((1 << LUT_BITS) + 1) {
            *item = (FRAC_PI_2 * i as f64 / (1u32 << LUT_BITS) as f64).sin();
        }
        // 미러 경계(frac = 0 에서만 도달)의 안전 복제
        t[(1 << LUT_BITS) + 1] = t[1 << LUT_BITS];
        t
    })
}

/// sin(2*pi*q / 2^20) 을 사분파 LUT + 선형 보간으로 평가 (f64 코어)
pub fn sin_phase_f64(q: u32) -> f64 {
    let q = q & PHASE_MASK;
    let quadrant = q >> QUAD_BITS;
    let pos = q & ((1 << QUAD_BITS) - 1);
    // 홀수 사분면은 감소 구간이므로 미러링
    let mirrored = if quadrant & 1 == 1 {
        (1u32 << QUAD_BITS) - pos
    } else {
        pos
    };
    let idx = (mirrored >> FRAC_BITS) as usize;
    let frac = (mirrored & ((1 << FRAC_BITS) - 1)) as f64 / (1u32 << FRAC_BITS) as f64;
    let t = table();
    let v = t[idx] + (t[idx + 1] - t[idx]) * frac;
    if quadrant >= 2 {
        -v
    } else {
        v
    }
}

/// cos(2*pi*q / 2^20) = sin(위상 + pi/2) — 미분과 동일한 위상 이동으로 얻는다 (정리 3.2)
pub fn cos_phase_f64(q: u32) -> f64 {
    sin_phase_f64(q.wrapping_add(PHASE_QUARTER) & PHASE_MASK)
}

/// f32 출력 래퍼. 추가 오차는 출력 반올림 U32/2 뿐 (합성 상계는 bounds 참조)
pub fn sin_phase(q: u32) -> f32 {
    sin_phase_f64(q) as f32
}

pub fn cos_phase(q: u32) -> f32 {
    cos_phase_f64(q) as f32
}

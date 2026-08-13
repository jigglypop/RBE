//! 위상-비트 상태 (논문 3장·4.1절, 하네스 9.1 규범 레이아웃)
//!
//! hi 64비트 필드:
//! [63:62] trk    (2)  트랙: 00 원형, 01 쌍곡, 10 tanh, 11 예약
//! [61:42] q      (20) 위상 레지스터. phi = 2*pi*q / 2^20. 상위 2비트 = 사분면
//! [41:32] m_r    (10) 반경 주파수 지수
//! [31:26] m_psi  (6)  각 파수 (정수)
//! [25]    s_h    (1)  쌍곡 라벨 (sinh=0 / cosh=1)
//! [24]    sign   (1)  부호
//! [23:8]  a      (16) 로그 진폭 코드: A = 2^((a - 32768) / 2048)
//! [7:0]   -      (8)  예약 (0)
//!
//! 미분 규칙 (전부 정확한 정수 연산):
//! - 원형 트랙: q += 2^18 (mod 2^20)  [정리 3.2]
//! - 쌍곡 트랙: s_h ^= 1              [정리 3.3]
//! - n계 미분: q += n * dq (단일 덧셈) [부록 A.3]

use std::f64::consts::PI;

pub const PHASE_BITS: u32 = 20;
/// pi/2 에 해당하는 위상 스텝 = 2^18
pub const PHASE_QUARTER: u32 = 1 << (PHASE_BITS - 2);
pub const PHASE_MASK: u32 = (1 << PHASE_BITS) - 1;

const TRK_SHIFT: u32 = 62;
const Q_SHIFT: u32 = 42;
const MR_SHIFT: u32 = 32;
const MR_MASK: u64 = 0x3FF;
const MPSI_SHIFT: u32 = 26;
const MPSI_MASK: u64 = 0x3F;
const SH_SHIFT: u32 = 25;
const SIGN_SHIFT: u32 = 24;
const AMP_SHIFT: u32 = 8;
const AMP_MASK: u64 = 0xFFFF;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PhaseState {
    pub hi: u64,
}

impl PhaseState {
    pub fn new(hi: u64) -> Self {
        Self { hi }
    }

    pub fn track(&self) -> u8 {
        (self.hi >> TRK_SHIFT) as u8 & 0x3
    }

    pub fn phase(&self) -> u32 {
        ((self.hi >> Q_SHIFT) as u32) & PHASE_MASK
    }

    pub fn set_phase(&mut self, q: u32) {
        let q = (q & PHASE_MASK) as u64;
        self.hi = (self.hi & !((PHASE_MASK as u64) << Q_SHIFT)) | (q << Q_SHIFT);
    }

    /// 사분면 = 위상 레지스터 상위 2비트 (따름정리 3.2.1: 미분 차수 카운터 mod 4)
    pub fn quadrant(&self) -> u8 {
        (self.phase() >> (PHASE_BITS - 2)) as u8
    }

    pub fn m_r(&self) -> u16 {
        ((self.hi >> MR_SHIFT) & MR_MASK) as u16
    }

    pub fn m_psi(&self) -> u8 {
        ((self.hi >> MPSI_SHIFT) & MPSI_MASK) as u8
    }

    pub fn s_h(&self) -> u8 {
        ((self.hi >> SH_SHIFT) & 1) as u8
    }

    pub fn sign(&self) -> u8 {
        ((self.hi >> SIGN_SHIFT) & 1) as u8
    }

    pub fn amp_code(&self) -> u16 {
        ((self.hi >> AMP_SHIFT) & AMP_MASK) as u16
    }

    /// 위상 실수값 phi = 2*pi*q / 2^20
    pub fn phase_value(q: u32) -> f64 {
        2.0 * PI * (q & PHASE_MASK) as f64 / (1u64 << PHASE_BITS) as f64
    }

    /// 위상 양자화: round(phi / 2pi * 2^20) mod 2^20
    pub fn quantize_phase(phi: f64) -> u32 {
        let t = phi / (2.0 * PI) * (1u64 << PHASE_BITS) as f64;
        (t.round() as i64).rem_euclid(1i64 << PHASE_BITS) as u32
    }

    /// 위상 전진 (mod 2^20)
    pub fn advance_phase(&mut self, dq: u32) {
        let q = (self.phase().wrapping_add(dq)) & PHASE_MASK;
        self.set_phase(q);
    }

    /// n계 미분의 위상 이동을 단일 덧셈으로: q += n * dq (mod 2^20)  [부록 A.3]
    pub fn advance_phase_n(&mut self, dq: u32, n: u32) {
        self.advance_phase(dq.wrapping_mul(n) & PHASE_MASK);
    }

    /// 원형 트랙 미분: 위상 += pi/2  [정리 3.2]
    /// 진폭 스케일(x omega)은 로그 진폭 필드에서 호출자가 별도 처리한다.
    pub fn differentiate_circular(&mut self) {
        self.advance_phase(PHASE_QUARTER);
    }

    /// 원형 트랙 적분: 위상 -= pi/2 (미분의 정확한 역원)
    pub fn integrate_circular(&mut self) {
        self.advance_phase((1 << PHASE_BITS) - PHASE_QUARTER);
    }

    /// 쌍곡 트랙 미분: 라벨 플립 sinh <-> cosh  [정리 3.3]
    pub fn differentiate_hyperbolic(&mut self) {
        self.hi ^= 1 << SH_SHIFT;
    }
}

/// 경계 안정 1 - r^2 = (1-r)(1+r)  [부록 E.1 Sterbenz, E.2]
/// r in [0.5, 1] 에서 1-r 이 정확하므로 상대오차가 r 에 무관하게 2u + u^2 이하.
/// 소박한 1.0 - r*r 형태는 경계에서 파국적 상쇄를 일으키므로 금지 (하네스 9.3).
pub fn one_minus_r_sq(r: f32) -> f32 {
    (1.0 - r) * (1.0 + r)
}

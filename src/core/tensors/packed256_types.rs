//! # `Packed256` - RBE 256-bit Tensor Core
//!
//! 이 모듈은 **상태(State)**와 **연속 파라미터(Continuous Parameter)**를 완벽히 분리하는
//! 새로운 `Packed256` 비트 필드 설계를 제공합니다.
//! 이 설계는 레거시 시스템의 핵심 철학을 계승하여 수치적 안정성과 코드 명확성을 목표로 합니다.
//!
//! ## 설계 철학
//! - **`hi: u128` (상태 및 제어 필드)**: 기저 함수, 미분 차수, 곡률 등 이산적인 상태를 제어.
//! - **`lo: u128` (고정밀 연속 파라미터 필드)**: 좌표, 주파수 등 연속적인 값을 고정소수점으로 표현.

use serde::{Deserialize, Serialize};
use rand::Rng;
use std::f32::consts::PI;

// --- 비트 필드 상수 정의 ---

// lo: u128 (Continuous Parameters)
const R_BITS: u32 = 32;
const THETA_BITS: u32 = 32;
const PARAM1_BITS: u32 = 32;
const PARAM2_BITS: u32 = 32;

const R_SHIFT: u32 = 0;
const THETA_SHIFT: u32 = R_SHIFT + R_BITS;
const PARAM1_SHIFT: u32 = THETA_SHIFT + THETA_BITS;
const PARAM2_SHIFT: u32 = PARAM1_SHIFT + PARAM1_BITS;

// hi: u128 (State & Control)
const BASIS_ID_BITS: u32 = 8;
const D_R_BITS: u32 = 4;
const D_THETA_BITS: u32 = 4;
const LOG2_C_BITS: u32 = 8;
const ACTIVATION_ID_BITS: u32 = 8;
const Q_VALUE_BITS: u32 = 8;
const K_VALUE_BITS: u32 = 8;
const FLAGS_BITS: u32 = 8;

const BASIS_ID_SHIFT: u32 = 0;
const D_R_SHIFT: u32 = BASIS_ID_SHIFT + BASIS_ID_BITS;
const D_THETA_SHIFT: u32 = D_R_SHIFT + D_R_BITS;
const LOG2_C_SHIFT: u32 = D_THETA_SHIFT + D_THETA_BITS;
const ACTIVATION_ID_SHIFT: u32 = LOG2_C_SHIFT + LOG2_C_BITS;
const Q_VALUE_SHIFT: u32 = ACTIVATION_ID_SHIFT + ACTIVATION_ID_BITS;
const K_VALUE_SHIFT: u32 = Q_VALUE_SHIFT + Q_VALUE_BITS;
const FLAGS_SHIFT: u32 = K_VALUE_SHIFT + K_VALUE_BITS;

const fn make_mask(bits: u32) -> u128 {
    (1u128 << bits) - 1
}

const R_MASK: u128 = make_mask(R_BITS);
const THETA_MASK: u128 = make_mask(THETA_BITS);
const PARAM1_MASK: u128 = make_mask(PARAM1_BITS);
const PARAM2_MASK: u128 = make_mask(PARAM2_BITS);

const BASIS_ID_MASK: u128 = make_mask(BASIS_ID_BITS);
const D_R_MASK: u128 = make_mask(D_R_BITS);
const D_THETA_MASK: u128 = make_mask(D_THETA_BITS);
const LOG2_C_MASK: u128 = make_mask(LOG2_C_BITS);
const ACTIVATION_ID_MASK: u128 = make_mask(ACTIVATION_ID_BITS);
const Q_VALUE_MASK: u128 = make_mask(Q_VALUE_BITS);
const K_VALUE_MASK: u128 = make_mask(K_VALUE_BITS);
const FLAGS_MASK: u128 = make_mask(FLAGS_BITS);

/// 32비트 고정소수점 (Q24.8)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct FixedPoint32(i32);

impl FixedPoint32 {
    pub fn from_f32(value: f32) -> Self {
        FixedPoint32((value * 256.0).round() as i32)
    }

    pub fn to_f32(&self) -> f32 {
        self.0 as f32 / 256.0
    }
}

/// `Packed256`의 디코딩된 f32 파라미터 표현
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Packed256Params {
    // Continuous Parameters
    pub r: f32,
    pub theta: f32,
    pub param1: f32,
    pub param2: f32,
    // State & Control
    pub basis_id: u8,
    pub d_r: u8,
    pub d_theta: u8,
    pub log2_c: i8,
    pub activation_id: u8,
    pub q_value: u8,
    pub k_value: u8,
    pub flags: u8,
}

/// 256비트 RBE 텐서 코어
/// 상태(hi)와 연속 파라미터(lo)를 분리하여 관리
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Packed256 {
    /// 상태 및 제어 필드 (이산적)
    pub hi: u128,
    /// 연속 파라미터 필드 (고정소수점)
    pub lo: u128,
}

impl Packed256 {
    /// Packed256Params로 디코딩
    pub fn decode(&self) -> Packed256Params {
        Packed256Params {
            r: self.get_r(),
            theta: self.get_theta(),
            param1: self.get_param1(),
            param2: self.get_param2(),
            basis_id: self.get_basis_id(),
            d_r: self.get_d_r(),
            d_theta: self.get_d_theta(),
            log2_c: self.get_log2_c(),
            activation_id: self.get_activation_id(),
            q_value: self.get_q_value(),
            k_value: self.get_k_value(),
            flags: self.get_flags(),
        }
    }
    /// 파라미터로부터 새로운 `Packed256` 인스턴스를 생성합니다.
    pub fn new(params: &Packed256Params) -> Self {
        let mut seed = Self::default();
        seed.set_r(params.r);
        seed.set_theta(params.theta);
        seed.set_param1(params.param1);
        seed.set_param2(params.param2);
        seed.set_basis_id(params.basis_id);
        seed.set_d_r(params.d_r);
        seed.set_d_theta(params.d_theta);
        seed.set_log2_c(params.log2_c);
        seed.set_activation_id(params.activation_id);
        seed.set_q_value(params.q_value);
        seed.set_k_value(params.k_value);
        seed.set_flags(params.flags);
        seed
    }

    /// 안정적인 범위 내에서 랜덤 파라미터를 가진 `Packed256`을 생성합니다.
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let params = Packed256Params {
            r: rng.gen_range(0.1..0.9),
            theta: rng.gen_range(0.0..2.0 * PI),
            param1: rng.gen_range(-2.0..2.0),
            param2: rng.gen_range(-2.0..2.0),
            basis_id: rng.gen_range(0..=4), // 초반에는 안정적인 함수만 선택
            d_r: rng.gen_range(0..=1),      // 1차 미분까지만
            d_theta: rng.gen_range(0..=1),  // 1차 미분까지만
            log2_c: rng.gen_range(-4..=4),
            activation_id: 0, // 기본값
            q_value: 0,
            k_value: 0,
            flags: 0,
        };
        Self::new(&params)
    }

    // 비트 필드 접근자 (Getter)
    pub fn get_r(&self) -> f32 {
        let val = (self.lo >> R_SHIFT) & R_MASK;
        FixedPoint32(val as i32).to_f32()
    }

    pub fn get_theta(&self) -> f32 {
        let val = (self.lo >> THETA_SHIFT) & THETA_MASK;
        FixedPoint32(val as i32).to_f32()
    }

    pub fn get_param1(&self) -> f32 {
        let val = (self.lo >> PARAM1_SHIFT) & PARAM1_MASK;
        FixedPoint32(val as i32).to_f32()
    }

    pub fn get_param2(&self) -> f32 {
        let val = (self.lo >> PARAM2_SHIFT) & PARAM2_MASK;
        FixedPoint32(val as i32).to_f32()
    }

    // 비트 필드 접근자 (Setter)
    pub fn set_r(&mut self, value: f32) {
        let fixed_val = FixedPoint32::from_f32(value).0 as u128;
        self.lo = (self.lo & !(R_MASK << R_SHIFT)) | ((fixed_val & R_MASK) << R_SHIFT);
    }

    pub fn set_theta(&mut self, value: f32) {
        let fixed_val = FixedPoint32::from_f32(value).0 as u128;
        self.lo = (self.lo & !(THETA_MASK << THETA_SHIFT)) | ((fixed_val & THETA_MASK) << THETA_SHIFT);
    }

    pub fn set_param1(&mut self, value: f32) {
        let fixed_val = FixedPoint32::from_f32(value).0 as u128;
        self.lo = (self.lo & !(PARAM1_MASK << PARAM1_SHIFT)) | ((fixed_val & PARAM1_MASK) << PARAM1_SHIFT);
    }

    pub fn set_param2(&mut self, value: f32) {
        let fixed_val = FixedPoint32::from_f32(value).0 as u128;
        self.lo = (self.lo & !(PARAM2_MASK << PARAM2_SHIFT)) | ((fixed_val & PARAM2_MASK) << PARAM2_SHIFT);
    }

    // --- hi: u128 필드 접근자 ---

    pub fn get_basis_id(&self) -> u8 { ((self.hi >> BASIS_ID_SHIFT) & BASIS_ID_MASK) as u8 }
    pub fn set_basis_id(&mut self, value: u8) {
        self.hi = (self.hi & !(BASIS_ID_MASK << BASIS_ID_SHIFT)) | ((value as u128 & BASIS_ID_MASK) << BASIS_ID_SHIFT);
    }

    pub fn get_d_r(&self) -> u8 { ((self.hi >> D_R_SHIFT) & D_R_MASK) as u8 }
    pub fn set_d_r(&mut self, value: u8) {
        self.hi = (self.hi & !(D_R_MASK << D_R_SHIFT)) | ((value as u128 & D_R_MASK) << D_R_SHIFT);
    }

    pub fn get_d_theta(&self) -> u8 { ((self.hi >> D_THETA_SHIFT) & D_THETA_MASK) as u8 }
    pub fn set_d_theta(&mut self, value: u8) {
        self.hi = (self.hi & !(D_THETA_MASK << D_THETA_SHIFT)) | ((value as u128 & D_THETA_MASK) << D_THETA_SHIFT);
    }

    pub fn get_log2_c(&self) -> i8 { ((self.hi >> LOG2_C_SHIFT) & LOG2_C_MASK) as i8 }
    pub fn set_log2_c(&mut self, value: i8) {
        self.hi = (self.hi & !(LOG2_C_MASK << LOG2_C_SHIFT)) | ((value as u128 & LOG2_C_MASK) << LOG2_C_SHIFT);
    }

    pub fn get_activation_id(&self) -> u8 { ((self.hi >> ACTIVATION_ID_SHIFT) & ACTIVATION_ID_MASK) as u8 }
    pub fn set_activation_id(&mut self, value: u8) {
        self.hi = (self.hi & !(ACTIVATION_ID_MASK << ACTIVATION_ID_SHIFT)) | ((value as u128 & ACTIVATION_ID_MASK) << ACTIVATION_ID_SHIFT);
    }

    pub fn get_q_value(&self) -> u8 { ((self.hi >> Q_VALUE_SHIFT) & Q_VALUE_MASK) as u8 }
    pub fn set_q_value(&mut self, value: u8) {
        self.hi = (self.hi & !(Q_VALUE_MASK << Q_VALUE_SHIFT)) | ((value as u128 & Q_VALUE_MASK) << Q_VALUE_SHIFT);
    }

    pub fn get_k_value(&self) -> u8 { ((self.hi >> K_VALUE_SHIFT) & K_VALUE_MASK) as u8 }
    pub fn set_k_value(&mut self, value: u8) {
        self.hi = (self.hi & !(K_VALUE_MASK << K_VALUE_SHIFT)) | ((value as u128 & K_VALUE_MASK) << K_VALUE_SHIFT);
    }

    pub fn get_flags(&self) -> u8 { ((self.hi >> FLAGS_SHIFT) & FLAGS_MASK) as u8 }
    pub fn set_flags(&mut self, value: u8) {
        self.hi = (self.hi & !(FLAGS_MASK << FLAGS_SHIFT)) | ((value as u128 & FLAGS_MASK) << FLAGS_SHIFT);
    }

    pub fn adam_update(
        &mut self,
        m_hat_r: f32,
        m_hat_theta: f32,
        v_hat_r: f32,
        v_hat_theta: f32,
        learning_rate: f32,
        epsilon: f32,
    ) {
        // 현재 파라미터 가져오기
        let mut r = self.get_r();
        let mut theta = self.get_theta();
        
        // r 업데이트 (안정화된 스텝)
        r = (r - learning_rate * m_hat_r / (v_hat_r.sqrt() + epsilon)).clamp(0.0, 0.9999);
        
        // theta 업데이트
        theta -= learning_rate * m_hat_theta / (v_hat_theta.sqrt() + epsilon);
        
        // 업데이트된 파라미터 설정
        self.set_r(r);
        self.set_theta(theta);
    }
}
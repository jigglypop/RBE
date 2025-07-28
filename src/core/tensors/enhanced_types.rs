//! Enhanced128 - Legacy 수학 모델을 비트 도메인으로 포팅
//! 
//! Legacy의 정교한 12가지 기저 함수를 128비트 구조에서 효율적으로 구현

use super::hyperbolic_lut::HYPERBOLIC_LUT_DATA;
use rand::Rng;
use std::f32::consts::PI;

/// Enhanced 128비트 압축 시드 (Legacy 호환)
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct Enhanced128 {
    pub hi: u64,  // 미분 사이클 + 메타데이터
    pub lo: u64,  // Legacy 파라미터 (r, θ, basis_id, rot_code, log2_c, d_r)
}

/// Enhanced 파라미터 구조체
#[derive(Debug, Clone, Default)]
pub struct EnhancedParams {
    pub r_fp32: f32,        // 반지름 [0, 0.9999]
    pub theta_fp32: f32,    // 각도 [0, 2π)
    pub basis_id: u8,       // 기저 함수 ID (0-11)
    pub rot_code: u8,       // 회전 코드 (0-15)
    pub log2_c: i8,         // 곡률 (-4 ~ +3)
    pub d_r: bool,          // 반지름 미분 차수
    pub d_theta: u8,        // 각도 미분 차수 (0-3) - hi에서 추출
}

/// 고정소수점 Q16.16 연산 헬퍼
pub struct FixedPoint;

impl FixedPoint {
    pub const SCALE: i32 = 65536; // 2^16
    
    #[inline(always)]
    pub fn from_f32(f: f32) -> i32 {
        (f * Self::SCALE as f32).round() as i32
    }
    
    #[inline(always)]
    pub fn to_f32(fixed: i32) -> f32 {
        fixed as f32 / Self::SCALE as f32
    }
    
    #[inline(always)]
    pub fn mul(a: i32, b: i32) -> i32 {
        ((a as i64 * b as i64) >> 16) as i32
    }
}

impl Enhanced128 {
    /// Legacy 파라미터로부터 Enhanced128 생성
    pub fn from_legacy_params(
        r: f32,
        theta: f32,
        basis_id: u8,
        d_theta: u8,
        d_r: bool,
        rot_code: u8,
        log2_c: i8,
    ) -> Self {
        // r, theta 정규화 및 양자화
        let r_clamped = r.clamp(0.0, 0.999999);
        let theta_norm = theta.rem_euclid(2.0 * PI);
        
        // 24비트 r, 28비트 theta
        let r_bits = ((r_clamped as f64) * ((1u64 << 24) - 1) as f64).round() as u64;
        let theta_bits = ((theta_norm as f64) / (2.0 * PI as f64) * ((1u64 << 28) - 1) as f64).round() as u64;
        
        // lo 필드 패킹
        let mut lo = 0u64;
        lo |= (r_bits & 0xFFFFFF) << 40;                    // bits 63-40: r (24bit)
        lo |= (theta_bits & 0xFFFFFFF) << 12;               // bits 39-12: theta (28bit)
        lo |= ((basis_id as u64) & 0xF) << 8;               // bits 11-8: basis_id (4bit)
        lo |= ((rot_code as u64) & 0xF) << 4;               // bits 7-4: rot_code (4bit)
        lo |= ((log2_c as u64) & 0x7) << 1;                 // bits 3-1: log2_c (3bit)
        lo |= (d_r as u64) & 0x1;                           // bit 0: d_r (1bit)
        
        // hi 필드: 미분 사이클 + 메타데이터
        let mut hi = 0u64;
        hi |= ((d_theta as u64) & 0xF) << 60;               // bits 63-60: d_theta (4bit)
        hi |= 0x1234_5678_9ABC_DEF0u64 & 0x0FFF_FFFF_FFFF_FFFFu64; // 나머지는 메타데이터/예약
        
        Self { hi, lo }
    }
    
    /// Enhanced 파라미터 디코딩
    pub fn decode_enhanced(&self) -> EnhancedParams {
        // lo 필드 비트 추출
        let r_bits = (self.lo >> 40) & 0xFFFFFF;
        let theta_bits = (self.lo >> 12) & 0xFFFFFFF;
        let basis_id = ((self.lo >> 8) & 0xF) as u8;
        let rot_code = ((self.lo >> 4) & 0xF) as u8;
        let log2_c_bits = ((self.lo >> 1) & 0x7) as u8;
        let d_r = (self.lo & 0x1) != 0;
        
        // hi 필드에서 d_theta 추출
        let d_theta = ((self.hi >> 60) & 0xF) as u8;
        
        // 부동소수점 복원
        let r_fp32 = (r_bits as f32) / ((1u64 << 24) - 1) as f32;
        let theta_fp32 = (theta_bits as f32) / ((1u64 << 28) - 1) as f32 * 2.0 * PI;
        
        // 3비트 부호있는 정수 복원
        let log2_c = if (log2_c_bits & 0x4) != 0 {
            (log2_c_bits as i8) | -8  // 음수 확장
        } else {
            log2_c_bits as i8
        };
        
        EnhancedParams {
            r_fp32,
            theta_fp32,
            basis_id,
            rot_code,
            log2_c,
            d_r,
            d_theta,
        }
    }
    
    /// Legacy 스타일 회전 각도 계산
    fn get_rotation_angle(rot_code: u8) -> f32 {
        match rot_code & 0xF {
            0 => 0.0,
            1 => PI / 8.0,
            2 => PI / 6.0,
            3 => PI / 4.0,
            4 => PI / 3.0,
            5 => PI / 2.0,
            6 => 2.0 * PI / 3.0,
            7 => 3.0 * PI / 4.0,
            8 => 5.0 * PI / 6.0,
            9 => 7.0 * PI / 8.0,
            10 => PI,
            11 => 9.0 * PI / 8.0,
            12 => 4.0 * PI / 3.0,
            13 => 3.0 * PI / 2.0,
            14 => 5.0 * PI / 3.0,
            15 => 7.0 * PI / 4.0,
            _ => 0.0,
        }
    }
    
    /// Legacy 스타일 각도 미분 적용
    fn apply_angular_derivative(theta: f32, d_theta: u8, basis_id: u8) -> f32 {
        let is_sin_based = (basis_id & 0x1) == 0;
        
        match (is_sin_based, d_theta % 4) {
            (true, 0) => theta.sin(),
            (true, 1) => theta.cos(),
            (true, 2) => -theta.sin(),
            (true, 3) => -theta.cos(),
            (false, 0) => theta.cos(),
            (false, 1) => -theta.sin(),
            (false, 2) => -theta.cos(),
            (false, 3) => theta.sin(),
            // 이 케이스들은 % 4 연산으로 인해 실제로는 도달할 수 없지만 컴파일러 만족용
            _ => theta.sin(), // 기본값
        }
    }
    
    /// Legacy 스타일 반지름 미분 적용
    fn apply_radial_derivative(r: f32, d_r: bool, basis_id: u8) -> f32 {
        let is_sinh_based = (basis_id & 0x2) == 0;
        
        match (is_sinh_based, d_r) {
            (true, false) => r.sinh(),
            (true, true) => r.cosh(),
            (false, false) => r.cosh(),
            (false, true) => r.sinh(),
        }
    }
    
    /// 임시 Bessel J0 근사 (나중에 LUT로 교체)
    fn bessel_j0_approx(x: f32) -> f32 {
        let ax = x.abs();
        if ax < 8.0 {
            let y = x * x;
            let ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
            let ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))));
            ans1 / ans2
        } else {
            let z = 8.0 / ax;
            let y = z * z;
            let xx = ax - 0.785398164;
            let ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
            let ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
            (2.0 / (PI * ax)).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2)
        }
    }
    
    /// Enhanced fused forward (Legacy 수학 + 비트 최적화)
    pub fn fused_forward_enhanced(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        let params = self.decode_enhanced();
        
        // 곡률 계산
        let c = 2.0f32.powi(params.log2_c as i32);
        
        // 좌표를 [-1, 1] 범위로 정규화
        let x = 2.0 * (j as f32) / ((cols - 1) as f32) - 1.0;
        let y = 2.0 * (i as f32) / ((rows - 1) as f32) - 1.0;
        
        // 로컬 극좌표
        let r_local = (x * x + y * y).sqrt().min(0.999999);
        let theta_local = y.atan2(x);
        
        // 회전 적용
        let rotation = Self::get_rotation_angle(params.rot_code);
        let theta_final = params.theta_fp32 + theta_local + rotation;
        
        // 미분 순환성 적용
        let angular_value = Self::apply_angular_derivative(theta_final, params.d_theta, params.basis_id);
        let radial_value = Self::apply_radial_derivative(c * params.r_fp32, params.d_r, params.basis_id);
        
        // 기저 함수에 따른 계산
        let basis_value = match params.basis_id {
            0..=3 => angular_value * radial_value,
            4 => Self::bessel_j0_approx(r_local * 10.0),
            5 => Self::bessel_j0_approx(r_local * 10.0).cosh(), // I0 근사
            6 => (-r_local * 10.0).exp(), // K0 근사  
            7 => Self::bessel_j0_approx(r_local * 10.0) * (r_local * 10.0).ln().max(-10.0), // Y0 근사
            8 => (c * r_local).tanh() * theta_final.cos().signum(),
            9 => Self::sech(c * r_local) * Self::triangle_wave(theta_final),
            10 => (-c * r_local).exp() * theta_final.sin(),
            11 => Self::morlet_wavelet(r_local, theta_final, 5.0),
            _ => 0.0,
        };
        
        // 야코비안 계산
        let jacobian = (1.0 - c * params.r_fp32 * params.r_fp32).powi(-2).sqrt();
        
        basis_value * jacobian
    }
    
    /// 헬퍼 함수들
    fn sech(x: f32) -> f32 {
        1.0 / x.cosh()
    }
    
    fn triangle_wave(x: f32) -> f32 {
        let normalized = x / PI;
        let t = normalized - normalized.floor();
        if t < 0.5 { 2.0 * t } else { 2.0 * (1.0 - t) }
    }
    
    fn morlet_wavelet(r: f32, theta: f32, omega: f32) -> f32 {
        let sigma = 1.0;
        let t = r * theta.cos();
        (-(t * t) / (2.0 * sigma * sigma)).exp() * (omega * t).cos()
    }
    
    /// 랜덤 Enhanced128 생성
    pub fn random(rng: &mut impl Rng) -> Self {
        Self::from_legacy_params(
            rng.gen_range(0.1..0.9),        // r
            rng.gen_range(0.0..2.0*PI),     // theta
            rng.gen_range(0..12),           // basis_id
            rng.gen_range(0..4),            // d_theta
            rng.gen::<bool>(),              // d_r
            rng.gen_range(0..16),           // rot_code
            rng.gen_range(-3..4),           // log2_c
        )
    }
    
    /// 현재 구현과의 호환성을 위한 기본 fused_forward
    pub fn fused_forward(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        self.fused_forward_enhanced(i, j, rows, cols)
    }
}

/// AnalyticalGradient 트레이트 구현 (기존 시스템 호환)
impl super::packed_types::AnalyticalGradient for Enhanced128 {
    fn analytical_gradient_r(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        let h = 1e-6;
        let mut params = self.decode_enhanced();
        
        let original = self.fused_forward_enhanced(i, j, rows, cols);
        
        params.r_fp32 += h;
        let enhanced_plus = Enhanced128::from_legacy_params(
            params.r_fp32, params.theta_fp32, params.basis_id, 
            params.d_theta, params.d_r, params.rot_code, params.log2_c
        );
        let forward_plus = enhanced_plus.fused_forward_enhanced(i, j, rows, cols);
        
        (forward_plus - original) / h
    }
    
    fn analytical_gradient_theta(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        let h = 1e-6;
        let mut params = self.decode_enhanced();
        
        let original = self.fused_forward_enhanced(i, j, rows, cols);
        
        params.theta_fp32 += h;
        let enhanced_plus = Enhanced128::from_legacy_params(
            params.r_fp32, params.theta_fp32, params.basis_id, 
            params.d_theta, params.d_r, params.rot_code, params.log2_c
        );
        let forward_plus = enhanced_plus.fused_forward_enhanced(i, j, rows, cols);
        
        (forward_plus - original) / h
    }
} 
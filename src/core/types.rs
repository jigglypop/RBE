use rand::Rng;
use std::f32;
use crate::math::{ste_quant_q0x, ste_quant_phase};

/// 64-bit Packed Poincaré 시드 표현 (CORDIC 통합)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Packed64 {
    pub rotations: u64,  // CORDIC 회전 시퀀스
}

impl Packed64 {
    pub fn new(rotations: u64) -> Self {
        Packed64 { rotations }
    }

    pub fn compute_weight(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 1. hi 비트필드에서 r, theta 디코딩
        let r_quant = (self.rotations >> 44) & 0xFFFFF; // 20 bits
        let theta_quant = (self.rotations >> 20) & 0xFFFFFF; // 24 bits
        
        let r_val = r_quant as f32 / ((1u64 << 20) - 1) as f32; // [0, 1] 범위로 정규화
        let theta_val = (theta_quant as f32 / ((1u64 << 24) - 1) as f32) * 2.0 * std::f32::consts::PI; // [0, 2PI] 범위로 정규화

        let rotations = self.rotations;

        // 2. 좌표 기반 초기 각도 계산
        let x_norm = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
        let y_norm = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
        let base_angle = y_norm.atan2(x_norm);
        
        // 3. r, theta를 적용하여 초기 벡터 (x, y) 설정
        let mut x = r_val * (base_angle + theta_val).cos();
        let mut y = r_val * (base_angle + theta_val).sin();

        for k in 0..20 { // CORDIC 반복 횟수를 r, theta를 제외한 나머지 비트(20)만큼으로 조정
            let sigma = if (rotations >> k) & 1 == 1 { 1.0 } else { -1.0 };
            
            let power_of_2 = (2.0f32).powi(-(k as i32));

            let x_new = x - sigma * y * power_of_2;
            let y_new = y + sigma * x * power_of_2;
            
            x = x_new;
            y = y_new;

            // 쌍곡 변환 추가
            if k % 4 == 0 {
                let r = (x*x + y*y).sqrt();
                if r > 1e-9 { // 0에 가까운 값 방지
                    let tanh_r = r.tanh();
                    x *= tanh_r;
                    y *= tanh_r;
                }
            }
        }
        
        // CORDIC 게인 보정.
        let gain = 1.64676; 
        x / gain
    }
}

/// 128-bit 시드 (Seed0: 비트필드, Seed1: 연속 FP32×2)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Packed128 {
    pub hi: u64,   // Seed0 : 기존 Packed64 비트필드
    pub lo: u64,   // Seed1 : [63:32] r_fp32  |  [31:0] θ_fp32
}

/// 연속 파라미터까지 포함해 디코딩
#[derive(Debug, Clone, Default)]
pub struct DecodedParams { // DecodedParams128 -> DecodedParams. 기존것과 통합
    // pub base: DecodedParams, // 기존 필드 모두
    pub r_fp32: f32,
    pub theta_fp32: f32,
}

impl Packed128 {
    /// Seed0+1 디코딩
    pub fn decode(&self) -> DecodedParams {
        let r_fp32     = f32::from_bits((self.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(self.lo as u32);
        DecodedParams { r_fp32, theta_fp32, ..Default::default() }
    }
    /// 연속 파라미터 → 128 bit 시드
    pub fn from_continuous(p: &DecodedParams) -> Self {
        // new.md의 비트 레이아웃에 따라 hi 필드를 구성합니다.
        // r (Q0.20) -> [63:44], theta (Q0.24) -> [43:20]
        let r_quant = ste_quant_q0x(p.r_fp32, 20);
        let theta_quant = ste_quant_phase(p.theta_fp32, 24);

        let hi = (r_quant << 44) | (theta_quant << 20); // 다른 필드는 0으로 가정
        let lo = ((p.r_fp32.to_bits() as u64) << 32) | p.theta_fp32.to_bits() as u64;
        
        Packed128 { hi, lo }
    }

    /// 무작위 초기화
    pub fn random(rng: &mut impl Rng) -> Self {
        let r = 0.8 + rng.gen::<f32>() * 0.2; // [0.8, 1.0] 범위로 증가
        let theta: f32 = rng.gen_range(-0.5..0.5); // [-0.5, 0.5] 범위
        
        let lo = ((r.to_bits() as u64) << 32) | theta.to_bits() as u64;
        let hi = rng.gen::<u64>() & 0xFFFFF; // 하위 20비트만 랜덤
        
        Packed128 { hi, lo }
    }

    /// 정밀한 순전파: hi(상태 전이) + lo(연속) 융합
    /// 상태 전이 미분의 핵심 구현
    #[inline(always)]
    pub fn fused_forward(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 1. 연속 파라미터 추출 (lo)
        let r_fp32 = f32::from_bits((self.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(self.lo as u32);
        
        // 2. 상태 비트 추출 (hi)
        let state_bits = self.hi & 0xFFFFF; // 하위 20비트
        
        // 3. 좌표 정규화
        let x_norm = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
        let y_norm = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
        
        // 4. 연속 기저 패턴 계산
        let dist = (x_norm * x_norm + y_norm * y_norm).sqrt();
        let base_angle = y_norm.atan2(x_norm);
        let base_pattern = (r_fp32 - dist * r_fp32 + theta_fp32).clamp(0.0, 1.0);
        
        // 5. 상태 기반 함수 선택 및 변조
        let coord_hash = ((i * 31 + j) & 0x3) as u64; // 위치별 상태 선택
        let state_selector = (state_bits >> (coord_hash * 2)) & 0x3;
        
        let modulated_value = match state_selector {
            0 => base_pattern * (base_angle + theta_fp32).sin().abs(),      // sin 상태
            1 => base_pattern * (base_angle + theta_fp32).cos().abs(),      // cos 상태  
            2 => base_pattern * (dist * r_fp32 + theta_fp32).tanh(),        // tanh 상태
            3 => base_pattern * (1.0 - (dist * r_fp32).exp() * 0.5),       // exp 상태
            _ => base_pattern,
        };
        
        // 6. 고주파 세부사항 추가 (나머지 상태 비트 활용)
        let detail_bits = (state_bits >> 8) & 0xFFF; // 상위 12비트
        let detail_factor = 1.0 + 0.1 * (detail_bits as f32 / 4095.0 - 0.5);
        
        (modulated_value * detail_factor).clamp(0.0, 1.0)
    }

    /// 추론 전용: hi(Seed0) → weight
    #[inline(always)]
    pub fn compute_weight(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        Packed64{ rotations: self.hi }.compute_weight(i, j, rows, cols)
    }
    
    /// 해석적 미분: r 파라미터에 대한 정확한 그래디언트 계산 (수치 미분 대체)
    #[inline(always)]
    pub fn analytical_gradient_r(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 1. 연속 파라미터 추출
        let r_fp32 = f32::from_bits((self.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(self.lo as u32);
        
        // 2. 좌표 정규화
        let x_norm = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
        let y_norm = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
        let dist = (x_norm * x_norm + y_norm * y_norm).sqrt();
        let base_angle = y_norm.atan2(x_norm);
        
        // 3. 상태 선택
        let state_bits = self.hi & 0xFFFFF;
        let coord_hash = ((i * 31 + j) & 0x3) as u64;
        let state_selector = (state_bits >> (coord_hash * 2)) & 0x3;
        
        // 4. base_pattern과 그 미분 (첫 번째 clamp)
        let base_pattern_unclamped = r_fp32 - dist * r_fp32 + theta_fp32;
        let base_pattern = base_pattern_unclamped.clamp(0.0, 1.0);
        
        // base_pattern의 clamp 미분
        let base_pattern_clamp_derivative = if base_pattern_unclamped > 0.0 && base_pattern_unclamped < 1.0 {
            1.0
        } else {
            0.0
        };
        
        let d_base_pattern_dr = (1.0 - dist) * base_pattern_clamp_derivative;
        
        // 5. 상태별 함수값과 그 미분 계산
        let (modulated_value, d_modulated_value_dr) = match state_selector {
            0 => {
                // sin 상태: base_pattern * sin(base_angle + theta).abs()
                let sin_arg = base_angle + theta_fp32;
                let sin_val = sin_arg.sin();
                let sin_abs = sin_val.abs();
                
                let value = base_pattern * sin_abs;
                let gradient = d_base_pattern_dr * sin_abs; // sin_abs는 r에 의존하지 않음
                
                (value, gradient)
            },
            1 => {
                // cos 상태: base_pattern * cos(base_angle + theta).abs()
                let cos_arg = base_angle + theta_fp32;
                let cos_val = cos_arg.cos();
                let cos_abs = cos_val.abs();
                
                let value = base_pattern * cos_abs;
                let gradient = d_base_pattern_dr * cos_abs;
                
                (value, gradient)
            },
            2 => {
                // tanh 상태: base_pattern * tanh(dist*r + theta)
                let tanh_arg = dist * r_fp32 + theta_fp32;
                let tanh_val = tanh_arg.tanh();
                let sech_sq = 1.0 - tanh_val * tanh_val;
                
                let value = base_pattern * tanh_val;
                let gradient = d_base_pattern_dr * tanh_val + base_pattern * sech_sq * dist;
                
                (value, gradient)
            },
            3 => {
                // exp 상태: base_pattern * (1 - exp(dist*r) * 0.5)
                let exp_arg = dist * r_fp32;
                let exp_val = exp_arg.exp();
                let exp_term = 1.0 - exp_val * 0.5;
                
                let value = base_pattern * exp_term;
                let gradient = d_base_pattern_dr * exp_term + base_pattern * (-0.5 * exp_val * dist);
                
                (value, gradient)
            },
            _ => {
                let value = base_pattern;
                let gradient = d_base_pattern_dr;
                (value, gradient)
            }
        };
        
        // 6. detail_factor 적용
        let detail_bits = (state_bits >> 8) & 0xFFF;
        let detail_factor = 1.0 + 0.1 * (detail_bits as f32 / 4095.0 - 0.5);
        
        let detailed_value = modulated_value * detail_factor;
        let detailed_gradient = d_modulated_value_dr * detail_factor;
        
        // 7. 최종 clamp(0,1) 적용 및 미분
        let final_value_unclamped = detailed_value;
        
        // 최종 clamp의 미분: 범위 [0,1] 내에 있으면 1, 외부면 0
        if final_value_unclamped > 0.0 && final_value_unclamped < 1.0 {
            detailed_gradient
        } else {
            0.0
        }
    }
    
    /// 해석적 미분: theta 파라미터에 대한 정확한 그래디언트 계산 (수치 미분 대체)
    #[inline(always)]
    pub fn analytical_gradient_theta(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 1. 연속 파라미터 추출
        let r_fp32 = f32::from_bits((self.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(self.lo as u32);
        
        // 2. 좌표 정규화
        let x_norm = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
        let y_norm = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
        let dist = (x_norm * x_norm + y_norm * y_norm).sqrt();
        let base_angle = y_norm.atan2(x_norm);
        
        // 3. 상태 선택
        let state_bits = self.hi & 0xFFFFF;
        let coord_hash = ((i * 31 + j) & 0x3) as u64;
        let state_selector = (state_bits >> (coord_hash * 2)) & 0x3;
        
        // 4. base_pattern과 그 미분 (첫 번째 clamp)
        let base_pattern_unclamped = r_fp32 - dist * r_fp32 + theta_fp32;
        let base_pattern = base_pattern_unclamped.clamp(0.0, 1.0);
        
        // base_pattern의 clamp 미분
        let base_pattern_clamp_derivative = if base_pattern_unclamped > 0.0 && base_pattern_unclamped < 1.0 {
            1.0
        } else {
            0.0
        };
        
        let d_base_pattern_dtheta = 1.0 * base_pattern_clamp_derivative;
        
        // 5. 상태별 함수값과 그 미분 계산
        let (modulated_value, d_modulated_value_dtheta) = match state_selector {
            0 => {
                // sin 상태: base_pattern * sin(base_angle + theta).abs()
                let sin_arg = base_angle + theta_fp32;
                let sin_val = sin_arg.sin();
                let cos_val = sin_arg.cos();
                let sin_abs = sin_val.abs();
                
                let value = base_pattern * sin_abs;
                
                // d/dtheta[base_pattern * sin(base_angle + theta).abs()]
                let d_sin_abs_dtheta = if sin_val >= 0.0 { cos_val } else { -cos_val };
                let gradient = d_base_pattern_dtheta * sin_abs + base_pattern * d_sin_abs_dtheta;
                
                (value, gradient)
            },
            1 => {
                // cos 상태: base_pattern * cos(base_angle + theta).abs()
                let cos_arg = base_angle + theta_fp32;
                let cos_val = cos_arg.cos();
                let sin_val = cos_arg.sin();
                let cos_abs = cos_val.abs();
                
                let value = base_pattern * cos_abs;
                
                let d_cos_abs_dtheta = if cos_val >= 0.0 { -sin_val } else { sin_val };
                let gradient = d_base_pattern_dtheta * cos_abs + base_pattern * d_cos_abs_dtheta;
                
                (value, gradient)
            },
            2 => {
                // tanh 상태: base_pattern * tanh(dist*r + theta)
                let tanh_arg = dist * r_fp32 + theta_fp32;
                let tanh_val = tanh_arg.tanh();
                let sech_sq = 1.0 - tanh_val * tanh_val;
                
                let value = base_pattern * tanh_val;
                let gradient = d_base_pattern_dtheta * tanh_val + base_pattern * sech_sq;
                
                (value, gradient)
            },
            3 => {
                // exp 상태: base_pattern * (1 - exp(dist*r) * 0.5)
                let exp_val = (dist * r_fp32).exp();
                let exp_term = 1.0 - exp_val * 0.5;
                
                let value = base_pattern * exp_term;
                let gradient = d_base_pattern_dtheta * exp_term; // exp_term은 theta에 의존하지 않음
                
                (value, gradient)
            },
            _ => {
                let value = base_pattern;
                let gradient = d_base_pattern_dtheta;
                (value, gradient)
            }
        };
        
        // 6. detail_factor 적용
        let detail_bits = (state_bits >> 8) & 0xFFF;
        let detail_factor = 1.0 + 0.1 * (detail_bits as f32 / 4095.0 - 0.5);
        
        let detailed_value = modulated_value * detail_factor;
        let detailed_gradient = d_modulated_value_dtheta * detail_factor;
        
        // 7. 최종 clamp(0,1) 적용 및 미분
        let final_value_unclamped = detailed_value;
        
        // 최종 clamp의 미분: 범위 [0,1] 내에 있으면 1, 외부면 0
        if final_value_unclamped > 0.0 && final_value_unclamped < 1.0 {
            detailed_gradient
        } else {
            0.0
        }
    }

    /// 학습 전용: lo(Seed1)의 연속 FP32 직접 사용
    #[inline(always)]
    pub fn compute_weight_continuous(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // lo에서 r_fp32, theta_fp32 직접 추출
        let r_fp32 = f32::from_bits((self.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(self.lo as u32);
        
        // 좌표를 [-1, 1] 범위로 정규화
        let x = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
        let y = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
        
        // 극좌표로 변환
        let dist = (x*x + y*y).sqrt();
        
        // radial gradient와 유사한 함수
        // r_fp32는 스케일, theta_fp32는 오프셋으로 사용
        let value = (r_fp32 - dist * r_fp32 + theta_fp32).max(0.0).min(1.0);
        
        value
    }

    /// 고급 상태 전이 미분: 다중 비트 상태와 함수별 전이 규칙
    pub fn advanced_state_transition(&mut self, gradient_signal: f32, i: usize, j: usize) {
        // 다중 해시를 사용한 위치별 상태 관리
        let primary_hash = ((i * 31 + j) & 0x7) as u64; // 3비트 선택자
        let primary_bit_pos = primary_hash * 3;
        let current_primary_state = (self.hi >> primary_bit_pos) & 0x7;
        // 보조 상태 (2비트): 4가지 변조 상태
        let secondary_bit_pos = (primary_hash + 8) * 2;
        let current_secondary_state = (self.hi >> secondary_bit_pos) & 0x3;
        // 그래디언트 강도에 따른 전이 결정
        let abs_gradient = gradient_signal.abs();
        let gradient_sign = gradient_signal.signum();
        let new_primary_state = if abs_gradient > 0.2 {
            // 강한 그래디언트: 명확한 미분 전이
            match (current_primary_state, gradient_sign > 0.0) {
                (0, true) => 1,   // sin -> cos (미분)
                (1, true) => 0,   // cos -> -sin (미분, 부호는 별도 처리)
                (2, true) => 3,   // tanh -> sech^2 (미분)
                (3, true) => 2,   // sech^2 -> tanh (역미분)
                (4, true) => 5,   // exp -> exp (자기 자신)
                (5, true) => 6,   // log -> 1/x (미분)
                (6, true) => 7,   // 1/x -> -1/x^2 (미분)
                (7, true) => 4,   // 다항식 -> exp (전이)
                
                // 음의 그래디언트: 역방향 전이
                (0, false) => 7,  // sin -> 다항식
                (1, false) => 6,  // cos -> 1/x
                (2, false) => 5,  // tanh -> log
                (3, false) => 4,  // sech^2 -> exp
                (4, false) => 3,  // exp -> sech^2
                (5, false) => 2,  // log -> tanh
                (6, false) => 1,  // 1/x -> cos
                (7, false) => 0,  // 다항식 -> sin
                _ => current_primary_state,
            }
        } else if abs_gradient > 0.05 {
            // 중간 그래디언트: 보조 상태 기반 미세 조정
            match current_secondary_state {
                0 => (current_primary_state + 1) & 0x7, // 순환 전진
                1 => if current_primary_state > 0 { current_primary_state - 1 } else { 7 }, // 순환 후진
                2 => current_primary_state ^ 0x1, // 비트 토글
                3 => current_primary_state ^ 0x2, // 다른 비트 토글
                _ => current_primary_state,
            }
        } else {
            current_primary_state // 약한 그래디언트는 상태 유지
        };
        
        // 보조 상태 업데이트 (적응적 학습)
        let new_secondary_state = if abs_gradient > 0.1 {
            // 그래디언트 방향에 따른 보조 상태 조정
            if gradient_sign > 0.0 {
                (current_secondary_state + 1) & 0x3
            } else {
                if current_secondary_state > 0 { current_secondary_state - 1 } else { 3 }
            }
        } else {
            current_secondary_state
        };
        
        // 비트 필드 업데이트
        self.hi = (self.hi & !(0x7 << primary_bit_pos)) | (new_primary_state << primary_bit_pos);
        self.hi = (self.hi & !(0x3 << secondary_bit_pos)) | (new_secondary_state << secondary_bit_pos);
    }

    /// 상태 전이 미분: 그래디언트 신호에 따른 비트 상태 업데이트
    pub fn apply_state_transition(&mut self, gradient_signal: f32, i: usize, j: usize) {
        let coord_hash = ((i * 31 + j) & 0x3) as u64;
        let bit_pos = coord_hash * 2;
        let current_state = (self.hi >> bit_pos) & 0x3;
        
        // 그래디언트 신호에 따른 상태 전이 결정
        let new_state = if gradient_signal > 0.1 {
            // 양의 그래디언트: sin -> cos, tanh -> exp 등으로 전이
            match current_state {
                0 => 1, // sin -> cos (미분)
                1 => 0, // cos -> -sin (미분)
                2 => 3, // tanh -> sech^2 (미분 근사)
                3 => 2, // exp -> exp (자기 자신이 미분)
                _ => current_state,
            }
        } else if gradient_signal < -0.1 {
            // 음의 그래디언트: 역방향 전이
            match current_state {
                0 => 3, // sin -> exp
                1 => 2, // cos -> tanh
                2 => 1, // tanh -> cos
                3 => 0, // exp -> sin
                _ => current_state,
            }
        } else {
            current_state // 작은 그래디언트는 상태 유지
        };
        
        // 비트 업데이트
        self.hi = (self.hi & !(0x3 << bit_pos)) | (new_state << bit_pos);
    }

    /// 상태별 함수 계산 (고급 버전) - 안정성 개선
    pub fn compute_state_function(&self, state: u64, input: f32, phase: f32) -> f32 {
        let safe_input = input.clamp(-10.0, 10.0); // 입력 범위 제한
        let safe_phase = phase.clamp(-10.0, 10.0); // 위상 범위 제한
        
        let result = match state {
            0 => (safe_input + safe_phase).sin(),                    // sin 상태
            1 => (safe_input + safe_phase).cos(),                    // cos 상태
            2 => (safe_input * safe_phase).tanh(),                   // tanh 상태
            3 => {
                let cosh_val = (safe_input * safe_phase).cosh();
                1.0 / (cosh_val * cosh_val).max(1e-6)               // sech^2 상태 (0 나누기 방지)
            },
            4 => (safe_input * safe_phase * 0.1).exp().min(10.0),   // exp 상태 (폭발 방지)
            5 => {
                let arg = (safe_input * safe_phase).abs() + 1e-6;   // 음수와 0 방지
                arg.ln().clamp(-10.0, 10.0)                         // log 상태
            },
            6 => {
                let denom = safe_input * safe_phase + 1e-3;         // 0 나누기 방지
                (1.0 / denom).clamp(-100.0, 100.0)                 // 1/x 상태
            },
            7 => {
                let linear = safe_input * safe_phase;
                let quadratic = 0.1 * safe_input * safe_input;     // 계수 축소
                (linear + quadratic).clamp(-10.0, 10.0)            // 다항식 상태
            },
            _ => safe_input * safe_phase, // 기본값
        };
        
        // 최종 안전성 검사
        if result.is_finite() {
            result.clamp(-1.0, 1.0) // 출력 범위 [-1, 1] 제한
        } else {
            0.0 // NaN이나 무한대면 0으로 대체
        }
    }

    /// 개선된 융합 순전파: 고급 상태 전이 적용
    #[inline(always)]
    pub fn fused_forward_advanced(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // 1. 연속 파라미터 추출 (lo)
        let r_fp32 = f32::from_bits((self.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(self.lo as u32);
        
        // 2. 상태 비트 추출 (hi) - 고급 버전
        let state_bits = self.hi & 0xFFFFF; // 하위 20비트
        
        // 3. 좌표 정규화
        let x_norm = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
        let y_norm = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
        
        // 4. 연속 기저 패턴 계산
        let dist = (x_norm * x_norm + y_norm * y_norm).sqrt();
        let base_angle = y_norm.atan2(x_norm);
        let base_pattern = (r_fp32 - dist * r_fp32 + theta_fp32).clamp(0.0, 1.0);
        
        // 5. 다중 상태 기반 함수 선택 및 변조
        let primary_hash = ((i * 31 + j) & 0x7) as u64;
        let primary_state = (state_bits >> (primary_hash * 3)) & 0x7;
        let secondary_state = (state_bits >> ((primary_hash + 8) * 2)) & 0x3;
        // 주요 함수 계산
        let primary_value = self.compute_state_function(
            primary_state, 
            base_angle + theta_fp32 * 0.5, 
            r_fp32
        );
        
        // 보조 변조 적용
        let modulation_factor = match secondary_state {
            0 => 1.0,                                    // 변조 없음
            1 => 0.8 + 0.4 * (dist * 3.14159).sin(),    // 사인 변조
            2 => 1.0 - 0.3 * dist,                      // 거리 기반 감쇠
            3 => (1.0 + (base_angle * 2.0).cos()) * 0.5, // 각도 기반 변조
            _ => 1.0,
        };
        
        let modulated_value = base_pattern * primary_value.abs() * modulation_factor;
        
        // 6. 고주파 세부사항 추가 (나머지 상태 비트 활용)
        let detail_bits = (state_bits >> 16) & 0xF; // 상위 4비트
        let detail_factor = 1.0 + 0.05 * (detail_bits as f32 / 15.0 - 0.5);
        
        (modulated_value * detail_factor).clamp(0.0, 1.0)
    }
}

/// 잔차 인코딩에 사용할 변환 타입
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TransformType {
    Dct,
    Dwt,
    Adaptive,
}

/// DCT/웨이블릿 잔차 계수
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ResidualCoefficient {
    pub index: (u16, u16), // 블록 내 좌표 (최대 65535x65535)
    pub value: f32,
}

/// RBE 기본 패턴 + 잔차 계수를 포함하는 하이브리드 압축 블록
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HybridEncodedBlock {
    /// RBE 기본 패턴을 생성하는 8개의 연속 파라미터
    pub rbe_params: RbeParameters,
    /// 잔차 보정을 위한 상위 K개의 DCT 또는 웨이블릿 계수
    pub residuals: Vec<ResidualCoefficient>,
    /// 블록의 원래 크기
    pub rows: usize,
    pub cols: usize,
    /// 적용된 변환 타입
    pub transform_type: TransformType,
}

/// 단일 인코딩 블록의 그래디언트를 저장하는 구조체
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedBlockGradients {
    pub rbe_params_grad: RbeParameters,
    pub residuals_grad: Vec<ResidualCoefficient>,
}

/// RBE 기본 패턴 파라미터 타입 별칭
pub type RbeParameters = [f32; 8];

/// 기저 함수 타입 (기존 유지)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BasisFunction {
    SinCosh = 0,
    SinSinh = 1,
    CosCosh = 2,
    CosSinh = 3,
    BesselJ = 4,
    BesselI = 5,
    BesselK = 6,
    BesselY = 7,
    TanhSign = 8,
    SechTri = 9,
    ExpSin = 10,
    Morlet = 11,
}

/// 행렬 압축 및 복원 (PoincareMatrix가 Packed128을 사용하도록 변경)
pub struct PoincareMatrix {
    pub seed: Packed128,
    pub rows: usize,
    pub cols: usize,
} 

// ================================
// 1장: 푸앵카레 볼 기반 데이터 구조 구현
// ================================

/// 푸앵카레 볼 128비트 인코딩 (1장 문서 정확한 설계)
/// hi: 푸앵카레 상태 코어, lo: 연속 파라미터 코어
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoincarePackedBit128 {
    /// 푸앵카레 상태 코어 (64비트)
    /// [63:62] poincare_quadrant (2비트)
    /// [61:50] hyperbolic_frequency (12비트) 
    /// [49:38] geodesic_amplitude (12비트)
    /// [37:32] basis_function_selector (6비트)
    /// [31:0]  cordic_rotation_sequence (32비트)
    pub hi: u64,
    
    /// 연속 파라미터 코어 (64비트)
    /// [63:32] r_poincare (32비트 IEEE754 float)
    /// [31:0]  theta_poincare (32비트 IEEE754 float)
    pub lo: u64,
}

/// 푸앵카레 사분면 열거형 (2비트 인코딩)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoincareQuadrant {
    First,  // 00: sinh(x) - 양의 지수 증가
    Second, // 01: cosh(x) - 대칭적 지수 증가  
    Third,  // 10: tanh(x) - S자형 포화 함수
    Fourth, // 11: sech²(x) - 종 모양 함수
}

/// 쌍곡 기저 함수 열거형 (6비트 = 64가지 조합)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HyperbolicBasisFunction {
    Sinh = 0,     // sinh
    Cosh = 1,     // cosh  
    Tanh = 2,     // tanh
    SechSquared = 3, // sech²
    // ... 60가지 더 (조합)
}

impl PoincarePackedBit128 {
    /// 새로운 푸앵카레 볼 인코딩 생성
    pub fn new(
        quadrant: PoincareQuadrant,
        frequency: u16,  // 12비트
        amplitude: u16,  // 12비트  
        basis_func: u8,  // 6비트
        cordic_seq: u32, // 32비트
        r_poincare: f32, // [0, 1)
        theta_poincare: f32, // [0, 2π]
    ) -> Self {
        let hi = Self::encode_hi_field(quadrant, frequency, amplitude, basis_func, cordic_seq);
        let lo = Self::encode_lo_field(r_poincare, theta_poincare);
        
        Self { hi, lo }
    }
    
    /// hi 필드 인코딩 (푸앵카레 상태 코어)
    fn encode_hi_field(
        quadrant: PoincareQuadrant,
        frequency: u16,
        amplitude: u16, 
        basis_func: u8,
        cordic_seq: u32,
    ) -> u64 {
        let quadrant_bits = quadrant as u64;
        let frequency_bits = (frequency as u64) & 0xFFF; // 12비트
        let amplitude_bits = (amplitude as u64) & 0xFFF; // 12비트
        let basis_bits = (basis_func as u64) & 0x3F;     // 6비트
        let cordic_bits = cordic_seq as u64;             // 32비트
        
        (quadrant_bits << 62) |
        (frequency_bits << 50) |
        (amplitude_bits << 38) |
        (basis_bits << 32) |
        cordic_bits
    }
    
    /// lo 필드 인코딩 (연속 파라미터 코어)
    fn encode_lo_field(r_poincare: f32, theta_poincare: f32) -> u64 {
        // r을 [0, 1) 범위로 클램핑
        let r_clamped = r_poincare.clamp(0.0, 0.99999);
        let theta_normalized = theta_poincare.rem_euclid(2.0 * std::f32::consts::PI);
        
        ((r_clamped.to_bits() as u64) << 32) | (theta_normalized.to_bits() as u64)
    }
    
    /// 푸앵카레 사분면 추출
    pub fn get_quadrant(&self) -> PoincareQuadrant {
        match (self.hi >> 62) & 0x3 {
            0 => PoincareQuadrant::First,
            1 => PoincareQuadrant::Second,
            2 => PoincareQuadrant::Third,
            3 => PoincareQuadrant::Fourth,
            _ => unreachable!(),
        }
    }
    
    /// 쌍곡 주파수 추출 (12비트)
    pub fn get_hyperbolic_frequency(&self) -> u16 {
        ((self.hi >> 50) & 0xFFF) as u16
    }
    
    /// 측지선 진폭 추출 (12비트)
    pub fn get_geodesic_amplitude(&self) -> u16 {
        ((self.hi >> 38) & 0xFFF) as u16
    }
    
    /// 기저 함수 선택자 추출 (6비트)
    pub fn get_basis_function_selector(&self) -> u8 {
        ((self.hi >> 32) & 0x3F) as u8
    }
    
    /// CORDIC 회전 시퀀스 추출 (32비트)
    pub fn get_cordic_rotation_sequence(&self) -> u32 {
        (self.hi & 0xFFFFFFFF) as u32
    }
    
    /// 푸앵카레 반지름 추출 ([0, 1))
    pub fn get_r_poincare(&self) -> f32 {
        f32::from_bits((self.lo >> 32) as u32)
    }
    
    /// 푸앵카레 각도 추출 ([0, 2π])
    pub fn get_theta_poincare(&self) -> f32 {
        f32::from_bits((self.lo & 0xFFFFFFFF) as u32)
    }
    
    /// 푸앵카레 볼 내 실제 쌍곡거리 계산
    pub fn compute_hyperbolic_distance(&self) -> f32 {
        let r = self.get_r_poincare();
        if r >= 1.0 {
            return f32::INFINITY;
        }
        // d_h = artanh(r) = 0.5 * ln((1+r)/(1-r))
        0.5 * ((1.0 + r) / (1.0 - r)).ln()
    }
    
    /// 정보 밀도 계산 (1/(1-r²)²)
    pub fn compute_information_density(&self) -> f32 {
        let r = self.get_r_poincare();
        let r_squared = r * r;
        let denominator = 1.0 - r_squared;
        if denominator.abs() < 1e-10 {
            return f32::INFINITY;
        }
        1.0 / (denominator * denominator)
    }
}

/// 쌍곡 함수 계산 유틸리티
impl PoincarePackedBit128 {
    /// 사분면에 따른 기본 쌍곡 함수 계산
    pub fn compute_hyperbolic_function(&self, input: f32) -> f32 {
        match self.get_quadrant() {
            PoincareQuadrant::First => input.sinh(),
            PoincareQuadrant::Second => input.cosh(),
            PoincareQuadrant::Third => input.tanh(),
            PoincareQuadrant::Fourth => {
                // sech²(x) = 1/cosh²(x)
                let cosh_val = input.cosh();
                1.0 / (cosh_val * cosh_val)
            }
        }
    }
    
    /// 쌍곡 주파수를 실제 주파수로 변환
    pub fn get_real_frequency(&self, max_frequency: f32) -> f32 {
        let freq_quantized = self.get_hyperbolic_frequency() as f32;
        (freq_quantized / 4095.0) * max_frequency
    }
    
    /// 측지선 진폭을 실제 진폭으로 변환  
    pub fn get_real_amplitude(&self, max_amplitude: f32) -> f32 {
        let amp_quantized = self.get_geodesic_amplitude() as f32;
        (amp_quantized / 4095.0) * max_amplitude
    }
}

/// 랜덤 생성 및 유틸리티
impl PoincarePackedBit128 {
    /// 랜덤 푸앵카레 볼 인코딩 생성
    pub fn random(rng: &mut impl Rng) -> Self {
        let quadrant = match rng.gen_range(0..4) {
            0 => PoincareQuadrant::First,
            1 => PoincareQuadrant::Second, 
            2 => PoincareQuadrant::Third,
            3 => PoincareQuadrant::Fourth,
            _ => unreachable!(),
        };
        
        let frequency: u16 = rng.gen_range(0..4096);
        let amplitude: u16 = rng.gen_range(0..4096);
        let basis_func: u8 = rng.gen_range(0..64);
        let cordic_seq: u32 = rng.gen();
        
        // r은 [0, 0.9) 범위로 제한 (수치 안정성)
        let r_poincare: f32 = rng.gen_range(0.0..0.9);
        let theta_poincare: f32 = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
        
        Self::new(quadrant, frequency, amplitude, basis_func, cordic_seq, r_poincare, theta_poincare)
    }
    
    /// 푸앵카레 볼 경계 조건 검증
    pub fn is_valid_poincare(&self) -> bool {
        let r = self.get_r_poincare();
        let theta = self.get_theta_poincare();
        
        r >= 0.0 && r < 1.0 && theta >= 0.0 && theta <= 2.0 * std::f32::consts::PI
    }
} 
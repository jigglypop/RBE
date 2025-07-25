//! 비트 도메인 푸앵카레볼 리만 Adam 최적화기
//! 모든 리만 미분, adam 연산은 비트 상태에서 수행

use crate::core::tensors::packed_types::*;
use std::collections::HashMap;
use std::f32::consts::PI;

/// 비트 도메인 리만 Adam 상태 - 11비트 미분 사이클 시스템 기반
/// 진정한 푸앵카레볼 기하학에서의 비트 상태 최적화
#[derive(Debug, Clone)]
pub struct BitRiemannianAdamState {
    /// 128비트 packed 푸앵카레볼 상태
    pub poincare_state: Packed128,
    /// r, θ 각각의 11비트 사이클 상태
    pub r_cycle: CycleState,     // 반지름 미분 사이클
    pub theta_cycle: CycleState, // 각도 미분 사이클
    /// 리만 메트릭 비트 인코딩
    pub metric_r_bits: u32,      // g_rr (Q16.16)
    pub metric_theta_bits: u32,  // g_θθ (Q16.16)
    /// 비트 인코딩된 하이퍼파라미터
    pub beta1_bits: u32,         // Q16.16 고정소수점
    pub beta2_bits: u32,         // Q16.16 고정소수점
    pub epsilon_bits: u32,       // Q16.16 고정소수점
    pub t: u16,                  // 시간 스텝
    /// 비트 도메인 모멘텀 (4개의 32비트 값)
    pub m_r_bits: u32,           // r 1차 모멘텀
    pub v_r_bits: u32,           // r 2차 모멘텀
    pub m_theta_bits: u32,       // θ 1차 모멘텀
    pub v_theta_bits: u32,       // θ 2차 모멘텀
    /// 비트 그래디언트 추적
    pub bit_tracker: BitGradientTracker,
    /// 쌍곡함수 캐시 (성능 최적화)
    hyperbolic_cache: HashMap<u32, u32>, // input_bits -> output_bits
}

impl Default for BitRiemannianAdamState {
    fn default() -> Self {
        Self::new()
    }
}

impl BitRiemannianAdamState {
    pub fn new() -> Self {
        let mut state = Self {
            poincare_state: Packed128::default(),
            r_cycle: CycleState::from_bits(0x080),    // r 전용 사이클
            theta_cycle: CycleState::from_bits(0x180), // θ 전용 사이클
            metric_r_bits: Self::f32_to_q16(4.0),    // 초기 메트릭
            metric_theta_bits: Self::f32_to_q16(1.0),
            beta1_bits: Self::f32_to_q16(0.9),
            beta2_bits: Self::f32_to_q16(0.999),
            epsilon_bits: Self::f32_to_q16(1e-8),
            t: 0,
            m_r_bits: 0,
            v_r_bits: 0,
            m_theta_bits: 0,
            v_theta_bits: 0,
            bit_tracker: BitGradientTracker::new(2),
            hyperbolic_cache: HashMap::new(),
        };
        
        // 초기 사이클 상태 설정
        state.poincare_state.set_cycle_state(state.r_cycle);
        state
    }
    
    pub fn with_config(beta1: f32, beta2: f32, epsilon: f32) -> Self {
        let mut state = Self {
            poincare_state: Packed128::default(),
            r_cycle: CycleState::from_bits(0x080),
            theta_cycle: CycleState::from_bits(0x180),
            metric_r_bits: Self::f32_to_q16(4.0),
            metric_theta_bits: Self::f32_to_q16(1.0),
            beta1_bits: Self::f32_to_q16(beta1),
            beta2_bits: Self::f32_to_q16(beta2),
            epsilon_bits: Self::f32_to_q16(epsilon),
            t: 0,
            m_r_bits: 0,
            v_r_bits: 0,
            m_theta_bits: 0,
            v_theta_bits: 0,
            bit_tracker: BitGradientTracker::new(2),
            hyperbolic_cache: HashMap::new(),
        };
        
        state.poincare_state.set_cycle_state(state.r_cycle);
        state
    }
    
    /// f32를 Q16.16 고정소수점으로 변환
    #[inline]
    fn f32_to_q16(val: f32) -> u32 {
        (val * 65536.0) as u32
    }
    
    /// Q16.16을 f32로 변환
    #[inline]
    fn q16_to_f32(bits: u32) -> f32 {
        (bits as i32) as f32 / 65536.0
    }
    
    /// 비트 도메인 곱셈 (고정소수점)
    #[inline]
    fn bit_multiply(a_bits: u32, b_bits: u32) -> u32 {
        let result = ((a_bits as u64) * (b_bits as u64)) >> 16;
        result as u32
    }
    
    /// 비트 도메인 나눗셈 (고정소수점)
    #[inline]
    fn bit_divide(numerator: u32, denominator: u32) -> u32 {
        if denominator == 0 { return 0; }
        let extended_num = (numerator as u64) << 16;
        (extended_num / denominator as u64) as u32
    }
    
    /// 비트 도메인 덧셈/뺄셈
    #[inline]
    fn bit_add(a: u32, b: u32) -> u32 { a.saturating_add(b) }
    #[inline]
    fn bit_sub(a: u32, b: u32) -> u32 { a.saturating_sub(b) }
    
    /// **핵심: 비트 도메인 리만 메트릭 텐서 계산**
    /// 푸앵카레볼의 리만 메트릭을 11비트 사이클 시스템으로 계산
    fn compute_bit_metric_tensor(&mut self, r_bits: u32) -> (u32, u32) {
        // 캐시 확인
        if let Some(&cached_metric) = self.hyperbolic_cache.get(&r_bits) {
            let g_rr = cached_metric >> 16;
            let g_theta_theta = cached_metric & 0xFFFF;
            return (g_rr, g_theta_theta);
        }
        
        // r을 f32로 변환하여 안전한 범위로 클램핑
        let r_f32 = Self::q16_to_f32(r_bits).clamp(0.0, 0.9999);
        let r_f32_bits = Self::f32_to_q16(r_f32);
        
        // r² 계산 (비트 도메인)
        let r_squared_bits = Self::bit_multiply(r_f32_bits, r_f32_bits);
        
        // (1 - r²) 계산
        let one_bits = Self::f32_to_q16(1.0);
        let one_minus_r2_bits = Self::bit_sub(one_bits, r_squared_bits);
        
        // 분모 보호: (1 - r²)이 너무 작으면 클램핑
        let safe_denom = if one_minus_r2_bits < 1000 { 1000 } else { one_minus_r2_bits };
        
        // 4/(1-r²)² 계산
        let four_bits = Self::f32_to_q16(4.0);
        let denom_squared = Self::bit_multiply(safe_denom, safe_denom);
        let base_factor = Self::bit_divide(four_bits, denom_squared);
        
        // g_rr = 4/(1-r²)²
        let g_rr = base_factor;
        
        // g_θθ = r² * 4/(1-r²)² (각도 메트릭)
        let g_theta_theta = Self::bit_multiply(r_squared_bits, base_factor);
        
        // 캐시 저장 (상위 16비트에 g_rr, 하위 16비트에 g_θθ)
        let cache_value = ((g_rr & 0xFFFF) << 16) | (g_theta_theta & 0xFFFF);
        if self.hyperbolic_cache.len() < 5000 {
            self.hyperbolic_cache.insert(r_bits, cache_value);
        }
        
        (g_rr, g_theta_theta)
    }
    
    /// **핵심: 비트 도메인 리만 그래디언트 계산**
    /// 11비트 사이클 시스템을 활용한 푸앵카레볼 그래디언트 계산
    fn compute_bit_riemannian_gradients(&mut self, packed: &Packed128, i: usize, j: usize,
                                       target: f32, rows: usize, cols: usize) -> (u32, u32, CycleState, CycleState) {
        // 1. 현재 상태에서 디코딩
        let decoded = packed.decode();
        let r_bits = Self::f32_to_q16(decoded.r_fp32);
        let theta_bits = Self::f32_to_q16(decoded.theta_fp32);
        
        // 2. 현재 출력과 오차 계산
        let current_output = packed.fused_forward(i, j, rows, cols);
        let error = target - current_output;
        let error_bits = Self::f32_to_q16(error);
        
        // 3. 오차를 11비트 사이클 상태로 변환
        let error_cycle = CycleState::from_bits((error_bits >> 5) as u16 & 0x7FF);
        
        // 4. r과 θ에 대한 편미분을 사이클 전이로 계산
        let current_cycle = packed.get_cycle_state();
        
        // r 그래디언트 사이클 (함수 인덱스 0 = sinh)
        let r_grad_cycle_base = CycleState::from_bits(0x000 | (current_cycle.to_bits() & 0xFF));
        let r_grad_cycle = r_grad_cycle_base.apply_transition(&error_cycle);
        
        // θ 그래디언트 사이클 (함수 인덱스 1 = cosh) 
        let theta_grad_cycle_base = CycleState::from_bits(0x100 | (current_cycle.to_bits() & 0xFF));
        let theta_grad_cycle = theta_grad_cycle_base.apply_transition(&error_cycle);
        
        // 5. 좌표 기반 추가 변조
        let coord_hash = ((i * 0x9E3779B9) ^ (j * 0x517CC1B7)) as u16 & 0x7FF;
        let coord_cycle = CycleState::from_bits(coord_hash);
        
        let final_r_cycle = r_grad_cycle.apply_transition(&coord_cycle);
        let final_theta_cycle = theta_grad_cycle.apply_transition(&coord_cycle);
        
        // 6. 사이클 상태를 그래디언트 크기로 변환
        let grad_r_bits = Self::cycle_to_gradient_magnitude(&final_r_cycle);
        let grad_theta_bits = Self::cycle_to_gradient_magnitude(&final_theta_cycle);
        
        (grad_r_bits, grad_theta_bits, final_r_cycle, final_theta_cycle)
    }
    
    /// 사이클 상태를 그래디언트 크기로 변환 (푸앵카레볼 특화)
    fn cycle_to_gradient_magnitude(cycle: &CycleState) -> u32 {
        let bits = cycle.to_bits();
        
        let func_idx = (bits >> 8) & 0x7;
        let cycle_pos = (bits >> 4) & 0xF;
        let modulation = bits & 0xF;
        
        // 푸앵카레볼 전용 함수별 가중치
        let func_weight = match func_idx {
            0 => 1.2,    // sinh (r 방향)
            1 => 0.8,    // cosh (θ 방향)
            2 => 1.5,    // tanh (경계 근처)
            3 => 2.0,    // sech² (곡률 높은 영역)
            _ => 1.0,
        };
        
        // 사이클 위치와 변조 조합 (비선형 스케일링)
        let cycle_factor = (cycle_pos as f32 / 15.0).powf(1.2);
        let mod_factor = (modulation as f32 / 15.0) * 0.3;
        
        let magnitude = (cycle_factor + mod_factor) * func_weight * 0.05;
        Self::f32_to_q16(magnitude)
    }
    
    /// **핵심: 비트 도메인 뫼비우스 덧셈**
    /// 푸앵카레볼에서의 비트 상태 결합 연산
    fn bit_mobius_add(&self, x_bits: u32, y_bits: u32) -> u32 {
        // 극초기 종료
        if y_bits < 100 { return x_bits; }
        if x_bits < 100 { return y_bits; }
        
        // x + y 계산 (분자)
        let numerator = Self::bit_add(x_bits, y_bits);
        
        // 1 + xy 계산 (분모)
        let xy = Self::bit_multiply(x_bits, y_bits);
        let one_bits = Self::f32_to_q16(1.0);
        let denominator = Self::bit_add(one_bits, xy);
        
        // 분모 보호
        if denominator < 1000 {
            return x_bits; // 안전한 값 반환
        }
        
        // (x + y) / (1 + xy)
        let result = Self::bit_divide(numerator, denominator);
        
        // 푸앵카레볼 경계 클리핑
        let boundary_bits = Self::f32_to_q16(0.9999);
        if result > boundary_bits {
            boundary_bits
        } else {
            result
        }
    }
    
    /// **핵심: 비트 도메인 지수 사상**
    /// 푸앵카레볼에서의 탄젠트 벡터를 매니폴드 위의 점으로 변환
    fn bit_exponential_map(&self, x_bits: u32, v_bits: u32) -> u32 {
        if v_bits < 100 { return x_bits; }
        
        // |v| 계산 (비트 도메인)
        let v_norm_bits = v_bits; // 단순화: 스칼라로 처리
        
        // tanh(|v|/2) 계산을 비트 도메인에서 근사
        let half_bits = Self::f32_to_q16(0.5);
        let tanh_arg_bits = Self::bit_multiply(v_norm_bits, half_bits);
        
        // tanh 근사: tanh(x) ≈ x / (1 + 0.3x) for small x
        let point_three_bits = Self::f32_to_q16(0.3);
        let tanh_denom = Self::bit_add(
            Self::f32_to_q16(1.0),
            Self::bit_multiply(point_three_bits, tanh_arg_bits)
        );
        let tanh_result = Self::bit_divide(tanh_arg_bits, tanh_denom);
        
        // 방향 벡터 (단순화: v의 부호)
        let direction_bits = if v_bits & 0x80000000 != 0 {
            Self::f32_to_q16(-1.0)
        } else {
            Self::f32_to_q16(1.0)
        };
        
        let tangent_vector = Self::bit_multiply(tanh_result, direction_bits);
        
        // 뫼비우스 덧셈으로 최종 결과
        self.bit_mobius_add(x_bits, tangent_vector)
    }
    
    /// **핵심: 비트 도메인 리만 Adam 업데이트**
    /// 모든 연산이 푸앵카레볼 비트 상태에서 수행됨
    pub fn bit_riemannian_update(&mut self, packed: &mut Packed128, i: usize, j: usize,
                                 target: f32, learning_rate: f32, rows: usize, cols: usize) {
        // 1. 비트 도메인 리만 그래디언트 계산
        let (grad_r_bits, grad_theta_bits, r_cycle, theta_cycle) = 
            self.compute_bit_riemannian_gradients(packed, i, j, target, rows, cols);
        
        // 2. 그래디언트가 0에 가까우면 조기 종료
        if grad_r_bits < 50 && grad_theta_bits < 50 {
            return;
        }
        
        self.t = self.t.saturating_add(1);
        
        // 3. 현재 상태에서 r, θ 추출
        let decoded = packed.decode();
        let r_bits = Self::f32_to_q16(decoded.r_fp32);
        let theta_bits = Self::f32_to_q16(decoded.theta_fp32);
        
        // 4. 비트 도메인 리만 메트릭 텐서 계산
        let (g_rr, g_theta_theta) = self.compute_bit_metric_tensor(r_bits);
        self.metric_r_bits = g_rr;
        self.metric_theta_bits = g_theta_theta;
        
        // 5. 리만 그래디언트 = g^(-1) * ∇f (비트 도메인)
        let riem_grad_r = if g_rr > 100 { Self::bit_divide(grad_r_bits, g_rr) } else { 0 };
        let riem_grad_theta = if g_theta_theta > 100 { Self::bit_divide(grad_theta_bits, g_theta_theta) } else { 0 };
        
        // 6. 모멘텀 업데이트 (비트 도메인)
        let one_minus_beta1 = Self::bit_sub(Self::f32_to_q16(1.0), self.beta1_bits);
        let one_minus_beta2 = Self::bit_sub(Self::f32_to_q16(1.0), self.beta2_bits);
        
        // r 모멘텀
        let beta1_m_r = Self::bit_multiply(self.beta1_bits, self.m_r_bits);
        let grad_r_term = Self::bit_multiply(one_minus_beta1, riem_grad_r);
        self.m_r_bits = Self::bit_add(beta1_m_r, grad_r_term);
        
        let beta2_v_r = Self::bit_multiply(self.beta2_bits, self.v_r_bits);
        let grad_r_squared = Self::bit_multiply(riem_grad_r, riem_grad_r);
        let grad_r2_term = Self::bit_multiply(one_minus_beta2, grad_r_squared);
        self.v_r_bits = Self::bit_add(beta2_v_r, grad_r2_term);
        
        // θ 모멘텀
        let beta1_m_theta = Self::bit_multiply(self.beta1_bits, self.m_theta_bits);
        let grad_theta_term = Self::bit_multiply(one_minus_beta1, riem_grad_theta);
        self.m_theta_bits = Self::bit_add(beta1_m_theta, grad_theta_term);
        
        let beta2_v_theta = Self::bit_multiply(self.beta2_bits, self.v_theta_bits);
        let grad_theta_squared = Self::bit_multiply(riem_grad_theta, riem_grad_theta);
        let grad_theta2_term = Self::bit_multiply(one_minus_beta2, grad_theta_squared);
        self.v_theta_bits = Self::bit_add(beta2_v_theta, grad_theta2_term);
        
        // 7. 편향 보정 (비트 도메인)
        let beta1_power = self.bit_power(self.beta1_bits, self.t);
        let beta2_power = self.bit_power(self.beta2_bits, self.t);
        
        let beta1_complement = Self::bit_sub(Self::f32_to_q16(1.0), beta1_power);
        let beta2_complement = Self::bit_sub(Self::f32_to_q16(1.0), beta2_power);
        
        if beta1_complement < 100 || beta2_complement < 100 {
            return;
        }
        
        // 8. m_hat, v_hat 계산
        let m_r_hat = Self::bit_divide(self.m_r_bits, beta1_complement);
        let v_r_hat = Self::bit_divide(self.v_r_bits, beta2_complement);
        let m_theta_hat = Self::bit_divide(self.m_theta_bits, beta1_complement);
        let v_theta_hat = Self::bit_divide(self.v_theta_bits, beta2_complement);
        
        // 9. 업데이트 벡터 계산 (비트 sqrt)
        let v_r_sqrt = self.bit_sqrt(v_r_hat);
        let v_theta_sqrt = self.bit_sqrt(v_theta_hat);
        
        let denom_r = Self::bit_add(v_r_sqrt, self.epsilon_bits);
        let denom_theta = Self::bit_add(v_theta_sqrt, self.epsilon_bits);
        
        let lr_bits = Self::f32_to_q16(learning_rate);
        let update_r_bits = Self::bit_multiply(lr_bits, Self::bit_divide(m_r_hat, denom_r));
        let update_theta_bits = Self::bit_multiply(lr_bits, Self::bit_divide(m_theta_hat, denom_theta));
        
        // 10. 비트 도메인 지수 사상을 통한 업데이트 적용
        let new_r_bits = self.bit_exponential_map(r_bits, update_r_bits);
        
        // θ는 단순 덧셈 (각도 공간)
        let two_pi_bits = Self::f32_to_q16(2.0 * PI);
        let new_theta_bits = Self::bit_add(theta_bits, update_theta_bits) % two_pi_bits;
        
        // 11. 새로운 연속 파라미터로 packed 업데이트
        let new_r_f32 = Self::q16_to_f32(new_r_bits).clamp(0.0, 0.9999);
        let new_theta_f32 = Self::q16_to_f32(new_theta_bits);
        
        let new_params = DecodedParams {
            r_fp32: new_r_f32,
            theta_fp32: new_theta_f32,
        };
        
        *packed = Packed128::from_continuous(&new_params);
        
        // 12. 사이클 상태 업데이트
        self.r_cycle = self.r_cycle.apply_transition(&r_cycle);
        self.theta_cycle = self.theta_cycle.apply_transition(&theta_cycle);
        
        let combined_cycle = self.r_cycle.apply_transition(&self.theta_cycle);
        packed.set_cycle_state(combined_cycle);
        
        // 13. 비트 그래디언트 추적
        self.bit_tracker.register_dependency(0, &self.poincare_state, packed);
        self.poincare_state = *packed;
    }
    
    /// 비트 도메인 거듭제곱
    fn bit_power(&self, base_bits: u32, exp: u16) -> u32 {
        if exp == 0 { return Self::f32_to_q16(1.0); }
        if exp == 1 { return base_bits; }
        
        let mut result = Self::f32_to_q16(1.0);
        let mut base = base_bits;
        let mut exponent = exp;
        
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = Self::bit_multiply(result, base);
            }
            base = Self::bit_multiply(base, base);
            exponent >>= 1;
        }
        
        result
    }
    
    /// 비트 도메인 제곱근
    fn bit_sqrt(&self, value_bits: u32) -> u32 {
        if value_bits == 0 { return 0; }
        
        let mut x = value_bits >> 1;
        for _ in 0..3 { // 더 정확한 근사를 위해 3회 반복
            let x_squared = Self::bit_multiply(x, x);
            if x > 0 {
                let error = Self::bit_sub(value_bits, x_squared);
                let correction = Self::bit_divide(error, x * 2);
                x = Self::bit_add(x, correction);
            }
        }
        x
    }
    
    /// 배치 업데이트
    pub fn batch_riemannian_update(&mut self, packed: &mut Packed128, coordinates: &[(usize, usize)],
                                  targets: &[f32], learning_rate: f32, rows: usize, cols: usize) {
        for (&(i, j), &target) in coordinates.iter().zip(targets.iter()) {
            self.bit_riemannian_update(packed, i, j, target, learning_rate, rows, cols);
        }
    }
    
    /// 상태 초기화
    pub fn reset(&mut self) {
        self.poincare_state = Packed128::default();
        self.r_cycle = CycleState::from_bits(0x080);
        self.theta_cycle = CycleState::from_bits(0x180);
        self.t = 0;
        self.m_r_bits = 0;
        self.v_r_bits = 0;
        self.m_theta_bits = 0;
        self.v_theta_bits = 0;
        self.bit_tracker = BitGradientTracker::new(2);
        self.hyperbolic_cache.clear();
        
        self.poincare_state.set_cycle_state(self.r_cycle);
    }
    
    /// 수렴 여부 확인
    pub fn is_converged(&self, threshold_bits: u32) -> bool {
        self.m_r_bits < threshold_bits && 
        self.v_r_bits < threshold_bits &&
        self.m_theta_bits < threshold_bits && 
        self.v_theta_bits < threshold_bits
    }
    
    /// 현재 상태 정보
    pub fn get_riemannian_state_info(&self) -> (u16, CycleState, CycleState, u32, u32, u32, u32) {
        (
            self.t,
            self.r_cycle, 
            self.theta_cycle,
            self.m_r_bits,
            self.v_r_bits,
            self.m_theta_bits,
            self.v_theta_bits,
        )
    }
    
    /// 비트 도메인 리만 성능 측정
    pub fn benchmark_riemannian_operations(iterations: usize) {
        use std::time::Instant;
        
        println!("=== 비트 도메인 리만 Adam 성능 측정 ===");
        
        let mut optimizer = Self::new();
        let mut packed = Packed128::random(&mut rand::thread_rng());
        
        let start = Instant::now();
        
        for i in 0..iterations {
            let target = (i as f32 % 100.0) * 0.01;
            let row = i % 50;
            let col = (i * 7) % 75;
            
            optimizer.bit_riemannian_update(&mut packed, row, col, target, 0.001, 50, 75);
        }
        
        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;
        
        println!("비트 리만 Adam 업데이트: {:.1} ns/op ({:.1} MHz)", 
                ns_per_op, 1000.0 / ns_per_op);
        
        let (t, r_cycle, theta_cycle, m_r, v_r, m_theta, v_theta) = optimizer.get_riemannian_state_info();
        println!("최종 상태: t={}, r_cycle={:011b}, theta_cycle={:011b}", 
                t, r_cycle.to_bits(), theta_cycle.to_bits());
        println!("r 모멘텀: m={}, v={}", m_r, v_r);
        println!("θ 모멘텀: m={}, v={}", m_theta, v_theta);
        
        let decoded = packed.decode();
        println!("최종 좌표: r={:.6}, θ={:.6}", decoded.r_fp32, decoded.theta_fp32);
    }
} 
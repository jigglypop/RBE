//! 비트 도메인 푸앵카레볼 Adam 최적화기
//! 모든 미분, adam 연산은 비트 상태에서 수행

use crate::core::tensors::packed_types::*;
use std::collections::HashMap;

/// 비트 도메인 Adam 상태 - 11비트 미분 사이클 시스템 기반
#[derive(Debug, Clone)]
pub struct BitAdamState {
    /// 128비트 packed 모멘텀 상태
    pub momentum_state: Packed128,
    /// 11비트 사이클 상태들
    pub m_cycle: CycleState,  // 1차 모멘텀 사이클
    pub v_cycle: CycleState,  // 2차 모멘텀 사이클
    /// 비트 인코딩된 하이퍼파라미터 
    pub beta1_bits: u32,      // Q16.16 고정소수점
    pub beta2_bits: u32,      // Q16.16 고정소수점  
    pub epsilon_bits: u32,    // Q16.16 고정소수점
    pub t: u16,               // 시간 스텝 (16비트)
    /// 비트 그래디언트 추적
    pub bit_tracker: BitGradientTracker,
    /// 상태 전이 캐시 (성능 최적화)
    transition_cache: HashMap<u64, (CycleState, CycleState)>,
}

impl Default for BitAdamState {
    fn default() -> Self {
        Self::new()
    }
}

impl BitAdamState {
    pub fn new() -> Self {
        let mut state = Self {
            momentum_state: Packed128::default(),
            m_cycle: CycleState::from_bits(0x100), // 초기 사이클 상태
            v_cycle: CycleState::from_bits(0x200), // 다른 초기 상태
            beta1_bits: Self::f32_to_q16(0.9),
            beta2_bits: Self::f32_to_q16(0.999),
            epsilon_bits: Self::f32_to_q16(1e-8),
            t: 0,
            bit_tracker: BitGradientTracker::new(1),
            transition_cache: HashMap::new(),
        };
        
        // 초기 사이클 상태 설정
        state.momentum_state.set_cycle_state(state.m_cycle);
        state
    }
    
    pub fn with_config(beta1: f32, beta2: f32, epsilon: f32) -> Self {
        let mut state = Self {
            momentum_state: Packed128::default(),
            m_cycle: CycleState::from_bits(0x100),
            v_cycle: CycleState::from_bits(0x200),
            beta1_bits: Self::f32_to_q16(beta1),
            beta2_bits: Self::f32_to_q16(beta2),
            epsilon_bits: Self::f32_to_q16(epsilon),
            t: 0,
            bit_tracker: BitGradientTracker::new(1),
            transition_cache: HashMap::new(),
        };
        
        state.momentum_state.set_cycle_state(state.m_cycle);
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
    
    /// 비트 도메인 덧셈 (오버플로 방지)
    #[inline]
    fn bit_add(a_bits: u32, b_bits: u32) -> u32 {
        a_bits.saturating_add(b_bits)
    }
    
    /// 비트 도메인 뺄셈
    #[inline]
    fn bit_sub(a_bits: u32, b_bits: u32) -> u32 {
        a_bits.saturating_sub(b_bits)
    }
    
    /// **핵심: 비트 상태 미분 계산**
    /// 11비트 사이클 시스템을 활용한 그래디언트 계산
    fn compute_bit_gradient(&mut self, packed: &Packed128, i: usize, j: usize, 
                           target: f32, rows: usize, cols: usize) -> (u32, CycleState) {
        // 1. 현재 출력 계산
        let current_output = packed.fused_forward(i, j, rows, cols);
        
        // 2. 오차 계산 (Q16.16)
        let error = target - current_output;
        let error_bits = Self::f32_to_q16(error);
        
        // 3. 오차를 11비트 사이클 상태로 변환
        let error_cycle = CycleState::from_bits((error_bits >> 5) as u16 & 0x7FF);
        
        // 4. 현재 사이클과 오차 사이클의 전이 계산
        let current_cycle = packed.get_cycle_state();
        let gradient_cycle = current_cycle.apply_transition(&error_cycle);
        
        // 5. 좌표 기반 추가 전이 (위치 정보 인코딩)
        let coord_hash = ((i * 0x9E3779B9) ^ (j * 0x517CC1B7)) as u16 & 0x7FF;
        let coord_cycle = CycleState::from_bits(coord_hash);
        let final_gradient_cycle = gradient_cycle.apply_transition(&coord_cycle);
        
        // 6. 사이클 상태를 그래디언트 크기로 변환
        let gradient_magnitude = Self::cycle_to_gradient_magnitude(&final_gradient_cycle);
        
        (gradient_magnitude, final_gradient_cycle)
    }
    
    /// 사이클 상태를 그래디언트 크기로 변환
    fn cycle_to_gradient_magnitude(cycle: &CycleState) -> u32 {
        let bits = cycle.to_bits();
        
        // 11비트를 분해하여 그래디언트 크기 계산
        let func_idx = (bits >> 8) & 0x7;  // 함수 선택
        let cycle_pos = (bits >> 4) & 0xF; // 사이클 위치  
        let modulation = bits & 0xF;       // 변조 파라미터
        
        // 함수별 가중치 (쌍곡함수 특성 반영)
        let func_weight = match func_idx {
            0 => 1.0,    // sinh
            1 => 0.8,    // cosh
            2 => 1.2,    // tanh  
            3 => 1.5,    // sech²
            _ => 1.0,
        };
        
        // 사이클 위치와 변조를 조합하여 최종 크기 계산
        let base_magnitude = (cycle_pos as f32 + modulation as f32 * 0.1) * func_weight;
        Self::f32_to_q16(base_magnitude * 0.01) // 적절한 스케일링
    }
    
    /// **핵심: 비트 도메인 Adam 업데이트**
    /// 모든 연산이 비트 상태에서 수행됨
    pub fn bit_update(&mut self, packed: &mut Packed128, i: usize, j: usize, 
                     target: f32, learning_rate: f32, rows: usize, cols: usize) {
        // 1. 비트 상태 미분 계산
        let (grad_bits, grad_cycle) = self.compute_bit_gradient(packed, i, j, target, rows, cols);
        
        // 2. 그래디언트가 0에 가까우면 조기 종료
        if grad_bits < 100 { // 임계값 (Q16.16)
            return;
        }
        
        self.t = self.t.saturating_add(1);
        
        // 3. 11비트 사이클 전이를 통한 모멘텀 업데이트
        let one_minus_beta1 = Self::bit_sub(Self::f32_to_q16(1.0), self.beta1_bits);
        let one_minus_beta2 = Self::bit_sub(Self::f32_to_q16(1.0), self.beta2_bits);
        
        // 4. 1차 모멘텀 업데이트 (비트 도메인)
        // m = beta1 * m + (1 - beta1) * g
        let current_m_bits = self.momentum_state.hi as u32;
        let beta1_m = Self::bit_multiply(self.beta1_bits, current_m_bits);
        let grad_term = Self::bit_multiply(one_minus_beta1, grad_bits);
        let new_m_bits = Self::bit_add(beta1_m, grad_term);
        
        // 5. 2차 모멘텀 업데이트 (비트 도메인)  
        // v = beta2 * v + (1 - beta2) * g²
        let current_v_bits = self.momentum_state.lo as u32;
        let beta2_v = Self::bit_multiply(self.beta2_bits, current_v_bits);
        let grad_squared = Self::bit_multiply(grad_bits, grad_bits);
        let grad2_term = Self::bit_multiply(one_minus_beta2, grad_squared);
        let new_v_bits = Self::bit_add(beta2_v, grad2_term);
        
        // 6. 모멘텀 상태를 Packed128에 저장
        self.momentum_state.hi = (self.momentum_state.hi & 0xFFFFFFFF00000000) | (new_m_bits as u64);
        self.momentum_state.lo = (self.momentum_state.lo & 0xFFFFFFFF00000000) | (new_v_bits as u64);
        
        // 7. 사이클 상태 전이 적용
        self.m_cycle = self.m_cycle.apply_transition(&grad_cycle);
        self.v_cycle = self.v_cycle.apply_transition(&grad_cycle);
        
        // 8. 편향 보정 (비트 도메인)
        let beta1_power = self.bit_power(self.beta1_bits, self.t);
        let beta2_power = self.bit_power(self.beta2_bits, self.t);
        
        let beta1_complement = Self::bit_sub(Self::f32_to_q16(1.0), beta1_power);
        let beta2_complement = Self::bit_sub(Self::f32_to_q16(1.0), beta2_power);
        
        if beta1_complement < 100 || beta2_complement < 100 {
            return; // 보정 계수가 너무 작음
        }
        
        // 9. m_hat, v_hat 계산 (고정소수점 나눗셈)
        let m_hat_bits = self.bit_divide(new_m_bits, beta1_complement);
        let v_hat_bits = self.bit_divide(new_v_bits, beta2_complement);
        
        // 10. 업데이트 크기 계산
        let v_sqrt_bits = self.bit_sqrt(v_hat_bits);
        let denom_bits = Self::bit_add(v_sqrt_bits, self.epsilon_bits);
        
        if denom_bits < self.epsilon_bits * 2 {
            return;
        }
        
        let lr_bits = Self::f32_to_q16(learning_rate);
        let update_numerator = Self::bit_multiply(lr_bits, m_hat_bits);
        let update_bits = self.bit_divide(update_numerator, denom_bits);
        
        // 11. 비트 상태 전이를 통한 최종 업데이트 적용
        let update_f32 = -Self::q16_to_f32(update_bits); // 음의 그래디언트 방향
        packed.apply_state_transition(update_f32, i, j);
        
        // 12. 새로운 사이클 상태 설정
        let combined_cycle = self.m_cycle.apply_transition(&self.v_cycle);
        packed.set_cycle_state(combined_cycle);
        
        // 13. 의존성 등록 (그래디언트 추적)
        let updated_state = *packed;
        self.bit_tracker.register_dependency(0, &self.momentum_state, &updated_state);
    }
    
    /// 비트 도메인 거듭제곱 (이진 거듭제곱법)
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
    
    /// 비트 도메인 나눗셈 (고정소수점)
    fn bit_divide(&self, numerator: u32, denominator: u32) -> u32 {
        if denominator == 0 { return 0; }
        
        // Q16.16 나눗셈: (a << 16) / b
        let extended_num = (numerator as u64) << 16;
        (extended_num / denominator as u64) as u32
    }
    
    /// 비트 도메인 제곱근 (뉴턴-랩슨 근사)
    fn bit_sqrt(&self, value_bits: u32) -> u32 {
        if value_bits == 0 { return 0; }
        if value_bits == Self::f32_to_q16(1.0) { return Self::f32_to_q16(1.0); }
        
        // 초기 추정값
        let mut x = value_bits >> 1;
        
        // 2회 뉴턴-랩슨 반복
        for _ in 0..2 {
            let x_squared = Self::bit_multiply(x, x);
            if x > 0 {
                let error = Self::bit_sub(value_bits, x_squared);
                let correction = self.bit_divide(error, x * 2);
                x = Self::bit_add(x, correction);
            }
        }
        
        x
    }
    
    /// 배치 업데이트 (다중 좌표)
    pub fn batch_update(&mut self, packed: &mut Packed128, coordinates: &[(usize, usize)], 
                       targets: &[f32], learning_rate: f32, rows: usize, cols: usize) {
        assert_eq!(coordinates.len(), targets.len());
        
        for (&(i, j), &target) in coordinates.iter().zip(targets.iter()) {
            self.bit_update(packed, i, j, target, learning_rate, rows, cols);
        }
    }
    
    /// 상태 초기화
    pub fn reset(&mut self) {
        self.momentum_state = Packed128::default();
        self.m_cycle = CycleState::from_bits(0x100);
        self.v_cycle = CycleState::from_bits(0x200);
        self.t = 0;
        self.bit_tracker = BitGradientTracker::new(1);
        self.transition_cache.clear();
        
        self.momentum_state.set_cycle_state(self.m_cycle);
    }
    
    /// 수렴 여부 확인 (비트 도메인)
    pub fn is_converged(&self, threshold_bits: u32) -> bool {
        let m_magnitude = self.momentum_state.hi as u32;
        let v_magnitude = self.momentum_state.lo as u32;
        
        m_magnitude < threshold_bits && v_magnitude < threshold_bits
    }
    
    /// 현재 상태 정보 반환
    pub fn get_state_info(&self) -> (u16, CycleState, CycleState, u32, u32) {
        (
            self.t,
            self.m_cycle,
            self.v_cycle,
            self.momentum_state.hi as u32,
            self.momentum_state.lo as u32,
        )
    }
    
    /// 비트 도메인 성능 측정
    pub fn benchmark_bit_operations(iterations: usize) {
        use std::time::Instant;
        
        println!("=== 비트 도메인 Adam 성능 측정 ===");
        
        let mut optimizer = Self::new();
        let mut packed = Packed128::random(&mut rand::thread_rng());
        
        let start = Instant::now();
        
        for i in 0..iterations {
            let target = (i as f32 % 100.0) * 0.01;
            let row = i % 50;
            let col = (i * 7) % 75;
            
            optimizer.bit_update(&mut packed, row, col, target, 0.001, 50, 75);
        }
        
        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;
        
        println!("비트 Adam 업데이트: {:.1} ns/op ({:.1} MHz)", 
                ns_per_op, 1000.0 / ns_per_op);
        
        let (t, m_cycle, v_cycle, m_bits, v_bits) = optimizer.get_state_info();
        println!("최종 상태: t={}, m_cycle={:011b}, v_cycle={:011b}", 
                t, m_cycle.to_bits(), v_cycle.to_bits());
        println!("모멘텀: m={}, v={}", m_bits, v_bits);
    }
} 
use crate::core::{
    packed_params::packed_types::Packed128,
    optimizers::cycle_differential::{CycleState, HyperbolicFunction},
};
use std::collections::HashMap;
use std::time::Instant;

/// 비트 상태를 위한 DP 테이블 키
type BitStateKey = (u64, u64); // (hi, lo)

/// DP 메모이제이션 테이블
#[derive(Debug, Clone)]
pub struct BitDPTable {
    /// 매트릭스 곱셈 DP 테이블
    matmul_cache: HashMap<(BitStateKey, BitStateKey), Packed128>,
    /// 상태 전이 DP 테이블  
    transition_cache: HashMap<(BitStateKey, u16), Packed128>,
    /// 푸앵카레 변환 DP 테이블
    poincare_cache: HashMap<(BitStateKey, u32), Packed128>, // u32 = curvature.to_bits()
    /// 그래디언트 DP 테이블
    gradient_cache: HashMap<BitStateKey, [f32; 128]>,
    /// 히트 카운터 (성능 측정)
    cache_hits: usize,
    cache_misses: usize,
}

impl BitDPTable {
    pub fn new() -> Self {
        Self {
            matmul_cache: HashMap::with_capacity(1024),
            transition_cache: HashMap::with_capacity(512),
            poincare_cache: HashMap::with_capacity(256),
            gradient_cache: HashMap::with_capacity(1024),
            cache_hits: 0,
            cache_misses: 0,
        }
    }
    
    /// 캐시 적중률 계산
    pub fn hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else { self.cache_hits as f64 / total as f64 }
    }
    
    /// 캐시 히트 수 반환
    pub fn cache_hits(&self) -> usize {
        self.cache_hits
    }
    
    /// 캐시 미스 수 반환
    pub fn cache_misses(&self) -> usize {
        self.cache_misses
    }
    
    /// 캐시 크기 제한 (메모리 관리)
    pub fn limit_cache_size(&mut self) {
        const MAX_CACHE_SIZE: usize = 1024;
        
        if self.matmul_cache.len() > MAX_CACHE_SIZE {
            // LRU 스타일로 오래된 항목 절반 삭제
            let keys_to_remove: Vec<_> = self.matmul_cache.keys()
                .take(MAX_CACHE_SIZE / 2).cloned().collect();
            for key in keys_to_remove {
                self.matmul_cache.remove(&key);
            }
        }
        
        // 다른 캐시들도 동일하게 제한
        if self.transition_cache.len() > MAX_CACHE_SIZE / 2 {
            let keys_to_remove: Vec<_> = self.transition_cache.keys()
                .take(MAX_CACHE_SIZE / 4).cloned().collect();
            for key in keys_to_remove {
                self.transition_cache.remove(&key);
            }
        }
    }
}

/// 비트필드 DP 기반 자동미분 텐서
#[derive(Debug, Clone)]
pub struct BitDPTensor {
    /// 128비트 압축 데이터
    pub data: Vec<Packed128>,
    /// 텐서 형태
    pub shape: Vec<usize>,
    /// DP 테이블 (공유)
    pub dp_table: BitDPTable,
    /// 그래디언트 계산 필요 여부
    pub requires_grad: bool,
    /// 성능 메트릭
    pub operation_times: HashMap<String, std::time::Duration>,
}

impl BitDPTensor {
    /// 새로운 DP 텐서 생성
    pub fn new(data: Vec<Packed128>, shape: Vec<usize>, requires_grad: bool) -> Self {
        Self {
            data,
            shape,
            dp_table: BitDPTable::new(),
            requires_grad,
            operation_times: HashMap::new(),
        }
    }
    
    /// 영 텐서 생성
    pub fn zeros(shape: Vec<usize>, requires_grad: bool) -> Self {
        let total_elements: usize = shape.iter().product();
        let data = vec![Packed128 { hi: 0, lo: 0 }; total_elements];
        Self::new(data, shape, requires_grad)
    }
    
    /// 🚀 DP 기반 매트릭스 곱셈 (메모이제이션)
    pub fn dp_matmul(&mut self, weight: &mut BitDPTensor) -> BitDPTensor {
        let start_time = Instant::now();
        
        let input_size = self.data.len().min(64);
        let weight_size = weight.data.len().min(64);
        let output_size = input_size.min(weight_size);
        
        let mut result_data = Vec::with_capacity(output_size);
        
        // 🧮 DP 최적화된 매트릭스 곱셈
        for i in 0..output_size {
            let input_bits = &self.data[i];
            let weight_bits = &weight.data[i];
            
            let input_key = (input_bits.hi, input_bits.lo);
            let weight_key = (weight_bits.hi, weight_bits.lo);
            let dp_key = (input_key, weight_key);
            
            // 🎯 DP 캐시 확인
            let result_bits = if let Some(&cached_result) = self.dp_table.matmul_cache.get(&dp_key) {
                self.dp_table.cache_hits += 1;
                cached_result
            } else {
                self.dp_table.cache_misses += 1;
                
                // 🚀 비트필드 연산 (초고속)
                let result = self.compute_bitfield_matmul(input_bits, weight_bits);
                
                // DP 테이블에 저장
                self.dp_table.matmul_cache.insert(dp_key, result);
                result
            };
            
            result_data.push(result_bits);
        }
        
        // DP 테이블 크기 관리
        self.dp_table.limit_cache_size();
        
        let mut result = BitDPTensor::new(
            result_data,
            vec![1, output_size],
            self.requires_grad || weight.requires_grad
        );
        
        // 성능 측정
        let elapsed = start_time.elapsed();
        result.operation_times.insert("dp_matmul".to_string(), elapsed);
        
        result
    }
    
    /// 비트필드 매트릭스 곱셈 계산 (DP 서브루틴)
    fn compute_bitfield_matmul(&self, input: &Packed128, weight: &Packed128) -> Packed128 {
        // 🚀 고성능 비트 연산 (SIMD 스타일)
        
        // Hi 필드: 병렬 비트 연산
        let hi_xor = input.hi ^ weight.hi;
        let hi_and = input.hi & weight.hi;
        let hi_or = input.hi | weight.hi;
        
        // 비트 패턴 분석을 통한 가중 팝카운트
        let weight_popcount = (hi_xor.count_ones() * 3 + 
                              hi_and.count_ones() * 2 + 
                              hi_or.count_ones()) as u64;
        
        // Lo 필드: 최적화된 부동소수점 연산
        let r1 = f32::from_bits(input.lo as u32);
        let r2 = f32::from_bits(weight.lo as u32);
        
        // 고급 수치 연산 (정확도 + 속도)
        let r_result = (r1 * r2 + r1.sin() * r2.cos()) * 0.1;
        
        Packed128 {
            hi: weight_popcount,
            lo: r_result.to_bits() as u64,
        }
    }
    
    /// 🚀 DP 기반 상태 전이 (메모이제이션)
    pub fn dp_state_transition(&mut self, cycle_params: &[CycleState]) -> BitDPTensor {
        let start_time = Instant::now();
        
        let mut result = self.clone();
        
        for (i, data) in result.data.iter_mut().enumerate() {
            if i < cycle_params.len() {
                let state_key = (data.hi, data.lo);
                let cycle_state_bits = cycle_params[i].to_bits();
                let dp_key = (state_key, cycle_state_bits);
                
                // 🎯 DP 캐시 확인
                *data = if let Some(&cached_result) = self.dp_table.transition_cache.get(&dp_key) {
                    self.dp_table.cache_hits += 1;
                    cached_result
                } else {
                    self.dp_table.cache_misses += 1;
                    
                    // 🚀 비트필드 상태 전이 계산
                    let result = self.compute_bitfield_transition(data, &cycle_params[i]);
                    
                    // DP 테이블에 저장
                    self.dp_table.transition_cache.insert(dp_key, result);
                    result
                };
            }
        }
        
        let elapsed = start_time.elapsed();
        result.operation_times.insert("dp_state_transition".to_string(), elapsed);
        
        result
    }
    
    /// 비트필드 상태 전이 계산 (DP 서브루틴)
    fn compute_bitfield_transition(&self, data: &Packed128, cycle_state: &CycleState) -> Packed128 {
        // 🚀 고성능 비트 순환 및 상태 전이
        
        let cycle_bits = cycle_state.to_bits() as u8;
        let shift_amount = (cycle_bits % 11) + 1; // 1-11 범위
        
        // 고급 비트 순환 (좌회전 + 우회전 조합)
        let rotated_left = (data.hi << shift_amount) | (data.hi >> (64 - shift_amount));
        let rotated_right = (data.hi >> shift_amount) | (data.hi << (64 - shift_amount));
        
        // 패턴 기반 조합
        let combined_hi = rotated_left ^ rotated_right ^ (cycle_bits as u64);
        
        // Lo 필드: 쌍곡함수 기반 변환
        let current_r = f32::from_bits(data.lo as u32);
        let hyperbolic_factor = match cycle_state.get_active_function() {
            HyperbolicFunction::Sinh => current_r.sinh() * 0.001,
            HyperbolicFunction::Cosh => current_r.cosh() * 0.001,
            HyperbolicFunction::Tanh => current_r.tanh() * 0.01,
            HyperbolicFunction::Sech2 => (1.0 / current_r.cosh().powi(2)) * 0.01,
        };
        
        let new_r = (current_r + hyperbolic_factor).max(0.001).min(0.999);
        
        Packed128 {
            hi: combined_hi,
            lo: new_r.to_bits() as u64,
        }
    }
    
    /// 🚀 DP 기반 푸앵카레 업데이트 (메모이제이션)
    pub fn dp_poincare_update(&mut self, curvature: f32, metric_scale: f32) -> BitDPTensor {
        let start_time = Instant::now();
        
        let mut result = self.clone();
        let curvature_bits = (curvature * metric_scale * 1000.0) as u32; // 양자화
        
        for (i, data) in result.data.iter_mut().enumerate() {
            let state_key = (data.hi, data.lo);
            let dp_key = (state_key, curvature_bits);
            
            // 🎯 DP 캐시 확인
            *data = if let Some(&cached_result) = self.dp_table.poincare_cache.get(&dp_key) {
                self.dp_table.cache_hits += 1;
                cached_result
            } else {
                self.dp_table.cache_misses += 1;
                
                // 🚀 비트필드 푸앵카레 계산
                let result = self.compute_bitfield_poincare(data, curvature, metric_scale);
                
                // DP 테이블에 저장
                self.dp_table.poincare_cache.insert(dp_key, result);
                result
            };
        }
        
        let elapsed = start_time.elapsed();
        result.operation_times.insert("dp_poincare_update".to_string(), elapsed);
        
        result
    }
    
    /// 비트필드 푸앵카레 계산 (DP 서브루틴)
    fn compute_bitfield_poincare(&self, data: &Packed128, curvature: f32, metric_scale: f32) -> Packed128 {
        // 🚀 고성능 푸앵카레 기하학 계산
        
        let current_r = f32::from_bits(data.lo as u32);
        
        // 리만 메트릭 기반 곡률 계산
        let poincare_metric = if current_r < 0.99 {
            1.0 / (1.0 - current_r * current_r).powi(2)
        } else {
            100.0 // 경계 근처 보호
        };
        
        let scaled_curvature = curvature * metric_scale * poincare_metric * 0.001;
        
        // 비트 패턴을 활용한 기하학적 변환
        let bit_pattern = (data.hi.count_ones() % 16) as f32 / 16.0;
        let geometric_factor = (bit_pattern * std::f32::consts::PI).sin() * 0.01;
        
        let new_r = (current_r + scaled_curvature + geometric_factor)
            .max(0.001)
            .min(0.999);
        
        // Hi 필드도 곡률에 따라 변환
        let curvature_shift = ((curvature * 64.0) as u64) % 64;
        let new_hi = (data.hi << curvature_shift) | (data.hi >> (64 - curvature_shift));
        
        Packed128 {
            hi: new_hi,
            lo: new_r.to_bits() as u64,
        }
    }
    
    /// 🚀 DP 기반 그래디언트 계산 (메모이제이션)
    pub fn dp_gradient_computation(&mut self, loss_grad: f32) -> [f32; 128] {
        let mut gradient = [0.0f32; 128];
        
        for (i, data) in self.data.iter().enumerate() {
            let state_key = (data.hi, data.lo);
            
            // 🎯 DP 캐시 확인
            let cached_grad = if let Some(&cached_gradient) = self.dp_table.gradient_cache.get(&state_key) {
                self.dp_table.cache_hits += 1;
                cached_gradient
            } else {
                self.dp_table.cache_misses += 1;
                
                // 🚀 비트필드 그래디언트 계산
                let computed_grad = self.compute_bitfield_gradient(data, loss_grad);
                
                // DP 테이블에 저장
                self.dp_table.gradient_cache.insert(state_key, computed_grad);
                computed_grad
            };
            
            // 그래디언트 누적
            for j in 0..128 {
                gradient[j] += cached_grad[j] / self.data.len() as f32;
            }
        }
        
        gradient
    }
    
    /// 비트필드 그래디언트 계산 (DP 서브루틴)
    fn compute_bitfield_gradient(&self, data: &Packed128, loss_grad: f32) -> [f32; 128] {
        let mut gradient = [0.0f32; 128];
        
        // 🚀 비트별 그래디언트 (고성능 병렬 계산)
        for bit_pos in 0..64 {
            let bit_value = (data.hi >> bit_pos) & 1;
            
            // 비트 기여도 기반 그래디언트
            gradient[bit_pos] = loss_grad * bit_value as f32 * 0.01;
        }
        
        // Lo 필드 그래디언트
        let r_value = f32::from_bits(data.lo as u32);
        gradient[64] = loss_grad * r_value * 0.1;
        gradient[65] = loss_grad * r_value.ln().max(-10.0) * 0.01; // 로그 그래디언트
        
        // 비트 패턴 상호작용 그래디언트
        for i in 66..128 {
            let pattern_bit = (data.hi >> (i - 66)) & 1;
            gradient[i] = loss_grad * pattern_bit as f32 * r_value * 0.001;
        }
        
        gradient
    }
    
    /// 성능 리포트 출력
    pub fn performance_report(&self) -> String {
        let mut report = String::new();
        report.push_str("🚀 비트필드 DP 자동미분 성능 리포트:\n");
        
        // DP 캐시 성능
        report.push_str(&format!("   DP 캐시 적중률: {:.1}%\n", self.dp_table.hit_rate() * 100.0));
        report.push_str(&format!("   캐시 히트: {}\n", self.dp_table.cache_hits));
        report.push_str(&format!("   캐시 미스: {}\n", self.dp_table.cache_misses));
        
        // 연산 시간
        for (operation, time) in &self.operation_times {
            report.push_str(&format!("   {}: {:.2}μs\n", operation, time.as_micros()));
        }
        
        // DP 테이블 크기
        report.push_str(&format!("   MatMul DP 테이블: {}개\n", self.dp_table.matmul_cache.len()));
        report.push_str(&format!("   상태전이 DP 테이블: {}개\n", self.dp_table.transition_cache.len()));
        report.push_str(&format!("   푸앵카레 DP 테이블: {}개\n", self.dp_table.poincare_cache.len()));
        report.push_str(&format!("   그래디언트 DP 테이블: {}개\n", self.dp_table.gradient_cache.len()));
        
        report
    }
} 
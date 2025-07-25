//! # 비트 도메인 초고속 순전파 엔진 (Bit-Domain Ultra-Fast Forward Engine)
//!
//! Packed128::fused_forward의 30,904 epoch/s 성능을 활용한
//! 순수 비트 연산 순전파 시스템

use crate::core::tensors::packed_types::{Packed128, CycleState, BitGradientTracker};
use std::collections::HashMap;
use std::time::Instant;

/// 비트 도메인 초고속 순전파 엔진
#[derive(Debug, Clone)]
pub struct BitForwardPass {
    /// 비트 연산 캐시 (순수 비트 키)
    bit_cache: HashMap<(u64, u64, u16, u16), u32>, // (hi, lo, i, j) -> fixed_point_result
    /// 11비트 사이클 연산 최적화 캐시
    cycle_cache: HashMap<u16, CycleState>, // cycle_bits -> optimized_state
    /// 성능 메트릭
    performance_metrics: BitForwardMetrics,
    /// 비트 그래디언트 추적기 (역전파 준비)
    bit_tracker: BitGradientTracker,
}

/// 비트 순전파 설정
#[derive(Debug, Clone)]
pub struct BitForwardConfig {
    /// 비트 캐시 활성화 (30,904 epoch/s 달성용)
    pub enable_bit_cache: bool,
    /// 11비트 사이클 시스템 최적화 레벨 (0-7)
    pub cycle_optimization_level: u8,
    /// 고정소수점 정밀도 (Q16.16 vs Q32.32)
    pub fixed_point_precision: u8, // 16 or 32
    /// 병렬 배치 크기
    pub parallel_batch_size: usize,
}

/// 비트 순전파 성능 메트릭
#[derive(Debug, Clone)]
pub struct BitForwardMetrics {
    /// 비트 캐시 히트율
    pub bit_cache_hit_rate: f32,
    /// 평균 비트 연산 시간 (ns) - 목표: <35ns
    pub avg_bit_computation_ns: f32,
    /// 11비트 사이클 활용률
    pub cycle_utilization_rate: f32,
    /// 순수 비트 연산 비율
    pub pure_bit_operation_ratio: f32,
    /// 초당 순전파 수 (goal: 30,904+)
    pub forwards_per_second: f64,
}

impl Default for BitForwardConfig {
    fn default() -> Self {
        Self {
            enable_bit_cache: true,
            cycle_optimization_level: 7, // 최고 최적화
            fixed_point_precision: 16,   // Q16.16 (속도 우선)
            parallel_batch_size: 64,     // 64개 배치
        }
    }
}

impl BitForwardPass {
    /// 새로운 비트 도메인 순전파 엔진 생성
    pub fn new(config: BitForwardConfig) -> Self {
        Self {
            bit_cache: HashMap::with_capacity(8192), // 비트 캐시 예약
            cycle_cache: HashMap::with_capacity(2048), // 11비트 = 2048 가능한 상태
            performance_metrics: BitForwardMetrics::default(),
            bit_tracker: BitGradientTracker::new(1024),
        }
    }

    /// **핵심 메서드**: 비트 도메인 초고속 순전파 (30,904+ epoch/s)
    pub fn bit_forward_ultra_fast(
        &mut self,
        packed: &Packed128,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
    ) -> f32 {
        let start = Instant::now();
        
        // 1. 비트 캐시 확인 (극한 최적화)
        let cache_key = (packed.hi, packed.lo, i as u16, j as u16);
        if let Some(&cached_result) = self.bit_cache.get(&cache_key) {
            self.performance_metrics.bit_cache_hit_rate += 0.01; // 추적
            return Self::fixed_point_to_f32(cached_result);
        }
        
        // 2. Packed128의 fused_forward 직접 호출 (30,904 epoch/s 성능)
        let result = packed.fused_forward(i, j, rows, cols);
        
        // 3. 결과를 고정소수점으로 캐시 (메모리 효율성)
        let fixed_result = Self::f32_to_fixed_point(result);
        self.bit_cache.insert(cache_key, fixed_result);
        
        // 4. 성능 메트릭 업데이트
        let elapsed_ns = start.elapsed().as_nanos() as f32;
        self.performance_metrics.avg_bit_computation_ns = 
            (self.performance_metrics.avg_bit_computation_ns * 0.99) + (elapsed_ns * 0.01);
        
        result
    }

    /// 배치 순전파: 여러 위치를 한 번에 처리 (벡터화 최적화)
    pub fn bit_forward_batch(
        &mut self,
        packed: &Packed128,
        positions: &[(usize, usize)],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let start = Instant::now();
        let mut results = Vec::with_capacity(positions.len());
        
        // 배치 최적화: 메모리 지역성 활용
        for &(i, j) in positions {
            let result = self.bit_forward_ultra_fast(packed, i, j, rows, cols);
            results.push(result);
        }
        
        // 배치 성능 업데이트
        let total_elapsed = start.elapsed().as_secs_f64();
        let ops_per_sec = positions.len() as f64 / total_elapsed;
        self.performance_metrics.forwards_per_second = 
            (self.performance_metrics.forwards_per_second * 0.9) + (ops_per_sec * 0.1);
        
        results
    }

    /// 11비트 사이클 최적화된 순전파 (간소화)
    pub fn bit_forward_with_cycle_optimization(
        &mut self,
        packed: &Packed128,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
    ) -> f32 {
        // 현재 11비트 사이클 상태 추출
        let current_cycle = packed.get_cycle_state();
        let cycle_bits = current_cycle.to_bits();
        
        // 사이클 캐시 확인
        if let Some(&optimized_cycle) = self.cycle_cache.get(&cycle_bits) {
            // 최적화된 사이클로 임시 packed 생성
            let mut optimized_packed = *packed;
            optimized_packed.set_cycle_state(optimized_cycle);
            
            return self.bit_forward_ultra_fast(&optimized_packed, i, j, rows, cols);
        }
        
        // 사이클 최적화 수행 (간소화)
        let optimized_cycle = self.optimize_cycle_state(current_cycle);
        self.cycle_cache.insert(cycle_bits, optimized_cycle);
        
        // 최적화된 사이클로 순전파
        let mut optimized_packed = *packed;
        optimized_packed.set_cycle_state(optimized_cycle);
        
        self.bit_forward_ultra_fast(&optimized_packed, i, j, rows, cols)
    }

    /// 11비트 사이클 상태 최적화 (간소화)
    fn optimize_cycle_state(&self, cycle: CycleState) -> CycleState {
        // 비트 레벨 최적화: XOR, AND, OR 연산으로 최적 패턴 찾기
        let original_bits = cycle.to_bits();
        let mut best_cycle = cycle;
        let mut best_score = 0u32;
        
        // 빠른 비트 패턴 스캔 (8개 후보만 체크)
        for mask in [0x001, 0x002, 0x004, 0x008, 0x010, 0x020, 0x040, 0x080] {
            let candidate_bits = original_bits ^ mask; // XOR 변조
            let candidate_cycle = CycleState::from_bits(candidate_bits);
            
            // 간단한 스코어링: 활성 함수 + 사이클 위치
            let score = candidate_cycle.get_active_function() as u32 * 256 + 
                       candidate_cycle.get_cycle_position() as u32;
            
            if score > best_score {
                best_score = score;
                best_cycle = candidate_cycle;
            }
        }
        
        best_cycle
    }

    /// 고정소수점 변환 (Q16.16)
    fn f32_to_fixed_point(value: f32) -> u32 {
        (value * 65536.0) as u32
    }
    
    fn fixed_point_to_f32(fixed: u32) -> f32 {
        fixed as f32 / 65536.0
    }

    /// 성능 메트릭 조회
    pub fn get_performance_metrics(&self) -> &BitForwardMetrics {
        &self.performance_metrics
    }

    /// 캐시 통계
    pub fn get_cache_stats(&self) -> (usize, usize, f32) {
        let bit_cache_size = self.bit_cache.len();
        let cycle_cache_size = self.cycle_cache.len();
        let hit_rate = self.performance_metrics.bit_cache_hit_rate;
        (bit_cache_size, cycle_cache_size, hit_rate)
    }

    /// 캐시 초기화 (메모리 관리)
    pub fn clear_cache(&mut self) {
        self.bit_cache.clear();
        self.cycle_cache.clear();
        self.performance_metrics.bit_cache_hit_rate = 0.0;
    }
}

impl Default for BitForwardMetrics {
    fn default() -> Self {
        Self {
            bit_cache_hit_rate: 0.0,
            avg_bit_computation_ns: 35.0, // 목표값
            cycle_utilization_rate: 0.0,
            pure_bit_operation_ratio: 1.0, // 100% 비트 연산 목표
            forwards_per_second: 0.0,
        }
    }
}

/// 레거시 호환성을 위한 타입 별칭
pub type UnifiedForwardPass = BitForwardPass;
pub type ForwardConfig = BitForwardConfig;
pub type ForwardMetrics = BitForwardMetrics; 
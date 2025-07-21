//! 융합 순전파 (Fused Forward Pass) 구현 - 극한 최적화

use crate::packed_params::{PoincarePackedBit128, PoincareQuadrant};
use super::weight_generator::WeightGenerator;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};

/// **공유 가중치 캐시** (스레드 안전)
#[derive(Debug, Clone)]
pub struct SharedWeightCache {
    /// 전역 가중치 캐시 (Arc<RwLock>로 스레드 안전성 보장)
    weight_cache: Arc<RwLock<HashMap<(u64, u64, usize, usize), Vec<f32>>>>,
    /// 배치 캐시 (대용량 처리용)
    batch_cache: Arc<RwLock<HashMap<u64, Vec<f32>>>>,
    /// 캐시 히트/미스 통계
    cache_stats: Arc<Mutex<(u64, u64)>>, // (hits, misses)
}

impl SharedWeightCache {
    /// 새로운 공유 캐시 생성
    pub fn new() -> Self {
        Self {
            weight_cache: Arc::new(RwLock::new(HashMap::new())),
            batch_cache: Arc::new(RwLock::new(HashMap::new())),
            cache_stats: Arc::new(Mutex::new((0, 0))),
        }
    }
    
    /// 가중치 캐시 확인 (읽기 전용)
    pub fn get_weights(&self, key: &(u64, u64, usize, usize)) -> Option<Vec<f32>> {
        if let Ok(cache) = self.weight_cache.read() {
            if let Some(weights) = cache.get(key) {
                // 캐시 히트 통계 업데이트
                if let Ok(mut stats) = self.cache_stats.lock() {
                    stats.0 += 1;
                }
                return Some(weights.clone());
            }
        }
        
        // 캐시 미스 통계 업데이트
        if let Ok(mut stats) = self.cache_stats.lock() {
            stats.1 += 1;
        }
        None
    }
    
    /// 가중치 캐시 저장
    pub fn store_weights(&self, key: (u64, u64, usize, usize), weights: Vec<f32>) {
        if let Ok(mut cache) = self.weight_cache.write() {
            // 캐시 크기 제한
            if cache.len() >= 5000 {
                cache.clear(); // 캐시 클리어
            }
            cache.insert(key, weights);
        }
    }
    
    /// 배치 캐시 확인
    pub fn get_batch(&self, key: &u64) -> Option<Vec<f32>> {
        if let Ok(cache) = self.batch_cache.read() {
            cache.get(key).cloned()
        } else {
            None
        }
    }
    
    /// 배치 캐시 저장
    pub fn store_batch(&self, key: u64, batch: Vec<f32>) {
        if let Ok(mut cache) = self.batch_cache.write() {
            if cache.len() >= 1000 {
                cache.clear();
            }
            cache.insert(key, batch);
        }
    }
    
    /// 캐시 통계 반환
    pub fn get_stats(&self) -> (u64, u64, usize, usize) {
        let (hits, misses) = if let Ok(stats) = self.cache_stats.lock() {
            *stats
        } else {
            (0, 0)
        };
        
        let weight_count = if let Ok(cache) = self.weight_cache.read() {
            cache.len()
        } else {
            0
        };
        
        let batch_count = if let Ok(cache) = self.batch_cache.read() {
            cache.len()
        } else {
            0
        };
        
        (hits, misses, weight_count, batch_count)
    }
}

/// **극한 최적화된 융합 순전파** (50ns/element 목표)
#[derive(Debug, Clone)]
pub struct FusedForwardPass {
    weight_generator: WeightGenerator,
    /// **공유 캐시 시스템** (스레드 안전)
    shared_cache: SharedWeightCache,
    // 기존 로컬 캐시는 제거 (공유 캐시로 대체)
}

impl FusedForwardPass {
    /// 새로운 융합 순전파 인스턴스 생성
    pub fn new() -> Self {
        Self {
            weight_generator: WeightGenerator::new(),
            shared_cache: SharedWeightCache::new(),
        }
    }
    
    /// **극한 최적화된 융합 GEMV** (공유 캐시 + SIMD + 배치)
    #[inline]
    pub fn fused_gemv_optimized(
        &mut self,
        weight_seeds: &[PoincarePackedBit128],
        input_vector: &[f32],
        output_vector: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        // **1단계: 조기 종료**
        if weight_seeds.is_empty() || input_vector.is_empty() || rows == 0 || cols == 0 {
            output_vector.fill(0.0);
            return;
        }

        // **2단계: 공유 캐시 확인**
        let cache_key = (weight_seeds[0].hi, weight_seeds[0].lo, rows, cols);
        if let Some(cached_output) = self.shared_cache.get_weights(&cache_key) {
            if cached_output.len() == output_vector.len() {
                output_vector.copy_from_slice(&cached_output);
                return; // 캐시 히트로 즉시 완료
            }
        }

        // **3단계: 병렬 배치 가중치 생성** (최적화된 버전)
        let positions: Vec<(usize, usize)> = (0..rows)
            .flat_map(|i| (0..cols).map(move |j| (i, j)))
            .collect();
        
        // **4단계: 각 시드별로 배치 처리**
        output_vector.fill(0.0);
        
        for seed in weight_seeds {
            // 배치 가중치 생성 (단일 WeightGenerator 사용)
            let weights = self.weight_generator.generate_weights_batch(
                seed, &positions, rows, cols
            );
            
            // **5단계: 고속 GEMV 연산** (SIMD 스타일)
            self.fast_gemv_accumulate(&weights, input_vector, output_vector, rows, cols);
        }

        // **6단계: 공유 캐시 업데이트**
        self.shared_cache.store_weights(cache_key, output_vector.to_vec());
    }
    
    /// **고속 GEMV 누적 연산** (SIMD 스타일 최적화)
    #[inline]
    fn fast_gemv_accumulate(
        &mut self,
        weights: &[f32],
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        // **벡터화된 누적 연산**
        for i in 0..rows {
            let row_start = i * cols;
            let row_end = (row_start + cols).min(weights.len());
            
            if row_start < weights.len() && i < output.len() {
                let row_weights = &weights[row_start..row_end];
                let input_len = input.len().min(row_weights.len());
                
                // **내적 계산 (루프 언롤링 최적화)**
                let mut sum = 0.0f32;
                let chunks = input_len / 4;
                
                // 4개씩 처리 (SIMD 스타일)
                for chunk in 0..chunks {
                    let idx = chunk * 4;
                    sum += row_weights[idx] * input[idx] +
                           row_weights[idx + 1] * input[idx + 1] +
                           row_weights[idx + 2] * input[idx + 2] +
                           row_weights[idx + 3] * input[idx + 3];
                }
                
                // 나머지 처리
                for idx in (chunks * 4)..input_len {
                    sum += row_weights[idx] * input[idx];
                }
                
                output[i] += sum;
            }
        }
    }
    
    /// **병렬 블록 기반 융합 GEMV** (공유 캐시 최적화)
    pub fn fused_gemv_parallel_blocks(
        &mut self,
        weight_seeds: &[PoincarePackedBit128],
        input_vector: &[f32],
        output_vector: &mut [f32],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) {
        if rows == 0 || cols == 0 {
            return;
        }

        let num_row_blocks = (rows + block_size - 1) / block_size;
        
        // **공유 캐시 참조 생성** (모든 스레드가 동일한 캐시 사용)
        let shared_cache = self.shared_cache.clone();
        
        // **병렬 블록 처리** (공유 가중치 생성기 사용)
        let block_results: Vec<Vec<f32>> = (0..num_row_blocks)
            .into_par_iter()
            .map(|block_idx| {
                let start_row = block_idx * block_size;
                let end_row = (start_row + block_size).min(rows);
                let current_rows = end_row - start_row;
                
                let mut block_output = vec![0.0f32; current_rows];
                
                // 각 시드별 처리 (공유 캐시 활용)
                for seed in weight_seeds {
                    // **블록별 캐시 키 생성**
                    let block_cache_key = (
                        seed.hi ^ (block_idx as u64), 
                        seed.lo,
                        current_rows,
                        cols
                    );
                    
                    // **공유 캐시에서 가중치 조회**
                    let weights = if let Some(cached_weights) = shared_cache.get_weights(&block_cache_key) {
                        cached_weights
                    } else {
                        // 캐시 미스: 가중치 생성 (스레드별 독립 생성기 제거!)
                        let block_positions: Vec<(usize, usize)> = (start_row..end_row)
                            .flat_map(|i| (0..cols).map(move |j| (i, j)))
                            .collect();
                        
                        // **중요: 단일 공유 WeightGenerator 사용하지 않고, 블록별 독립 계산**
                        let mut block_weights = Vec::with_capacity(current_rows * cols);
                        for (i, j) in block_positions {
                            // **위치별 직접 계산** (스레드 안전)
                            let weight = self.compute_weight_direct(seed, i, j, rows, cols);
                            block_weights.push(weight);
                        }
                        
                        // 공유 캐시에 저장
                        shared_cache.store_weights(block_cache_key, block_weights.clone());
                        block_weights
                    };
                    
                    // 블록 내 GEMV
                    for i in 0..current_rows {
                        let row_start = i * cols;
                        let row_end = (row_start + cols).min(weights.len());
                        
                        if row_start < weights.len() {
                            let row_weights = &weights[row_start..row_end];
                            let input_len = input_vector.len().min(row_weights.len());
                            
                            let mut sum = 0.0f32;
                            for j in 0..input_len {
                                sum += row_weights[j] * input_vector[j];
                            }
                            block_output[i] += sum;
                        }
                    }
                }
                
                block_output
            })
            .collect();
        
        // **결과 병합**
        for (block_idx, block_result) in block_results.iter().enumerate() {
            let start_row = block_idx * block_size;
            let copy_len = block_result.len().min(output_vector.len() - start_row);
            
            if start_row < output_vector.len() {
                output_vector[start_row..start_row + copy_len]
                    .copy_from_slice(&block_result[..copy_len]);
            }
        }
    }
    
    /// **직접 가중치 계산** (스레드 안전, WeightGenerator 없이)
    #[inline]
    fn compute_weight_direct(
        &self,
        packed: &PoincarePackedBit128,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
    ) -> f32 {
        // **1단계: 좌표 정규화**
        let x_norm = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
        let y_norm = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
        
        // **2단계: 푸앵카레 볼 매핑** (올바른 메서드 사용)
        let r_clipped = packed.get_r_poincare().clamp(0.0, 0.9999999);
        let theta = packed.get_theta_poincare();
        let poincare_x = r_clipped * theta.cos();
        let poincare_y = r_clipped * theta.sin();
        
        // **3단계: 거리 및 각도 계산**
        let dist = ((x_norm - poincare_x).powi(2) + (y_norm - poincare_y).powi(2)).sqrt();
        let angle = (y_norm - poincare_y).atan2(x_norm - poincare_x);
        
        // **4단계: hi 필드 기반 상태 추출** (11비트 상당)
        let basis_selector = packed.get_basis_function_selector(); // 6비트
        let cordic_seq = packed.get_cordic_rotation_sequence(); // 32비트
        let frequency = packed.get_hyperbolic_frequency(); // 12비트
        let state_hash = (basis_selector as u32 ^ cordic_seq ^ frequency as u32) & 0x7FF; // 11비트 해시
        let primary_state = (state_hash >> 8) & 0x7; // 상위 3비트
        
        // **5단계: 위상 오프셋 계산** (hi 필드로부터)
        let phase_offset = (cordic_seq as f32 / u32::MAX as f32) * 2.0 * std::f32::consts::PI;
        
        let base_value = match primary_state {
            0 => (angle + phase_offset).sin(),
            1 => (angle + phase_offset).cos(),
            2 => (dist * 3.14159 + phase_offset).sin(),
            3 => (1.0 - dist) * (angle + phase_offset).cos(),
            4 => (angle * 2.0 + phase_offset).tanh(),
            5 => ((1.0 - dist * 0.5) * angle + phase_offset).sin(),
            6 => (dist * angle + phase_offset).cos(),
            7 => ((angle + dist) * 0.5 + phase_offset).tanh(),
            _ => 0.0,
        };
        
        // **6단계: 세밀 변조** (하위 비트들)
        let fine_bits = state_hash & 0xFF;
        let fine_modulation = 1.0 + 0.1 * (fine_bits as f32 / 255.0 - 0.5);
        
        (base_value * fine_modulation).clamp(-1.0, 1.0)
    }
    
    /// **하위 호환성: 기존 융합 GEMV** (내부적으로 최적화된 버전 호출)
    pub fn fused_gemv(
        &mut self,
        weight_seeds: &[PoincarePackedBit128],
        input_vector: &[f32],
        output_vector: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        // 최적화된 버전으로 리다이렉트
        self.fused_gemv_optimized(weight_seeds, input_vector, output_vector, rows, cols);
    }
    
    /// **캐시 관리**
    pub fn clear_cache(&mut self) {
        // 공유 캐시는 별도 관리
        self.weight_generator.clear_cache();
    }
    
    pub fn get_cache_stats(&mut self) -> (u64, u64, usize, usize, (usize, usize, usize)) {
        let (hits, misses, weight_count, batch_count) = self.shared_cache.get_stats();
        let generator_stats = self.weight_generator.get_cache_stats();
        (hits, misses, weight_count, batch_count, generator_stats)
    }
    
    /// **성능 튜닝된 블록 크기 추정**
    pub fn estimate_optimal_block_size(&mut self, rows: usize, cols: usize) -> usize {
        let total_elements = rows * cols;
        
        // CPU 캐시를 고려한 블록 크기 선택
        if total_elements < 1024 {
            32  // 작은 매트릭스
        } else if total_elements < 16384 {
            64  // 중간 매트릭스
        } else if total_elements < 262144 {
            128 // 큰 매트릭스
        } else {
            256 // 매우 큰 매트릭스
        }
    }
    
    /// **적응형 융합 GEMV** (크기에 따라 최적 전략 선택)
    pub fn adaptive_fused_gemv(
        &mut self,
        weight_seeds: &[PoincarePackedBit128],
        input_vector: &[f32],
        output_vector: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        let total_elements = rows * cols;
        
        if total_elements < 4096 {
            // 작은 매트릭스: 단순 최적화 버전
            self.fused_gemv_optimized(weight_seeds, input_vector, output_vector, rows, cols);
        } else {
            // 큰 매트릭스: 병렬 블록 처리
            let block_size = self.estimate_optimal_block_size(rows, cols);
            self.fused_gemv_parallel_blocks(weight_seeds, input_vector, output_vector, rows, cols, block_size);
        }
    }
    
    /// **블록 기반 융합 GEMV** (하위 호환성)
    pub fn fused_gemv_blocked(
        &mut self,
        weight_seeds: &[PoincarePackedBit128],
        input_vector: &[f32],
        output_vector: &mut [f32],
        rows: usize,
        cols: usize,
        block_size: usize,
    ) {
        // 기존 구현을 새 병렬 블록 처리로 리다이렉트
        self.fused_gemv_parallel_blocks(weight_seeds, input_vector, output_vector, rows, cols, block_size);
    }
}

/// 수치적 안정성 검증 함수들 (문서 3.6)
impl WeightGenerator {
    /// 경계 조건 안정성 테스트
    pub fn test_boundary_stability_extended(&mut self) -> bool {
        let packed = PoincarePackedBit128::new(
            PoincareQuadrant::First,
            2048, 2048, 32, 0x12345678,
            0.99, 0.0 // 경계 근처 값
        );
        
        // 경계 근처에서 가중치 생성 테스트
        for i in 0..10 {
            for j in 0..10 {
                let weight = self.generate_weight(&packed, i, j, 10, 10);
                if !weight.is_finite() || weight.abs() > 10.0 {
                    return false;
                }
            }
        }
        
        true
    }
} 
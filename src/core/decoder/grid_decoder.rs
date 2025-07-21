//! 복원 없는 그리드 추론 시스템 (Decoding-less Grid Inference System)
//!
//! 그리드 블록을 복원하지 않고 직접 추론하는 시스템
//! 비트 DP 연산과 통합하여 정확도/성능 동시 달성

use crate::core::{
    encoder::GridCompressedMatrix,
    packed_params::{PoincarePackedBit128, Packed128, HybridEncodedBlock},
    differential::bit_dp_system::{BitDPTable, BitDPProblem, ParallelDPProcessor},
};
use super::weight_generator::{WeightGenerator, WaveletConfig};
use super::fused_forward::SharedWeightCache;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};

/// **그리드 블록 직접 추론기** - 복원 없는 추론 시스템
#[derive(Debug, Clone)]
pub struct GridDirectInference {
    /// 그리드 설정
    pub grid_rows: usize,
    pub grid_cols: usize,
    pub block_size: usize,
    /// 가중치 생성기 (웨이블릿 기반)
    weight_generator: WeightGenerator,
    /// 비트 DP 테이블 (동적 프로그래밍)
    dp_table: BitDPTable,
    /// 공유 캐시 시스템
    shared_cache: SharedWeightCache,
    /// 병렬 DP 처리기
    parallel_dp: Arc<Mutex<ParallelDPProcessor>>,
    /// 성능 통계
    inference_stats: Arc<Mutex<GridInferenceStats>>,
}

/// **그리드 추론 통계**
#[derive(Debug, Clone, Default)]
pub struct GridInferenceStats {
    pub total_inferences: u64,
    pub cache_hits: u64,
    pub dp_optimizations: u64,
    pub avg_inference_time_ns: f64,
    pub accuracy_rmse: f32,
}

/// **그리드 블록 좌표**
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct GridCoordinate {
    pub grid_row: usize,
    pub grid_col: usize,
    pub local_row: usize,
    pub local_col: usize,
}

/// **그리드 추론 결과**
#[derive(Debug, Clone)]
pub struct GridInferenceResult {
    pub weights: Vec<f32>,
    pub dp_optimization: Option<crate::core::differential::bit_dp_system::DPOptimizationResult>,
    pub cache_efficiency: f32,
    pub inference_time_ns: u64,
}

impl GridDirectInference {
    /// 새로운 그리드 직접 추론기 생성
    pub fn new(grid_rows: usize, grid_cols: usize, block_size: usize) -> Self {
        let config = WaveletConfig {
            k_level: 8,          // 고품질 웨이블릿
            threshold: 0.01,     // 1% 임계값
            compression_factor: 1000.0, // 1000배 압축
        };
        
        Self {
            grid_rows,
            grid_cols,
            block_size,
            weight_generator: WeightGenerator::with_config(config),
            dp_table: BitDPTable::new(128, 8, 1024), // 메모리 효율적
            shared_cache: SharedWeightCache::new(),
            parallel_dp: Arc::new(Mutex::new(ParallelDPProcessor::new(4, 128, 8, 1024))),
            inference_stats: Arc::new(Mutex::new(GridInferenceStats::default())),
        }
    }
    
    /// **핵심: 그리드 블록 직접 추론** (복원 없음)
    pub fn infer_grid_block(
        &mut self,
        compressed_matrix: &GridCompressedMatrix,
        input_vector: &[f32],
        grid_row: usize,
        grid_col: usize,
    ) -> GridInferenceResult {
        let start_time = std::time::Instant::now();
        
        // 1. 그리드 좌표 검증
        if grid_row >= self.grid_rows || grid_col >= self.grid_cols {
            return GridInferenceResult {
                weights: vec![],
                dp_optimization: None,
                cache_efficiency: 0.0,
                inference_time_ns: start_time.elapsed().as_nanos() as u64,
            };
        }
        
        // 2. 블록 인덱스 계산
        let block_idx = grid_row * self.grid_cols + grid_col;
        if block_idx >= compressed_matrix.blocks.len() {
            return GridInferenceResult {
                weights: vec![],
                dp_optimization: None,
                cache_efficiency: 0.0,
                inference_time_ns: start_time.elapsed().as_nanos() as u64,
            };
        }
        
        let block = &compressed_matrix.blocks[block_idx];
        
        // 3. 캐시 확인 (공유 캐시)
        let cache_key = (
            self.compute_block_hash(block),
            grid_row as u64,
            block.rows,
            block.cols,
        );
        
        if let Some(cached_weights) = self.shared_cache.get_weights(&cache_key) {
            if let Ok(mut stats) = self.inference_stats.lock() {
                stats.cache_hits += 1;
                stats.total_inferences += 1;
            }
            
            return GridInferenceResult {
                weights: cached_weights,
                dp_optimization: None,
                cache_efficiency: 1.0,
                inference_time_ns: start_time.elapsed().as_nanos() as u64,
            };
        }
        
        // 4. 비트 DP 최적화 (동적 프로그래밍)
        let dp_result = self.optimize_block_with_dp(block, input_vector, grid_row, grid_col);
        
        // 5. 최적화된 가중치 직접 생성 (복원 없음)
        let weights = self.generate_optimized_weights(block, &dp_result, grid_row, grid_col);
        
        // 6. 캐시 업데이트
        self.shared_cache.store_weights(cache_key, weights.clone());
        
        // 7. 통계 업데이트
        if let Ok(mut stats) = self.inference_stats.lock() {
            stats.total_inferences += 1;
            stats.dp_optimizations += 1;
            stats.avg_inference_time_ns = 
                (stats.avg_inference_time_ns * (stats.total_inferences - 1) as f64 + 
                 start_time.elapsed().as_nanos() as f64) / stats.total_inferences as f64;
        }
        
        GridInferenceResult {
            weights,
            dp_optimization: Some(dp_result),
            cache_efficiency: 0.0, // 새로 계산됨
            inference_time_ns: start_time.elapsed().as_nanos() as u64,
        }
    }
    
    /// **비트 DP 최적화** (11비트 상태 전이)
    fn optimize_block_with_dp(
        &mut self,
        block: &HybridEncodedBlock,
        input_vector: &[f32],
        grid_row: usize,
        grid_col: usize,
    ) -> crate::core::differential::bit_dp_system::DPOptimizationResult {
        // 1. 블록의 RBE 파라미터를 Packed128로 변환
        let packed = self.convert_block_to_packed128(block);
        
        // 2. 입력 벡터에서 에러 추정
        let errors = self.estimate_block_errors(input_vector, block.rows, block.cols);
        
        // 3. DP 문제 정의
        let problem = BitDPProblem {
            current_state: self.compute_initial_state(grid_row, grid_col),
            gradient_level: self.compute_gradient_level(&errors),
            position: 0,
            remaining_steps: 8, // 8단계 최적화
        };
        
        // 4. DP 최적화 실행
        self.dp_table.optimize_bit_sequence(
            &problem,
            &packed,
            &errors,
            block.rows,
            block.cols,
        )
    }
    
    /// **최적화된 가중치 직접 생성** (복원 없음)
    fn generate_optimized_weights(
        &mut self,
        block: &HybridEncodedBlock,
        dp_result: &crate::core::differential::bit_dp_system::DPOptimizationResult,
        grid_row: usize,
        grid_col: usize,
    ) -> Vec<f32> {
        let mut weights = Vec::with_capacity(block.rows * block.cols);
        
        // 1. DP 최적 경로를 사용하여 상태별 가중치 생성
        for (idx, &state) in dp_result.optimal_path.iter().enumerate() {
            let local_row = idx / block.cols;
            let local_col = idx % block.cols;
            
            if local_row >= block.rows {
                break;
            }
            
            // 2. 상태 기반 PoincarePackedBit128 생성
            let packed = self.create_optimized_packed(block, state, local_row, local_col);
            
            // 3. 글로벌 좌표 계산
            let global_row = grid_row * self.block_size + local_row;
            let global_col = grid_col * self.block_size + local_col;
            let total_rows = self.grid_rows * self.block_size;
            let total_cols = self.grid_cols * self.block_size;
            
            // 4. 직접 가중치 생성 (복원 없음)
            let weight = self.weight_generator.generate_weight(
                &packed,
                global_row,
                global_col,
                total_rows,
                total_cols,
            );
            
            weights.push(weight);
        }
        
        // 5. 남은 위치들은 기본 생성
        while weights.len() < block.rows * block.cols {
            let idx = weights.len();
            let local_row = idx / block.cols;
            let local_col = idx % block.cols;
            
            let packed = self.create_default_packed(block, local_row, local_col);
            let global_row = grid_row * self.block_size + local_row;
            let global_col = grid_col * self.block_size + local_col;
            let total_rows = self.grid_rows * self.block_size;
            let total_cols = self.grid_cols * self.block_size;
            
            let weight = self.weight_generator.generate_weight(
                &packed,
                global_row,
                global_col,
                total_rows,
                total_cols,
            );
            
            weights.push(weight);
        }
        
        weights
    }
    
    /// **병렬 그리드 추론** (전체 그리드 동시 처리)
    pub fn parallel_infer_full_grid(
        &mut self,
        compressed_matrix: &GridCompressedMatrix,
        input_vector: &[f32],
    ) -> Vec<Vec<GridInferenceResult>> {
        // 그리드 좌표 생성
        let grid_coords: Vec<(usize, usize)> = (0..self.grid_rows)
            .flat_map(|i| (0..self.grid_cols).map(move |j| (i, j)))
            .collect();
        
        // 병렬 처리를 위한 공유 참조 생성
        let shared_cache = self.shared_cache.clone();
        let inference_stats = self.inference_stats.clone();
        let grid_rows = self.grid_rows;
        let grid_cols = self.grid_cols;
        let block_size = self.block_size;
        
        // 병렬 추론 실행
        let results: Vec<GridInferenceResult> = grid_coords
            .into_par_iter()
            .map(|(grid_row, grid_col)| {
                // 각 스레드별 독립적인 처리
                let mut local_generator = WeightGenerator::new();
                let mut local_dp_table = BitDPTable::new(128, 8, 1024);
                
                // 로컬 인스턴스 생성
                let local_inference = GridDirectInference {
                    grid_rows,
                    grid_cols,
                    block_size,
                    weight_generator: local_generator.clone(),
                    dp_table: local_dp_table.clone(),
                    shared_cache: shared_cache.clone(),
                    parallel_dp: Arc::new(Mutex::new(ParallelDPProcessor::new(1, 128, 8, 1024))),
                    inference_stats: inference_stats.clone(),
                };
                
                local_inference.infer_grid_block_local(
                    compressed_matrix,
                    input_vector,
                    grid_row,
                    grid_col,
                    &mut local_generator,
                    &mut local_dp_table,
                    &shared_cache,
                    &inference_stats,
                )
            })
            .collect();
        
        // 결과를 그리드 형태로 재구성
        let mut grid_results = vec![vec![]; self.grid_rows];
        for (idx, result) in results.into_iter().enumerate() {
            let grid_row = idx / self.grid_cols;
            if grid_row < self.grid_rows {
                grid_results[grid_row].push(result);
            }
        }
        
        grid_results
    }
    
    /// **로컬 그리드 블록 추론** (병렬 처리용)
    fn infer_grid_block_local(
        &self,
        compressed_matrix: &GridCompressedMatrix,
        input_vector: &[f32],
        grid_row: usize,
        grid_col: usize,
        local_generator: &mut WeightGenerator,
        local_dp_table: &mut BitDPTable,
        shared_cache: &SharedWeightCache,
        stats: &Arc<Mutex<GridInferenceStats>>,
    ) -> GridInferenceResult {
        let start_time = std::time::Instant::now();
        
        // 블록 인덱스 및 검증
        let block_idx = grid_row * self.grid_cols + grid_col;
        if block_idx >= compressed_matrix.blocks.len() {
            return GridInferenceResult {
                weights: vec![],
                dp_optimization: None,
                cache_efficiency: 0.0,
                inference_time_ns: start_time.elapsed().as_nanos() as u64,
            };
        }
        
        let block = &compressed_matrix.blocks[block_idx];
        
        // 캐시 확인
        let cache_key = (
            self.compute_block_hash(block),
            grid_row as u64,
            block.rows,
            block.cols,
        );
        
        if let Some(cached_weights) = shared_cache.get_weights(&cache_key) {
            return GridInferenceResult {
                weights: cached_weights,
                dp_optimization: None,
                cache_efficiency: 1.0,
                inference_time_ns: start_time.elapsed().as_nanos() as u64,
            };
        }
        
                // 로컬 DP 최적화 및 가중치 생성
        let packed = self.convert_block_to_packed128(block);
        let errors = self.estimate_block_errors(input_vector, block.rows, block.cols);

        let problem = BitDPProblem {
            current_state: self.compute_initial_state(grid_row, grid_col),
            gradient_level: self.compute_gradient_level(&errors),
            position: 0,
            remaining_steps: 6, // 로컬 처리는 단축
        };

        let dp_result = local_dp_table.optimize_bit_sequence(
            &problem,
            &packed,
            &errors,
            block.rows,
            block.cols,
        );

        // 가중치 생성 (로컬 생성기 생성)
        let mut local_weight_generator = WeightGenerator::new();
        let weights = self.generate_weights_from_dp(
            block,
            &dp_result,
            grid_row,
            grid_col,
            &mut local_weight_generator,
        );
        
        // 캐시 업데이트
        shared_cache.store_weights(cache_key, weights.clone());
        
        // 통계 업데이트
        if let Ok(mut s) = stats.lock() {
            s.total_inferences += 1;
            s.dp_optimizations += 1;
        }
        
        GridInferenceResult {
            weights,
            dp_optimization: Some(dp_result),
            cache_efficiency: 0.0,
            inference_time_ns: start_time.elapsed().as_nanos() as u64,
        }
    }
    
    // 헬퍼 메서드들
    
    fn compute_block_hash(&self, block: &HybridEncodedBlock) -> u64 {
        // 블록의 고유 해시 계산
        let mut hash = 0u64;
        hash ^= block.rbe_params[0].to_bits() as u64;
        if !block.residuals.is_empty() {
            hash ^= block.residuals[0].value.to_bits() as u64;
        }
        hash ^= (block.rows as u64) << 32;
        hash ^= block.cols as u64;
        hash
    }
    
    fn convert_block_to_packed128(&self, block: &HybridEncodedBlock) -> Packed128 {
        // HybridEncodedBlock을 Packed128로 변환
        let hi = block.rbe_params[0].to_bits() as u64;
        let lo = if !block.residuals.is_empty() {
            (block.residuals[0].value.to_bits() as u64) << 32 | (block.rows as u64)
        } else {
            (block.rows as u64) << 32 | (block.cols as u64)
        };
        
        Packed128 { hi, lo }
    }
    
    fn estimate_block_errors(&self, input_vector: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        // 입력 벡터에서 블록별 에러 추정
        let block_size = rows * cols;
        let mut errors = Vec::with_capacity(block_size);
        
        for i in 0..block_size {
            let error = if i < input_vector.len() {
                input_vector[i].abs()
            } else {
                0.01 // 기본값
            };
            errors.push(error);
        }
        
        errors
    }
    
    fn compute_initial_state(&self, grid_row: usize, grid_col: usize) -> u16 {
        // 그리드 위치 기반 초기 상태 계산
        ((grid_row * self.grid_cols + grid_col) % 2048) as u16
    }
    
    fn compute_gradient_level(&self, errors: &[f32]) -> u8 {
        // 에러 크기 기반 그래디언트 레벨 계산
        let avg_error = errors.iter().sum::<f32>() / errors.len() as f32;
        ((avg_error * 15.0).min(15.0) as u8).max(1)
    }
    
    fn create_optimized_packed(
        &self,
        block: &HybridEncodedBlock,
        state: u16,
        local_row: usize,
        local_col: usize,
    ) -> PoincarePackedBit128 {
        // 최적화된 상태로부터 PoincarePackedBit128 생성
        let quadrant = (state & 0x3) as u8;
        let freq = ((state >> 2) & 0xFFF) as u32;
        let amp = ((state >> 4) & 0xFFF) as u32;
        let phase = ((state >> 6) & 0xFFF) as u32;
        
        PoincarePackedBit128::new(
            match quadrant {
                0 => crate::packed_params::PoincareQuadrant::First,
                1 => crate::packed_params::PoincareQuadrant::Second,
                2 => crate::packed_params::PoincareQuadrant::Third,
                _ => crate::packed_params::PoincareQuadrant::Fourth,
            },
            (freq & 0xFFFF) as u16,
            (amp & 0xFFFF) as u16,
            (phase & 0xFF) as u8,
            state as u32,
            0.5, // r
            0.0, // theta
        )
    }
    
    fn create_default_packed(
        &self,
        block: &HybridEncodedBlock,
        local_row: usize,
        local_col: usize,
    ) -> PoincarePackedBit128 {
        // 기본 PoincarePackedBit128 생성
        let seed = (local_row * block.cols + local_col) as u32;
        PoincarePackedBit128::new(
            crate::packed_params::PoincareQuadrant::First,
            (seed & 0xFFFF) as u16,
            ((seed * 2) & 0xFFFF) as u16,
            ((seed * 3) & 0xFF) as u8,
            seed * 4,
            0.5,
            0.0,
        )
    }
    
    fn generate_weights_from_dp(
        &self,
        block: &HybridEncodedBlock,
        dp_result: &crate::core::differential::bit_dp_system::DPOptimizationResult,
        grid_row: usize,
        grid_col: usize,
        generator: &mut WeightGenerator,
    ) -> Vec<f32> {
        // DP 결과로부터 가중치 생성
        let mut weights = Vec::with_capacity(block.rows * block.cols);
        
        for (idx, &state) in dp_result.optimal_path.iter().enumerate() {
            if idx >= block.rows * block.cols {
                break;
            }
            
            let local_row = idx / block.cols;
            let local_col = idx % block.cols;
            
            let packed = self.create_optimized_packed(block, state, local_row, local_col);
            
            let weight = generator.generate_weight(
                &packed,
                grid_row * self.block_size + local_row,
                grid_col * self.block_size + local_col,
                self.grid_rows * self.block_size,
                self.grid_cols * self.block_size,
            );
            
            weights.push(weight);
        }
        
        weights
    }
    
    /// 통계 조회
    pub fn get_inference_stats(&self) -> GridInferenceStats {
        if let Ok(stats) = self.inference_stats.lock() {
            stats.clone()
        } else {
            GridInferenceStats::default()
        }
    }
    
    /// 캐시 클리어
    pub fn clear_cache(&mut self) {
        self.weight_generator.clear_cache();
        self.dp_table.clear_cache();
        // shared_cache는 다른 곳에서도 사용되므로 클리어하지 않음
    }
}

/// **GridCompressedMatrix 확장 메서드**
impl GridCompressedMatrix {
    /// **복원 없는 그리드 추론** (기존 decompress_hybrid 대체)
    pub fn infer_direct(&self, input_vector: &[f32]) -> Vec<f32> {
        let mut grid_inference = GridDirectInference::new(
            self.grid_rows,
            self.grid_cols,
            self.block_size,
        );
        
        let grid_results = grid_inference.parallel_infer_full_grid(self, input_vector);
        
        // 결과를 평탄화하여 전체 행렬 형태로 변환
        let mut output = vec![0.0; self.total_rows * self.total_cols];
        
        for (grid_row, row_results) in grid_results.iter().enumerate() {
            for (grid_col, result) in row_results.iter().enumerate() {
                let start_row = grid_row * self.block_size;
                let start_col = grid_col * self.block_size;
                
                let block_idx = grid_row * self.grid_cols + grid_col;
                if block_idx < self.blocks.len() {
                    let block = &self.blocks[block_idx];
                    
                    for (idx, &weight) in result.weights.iter().enumerate() {
                        let local_row = idx / block.cols;
                        let local_col = idx % block.cols;
                        let global_row = start_row + local_row;
                        let global_col = start_col + local_col;
                        
                        if global_row < self.total_rows && global_col < self.total_cols {
                            output[global_row * self.total_cols + global_col] = weight;
                        }
                    }
                }
            }
        }
        
        output
    }
    
    /// **GEMV 연산 직접 수행** (복원 없이)
    pub fn direct_gemv(&self, input: &[f32], output: &mut [f32]) {
        if input.len() != self.total_cols || output.len() != self.total_rows {
            return;
        }
        
        output.fill(0.0);
        
        // 캐시 및 설정을 공유하기 위한 참조 준비
        let grid_rows = self.grid_rows;
        let grid_cols = self.grid_cols;
        let block_size = self.block_size;
        
        // 블록별 병렬 GEMV
        let block_outputs: Vec<Vec<f32>> = (0..self.grid_rows)
            .into_par_iter()
            .map(|grid_row| {
                let mut row_output = vec![0.0; self.block_size];
                let mut local_inference = GridDirectInference::new(grid_rows, grid_cols, block_size);
                
                for grid_col in 0..self.grid_cols {
                    let result = local_inference.infer_grid_block(self, input, grid_row, grid_col);
                    
                    // 블록 내 GEMV 계산
                    let start_col = grid_col * self.block_size;
                    let end_col = (start_col + self.block_size).min(self.total_cols);
                    
                    for (local_row, &weight) in result.weights.iter().enumerate() {
                        if local_row < row_output.len() {
                            for j in start_col..end_col {
                                if j < input.len() {
                                    row_output[local_row] += weight * input[j];
                                }
                            }
                        }
                    }
                }
                
                row_output
            })
            .collect();
        
        // 결과 병합
        for (grid_row, block_output) in block_outputs.iter().enumerate() {
            let start_row = grid_row * self.block_size;
            for (local_row, &value) in block_output.iter().enumerate() {
                let global_row = start_row + local_row;
                if global_row < output.len() {
                    output[global_row] = value;
                }
            }
        }
    }
} 
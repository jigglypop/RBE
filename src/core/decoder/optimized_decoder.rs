//! 최적화된 디코더 - 비트 DP 통합 버전

use crate::packed_params::{PoincarePackedBit128, PoincareQuadrant};
use crate::core::differential::{BitDPTable, BitDPProblem, DPOptimizationResult, ParallelDPProcessor};
use super::weight_generator::WeightGenerator;
use super::fused_forward::FusedForwardPass;
use rayon::prelude::*;
use std::collections::HashMap;

/// **비트 DP 통합 최적화 디코더** (완전한 동적 프로그래밍 지원)
#[derive(Debug, Clone)]
pub struct OptimizedDecoder {
    /// 가중치 생성기
    weight_generator: WeightGenerator,
    /// 융합 순전파 엔진
    fused_forward: FusedForwardPass,
    /// **진정한 비트 DP 시스템**
    bit_dp_table: BitDPTable,
    /// **병렬 DP 처리기**
    parallel_dp_processor: ParallelDPProcessor,
    /// 디코딩 성능 통계
    performance_stats: DecodingPerformanceStats,
}

/// 디코딩 성능 통계
#[derive(Debug, Clone)]
pub struct DecodingPerformanceStats {
    /// 총 디코딩 시간 (ns)
    pub total_decoding_time_ns: u64,
    /// DP 최적화 시간 (ns)
    pub dp_optimization_time_ns: u64,
    /// 가중치 생성 시간 (ns)
    pub weight_generation_time_ns: u64,
    /// DP 캐시 히트율
    pub dp_cache_hit_rate: f32,
    /// 병렬 처리 효율성
    pub parallel_efficiency: f32,
    /// 총 디코딩 연산 수
    pub total_operations: u64,
}

/// **DP 기반 디코딩 결과**
#[derive(Debug, Clone)]
pub struct DPDecodingResult {
    /// 복원된 매트릭스
    pub restored_matrix: Vec<Vec<f32>>,
    /// DP 최적화 결과
    pub dp_optimization: DPOptimizationResult,
    /// 디코딩 품질 메트릭
    pub quality_metrics: DecodingQualityMetrics,
    /// 성능 통계
    pub performance: DecodingPerformanceStats,
}

/// 디코딩 품질 메트릭
#[derive(Debug, Clone)]
pub struct DecodingQualityMetrics {
    /// 수치적 안정성 스코어
    pub numerical_stability: f32,
    /// 매트릭스 복원 오차
    pub reconstruction_error: f32,
    /// DP 최적성 스코어
    pub dp_optimality: f32,
}

impl OptimizedDecoder {
    /// 새로운 DP 통합 디코더 생성
    pub fn new() -> Self {
        let num_threads = rayon::current_num_threads();
        
        Self {
            weight_generator: WeightGenerator::new(),
            fused_forward: FusedForwardPass::new(),
            bit_dp_table: BitDPTable::new(128, 8, 256), // state_count, gradient_levels, max_positions (메모리 효율적)
            parallel_dp_processor: ParallelDPProcessor::new(num_threads, 128, 8, 256),
            performance_stats: DecodingPerformanceStats::default(),
        }
}

    /// **핵심: DP 최적화 기반 디코딩** (완전한 동적 프로그래밍)
    pub fn decode_with_dp_optimization(
        &mut self,
        encoded_params: &[PoincarePackedBit128],
        rows: usize,
        cols: usize,
        target_quality: f32,
    ) -> DPDecodingResult {
        let start_time = std::time::Instant::now();
        
        // **1단계: DP 문제 정의**
        let dp_problems = self.construct_dp_problems(encoded_params, rows, cols, target_quality);
    
        // **2단계: 병렬 DP 최적화**
        let dp_start = std::time::Instant::now();
        let optimization_results = self.parallel_dp_processor.parallel_optimize(
            &dp_problems,
            &self.convert_to_packed128(encoded_params),
            &self.generate_error_batches(encoded_params.len(), rows, cols),
            rows,
            cols,
        );
        let dp_time = dp_start.elapsed().as_nanos() as u64;
        
        // **3단계: 최적 상태 시퀀스 적용**
        let optimized_params = self.apply_dp_optimization(encoded_params, &optimization_results);
        
        // **4단계: 융합 순전파로 매트릭스 복원**
        let weight_start = std::time::Instant::now();
        let restored_matrix = self.restore_matrix_with_optimized_params(
            &optimized_params, rows, cols
        );
        let weight_time = weight_start.elapsed().as_nanos() as u64;
        
        // **5단계: 품질 메트릭 계산**
        let quality_metrics = self.compute_quality_metrics(&restored_matrix, &optimization_results);
            
        // **6단계: 성능 통계 업데이트**
        let total_time = start_time.elapsed().as_nanos() as u64;
        self.update_performance_stats(total_time, dp_time, weight_time);
        
        DPDecodingResult {
            restored_matrix,
            dp_optimization: optimization_results[0].clone(), // 주요 결과 반환
            quality_metrics,
            performance: self.performance_stats.clone(),
        }
}

    /// DP 문제 구성 (각 파라미터별로)
    fn construct_dp_problems(
        &self,
        encoded_params: &[PoincarePackedBit128],
    rows: usize, 
    cols: usize, 
        target_quality: f32,
    ) -> Vec<BitDPProblem> {
        encoded_params.iter().enumerate().map(|(idx, param)| {
            // 파라미터에서 현재 상태 추출
            let current_state = self.extract_state_from_param(param);
            
            // 목표 품질에 따른 그래디언트 레벨 결정
            let gradient_level = self.determine_gradient_level(target_quality);
            
            // 남은 최적화 단계 계산
            let remaining_steps = self.calculate_remaining_steps(idx, encoded_params.len());
            
            BitDPProblem {
                current_state,
                gradient_level,
                position: idx % (rows * cols),
                remaining_steps,
            }
        }).collect()
    }
    
    /// 파라미터에서 11비트 상태 추출
    fn extract_state_from_param(&self, param: &PoincarePackedBit128) -> u16 {
        // hi 필드에서 11비트 상태 정보 추출
        let basis_selector = param.get_basis_function_selector() as u16; // 6비트
        let cordic_seq = param.get_cordic_rotation_sequence();
        let frequency = param.get_hyperbolic_frequency(); // 12비트
        
        // 11비트 해시 생성 (기존 방식과 동일)
        let state_hash = (basis_selector ^ (cordic_seq as u16) ^ frequency) & 0x7FF;
        state_hash
    }
    
    /// 목표 품질에 따른 그래디언트 레벨 결정
    fn determine_gradient_level(&self, target_quality: f32) -> u8 {
        if target_quality > 0.95 {
            15 // 최고 품질
        } else if target_quality > 0.8 {
            12 // 고품질
        } else if target_quality > 0.6 {
            8  // 중품질
        } else {
            4  // 저품질
        }
    }
    
    /// 남은 최적화 단계 계산
    fn calculate_remaining_steps(&self, current_idx: usize, total_params: usize) -> u8 {
        let remaining_ratio = (total_params - current_idx) as f32 / total_params as f32;
        (remaining_ratio * 15.0) as u8 + 1 // 1-15 단계
}

    /// PoincarePackedBit128을 Packed128으로 변환
    fn convert_to_packed128(&self, params: &[PoincarePackedBit128]) -> Vec<crate::packed_params::Packed128> {
        params.iter().map(|param| {
            crate::packed_params::Packed128 {
                hi: param.hi,
                lo: param.lo,
            }
        }).collect()
    }
    
    /// 에러 배치 생성 (DP 최적화용)
    fn generate_error_batches(&self, num_params: usize, rows: usize, cols: usize) -> Vec<Vec<f32>> {
        (0..num_params).map(|_| {
            // 각 파라미터에 대한 가상의 에러 시퀀스 생성
            (0..(rows * cols)).map(|i| {
                let error = 0.1 * (i as f32 / (rows * cols) as f32); // 감소하는 에러 패턴
                error
            }).collect()
        }).collect()
    }
    
    /// DP 최적화 결과 적용
    fn apply_dp_optimization(
        &self,
        original_params: &[PoincarePackedBit128],
        optimization_results: &[DPOptimizationResult],
    ) -> Vec<PoincarePackedBit128> {
        original_params.iter().zip(optimization_results.iter()).map(|(param, result)| {
            if !result.optimal_path.is_empty() {
                // 최적 경로의 마지막 상태를 새로운 파라미터로 적용
                let optimal_state = result.optimal_path.last().unwrap_or(&0);
                self.create_optimized_param(param, *optimal_state)
            } else {
                *param // 최적화 실패 시 원본 유지
            }
        }).collect()
    }
    
    /// 최적화된 파라미터 생성
    fn create_optimized_param(&self, original: &PoincarePackedBit128, optimal_state: u16) -> PoincarePackedBit128 {
        // 원본 파라미터의 연속 부분은 유지하고, 이산 상태만 최적화
        let basis_selector = (optimal_state & 0x3F) as u8; // 6비트
        let frequency = (optimal_state >> 6) & 0x1F; // 5비트 (12비트에서 축소)
        let amplitude = original.get_geodesic_amplitude(); // 기존 값 유지
        let cordic_seq = original.get_cordic_rotation_sequence(); // 기존 값 유지
        
        PoincarePackedBit128::new(
            original.get_quadrant(),
            frequency as u16,
            amplitude,
            basis_selector,
            cordic_seq,
            original.get_r_poincare(),
            original.get_theta_poincare(),
        )
}

    /// 최적화된 파라미터로 매트릭스 복원
    fn restore_matrix_with_optimized_params(
        &mut self,
        optimized_params: &[PoincarePackedBit128],
        rows: usize,
        cols: usize,
    ) -> Vec<Vec<f32>> {
        let mut matrix = vec![vec![0.0; cols]; rows];
        
        // 각 파라미터에서 가중치 생성하여 누적
        for param in optimized_params {
            // 융합 순전파를 통한 고속 가중치 생성
            let mut input_vector = vec![1.0; cols]; // 단위 입력
            let mut output_vector = vec![0.0; rows];
            
            self.fused_forward.fused_gemv_optimized(
                &[*param],
                &input_vector,
                &mut output_vector,
                rows,
                cols,
            );
            
            // 결과를 매트릭스에 누적
            for i in 0..rows {
                for j in 0..cols {
                    if i < output_vector.len() {
                        matrix[i][j] += output_vector[i] / optimized_params.len() as f32;
    }
                }
            }
        }
        
        matrix
}

    /// 품질 메트릭 계산
    fn compute_quality_metrics(
        &self,
        restored_matrix: &[Vec<f32>],
        optimization_results: &[DPOptimizationResult],
    ) -> DecodingQualityMetrics {
        // 수치적 안정성 검사
        let mut total_elements = 0;
        let mut stable_elements = 0;
        
        for row in restored_matrix {
            for &value in row {
                total_elements += 1;
                if value.is_finite() && !value.is_nan() && value.abs() <= 10.0 {
                    stable_elements += 1;
                }
            }
        }
        
        let numerical_stability = if total_elements > 0 {
            stable_elements as f32 / total_elements as f32
        } else {
            1.0
        };
        
        // 재구성 오차 계산 (임시로 평균 절대값 사용)
        let mut total_abs_value = 0.0;
        let mut element_count = 0;
        
        for row in restored_matrix {
            for &value in row {
                total_abs_value += value.abs();
                element_count += 1;
            }
        }
        
        let reconstruction_error = if element_count > 0 {
            total_abs_value / element_count as f32
        } else {
            0.0
        };
        
        // DP 최적성 스코어 (최적 값의 평균)
        let dp_optimality = if !optimization_results.is_empty() {
            optimization_results.iter()
                .map(|result| 1.0 / (1.0 + result.optimal_value))
                .sum::<f32>() / optimization_results.len() as f32
        } else {
            0.0
        };
        
        DecodingQualityMetrics {
            numerical_stability,
            reconstruction_error,
            dp_optimality,
        }
    }
    
    /// 성능 통계 업데이트
    fn update_performance_stats(&mut self, total_time: u64, dp_time: u64, weight_time: u64) {
        self.performance_stats.total_decoding_time_ns += total_time;
        self.performance_stats.dp_optimization_time_ns += dp_time;
        self.performance_stats.weight_generation_time_ns += weight_time;
        self.performance_stats.total_operations += 1;
        
        // DP 캐시 히트율 계산
        let (dp_subproblems, dp_optimal, dp_filled) = self.bit_dp_table.get_dp_stats();
        self.performance_stats.dp_cache_hit_rate = if dp_subproblems > 0 {
            dp_filled as f32 / dp_subproblems as f32
        } else {
            0.0
        };
        
        // 병렬 처리 효율성
        let (global_cache, num_threads, thread_stats) = self.parallel_dp_processor.get_parallel_stats();
        self.performance_stats.parallel_efficiency = if num_threads > 0 {
            thread_stats.iter().map(|(cache, _, _)| *cache as f32).sum::<f32>() / 
            (num_threads as f32 * global_cache as f32).max(1.0)
        } else {
            0.0
        };
    }
    
    /// **기존 디코딩 인터페이스와의 호환성** (하위 호환)
    pub fn decode_compressed_blocks(
        &mut self,
        encoded_blocks: &[PoincarePackedBit128],
        rows: usize,
        cols: usize,
    ) -> Vec<Vec<f32>> {
        // 기본 품질(0.8)로 DP 최적화 디코딩 수행
        let result = self.decode_with_dp_optimization(encoded_blocks, rows, cols, 0.8);
        result.restored_matrix
    }
    
    /// 디코더 성능 통계 반환
    pub fn get_performance_stats(&self) -> &DecodingPerformanceStats {
        &self.performance_stats
    }
    
    /// DP 시스템 통계 반환
    pub fn get_dp_stats(&self) -> (usize, usize, usize) {
        self.bit_dp_table.get_dp_stats()
}

    /// 캐시 정리
    pub fn clear_cache(&mut self) {
        self.weight_generator.clear_cache();
        self.fused_forward.clear_cache();
        self.bit_dp_table.clear_cache();
        self.parallel_dp_processor.merge_caches(); // 캐시 병합
    }
}

impl Default for DecodingPerformanceStats {
    fn default() -> Self {
        Self {
            total_decoding_time_ns: 0,
            dp_optimization_time_ns: 0,
            weight_generation_time_ns: 0,
            dp_cache_hit_rate: 0.0,
            parallel_efficiency: 0.0,
            total_operations: 0,
        }
    }
}

/// **극한 최적화된 디코딩 인터페이스**
impl OptimizedDecoder {
    /// **원샷 최고 성능 디코딩** (모든 최적화 기법 동시 적용)
    pub fn ultimate_decode(
        &mut self,
        encoded_params: &[PoincarePackedBit128],
        rows: usize,
        cols: usize,
    ) -> Vec<Vec<f32>> {
        // DP 최적화 + 병렬 처리 + 캐시 활용을 모두 적용한 최고 성능 디코딩
        let result = self.decode_with_dp_optimization(encoded_params, rows, cols, 0.95);
        
        // 캐시 병합으로 다음 호출 최적화
        self.parallel_dp_processor.merge_caches();
        
        result.restored_matrix
    }
    
    /// **스트리밍 디코딩** (대용량 매트릭스용)
    pub fn streaming_decode(
        &mut self,
        encoded_params: &[PoincarePackedBit128],
        rows: usize,
        cols: usize,
        chunk_size: usize,
    ) -> Vec<Vec<f32>> {
        let mut result_matrix = vec![vec![0.0; cols]; rows];
        
        // 청크별로 DP 최적화 적용
        for chunk in encoded_params.chunks(chunk_size) {
            let chunk_result = self.decode_with_dp_optimization(chunk, rows, cols, 0.7);
            
            // 결과 누적
            for i in 0..rows {
                for j in 0..cols {
                    if i < chunk_result.restored_matrix.len() && j < chunk_result.restored_matrix[i].len() {
                        result_matrix[i][j] += chunk_result.restored_matrix[i][j];
                    }
                }
    }
}

        // 정규화
        let chunk_count = (encoded_params.len() + chunk_size - 1) / chunk_size;
        for row in &mut result_matrix {
            for value in row {
                *value /= chunk_count as f32;
            }
        }
        
        result_matrix
    }
} 
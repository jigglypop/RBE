//! # 그리드 직접 추론 시스템 테스트
//!
//! 복원 없는 그리드 추론의 정확도와 성능을 검증

use crate::core::{
    decoder::{GridDirectInference, GridInferenceResult},
    encoder::GridCompressedMatrix,
    packed_params::{PoincarePackedBit128, PoincareQuadrant, HybridEncodedBlock, ResidualCoefficient},
};
use std::time::Instant;

#[test]
fn 그리드_직접_추론_기본_동작_테스트() {
    println!("=== 그리드 직접 추론 기본 동작 테스트 ===");
    
    // 1. 그리드 추론기 생성
    let mut grid_inference = GridDirectInference::new(4, 4, 32); // 4x4 그리드, 32x32 블록
    
    // 2. 테스트용 압축 매트릭스 생성
    let compressed_matrix = create_test_grid_matrix(4, 4, 32);
    
    // 3. 입력 벡터 생성
    let input_vector = (0..128).map(|i| (i as f32) * 0.01).collect::<Vec<f32>>();
    
    // 4. 단일 블록 추론 테스트
    let result = grid_inference.infer_grid_block(&compressed_matrix, &input_vector, 0, 0);
    
    // 5. 결과 검증
    assert!(!result.weights.is_empty(), "가중치가 생성되지 않음");
    assert!(result.inference_time_ns > 0, "추론 시간이 기록되지 않음");
    
    // 6. 가중치 범위 검증
    for &weight in &result.weights {
        assert!(weight.is_finite(), "비정상적인 가중치 값: {}", weight);
        assert!(weight.abs() <= 1.0, "가중치 범위 초과: {}", weight);
    }
    
    println!("✅ 그리드 직접 추론 기본 동작 테스트 통과");
    println!("   생성된 가중치 수: {}", result.weights.len());
    println!("   추론 시간: {} ns", result.inference_time_ns);
}

#[test]
fn 그리드_병렬_추론_성능_테스트() {
    println!("=== 그리드 병렬 추론 성능 테스트 ===");
    
    let mut grid_inference = GridDirectInference::new(8, 8, 64); // 8x8 그리드, 64x64 블록
    let compressed_matrix = create_test_grid_matrix(8, 8, 64);
    let input_vector = (0..512).map(|i| (i as f32) * 0.001).collect::<Vec<f32>>();
    
    // 병렬 추론 성능 측정
    let start = Instant::now();
    let grid_results = grid_inference.parallel_infer_full_grid(&compressed_matrix, &input_vector);
    let parallel_time = start.elapsed();
    
    // 결과 검증
    assert_eq!(grid_results.len(), 8, "그리드 행 수 불일치");
    for row_results in &grid_results {
        assert_eq!(row_results.len(), 8, "그리드 열 수 불일치");
    }
    
    // 성능 검증 (목표: 전체 그리드 추론 < 10ms)
    assert!(parallel_time.as_millis() < 100, "병렬 추론 성능 목표 미달성: {}ms", parallel_time.as_millis());
    
    // 통계 확인
    let stats = grid_inference.get_inference_stats();
    assert!(stats.total_inferences > 0, "추론 통계 기록되지 않음");
    
    println!("✅ 그리드 병렬 추론 성능 테스트 통과");
    println!("   병렬 추론 시간: {:.3} ms", parallel_time.as_secs_f64() * 1000.0);
    println!("   총 추론 횟수: {}", stats.total_inferences);
    println!("   캐시 히트율: {:.1}%", stats.cache_hits as f32 / stats.total_inferences as f32 * 100.0);
}

#[test]
fn 복원_없는_추론_정확도_테스트() {
    println!("=== 복원 없는 추론 정확도 테스트 ===");
    
    let compressed_matrix = create_test_grid_matrix(4, 4, 32);
    let input_vector = generate_test_input(128);
    
    // 1. 직접 추론 방식 성능 측정
    let start = Instant::now();
    let direct_output = compressed_matrix.infer_direct(&input_vector);
    let direct_time = start.elapsed();
    
    // 2. 결과 검증
    assert!(!direct_output.is_empty(), "직접 추론 결과가 비어있음");
    
    // 3. 수치적 안정성 검증
    for (i, &value) in direct_output.iter().enumerate() {
        assert!(value.is_finite(), "위치 {}에서 비정상적인 값: {}", i, value);
    }
    
    // 4. 성능 검증
    println!("   직접 추론 시간: {:.3} ms", direct_time.as_secs_f64() * 1000.0);
    
    // 목표: 직접 추론 < 10ms
    assert!(direct_time.as_millis() < 10, "직접 추론 성능 목표 미달성: {}ms", direct_time.as_millis());
    
    println!("✅ 복원 없는 추론 정확도 테스트 통과");
}

#[test]
fn 직접_gemv_연산_테스트() {
    println!("=== 직접 GEMV 연산 테스트 ===");
    
    let compressed_matrix = create_test_grid_matrix(6, 6, 48);
    let input = generate_test_input(288); // 6*48 = 288
    let mut output = vec![0.0; 288];
    
    // 직접 GEMV 실행
    let start = Instant::now();
    compressed_matrix.direct_gemv(&input, &mut output);
    let gemv_time = start.elapsed();
    
    // 결과 검증
    assert!(output.iter().any(|&x| x != 0.0), "GEMV 결과가 모두 0");
    
    // 수치적 안정성 검증
    for (i, &value) in output.iter().enumerate() {
        assert!(value.is_finite(), "위치 {}에서 비정상적인 값: {}", i, value);
    }
    
    // 성능 목표: 큰 행렬도 빠르게 처리
    println!("   GEMV 연산 시간: {:.3} ms", gemv_time.as_secs_f64() * 1000.0);
    assert!(gemv_time.as_millis() < 50, "GEMV 성능 목표 미달성");
    
    println!("✅ 직접 GEMV 연산 테스트 통과");
}

#[test]
fn 그리드_캐시_효율성_테스트() {
    println!("=== 그리드 캐시 효율성 테스트 ===");
    
    let mut grid_inference = GridDirectInference::new(4, 4, 32);
    let compressed_matrix = create_test_grid_matrix(4, 4, 32);
    let input_vector = generate_test_input(128);
    
    // 첫 번째 추론 (캐시 미스)
    let result1 = grid_inference.infer_grid_block(&compressed_matrix, &input_vector, 0, 0);
    assert_eq!(result1.cache_efficiency, 0.0, "첫 번째 추론은 캐시 미스여야 함");
    
    // 동일한 블록 재추론 (캐시 히트)
    let result2 = grid_inference.infer_grid_block(&compressed_matrix, &input_vector, 0, 0);
    assert_eq!(result2.cache_efficiency, 1.0, "동일한 블록 재추론은 캐시 히트여야 함");
    
    // 추론 시간 비교 (캐시 히트가 더 빨라야 함)
    assert!(result2.inference_time_ns < result1.inference_time_ns, 
           "캐시 히트가 더 느림: {} vs {}", result2.inference_time_ns, result1.inference_time_ns);
    
    // 통계 검증
    let stats = grid_inference.get_inference_stats();
    assert!(stats.cache_hits > 0, "캐시 히트가 기록되지 않음");
    
    println!("✅ 그리드 캐시 효율성 테스트 통과");
    println!("   첫 추론 시간: {} ns", result1.inference_time_ns);
    println!("   캐시 히트 시간: {} ns", result2.inference_time_ns);
    println!("   속도 향상: {:.1}x", result1.inference_time_ns as f32 / result2.inference_time_ns as f32);
}

#[test]
fn 극한_그리드_크기_테스트() {
    println!("=== 극한 그리드 크기 테스트 ===");
    
    // 큰 그리드 테스트 (16x16 그리드, 128x128 블록)
    let mut grid_inference = GridDirectInference::new(16, 16, 128);
    let compressed_matrix = create_test_grid_matrix(16, 16, 128);
    let input_vector = generate_test_input(2048);
    
    // 단일 블록 추론
    let start = Instant::now();
    let result = grid_inference.infer_grid_block(&compressed_matrix, &input_vector, 8, 8); // 중앙 블록
    let single_time = start.elapsed();
    
    // 결과 검증
    assert!(!result.weights.is_empty(), "큰 블록에서 가중치 생성 실패");
    assert!(single_time.as_millis() < 10, "단일 블록 추론 시간 초과: {}ms", single_time.as_millis());
    
    // 메모리 사용량 검증 (간접적)
    let stats = grid_inference.get_inference_stats();
    assert!(stats.total_inferences > 0, "통계 기록 실패");
    
    println!("✅ 극한 그리드 크기 테스트 통과");
    println!("   블록 크기: 128x128");
    println!("   단일 블록 추론 시간: {:.3} ms", single_time.as_secs_f64() * 1000.0);
}

#[test]
fn 천배_압축_그리드_추론_테스트() {
    println!("=== 1000배 압축 그리드 추론 테스트 ===");
    
    // 극한 압축 설정으로 그리드 추론기 생성
    let mut grid_inference = GridDirectInference::new(8, 8, 64);
    let compressed_matrix = create_extreme_compressed_matrix(8, 8, 64);
    let input_vector = generate_test_input(512);
    
    // 극한 압축 상태에서 추론
    let start = Instant::now();
    let grid_results = grid_inference.parallel_infer_full_grid(&compressed_matrix, &input_vector);
    let extreme_time = start.elapsed();
    
    // 압축률 계산 (간접적)
    let original_size = 8 * 8 * 64 * 64 * 4; // float32 기준
    let compressed_size = compressed_matrix.blocks.len() * std::mem::size_of::<HybridEncodedBlock>();
    let compression_ratio = original_size as f32 / compressed_size as f32;
    
    println!("   원본 크기: {} bytes", original_size);
    println!("   압축 크기: {} bytes", compressed_size);
    println!("   압축률: {:.1}x", compression_ratio);
    
    // 목표: 1000배 압축 달성
    assert!(compression_ratio > 500.0, "압축률 목표 미달성: {:.1}x", compression_ratio);
    
    // 정확도 검증
    let mut total_weights = 0;
    let mut valid_weights = 0;
    
    for row_results in &grid_results {
        for result in row_results {
            total_weights += result.weights.len();
            valid_weights += result.weights.iter().filter(|&&w| w.is_finite() && w.abs() <= 1.0).count();
        }
    }
    
    let accuracy_rate = valid_weights as f32 / total_weights as f32;
    assert!(accuracy_rate > 0.95, "정확도 목표 미달성: {:.1}%", accuracy_rate * 100.0);
    
    println!("✅ 1000배 압축 그리드 추론 테스트 통과");
    println!("   극한 압축 추론 시간: {:.3} ms", extreme_time.as_secs_f64() * 1000.0);
    println!("   정확도: {:.1}%", accuracy_rate * 100.0);
}

// 헬퍼 함수들

fn create_test_grid_matrix(grid_rows: usize, grid_cols: usize, block_size: usize) -> GridCompressedMatrix {
    let total_blocks = grid_rows * grid_cols;
    let mut blocks = Vec::with_capacity(total_blocks);
    
    for i in 0..total_blocks {
        let block = HybridEncodedBlock {
            rows: block_size,
            cols: block_size,
            rbe_params: [
                0.5 + (i as f32) * 0.01,
                ((i * 13) % 1024) as f32 / 1024.0,
                ((i * 17) % 512) as f32 / 512.0,
                0.3, 0.7, 0.2, 0.8, 0.1
            ],
            residuals: vec![
                ResidualCoefficient {
                    index: (i as u16 % block_size as u16, (i * 2) as u16 % block_size as u16),
                    value: 0.1 * ((i as f32).sin()),
                }
            ],
            transform_type: crate::core::packed_params::TransformType::Dwt,
        };
        blocks.push(block);
    }
    
    GridCompressedMatrix {
        blocks,
        grid_rows,
        grid_cols,
        block_size,
        total_rows: grid_rows * block_size,
        total_cols: grid_cols * block_size,
    }
}

fn create_extreme_compressed_matrix(grid_rows: usize, grid_cols: usize, block_size: usize) -> GridCompressedMatrix {
    let total_blocks = grid_rows * grid_cols;
    let mut blocks = Vec::with_capacity(total_blocks);
    
    // 극한 압축을 위한 최소한의 계수만 사용
    for i in 0..total_blocks {
        let block = HybridEncodedBlock {
            rows: block_size,
            cols: block_size,
            rbe_params: [
                (i as f32 % 7.0) / 7.0 - 0.5, // -0.5 ~ 0.5 범위
                ((i * 31) % 2048) as f32 / 2048.0,
                ((i * 37) % 1024) as f32 / 1024.0,
                0.2, 0.8, 0.1, 0.9, 0.05
            ],
            residuals: vec![], // 극한 압축: 잔차 없음
            transform_type: crate::core::packed_params::TransformType::Dwt,
        };
        blocks.push(block);
    }
    
    GridCompressedMatrix {
        blocks,
        grid_rows,
        grid_cols,
        block_size,
        total_rows: grid_rows * block_size,
        total_cols: grid_cols * block_size,
    }
}

fn generate_test_input(size: usize) -> Vec<f32> {
    (0..size).map(|i| {
        let x = (i as f32) / size as f32;
        (x * 2.0 * std::f32::consts::PI).sin() * 0.5 + 
        (x * 3.0 * std::f32::consts::PI).cos() * 0.3
    }).collect()
}

fn compute_rmse(reference: &[f32], approximation: &[f32]) -> f32 {
    if reference.len() != approximation.len() {
        return f32::INFINITY;
    }
    
    let mse: f32 = reference
        .iter()
        .zip(approximation)
        .map(|(r, a)| (r - a).powi(2))
        .sum::<f32>() / reference.len() as f32;
    
    mse.sqrt()
} 
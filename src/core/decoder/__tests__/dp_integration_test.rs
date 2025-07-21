//! # 비트 DP 통합 테스트
//!
//! 비트 DP 시스템과 그리드 디코더의 통합 동작을 검증

use crate::core::{
    decoder::{GridDirectInference, WeightGenerator},
    differential::bit_dp_system::{BitDPTable, BitDPProblem, ParallelDPProcessor},
    packed_params::{PoincarePackedBit128, PoincareQuadrant, Packed128},
    encoder::GridCompressedMatrix,
};
use std::time::Instant;

#[test]
fn 비트_dp_기본_동작_통합_테스트() {
    println!("=== 비트 DP 기본 동작 통합 테스트 ===");
    
    // 1. DP 테이블 초기화
    let mut dp_table = BitDPTable::new(128, 8, 1024);
    
    // 2. 테스트 문제 정의
    let problem = BitDPProblem {
        current_state: 42,
        gradient_level: 3,
        position: 0,
        remaining_steps: 5,
    };
    
    // 3. 테스트 파라미터
    let packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0x3F8000003F000000,
    };
    
    let errors = vec![0.1, 0.05, 0.02, 0.01, 0.005];
    
    // 4. DP 최적화 실행
    let start = Instant::now();
    let result = dp_table.optimize_bit_sequence(&problem, &packed, &errors, 8, 8);
    let dp_time = start.elapsed();
    
    // 5. 결과 검증
    assert!(result.optimal_value.is_finite(), "DP 최적값이 비정상적임");
    assert!(!result.optimal_path.is_empty(), "DP 최적 경로가 비어있음");
    assert!(result.total_steps > 0, "DP 단계 수가 0");
    
    // 6. 성능 검증
    println!("   DP 최적화 시간: {:.3} ms", dp_time.as_secs_f64() * 1000.0);
    println!("   최적값: {:.6}", result.optimal_value);
    println!("   경로 길이: {}", result.optimal_path.len());
    println!("   총 단계: {}", result.total_steps);
    
    assert!(dp_time.as_millis() < 100, "DP 최적화 시간 초과");
    
    println!("✅ 비트 DP 기본 동작 통합 테스트 통과");
}

#[test]
fn dp_최적화된_가중치_생성_테스트() {
    println!("=== DP 최적화된 가중치 생성 테스트 ===");
    
    // 1. WeightGenerator 초기화
    let mut generator = WeightGenerator::new();
    
    // 2. 시드 생성
    let packed_seeds = vec![
        PoincarePackedBit128::new(
            PoincareQuadrant::First,
            1024, 512, 256, 0x12345678,
            0.5, 0.0
        ),
        PoincarePackedBit128::new(
            PoincareQuadrant::Second,
            2048, 1024, 512, 0x87654321,
            0.3, 0.25
        ),
    ];
    
    // 3. DP 결과 시뮬레이션
    let dp_result = create_mock_dp_result();
    
    // 4. 위치 배열 생성
    let positions: Vec<(usize, usize)> = (0..64).map(|i| (i / 8, i % 8)).collect();
    
    // 5. DP 최적화된 가중치 생성
    let start = Instant::now();
    let weights = generator.generate_weights_with_dp_optimization(
        &packed_seeds,
        &dp_result,
        &positions,
        8, 8
    );
    let generation_time = start.elapsed();
    
    // 6. 결과 검증
    assert_eq!(weights.len(), positions.len(), "가중치 수량 불일치");
    
    for (i, &weight) in weights.iter().enumerate() {
        assert!(weight.is_finite(), "위치 {}에서 비정상적인 가중치: {}", i, weight);
        assert!(weight.abs() <= 1.0, "위치 {}에서 가중치 범위 초과: {}", i, weight);
    }
    
    // 7. 성능 검증
    let ns_per_weight = generation_time.as_nanos() as f64 / weights.len() as f64;
    println!("   DP 가중치 생성 시간: {:.3} ms", generation_time.as_secs_f64() * 1000.0);
    println!("   가중치당 시간: {:.1} ns", ns_per_weight);
    
    // 목표: 가중치당 < 100ns
    assert!(ns_per_weight < 100.0, "가중치 생성 성능 목표 미달성: {:.1} ns", ns_per_weight);
    
    println!("✅ DP 최적화된 가중치 생성 테스트 통과");
}

#[test]
fn 병렬_dp_처리_성능_테스트() {
    println!("=== 병렬 DP 처리 성능 테스트 ===");
    
    // 1. 병렬 DP 처리기 초기화
    let mut parallel_dp = ParallelDPProcessor::new(4, 128, 8, 1024);
    
    // 2. 배치 문제 생성
    let problems: Vec<BitDPProblem> = (0..16).map(|i| BitDPProblem {
        current_state: (i * 137) % 2048,
        gradient_level: (i % 8) as u8,
        position: 0,
        remaining_steps: 4,
    }).collect();
    
    // 3. 배치 파라미터 생성
    let packed_params: Vec<Packed128> = (0..16).map(|i| Packed128 {
        hi: 0x123456789ABCDEF0 + i as u64 * 0x1111,
        lo: 0x3F8000003F000000 + i as u64 * 0x2222,
    }).collect();
    
    // 4. 배치 에러 생성
    let error_batches: Vec<Vec<f32>> = (0..16).map(|i| {
        vec![0.1 / (i + 1) as f32; 8]
    }).collect();
    
    // 5. 병렬 최적화 실행
    let start = Instant::now();
    let results = parallel_dp.parallel_optimize(
        &problems,
        &packed_params,
        &error_batches,
        8, 8
    );
    let parallel_time = start.elapsed();
    
    // 6. 결과 검증
    assert_eq!(results.len(), problems.len(), "결과 수량 불일치");
    
    for (i, result) in results.iter().enumerate() {
        assert!(result.optimal_value.is_finite(), "문제 {}의 최적값이 비정상적임", i);
        assert!(!result.optimal_path.is_empty(), "문제 {}의 최적 경로가 비어있음", i);
    }
    
    // 7. 성능 분석
    let ms_per_problem = parallel_time.as_secs_f64() * 1000.0 / problems.len() as f64;
    println!("   병렬 DP 처리 시간: {:.3} ms", parallel_time.as_secs_f64() * 1000.0);
    println!("   문제당 시간: {:.3} ms", ms_per_problem);
    
    // 목표: 문제당 < 10ms
    assert!(ms_per_problem < 10.0, "병렬 DP 성능 목표 미달성: {:.3} ms", ms_per_problem);
    
    // 8. 캐시 병합 테스트
    parallel_dp.merge_caches();
    let stats = parallel_dp.get_parallel_stats();
    println!("   전역 캐시 크기: {}", stats.0);
    println!("   스레드 수: {}", stats.1);
    
    println!("✅ 병렬 DP 처리 성능 테스트 통과");
}

#[test]
fn 그리드_dp_통합_워크플로우_테스트() {
    println!("=== 그리드 DP 통합 워크플로우 테스트 ===");
    
    // 1. 통합 시스템 초기화
    let mut grid_inference = GridDirectInference::new(4, 4, 32);
    let compressed_matrix = create_test_matrix_with_dp(4, 4, 32);
    let input_vector = generate_complex_input(128);
    
    // 2. DP 최적화가 포함된 추론 실행
    let start = Instant::now();
    let grid_results = grid_inference.parallel_infer_full_grid(&compressed_matrix, &input_vector);
    let total_time = start.elapsed();
    
    // 3. 결과 검증
    assert_eq!(grid_results.len(), 4, "그리드 행 수 불일치");
    
    let mut total_dp_optimizations = 0;
    let mut total_weights = 0;
    
    for row_results in &grid_results {
        assert_eq!(row_results.len(), 4, "그리드 열 수 불일치");
        
        for result in row_results {
            if result.dp_optimization.is_some() {
                total_dp_optimizations += 1;
            }
            total_weights += result.weights.len();
        }
    }
    
    println!("   총 DP 최적화 횟수: {}", total_dp_optimizations);
    println!("   총 생성 가중치 수: {}", total_weights);
    
    assert!(total_dp_optimizations > 0, "DP 최적화가 실행되지 않음");
    assert!(total_weights > 0, "가중치가 생성되지 않음");
    
    // 4. 성능 vs 정확도 분석
    let stats = grid_inference.get_inference_stats();
    println!("   통합 워크플로우 시간: {:.3} ms", total_time.as_secs_f64() * 1000.0);
    println!("   평균 추론 시간: {:.1} ns", stats.avg_inference_time_ns);
         println!("   DP 최적화율: {:.1}%", stats.dp_optimizations as f32 / stats.total_inferences as f32 * 100.0);
    
    // 목표 검증
    assert!(total_time.as_millis() < 50, "통합 워크플로우 시간 초과");
    assert!(stats.avg_inference_time_ns < 1000.0, "평균 추론 시간 초과");
    
    println!("✅ 그리드 DP 통합 워크플로우 테스트 통과");
}

#[test]
fn dp_캐시_효율성_검증_테스트() {
    println!("=== DP 캐시 효율성 검증 테스트 ===");
    
    let mut generator = WeightGenerator::new();
    
    // 동일한 시드와 DP 결과로 반복 생성
    let seeds = vec![
        PoincarePackedBit128::new(
            PoincareQuadrant::First,
            1024, 512, 256, 0xABCDEF12,
            0.5, 0.25
        )
    ];
    
    let dp_result = create_mock_dp_result();
    let positions: Vec<(usize, usize)> = (0..16).map(|i| (i / 4, i % 4)).collect();
    
    // 첫 번째 생성 (캐시 미스)
    let start = Instant::now();
    let weights1 = generator.generate_weights_with_dp_optimization(
        &seeds, &dp_result, &positions, 4, 4
    );
    let first_time = start.elapsed();
    
    // 두 번째 생성 (캐시 히트)
    let start = Instant::now();
    let weights2 = generator.generate_weights_with_dp_optimization(
        &seeds, &dp_result, &positions, 4, 4
    );
    let second_time = start.elapsed();
    
    // 결과 일치성 검증
    assert_eq!(weights1.len(), weights2.len(), "캐시된 가중치 수량 불일치");
    
    for (i, (&w1, &w2)) in weights1.iter().zip(&weights2).enumerate() {
        assert!((w1 - w2).abs() < f32::EPSILON, "위치 {}에서 캐시된 가중치 불일치: {} vs {}", i, w1, w2);
    }
    
    // 캐시 효율성 검증
    let speedup = first_time.as_nanos() as f64 / second_time.as_nanos() as f64;
    println!("   첫 번째 생성: {:.3} ms", first_time.as_secs_f64() * 1000.0);
    println!("   두 번째 생성: {:.3} ms", second_time.as_secs_f64() * 1000.0);
    println!("   캐시 속도 향상: {:.1}x", speedup);
    
    assert!(speedup > 2.0, "캐시 효율성 목표 미달성: {:.1}x", speedup);
    
    // 캐시 통계 확인
    let stats = generator.get_cache_stats();
    assert!(stats.0 > 0, "캐시 히트가 기록되지 않음"); // cache_hits
    
    println!("✅ DP 캐시 효율성 검증 테스트 통과");
}

#[test]
fn 극한_상황_dp_안정성_테스트() {
    println!("=== 극한 상황 DP 안정성 테스트 ===");
    
    let mut dp_table = BitDPTable::new(128, 8, 1024);
    
    // 극한 상황들 테스트
    let extreme_cases = vec![
        // 최대 상태값
        BitDPProblem {
            current_state: 127,
            gradient_level: 7,
            position: 0,
            remaining_steps: 8,
        },
        // 최소 상태값
        BitDPProblem {
            current_state: 0,
            gradient_level: 0,
            position: 1023,
            remaining_steps: 1,
        },
        // 중간값들
        BitDPProblem {
            current_state: 64,
            gradient_level: 4,
            position: 512,
            remaining_steps: 4,
        },
    ];
    
    let packed = Packed128 {
        hi: 0xFFFFFFFFFFFFFFFF, // 최대값
        lo: 0x0000000000000000, // 최소값
    };
    
    for (i, problem) in extreme_cases.iter().enumerate() {
        // 극한 에러 배열
        let errors = match i {
            0 => vec![1.0; 1024], // 최대 에러
            1 => vec![0.0; 1024], // 최소 에러
            _ => vec![0.5; 1024], // 중간값
        };
        
        // DP 최적화 실행
        let result = dp_table.optimize_bit_sequence(problem, &packed, &errors, 32, 32);
        
        // 안정성 검증
        assert!(result.optimal_value.is_finite(), "극한 상황 {}에서 최적값이 비정상적임", i);
        assert!(!result.optimal_path.is_empty(), "극한 상황 {}에서 최적 경로가 비어있음", i);
        assert!(result.total_steps > 0, "극한 상황 {}에서 단계 수가 0", i);
        
        println!("   극한 상황 {}: 최적값 = {:.6}", i, result.optimal_value);
    }
    
    // DP 테이블 통계 확인
    let stats = dp_table.get_dp_stats();
    println!("   DP 캐시 크기: {}", stats.0);
    println!("   최적 부분구조 수: {}", stats.1);
    println!("   채워진 DP 엔트리: {}", stats.2);
    
    assert!(stats.0 > 0, "DP 캐시가 사용되지 않음");
    
    println!("✅ 극한 상황 DP 안정성 테스트 통과");
}

// 헬퍼 함수들

fn create_mock_dp_result() -> crate::core::differential::bit_dp_system::DPOptimizationResult {
    crate::core::differential::bit_dp_system::DPOptimizationResult {
        optimal_value: 0.123456,
        optimal_path: vec![42, 73, 156, 89, 234, 12, 67, 198],
        transition_sequence: vec![
            (42, 73, 0.01),
            (73, 156, 0.02),
            (156, 89, 0.015),
        ],
        total_steps: 8,
    }
}

fn create_test_matrix_with_dp(grid_rows: usize, grid_cols: usize, block_size: usize) -> GridCompressedMatrix {
    use crate::core::packed_params::{HybridEncodedBlock, ResidualCoefficient};
    
    let total_blocks = grid_rows * grid_cols;
    let mut blocks = Vec::with_capacity(total_blocks);
    
    for i in 0..total_blocks {
        let block = HybridEncodedBlock {
            rows: block_size,
            cols: block_size,
            rbe_params: [
                (i as f32 * 0.1).sin(),
                (i as f32 * 0.2).cos(),
                ((i * 137) % 2048) as f32 / 2048.0,
                ((i * 173) % 1024) as f32 / 1024.0,
                0.5, 0.3, 0.7, 0.2
            ],
            residuals: vec![
                ResidualCoefficient {
                    index: (i as u16 % block_size as u16, (i * 2) as u16 % block_size as u16),
                    value: (i as f32 * 0.15).sin(),
                },
                ResidualCoefficient {
                    index: ((i + 1) as u16 % block_size as u16, (i * 3) as u16 % block_size as u16),
                    value: (i as f32 * 0.25).cos(),
                },
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

fn generate_complex_input(size: usize) -> Vec<f32> {
    (0..size).map(|i| {
        let x = (i as f32) / size as f32;
        let y = x * 2.0 * std::f32::consts::PI;
        
        // 복잡한 주파수 성분
        0.5 * y.sin() + 
        0.3 * (y * 2.0).cos() + 
        0.2 * (y * 3.0).sin() +
        0.1 * (y * 5.0).cos()
    }).collect()
} 
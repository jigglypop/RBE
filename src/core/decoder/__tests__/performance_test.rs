use crate::core::decoder::{FusedForwardPass, WeightGenerator};
use crate::packed_params::{PoincarePackedBit128, PoincareQuadrant};
use std::time::Instant;

/// WeightGenerator 성능 벤치마크
#[test]
fn weight_generator_performance_benchmark() {
    let mut generator = WeightGenerator::new();
    
    // 테스트용 PoincarePackedBit128 생성
    let packed = PoincarePackedBit128::new(
        PoincareQuadrant::First,
        1000, // frequency
        800,  // amplitude  
        15,   // basis_func
        0x12345678, // cordic_seq
        0.5,  // r_poincare
        1.0   // theta_poincare
    );
    
    let iterations = 10000;
    let start = Instant::now();
    
    for i in 0..iterations {
        let row = i % 64;
        let col = (i * 17) % 64; // 다양한 패턴
        let _weight = generator.generate_weight(&packed, row, col, 64, 64);
    }
    
    let duration = start.elapsed();
    let avg_time_ns = duration.as_nanos() / iterations as u128;
    
    println!("WeightGenerator 평균 시간: {}ns", avg_time_ns);
    
    // 성능 목표: 100ns 이하 (현실적 목표)
    assert!(avg_time_ns < 100, "WeightGenerator가 너무 느립니다: {}ns", avg_time_ns);
}

/// FusedForwardPass GEMV 성능 벤치마크
#[test]
fn fused_forward_gemv_benchmark() {
    let mut fused_forward = FusedForwardPass::new();
    
    let rows = 128;
    let cols = 128;
    let num_seeds = 4;
    
    // 테스트 데이터 생성
    let weight_seeds: Vec<PoincarePackedBit128> = (0..num_seeds)
        .map(|i| PoincarePackedBit128::new(
            match i % 4 {
                0 => PoincareQuadrant::First,
                1 => PoincareQuadrant::Second,
                2 => PoincareQuadrant::Third,
                _ => PoincareQuadrant::Fourth,
            },
            1000 + i as u16 * 100, // frequency
            800 + i as u16 * 50,   // amplitude  
            10 + i as u8,          // basis_func
            0x12345678 + i as u32, // cordic_seq
            0.3 + i as f32 * 0.1,  // r_poincare
            0.5 + i as f32 * 0.2   // theta_poincare
        ))
        .collect();
    
    let input_vector = vec![1.0; cols];
    let mut output_vector = vec![0.0; rows];
    
    let iterations = 100;
    let start = Instant::now();
    
    for _ in 0..iterations {
        fused_forward.fused_gemv(&weight_seeds, &input_vector, &mut output_vector, rows, cols);
    }
    
    let duration = start.elapsed();
    let total_ops = iterations * rows * cols * num_seeds;
    let avg_time_ns = duration.as_nanos() / total_ops as u128;
    
    println!("FusedForwardPass 평균 시간 (element당): {}ns", avg_time_ns);
    println!("총 연산 수: {}", total_ops);
    
    // 성능 목표: 200ns/element 이하 (현실적 목표)
    assert!(avg_time_ns < 200, "FusedForwardPass가 너무 느립니다: {}ns", avg_time_ns);
}

/// 가중치 복원 vs 즉석 생성 비교 벤치마크
#[test]
fn weight_restoration_vs_generation_benchmark() {
    let mut generator = WeightGenerator::new();
    let packed = PoincarePackedBit128::new(
        PoincareQuadrant::Second,
        1200, // frequency
        900,  // amplitude  
        20,   // basis_func
        0x87654321, // cordic_seq
        0.6,  // r_poincare
        0.8   // theta_poincare
    );
    
    let size = 64;
    let total_elements = size * size;
    
    // 1. 즉석 생성 벤치마크
    let start = Instant::now();
    for i in 0..total_elements {
        let row = i / size;
        let col = i % size;
        let _weight = generator.generate_weight(&packed, row, col, size, size);
    }
    let generation_time = start.elapsed();
    
    // 2. 메모리 접근 시뮬레이션 (저장된 가중치 읽기)
    let weights: Vec<f32> = (0..total_elements).map(|i| i as f32 * 0.01).collect();
    let start = Instant::now();
    for weight in &weights {
        let _accessed = *weight; // 메모리 접근
    }
    let access_time = start.elapsed();
    
    let generation_ns = generation_time.as_nanos() / total_elements as u128;
    let access_ns = access_time.as_nanos() / total_elements as u128;
    
    println!("즉석 생성: {}ns/element", generation_ns);
    println!("메모리 접근: {}ns/element", access_ns);
    println!("메모리 절약량: {} bytes", total_elements * 4); // f32 = 4 bytes
    
    // 즉석 생성이 메모리 접근보다 50배 이내라면 허용 가능
    assert!(generation_ns < access_ns * 50, "즉석 생성이 메모리 접근보다 너무 느립니다");
}

/// 다양한 크기별 성능 테스트
#[test]
fn variable_size_performance_test() {
    let mut generator = WeightGenerator::new();
    let packed = PoincarePackedBit128::new(
        PoincareQuadrant::Third,
        800,  // frequency
        1100, // amplitude  
        25,   // basis_func
        0xABCDEF01, // cordic_seq
        0.4,  // r_poincare
        1.2   // theta_poincare
    );
    
    let sizes = vec![32, 64, 128, 256];
    
    for size in sizes {
        let total_elements = size * size;
        let start = Instant::now();
        
        for i in 0..total_elements {
            let row = i / size;
            let col = i % size;
            let _weight = generator.generate_weight(&packed, row, col, size, size);
        }
        
        let duration = start.elapsed();
        let avg_ns = duration.as_nanos() / total_elements as u128;
        
        println!("크기 {}x{}: {}ns/element", size, size, avg_ns);
        
        // 크기가 커져도 element당 시간은 일정해야 함
        assert!(avg_ns < 150, "{}x{} 크기에서 성능이 저하됨: {}ns", size, size, avg_ns);
    }
}

/// 병렬 처리 성능 테스트
#[test]
fn parallel_processing_benchmark() {
    use rayon::prelude::*;
    
    let mut generator = WeightGenerator::new();
    let packed = PoincarePackedBit128::new(
        PoincareQuadrant::Fourth,
        1500, // frequency
        750,  // amplitude  
        30,   // basis_func
        0xFEDCBA98, // cordic_seq
        0.7,  // r_poincare
        0.9   // theta_poincare
    );
    
    let size = 128;
    let total_elements = size * size;
    
    // 순차 처리
    let start = Instant::now();
    for i in 0..total_elements {
        let row = i / size;
        let col = i % size;
        let _weight = generator.generate_weight(&packed, row, col, size, size);
    }
    let sequential_time = start.elapsed();
    
    // 병렬 처리 (각 스레드별로 독립적인 generator 사용)
    let start = Instant::now();
    (0..total_elements).into_par_iter().for_each(|i| {
        let mut thread_generator = WeightGenerator::new(); // 각 스레드별 인스턴스
        let row = i / size;
        let col = i % size;
        let _weight = thread_generator.generate_weight(&packed, row, col, size, size);
    });
    let parallel_time = start.elapsed();
    
    let sequential_ns = sequential_time.as_nanos() / total_elements as u128;
    let parallel_ns = parallel_time.as_nanos() / total_elements as u128;
    
    println!("순차 처리: {}ns/element", sequential_ns);
    println!("병렬 처리: {}ns/element", parallel_ns);
    println!("병렬 가속비: {:.2}x", sequential_ns as f64 / parallel_ns as f64);
    
    // 병렬 처리가 적어도 1.5배는 빨라야 함
    assert!(parallel_ns * 3 < sequential_ns * 2, "병렬 처리 효과가 부족합니다");
} 
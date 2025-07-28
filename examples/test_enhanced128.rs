//! Enhanced128 간단 테스트 프로그램
//! 
//! Legacy 수학 모델의 정확성을 검증하고 성능을 측정합니다.

use rbe_llm::core::tensors::{Enhanced128, Packed128};
use std::time::Instant;
use std::f32::consts::PI;
use rand::{SeedableRng, Rng};

fn main() {
    println!("🚀 Enhanced128 테스트 프로그램 시작\n");
    
    // 1. 기본 생성 및 디코딩 테스트
    basic_creation_test();
    
    // 2. 성능 비교 테스트
    performance_comparison_test();
    
    // 3. 정확도 테스트
    accuracy_test();
    
    // 4. 압축률 테스트
    compression_test();
    
    println!("\n✅ 모든 테스트 완료!");
}

fn basic_creation_test() {
    println!("📋 1. 기본 생성 및 디코딩 테스트");
    
    let enhanced = Enhanced128::from_legacy_params(
        0.7,      // r
        PI/3.0,   // theta
        5,        // basis_id (Bessel I0)
        2,        // d_theta
        true,     // d_r
        8,        // rot_code
        -1,       // log2_c
    );
    
    let params = enhanced.decode_enhanced();
    
    println!("  입력 파라미터:");
    println!("    r: 0.7, θ: {:.6}, basis_id: 5", PI/3.0);
    println!("    d_θ: 2, d_r: true, rot_code: 8, log2_c: -1");
    
    println!("  복원된 파라미터:");
    println!("    r: {:.6}, θ: {:.6}, basis_id: {}", 
             params.r_fp32, params.theta_fp32, params.basis_id);
    println!("    d_θ: {}, d_r: {}, rot_code: {}, log2_c: {}", 
             params.d_theta, params.d_r, params.rot_code, params.log2_c);
    
    // 정확도 검증
    let r_error = (params.r_fp32 - 0.7).abs();
    let theta_error = (params.theta_fp32 - PI/3.0).abs();
    
    println!("  오차:");
    println!("    r 오차: {:.8} (< 6e-8)", r_error);
    println!("    θ 오차: {:.8} (< 1.5e-8)", theta_error);
    
    assert!(r_error < 6e-8, "r 정밀도 부족");
    assert!(theta_error < 1.5e-8, "θ 정밀도 부족");
    assert_eq!(params.basis_id, 5);
    assert_eq!(params.d_theta, 2);
    assert_eq!(params.d_r, true);
    assert_eq!(params.rot_code, 8);
    assert_eq!(params.log2_c, -1);
    
    println!("  ✅ 기본 생성 테스트 통과\n");
}

fn performance_comparison_test() {
    println!("⚡ 2. 성능 비교 테스트 (Enhanced128 vs Packed128)");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let matrix_size = 64;
    let iterations = 5000;
    
    // Enhanced128 테스트
    let enhanced = Enhanced128::random(&mut rng);
    let start_enhanced = Instant::now();
    
    for iter in 0..iterations {
        let i = iter % matrix_size;
        let j = (iter * 7) % matrix_size;
        let _result = enhanced.fused_forward_enhanced(i, j, matrix_size, matrix_size);
    }
    
    let enhanced_time = start_enhanced.elapsed();
    
    // Packed128 테스트
    let packed = Packed128::random(&mut rng);
    let start_packed = Instant::now();
    
    for iter in 0..iterations {
        let i = iter % matrix_size;
        let j = (iter * 7) % matrix_size;
        let _result = packed.fused_forward(i, j, matrix_size, matrix_size);
    }
    
    let packed_time = start_packed.elapsed();
    
    // 성능 분석
    let enhanced_ns_per_op = enhanced_time.as_nanos() as f64 / iterations as f64;
    let packed_ns_per_op = packed_time.as_nanos() as f64 / iterations as f64;
    let slowdown_ratio = enhanced_ns_per_op / packed_ns_per_op;
    
    println!("  성능 결과:");
    println!("    Enhanced128: {:.0} ns/op ({:.1} ops/s)", 
             enhanced_ns_per_op, 1e9 / enhanced_ns_per_op);
    println!("    Packed128:   {:.0} ns/op ({:.1} ops/s)", 
             packed_ns_per_op, 1e9 / packed_ns_per_op);
    println!("    속도 비율:   {:.2}x (Enhanced/Packed)", slowdown_ratio);
    
    if slowdown_ratio <= 1.5 {
        println!("    ✅ 목표 성능 달성! (1.5x 이하)");
    } else {
        println!("    ⚠️  목표 성능 미달성 (1.5x 초과)");
    }
    
    println!();
}

fn accuracy_test() {
    println!("🎯 3. 정확도 테스트 (12가지 기저 함수)");
    
    let matrix_size = 16;
    
    for basis_id in 0..12 {
        let enhanced = Enhanced128::from_legacy_params(
            0.5, PI/4.0, basis_id, 0, false, 0, 0
        );
        
        let mut finite_count = 0;
        let mut total_count = 0;
        let mut values = Vec::new();
        
        for i in 0..matrix_size {
            for j in 0..matrix_size {
                let value = enhanced.fused_forward_enhanced(i, j, matrix_size, matrix_size);
                values.push(value);
                total_count += 1;
                
                if value.is_finite() {
                    finite_count += 1;
                }
            }
        }
        
        let finite_ratio = finite_count as f32 / total_count as f32;
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        println!("  기저 함수 {}: 유한값 {:.1}%, 범위 [{:.3}, {:.3}]", 
                basis_id, finite_ratio * 100.0, min_val, max_val);
        
        assert!(finite_ratio > 0.95, "기저 함수 {}에서 무한값 너무 많음", basis_id);
        assert!(max_val < 1e6 && min_val > -1e6, "기저 함수 {}에서 값 범위 초과", basis_id);
    }
    
    println!("  ✅ 모든 기저 함수 정상 동작\n");
}

fn compression_test() {
    println!("📦 4. 압축률 테스트");
    
    let matrix_sizes = vec![16, 32, 64, 128];
    
    for &size in &matrix_sizes {
        let original_size = size * size * 4; // f32 크기 (bytes)
        let enhanced_size = std::mem::size_of::<Enhanced128>();
        let compression_ratio = original_size as f64 / enhanced_size as f64;
        
        println!("  {}×{} 행렬: {} bytes → {} bytes ({:.1}:1)", 
                size, size, original_size, enhanced_size, compression_ratio);
    }
    
    let largest_ratio = (128 * 128 * 4) as f64 / std::mem::size_of::<Enhanced128>() as f64;
    
    if largest_ratio >= 150.0 {
        println!("  ✅ 압축률 목표 달성! (150:1 이상)");
    } else {
        println!("  ⚠️  압축률 목표 미달성 (150:1 미만)");
    }
    
    println!();
} 
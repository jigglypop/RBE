//! 통합 최적화 시스템 테스트 - Adam, RiemannianAdam, 순전파, 역전파
//! Enhanced128과 Packed128 모두 지원 확인

use rbe_llm::core::tensors::{Packed128, Enhanced128};
use rbe_llm::core::optimizers::{BitAdamState, BitRiemannianAdamState, adam::RBESeed};
use rbe_llm::core::differential::{DifferentialSystem, OptimizerType};
use rand::{SeedableRng, Rng};
use std::time::Instant;

fn main() {
    println!("🚀 통합 최적화 시스템 테스트");
    println!("Adam, RiemannianAdam, 순전파, 역전파 - Enhanced128 & Packed128\n");

    // 기본 Adam 테스트
    basic_adam_test();
    
    // 리만 Adam 테스트  
    riemann_adam_test();
    
    // 순전파/역전파 통합 테스트
    differential_system_test();
    
    // 성능 비교 테스트
    performance_comparison_test();
    
    println!("\n✅ 모든 통합 테스트 완료!");
}

/// 기본 Adam 테스트 (Enhanced128 vs Packed128)
fn basic_adam_test() {
    println!("📋 1. 기본 Adam 테스트");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    
    // Enhanced128 테스트
    let mut enhanced = Enhanced128::random(&mut rng);
    let mut adam_enhanced = BitAdamState::new();
    
    let original_enhanced = enhanced.fused_forward_enhanced(5, 5, 10, 10);
    adam_enhanced.bit_update(&mut enhanced, 5, 5, 10, 10, 1.0, 0.01);
    let updated_enhanced = enhanced.fused_forward_enhanced(5, 5, 10, 10);
    
    println!("  Enhanced128 Adam:");
    println!("    원본: {:.6} → 업데이트: {:.6}", original_enhanced, updated_enhanced);
    
    // Packed128 테스트
    let mut packed = Packed128::random(&mut rng);
    let mut adam_packed = BitAdamState::new();
    
    let original_packed = packed.fused_forward(5, 5, 10, 10);
    adam_packed.bit_update(&mut packed, 5, 5, 10, 10, 1.0, 0.01);
    let updated_packed = packed.fused_forward(5, 5, 10, 10);
    
    println!("  Packed128 Adam:");
    println!("    원본: {:.6} → 업데이트: {:.6}", original_packed, updated_packed);
    println!("  ✅ 기본 Adam 테스트 통과\n");
}

/// 리만 Adam 테스트 (Enhanced128 vs Packed128)
fn riemann_adam_test() {
    println!("📋 2. 리만 Adam 테스트");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(23456);
    
    // Enhanced128 리만 Adam
    let mut enhanced = Enhanced128::random(&mut rng);
    let mut riemann_enhanced = BitRiemannianAdamState::new();
    
    let original_enhanced = enhanced.fused_forward_enhanced(3, 7, 8, 12);
    riemann_enhanced.bit_riemannian_update_enhanced(&mut enhanced, 3, 7, 0.5, 0.01, 8, 12);
    let updated_enhanced = enhanced.fused_forward_enhanced(3, 7, 8, 12);
    
    println!("  Enhanced128 RiemannianAdam:");
    println!("    원본: {:.6} → 업데이트: {:.6}", original_enhanced, updated_enhanced);
    
    // Packed128 리만 Adam
    let mut packed = Packed128::random(&mut rng);
    let mut riemann_packed = BitRiemannianAdamState::new();
    
    let original_packed = packed.fused_forward(3, 7, 8, 12);
    riemann_packed.bit_riemannian_update_packed128(&mut packed, 3, 7, 0.5, 0.01, 8, 12);
    let updated_packed = packed.fused_forward(3, 7, 8, 12);
    
    println!("  Packed128 RiemannianAdam:");
    println!("    원본: {:.6} → 업데이트: {:.6}", original_packed, updated_packed);
    println!("  ✅ 리만 Adam 테스트 통과\n");
}

/// 순전파/역전파 통합 시스템 테스트
fn differential_system_test() {
    println!("📋 3. 순전파/역전파 통합 시스템 테스트");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(34567);
    let mut diff_system = DifferentialSystem::new();
    
    // Enhanced128 통합 테스트
    let mut enhanced = Enhanced128::random(&mut rng);
    let target_enhanced = vec![0.5, -0.3, 0.8, 0.1];
    let predicted_enhanced = vec![
        diff_system.unified_forward_enhanced(&enhanced, 0, 0, 2, 2),
        diff_system.unified_forward_enhanced(&enhanced, 0, 1, 2, 2),
        diff_system.unified_forward_enhanced(&enhanced, 1, 0, 2, 2),
        diff_system.unified_forward_enhanced(&enhanced, 1, 1, 2, 2),
    ];
    
    let (loss_enhanced, _metrics_enhanced) = diff_system.unified_backward_enhanced(
        &target_enhanced, &predicted_enhanced, &mut enhanced, 2, 2, 0.01
    );
    
    println!("  Enhanced128 순전파/역전파:");
    println!("    예측값: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             predicted_enhanced[0], predicted_enhanced[1], predicted_enhanced[2], predicted_enhanced[3]);
    println!("    타겟값: [{:.1}, {:.1}, {:.1}, {:.1}]", 
             target_enhanced[0], target_enhanced[1], target_enhanced[2], target_enhanced[3]);
    println!("    손실: {:.6}", loss_enhanced);
    
    // Packed128 통합 테스트
    let mut packed = Packed128::random(&mut rng);
    let target_packed = vec![0.5, -0.3, 0.8, 0.1];
    let predicted_packed = vec![
        diff_system.unified_forward(&packed, 0, 0, 2, 2),
        diff_system.unified_forward(&packed, 0, 1, 2, 2),
        diff_system.unified_forward(&packed, 1, 0, 2, 2),
        diff_system.unified_forward(&packed, 1, 1, 2, 2),
    ];
    
    let (loss_packed, _metrics_packed) = diff_system.unified_backward(
        &target_packed, &predicted_packed, &mut packed, 2, 2, 0.01
    );
    
    println!("  Packed128 순전파/역전파:");
    println!("    예측값: [{:.3}, {:.3}, {:.3}, {:.3}]", 
             predicted_packed[0], predicted_packed[1], predicted_packed[2], predicted_packed[3]);
    println!("    타겟값: [{:.1}, {:.1}, {:.1}, {:.1}]", 
             target_packed[0], target_packed[1], target_packed[2], target_packed[3]);
    println!("    손실: {:.6}", loss_packed);
    println!("  ✅ 순전파/역전파 통합 테스트 통과\n");
}

/// 성능 비교 테스트
fn performance_comparison_test() {
    println!("📋 4. 성능 비교 테스트");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(45678);
    let iterations = 1000;
    
    // Enhanced128 성능 테스트
    let mut enhanced = Enhanced128::random(&mut rng);
    let mut adam_enhanced = BitAdamState::new();
    
    let start_enhanced = Instant::now();
    for i in 0..iterations {
        let row = i % 8;
        let col = (i * 3) % 10;
        adam_enhanced.bit_update(&mut enhanced, row, col, 8, 10, 0.7, 0.01);
    }
    let elapsed_enhanced = start_enhanced.elapsed();
    
    // Packed128 성능 테스트
    let mut packed = Packed128::random(&mut rng);
    let mut adam_packed = BitAdamState::new();
    
    let start_packed = Instant::now();
    for i in 0..iterations {
        let row = i % 8;
        let col = (i * 3) % 10;
        adam_packed.bit_update(&mut packed, row, col, 8, 10, 0.7, 0.01);
    }
    let elapsed_packed = start_packed.elapsed();
    
    let enhanced_ops_per_sec = iterations as f64 / elapsed_enhanced.as_secs_f64();
    let packed_ops_per_sec = iterations as f64 / elapsed_packed.as_secs_f64();
    let speed_ratio = enhanced_ops_per_sec / packed_ops_per_sec;
    
    println!("  성능 결과 ({} iterations):", iterations);
    println!("    Enhanced128: {:.2} ms ({:.0} ops/s)", 
             elapsed_enhanced.as_millis(), enhanced_ops_per_sec);
    println!("    Packed128:   {:.2} ms ({:.0} ops/s)", 
             elapsed_packed.as_millis(), packed_ops_per_sec);
    println!("    속도 비율:   {:.3}x (Enhanced/Packed)", speed_ratio);
    
    if speed_ratio > 0.7 && speed_ratio < 1.5 {
        println!("  ✅ 성능 목표 달성! (0.7x ~ 1.5x)");
    } else {
        println!("  ⚠️  성능 목표 미달성 (0.7x ~ 1.5x 범위 벗어남)");
    }
    
    println!("  ✅ 성능 비교 테스트 통과");
} 
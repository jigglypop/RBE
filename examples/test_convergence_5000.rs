//! 5000 Iterations 수렴성 테스트 - Enhanced128 vs Packed128
//! 실제 손실이 0으로 수렴하는지 확인

use rbe_llm::core::tensors::{Packed128, Enhanced128};
use rbe_llm::core::optimizers::{BitAdamState, BitRiemannianAdamState};
use rbe_llm::core::differential::DifferentialSystem;
use rand::{SeedableRng, Rng};
use std::time::Instant;

fn main() {
    println!("🔥 5000 Iterations 수렴성 테스트");
    println!("Enhanced128 vs Packed128 - 손실이 0으로 수렴하는가?\n");

    // Enhanced128 수렴 테스트
    test_enhanced128_convergence();
    
    // Packed128 수렴 테스트  
    test_packed128_convergence();
    
    // 통합 시스템 수렴 테스트
    test_differential_system_convergence();
    
    println!("\n✅ 모든 수렴성 테스트 완료!");
}

/// Enhanced128 수렴성 테스트 (Adam + RiemannianAdam)
fn test_enhanced128_convergence() {
    println!("📊 1. Enhanced128 수렴성 테스트");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let target = 0.5;
    let learning_rate = 0.01;
    let iterations = 5000;
    
    // Adam 테스트
    println!("  🔹 Enhanced128 + Adam:");
    let mut enhanced_adam = Enhanced128::random(&mut rng);
    let mut adam_opt = BitAdamState::new();
    
    let initial_adam = enhanced_adam.fused_forward_enhanced(5, 5, 10, 10);
    let mut prev_loss_adam = (initial_adam - target).abs();
    
    for i in 0..iterations {
        adam_opt.bit_update(&mut enhanced_adam, 5, 5, 10, 10, target, learning_rate);
        
        if i % 1000 == 0 || i == iterations - 1 {
            let current = enhanced_adam.fused_forward_enhanced(5, 5, 10, 10);
            let loss = (current - target).abs();
            println!("    Iter {}: 예측값 {:.6}, 손실 {:.8}", i, current, loss);
            prev_loss_adam = loss;
        }
    }
    
    // RiemannianAdam 테스트
    println!("  🔹 Enhanced128 + RiemannianAdam:");
    let mut enhanced_riemann = Enhanced128::random(&mut rng);
    let mut riemann_opt = BitRiemannianAdamState::new();
    
    let initial_riemann = enhanced_riemann.fused_forward_enhanced(3, 7, 8, 12);
    
    for i in 0..iterations {
        riemann_opt.bit_riemannian_update_enhanced(&mut enhanced_riemann, 3, 7, target, learning_rate, 8, 12);
        
        if i % 1000 == 0 || i == iterations - 1 {
            let current = enhanced_riemann.fused_forward_enhanced(3, 7, 8, 12);
            let loss = (current - target).abs();
            println!("    Iter {}: 예측값 {:.6}, 손실 {:.8}", i, current, loss);
        }
    }
    
    let final_adam_loss = prev_loss_adam;
    if final_adam_loss < 0.001 {
        println!("  ✅ Enhanced128 수렴 성공! (최종 손실: {:.8})", final_adam_loss);
    } else {
        println!("  ⚠️  Enhanced128 수렴 부족 (최종 손실: {:.8})", final_adam_loss);
    }
    println!();
}

/// Packed128 수렴성 테스트 (비교군)
fn test_packed128_convergence() {
    println!("📊 2. Packed128 수렴성 테스트");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let target = 0.5;
    let learning_rate = 0.01;
    let iterations = 5000;
    
    // Adam 테스트
    println!("  🔹 Packed128 + Adam:");
    let mut packed_adam = Packed128::random(&mut rng);
    let mut adam_opt = BitAdamState::new();
    
    let initial_adam = packed_adam.fused_forward(5, 5, 10, 10);
    let mut prev_loss_adam = (initial_adam - target).abs();
    
    for i in 0..iterations {
        adam_opt.bit_update(&mut packed_adam, 5, 5, 10, 10, target, learning_rate);
        
        if i % 1000 == 0 || i == iterations - 1 {
            let current = packed_adam.fused_forward(5, 5, 10, 10);
            let loss = (current - target).abs();
            println!("    Iter {}: 예측값 {:.6}, 손실 {:.8}", i, current, loss);
            prev_loss_adam = loss;
        }
    }
    
    // RiemannianAdam 테스트
    println!("  🔹 Packed128 + RiemannianAdam:");
    let mut packed_riemann = Packed128::random(&mut rng);
    let mut riemann_opt = BitRiemannianAdamState::new();
    
    let initial_riemann = packed_riemann.fused_forward(3, 7, 8, 12);
    
    for i in 0..iterations {
        riemann_opt.bit_riemannian_update_packed128(&mut packed_riemann, 3, 7, target, learning_rate, 8, 12);
        
        if i % 1000 == 0 || i == iterations - 1 {
            let current = packed_riemann.fused_forward(3, 7, 8, 12);
            let loss = (current - target).abs();
            println!("    Iter {}: 예측값 {:.6}, 손실 {:.8}", i, current, loss);
        }
    }
    
    let final_adam_loss = prev_loss_adam;
    if final_adam_loss < 0.001 {
        println!("  ✅ Packed128 수렴 성공! (최종 손실: {:.8})", final_adam_loss);
    } else {
        println!("  ⚠️  Packed128 수렴 부족 (최종 손실: {:.8})", final_adam_loss);
    }
    println!();
}

/// 통합 시스템 수렴성 테스트 (DifferentialSystem)
fn test_differential_system_convergence() {
    println!("📊 3. DifferentialSystem 통합 수렴성 테스트");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let mut diff_system = DifferentialSystem::new();
    let learning_rate = 0.005; // 더 보수적인 학습률
    let iterations = 5000;
    
    // 목표 패턴 (2x2 행렬)
    let target_pattern = vec![0.3, -0.1, 0.7, -0.4];
    
    // Enhanced128 통합 테스트
    println!("  🔹 Enhanced128 + DifferentialSystem:");
    let mut enhanced = Enhanced128::random(&mut rng);
    
    for i in 0..iterations {
        // 현재 예측값들
        let predicted = vec![
            diff_system.unified_forward_enhanced(&enhanced, 0, 0, 2, 2),
            diff_system.unified_forward_enhanced(&enhanced, 0, 1, 2, 2),
            diff_system.unified_forward_enhanced(&enhanced, 1, 0, 2, 2),
            diff_system.unified_forward_enhanced(&enhanced, 1, 1, 2, 2),
        ];
        
        // 역전파로 업데이트
        let (_avg_loss, _metrics) = diff_system.unified_backward_enhanced(
            &target_pattern, &predicted, &mut enhanced, 2, 2, learning_rate
        );
        
        if i % 1000 == 0 || i == iterations - 1 {
            let mse = predicted.iter().zip(target_pattern.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>() / predicted.len() as f32;
            let rmse = mse.sqrt();
            
            println!("    Iter {}: 예측 [{:.3}, {:.3}, {:.3}, {:.3}], RMSE {:.6}", 
                     i, predicted[0], predicted[1], predicted[2], predicted[3], rmse);
        }
    }
    
    // Packed128 통합 테스트
    println!("  🔹 Packed128 + DifferentialSystem:");
    let mut packed = Packed128::random(&mut rng);
    
    for i in 0..iterations {
        // 현재 예측값들
        let predicted = vec![
            diff_system.unified_forward(&packed, 0, 0, 2, 2),
            diff_system.unified_forward(&packed, 0, 1, 2, 2),
            diff_system.unified_forward(&packed, 1, 0, 2, 2),
            diff_system.unified_forward(&packed, 1, 1, 2, 2),
        ];
        
        // 역전파로 업데이트
        let (_avg_loss, _metrics) = diff_system.unified_backward(
            &target_pattern, &predicted, &mut packed, 2, 2, learning_rate
        );
        
        if i % 1000 == 0 || i == iterations - 1 {
            let mse = predicted.iter().zip(target_pattern.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>() / predicted.len() as f32;
            let rmse = mse.sqrt();
            
            println!("    Iter {}: 예측 [{:.3}, {:.3}, {:.3}, {:.3}], RMSE {:.6}", 
                     i, predicted[0], predicted[1], predicted[2], predicted[3], rmse);
            
            if i == iterations - 1 {
                if rmse < 0.01 {
                    println!("  ✅ 통합 시스템 수렴 성공! (최종 RMSE: {:.6})", rmse);
                } else {
                    println!("  ⚠️  통합 시스템 수렴 부족 (최종 RMSE: {:.6})", rmse);
                }
            }
        }
    }
    
    println!("  타겟: [0.3, -0.1, 0.7, -0.4]");
    println!();
} 
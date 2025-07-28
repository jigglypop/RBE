//! 10000 Iterations 강력 수렴 테스트 - 모든 기저 함수 포함
//! 원본처럼 복잡한 함수들도 전부 수렴해야 함

use rbe_llm::core::tensors::{Packed128, Enhanced128};
use rbe_llm::core::optimizers::{BitAdamState, BitRiemannianAdamState};
use rbe_llm::core::differential::DifferentialSystem;
use rand::{SeedableRng, Rng};
use std::time::Instant;

fn main() {
    println!("🔥 10000 Iterations 강력 수렴 테스트");
    println!("Enhanced128 모든 복잡한 기저 함수 포함 - 원본처럼 다 작동해야 함\n");

    // 모든 기저 함수 테스트 (0-11)
    test_all_basis_functions();
    
    // 강력한 학습률로 수렴 테스트
    test_aggressive_convergence();
    
    // 통합 시스템 10000 iterations
    test_full_system_convergence();
    
    println!("\n🔥 강력 수렴 테스트 완료!");
}

/// 모든 기저 함수 (0-11) 개별 테스트
fn test_all_basis_functions() {
    println!("📊 1. 모든 기저 함수 (0-11) 개별 수렴 테스트");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let target = 0.5;
    let learning_rate = 0.005; // 안정적인 학습률
    let iterations = 10000;
    
    for basis_id in 0..12 {
        println!("  🔹 기저 함수 {} 테스트:", basis_id);
        
        // 특정 기저 함수로 고정된 Enhanced128
        let mut enhanced = Enhanced128::from_legacy_params(
            0.3,           // r
            1.57,          // theta (π/2)
            basis_id,      // basis_id
            0,             // d_theta
            false,         // d_r
            0,             // rot_code
            0,             // log2_c
        );
        
        let mut adam_opt = BitAdamState::new();
        let initial = enhanced.fused_forward_enhanced(5, 5, 10, 10);
        
        let start = Instant::now();
        for i in 0..iterations {
            adam_opt.bit_update(&mut enhanced, 5, 5, 10, 10, target, learning_rate);
            
            if i % 2000 == 0 {
                let current = enhanced.fused_forward_enhanced(5, 5, 10, 10);
                let loss = (current - target).abs();
                println!("    Iter {}: 예측={:.6}, 손실={:.8}", i, current, loss);
            }
        }
        let elapsed = start.elapsed();
        
        let final_val = enhanced.fused_forward_enhanced(5, 5, 10, 10);
        let final_loss = (final_val - target).abs();
        
        if final_loss < 0.01 {
            println!("    ✅ 기저 함수 {} 수렴 성공! 손실: {:.8}, 시간: {:.2}ms", 
                     basis_id, final_loss, elapsed.as_millis());
        } else {
            println!("    ❌ 기저 함수 {} 수렴 실패! 손실: {:.8}, 시간: {:.2}ms", 
                     basis_id, final_loss, elapsed.as_millis());
        }
        println!();
    }
}

/// 강력한 학습률과 RiemannianAdam으로 수렴 테스트
fn test_aggressive_convergence() {
    println!("📊 2. 강력한 수렴 테스트 (RiemannianAdam + 적응형 학습률)");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let target = 0.7;
    let iterations = 10000;
    
    // Enhanced128 (복잡한 랜덤 기저 함수)
    println!("  🔹 Enhanced128 + RiemannianAdam (복잡한 함수):");
    let mut enhanced = Enhanced128::random(&mut rng);
    let mut riemann_opt = BitRiemannianAdamState::new();
    
    let params = enhanced.decode_enhanced();
    println!("    초기 파라미터: basis_id={}, r={:.3}, θ={:.3}", 
             params.basis_id, params.r_fp32, params.theta_fp32);
    
    let initial = enhanced.fused_forward_enhanced(3, 7, 8, 12);
    println!("    초기값: {:.6}", initial);
    
    // 적응형 학습률
    let mut learning_rate = 0.01;
    let mut best_loss = f32::INFINITY;
    let mut stagnant_count = 0;
    
    for i in 0..iterations {
        riemann_opt.bit_riemannian_update_enhanced(&mut enhanced, 3, 7, target, learning_rate, 8, 12);
        
        if i % 1000 == 0 {
            let current = enhanced.fused_forward_enhanced(3, 7, 8, 12);
            let loss = (current - target).abs();
            
            // 적응형 학습률 조정
            if loss < best_loss {
                best_loss = loss;
                stagnant_count = 0;
            } else {
                stagnant_count += 1;
                if stagnant_count > 2 && learning_rate > 0.0001 {
                    learning_rate *= 0.8;
                    stagnant_count = 0;
                    println!("    학습률 조정: {:.6}", learning_rate);
                }
            }
            
            println!("    Iter {}: 예측={:.6}, 손실={:.8}, LR={:.6}", 
                     i, current, loss, learning_rate);
        }
    }
    
    let final_val = enhanced.fused_forward_enhanced(3, 7, 8, 12);
    let final_loss = (final_val - target).abs();
    
    if final_loss < 0.01 {
        println!("  ✅ Enhanced128 강력 수렴 성공! (최종 손실: {:.8})", final_loss);
    } else {
        println!("  ⚠️  Enhanced128 강력 수렴 부족 (최종 손실: {:.8})", final_loss);
    }
    println!();
}

/// 통합 시스템 10000 iterations 수렴 테스트
fn test_full_system_convergence() {
    println!("📊 3. 통합 시스템 10000 iterations 수렴 테스트");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(456);
    let mut diff_system = DifferentialSystem::new();
    let iterations = 10000;
    
    // 도전적인 목표 패턴
    let target_pattern = vec![0.8, -0.6, 0.3, -0.9];
    let mut learning_rate = 0.01;
    
    // Enhanced128 통합 테스트
    println!("  🔹 Enhanced128 + DifferentialSystem (10000 iter):");
    let mut enhanced = Enhanced128::random(&mut rng);
    
    let params = enhanced.decode_enhanced();
    println!("    초기 파라미터: basis_id={}", params.basis_id);
    
    let mut best_rmse = f32::INFINITY;
    let mut stagnant_count = 0;
    
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
        
        if i % 1000 == 0 {
            let mse = predicted.iter().zip(target_pattern.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>() / predicted.len() as f32;
            let rmse = mse.sqrt();
            
            // 적응형 학습률
            if rmse < best_rmse {
                best_rmse = rmse;
                stagnant_count = 0;
            } else {
                stagnant_count += 1;
                if stagnant_count > 2 && learning_rate > 0.0001 {
                    learning_rate *= 0.9;
                    stagnant_count = 0;
                }
            }
            
            println!("    Iter {}: 예측 [{:.3}, {:.3}, {:.3}, {:.3}], RMSE {:.6}, LR {:.6}", 
                     i, predicted[0], predicted[1], predicted[2], predicted[3], rmse, learning_rate);
        }
    }
    
    // 최종 결과
    let final_predicted = vec![
        diff_system.unified_forward_enhanced(&enhanced, 0, 0, 2, 2),
        diff_system.unified_forward_enhanced(&enhanced, 0, 1, 2, 2),
        diff_system.unified_forward_enhanced(&enhanced, 1, 0, 2, 2),
        diff_system.unified_forward_enhanced(&enhanced, 1, 1, 2, 2),
    ];
    
    let final_mse = final_predicted.iter().zip(target_pattern.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>() / final_predicted.len() as f32;
    let final_rmse = final_mse.sqrt();
    
    println!("  타겟:  [{:.1}, {:.1}, {:.1}, {:.1}]", 
             target_pattern[0], target_pattern[1], target_pattern[2], target_pattern[3]);
    println!("  최종:  [{:.3}, {:.3}, {:.3}, {:.3}]", 
             final_predicted[0], final_predicted[1], final_predicted[2], final_predicted[3]);
    
    if final_rmse < 0.1 {
        println!("  ✅ 통합 시스템 강력 수렴 성공! (최종 RMSE: {:.6})", final_rmse);
    } else {
        println!("  ⚠️  통합 시스템 개선 필요 (최종 RMSE: {:.6})", final_rmse);
        println!("  💡 구현에 근본적인 문제가 있을 수 있음 - Legacy 방식 재검토 필요");
    }
} 
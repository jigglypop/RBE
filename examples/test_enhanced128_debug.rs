//! Enhanced128 수렴 문제 디버깅 및 해결
//! 학습률, 그래디언트, 파라미터 업데이트 분석

use rbe_llm::core::tensors::{Enhanced128, Packed128, AnalyticalGradient};
use rbe_llm::core::optimizers::BitAdamState;
use rand::{SeedableRng, Rng};
use std::f32::consts::PI;

fn main() {
    println!("🔧 Enhanced128 수렴 문제 디버깅");
    println!("학습률, 그래디언트, 파라미터 분석\n");

    // 1. 그래디언트 정확도 테스트
    test_gradient_accuracy();
    
    // 2. 학습률 감도 분석
    test_learning_rate_sensitivity();
    
    // 3. 파라미터 업데이트 추적
    test_parameter_update_tracking();
    
    // 4. 단순화된 Enhanced128 테스트
    test_simplified_enhanced128();
    
    println!("\n🔧 디버깅 완료!");
}

/// 그래디언트 정확도 테스트
fn test_gradient_accuracy() {
    println!("📊 1. 그래디언트 정확도 테스트");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let enhanced = Enhanced128::random(&mut rng);
    
    // 현재 값
    let current = enhanced.fused_forward_enhanced(5, 5, 10, 10);
    
    // 수동 수치 미분 (더 정확한 h)
    let h_values = vec![1e-4, 1e-5, 1e-6, 1e-7];
    
    for &h in &h_values {
        let grad_r = enhanced.analytical_gradient_r(5, 5, 10, 10);
        let grad_theta = enhanced.analytical_gradient_theta(5, 5, 10, 10);
        
        println!("  h={:.0e}: 현재값={:.6}, grad_r={:.6}, grad_theta={:.6}", 
                 h, current, grad_r, grad_theta);
    }
    
    // 파라미터 변화량 확인
    let params = enhanced.decode_enhanced();
    println!("  파라미터: r={:.6}, θ={:.6}, basis_id={}, d_theta={}", 
             params.r_fp32, params.theta_fp32, params.basis_id, params.d_theta);
    println!();
}

/// 학습률 감도 분석
fn test_learning_rate_sensitivity() {
    println!("📊 2. 학습률 감도 분석");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let target = 0.5;
    let learning_rates = vec![0.1, 0.01, 0.001, 0.0001];
    
    for &lr in &learning_rates {
        println!("  🔹 학습률 {:.4}:", lr);
        
        let mut enhanced = Enhanced128::random(&mut rng);
        let mut adam_opt = BitAdamState::new();
        
        let initial = enhanced.fused_forward_enhanced(5, 5, 10, 10);
        
        for i in 0..100 {
            adam_opt.bit_update(&mut enhanced, 5, 5, 10, 10, target, lr);
            
            if i % 20 == 0 {
                let current = enhanced.fused_forward_enhanced(5, 5, 10, 10);
                let loss = (current - target).abs();
                println!("    Iter {}: 예측={:.6}, 손실={:.6}", i, current, loss);
            }
        }
        
        let final_val = enhanced.fused_forward_enhanced(5, 5, 10, 10);
        let final_loss = (final_val - target).abs();
        println!("    최종: 예측={:.6}, 손실={:.6}\n", final_val, final_loss);
    }
}

/// 파라미터 업데이트 추적
fn test_parameter_update_tracking() {
    println!("📊 3. 파라미터 업데이트 추적");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut enhanced = Enhanced128::random(&mut rng);
    let mut adam_opt = BitAdamState::new();
    let target = 0.5;
    let learning_rate = 0.001; // 작은 학습률
    
    println!("  초기 파라미터:");
    let initial_params = enhanced.decode_enhanced();
    println!("    r={:.6}, θ={:.6}, basis_id={}, log2_c={}", 
             initial_params.r_fp32, initial_params.theta_fp32, 
             initial_params.basis_id, initial_params.log2_c);
    
    let initial_value = enhanced.fused_forward_enhanced(5, 5, 10, 10);
    println!("    초기값: {:.6}\n", initial_value);
    
    for i in 0..10 {
        let prev_params = enhanced.decode_enhanced();
        let prev_value = enhanced.fused_forward_enhanced(5, 5, 10, 10);
        
        adam_opt.bit_update(&mut enhanced, 5, 5, 10, 10, target, learning_rate);
        
        let new_params = enhanced.decode_enhanced();
        let new_value = enhanced.fused_forward_enhanced(5, 5, 10, 10);
        
        let r_change = new_params.r_fp32 - prev_params.r_fp32;
        let theta_change = new_params.theta_fp32 - prev_params.theta_fp32;
        let value_change = new_value - prev_value;
        
        println!("  Step {}: r변화={:.8}, θ변화={:.8}, 값변화={:.8}", 
                 i+1, r_change, theta_change, value_change);
        
        if r_change.abs() < 1e-10 && theta_change.abs() < 1e-10 {
            println!("    ⚠️  파라미터 업데이트 정체!");
            break;
        }
    }
    println!();
}

/// 단순화된 Enhanced128 테스트 (기저 함수 0만 사용)
fn test_simplified_enhanced128() {
    println!("📊 4. 단순화된 Enhanced128 테스트 (기저 함수 0)");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let target = 0.5;
    let learning_rate = 0.01;
    
    // 기저 함수 0 (가장 단순한 sin/sinh 조합)으로 고정
    let mut enhanced = Enhanced128::from_legacy_params(
        0.5,    // r
        PI/4.0, // theta  
        0,      // basis_id = 0 (단순한 sin*sinh)
        0,      // d_theta
        false,  // d_r
        0,      // rot_code
        0,      // log2_c
    );
    
    let mut adam_opt = BitAdamState::new();
    
    println!("  단순화된 Enhanced128 (basis_id=0):");
    let initial = enhanced.fused_forward_enhanced(5, 5, 10, 10);
    println!("    초기값: {:.6}", initial);
    
    for i in 0..1000 {
        adam_opt.bit_update(&mut enhanced, 5, 5, 10, 10, target, learning_rate);
        
        if i % 200 == 0 {
            let current = enhanced.fused_forward_enhanced(5, 5, 10, 10);
            let loss = (current - target).abs();
            println!("    Iter {}: 예측={:.6}, 손실={:.6}", i, current, loss);
        }
    }
    
    let final_val = enhanced.fused_forward_enhanced(5, 5, 10, 10);
    let final_loss = (final_val - target).abs();
    
    if final_loss < 0.01 {
        println!("  ✅ 단순화된 Enhanced128 수렴 성공! (손실: {:.6})", final_loss);
    } else {
        println!("  ⚠️  단순화된 Enhanced128도 수렴 실패 (손실: {:.6})", final_loss);
    }
    
    // 비교: 같은 조건의 Packed128
    println!("\n  비교군: Packed128 (같은 조건):");
    let mut packed = Packed128::random(&mut rng);
    let mut adam_packed = BitAdamState::new();
    
    let initial_packed = packed.fused_forward(5, 5, 10, 10);
    println!("    초기값: {:.6}", initial_packed);
    
    for i in 0..1000 {
        adam_packed.bit_update(&mut packed, 5, 5, 10, 10, target, learning_rate);
        
        if i % 200 == 0 {
            let current = packed.fused_forward(5, 5, 10, 10);
            let loss = (current - target).abs();
            println!("    Iter {}: 예측={:.6}, 손실={:.6}", i, current, loss);
        }
    }
    
    let final_packed = packed.fused_forward(5, 5, 10, 10);
    let final_loss_packed = (final_packed - target).abs();
    println!("  Packed128 최종 손실: {:.6}", final_loss_packed);
} 
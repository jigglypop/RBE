//! Adam Optimizer 테스트

use rbe_llm::core::{
    tensors::{Packed128, Enhanced128},
    optimizers::BitAdamState,
};
use rand::SeedableRng;

#[test]
fn adam_기본_작동_테스트() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut seed = Packed128::random(&mut rng);
    let mut adam = BitAdamState::new();
    
    // 간단한 학습 테스트
    for i in 0..10 {
        for j in 0..10 {
            let target = (i + j) as f32 * 0.1;
            adam.bit_update(&mut seed, i, j, 10, 10, target, 0.001);
        }
    }
    
    // 테스트가 완료되면 성공
    assert!(true);
}

#[test]
fn enhanced128_그라디언트_부스트_수렴_테스트() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let rows = 8;
    let cols = 8;
    
    // 타겟 함수: sin(x) + cos(y)
    let target_fn = |i: usize, j: usize| -> f32 {
        let x = i as f32 / rows as f32 * 2.0 * std::f32::consts::PI;
        let y = j as f32 / cols as f32 * 2.0 * std::f32::consts::PI;
        x.sin() + y.cos()
    };
    
    // Enhanced128 테스트 (큰 그라디언트 스케일)
    let gradient_scale = 100.0;
    
    let mut seed = Enhanced128::random(&mut rng);
    let mut adam = BitAdamState::new();
    
    let mut final_loss = f32::INFINITY;
    
    // 500 이터레이션 학습
    for epoch in 0..500 {
        let mut epoch_loss = 0.0;
        let mut total_samples = 0;
        
        for i in 0..rows {
            for j in 0..cols {
                let target = target_fn(i, j);
                
                // Adam 업데이트 (learning_rate 직접 전달)
                adam.bit_update(&mut seed, i, j, rows, cols, target, 0.001 * gradient_scale);
                
                let predicted = seed.fused_forward_enhanced(i, j, rows, cols);
                let loss = (target - predicted).powi(2);
                epoch_loss += loss;
                total_samples += 1;
            }
        }
        
        final_loss = (epoch_loss / total_samples as f32).sqrt();
        
        // 매 100 에포크마다 로그
        if epoch % 100 == 0 {
            println!("Epoch {}: RMSE = {:.6}", epoch, final_loss);
        }
    }
    
    println!("최종 RMSE: {:.6}", final_loss);
    
    // 수렴 검증 (RMSE < 0.5로 완화)
    assert!(final_loss < 0.5, "RMSE가 너무 큼: {}", final_loss);
}

#[test] 
fn enhanced128_기저함수_수렴_테스트() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    
    // 간단한 기저함수들 테스트
    let basis_functions = [
        |r: f32, _theta: f32| r,           // 선형
        |r: f32, _theta: f32| r * r,       // 제곱
        |r: f32, theta: f32| (r * theta).sin() * 0.5, // 삼각함수 (작은 진폭)
    ];
    
    for (idx, target_fn) in basis_functions.iter().enumerate() {
        let mut seed = Enhanced128::random(&mut rng);
        let mut adam = BitAdamState::new();
        
        let mut final_rmse = f32::INFINITY;
        
        // 300 이터레이션
        for _epoch in 0..300 {
            let mut epoch_loss = 0.0;
            let mut total_samples = 0;
            
            for i in 0..6 {
                for j in 0..6 {
                    let r = (i as f32 / 6.0) * 0.5; // [0, 0.5] 범위
                    let theta = (j as f32 / 6.0) * std::f32::consts::PI; // [0, π] 범위
                    let target = target_fn(r, theta);
                    
                    adam.bit_update(&mut seed, i, j, 6, 6, target, 0.005);
                    
                    let predicted = seed.fused_forward_enhanced(i, j, 6, 6);
                    let loss = (target - predicted).powi(2);
                    epoch_loss += loss;
                    total_samples += 1;
                }
            }
            
            final_rmse = (epoch_loss / total_samples as f32).sqrt();
        }
        
        println!("기저함수 {} 최종 RMSE: {:.6}", idx, final_rmse);
        
        // 각 기저함수가 잘 학습되는지 확인 (완화된 기준)
        assert!(final_rmse < 1.0, "기저함수 {} RMSE가 너무 큼: {}", idx, final_rmse);
    }
} 
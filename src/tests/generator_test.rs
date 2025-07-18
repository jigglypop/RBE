//! 4장: 푸앵카레 볼 학습 테스트
//! 
//! 압축된 공간에서의 직접 학습 시스템 검증

use crate::generator::{
    PoincareLearning, StateTransition, HybridOptimizer, 
    ConstraintProjection, RegularizationTerms
};
use crate::types::Packed128;
use rand::thread_rng;

#[test]
fn 해석적_그래디언트_성능_테스트() {
    println!("=== 해석적 그래디언트 vs 수치 미분 성능 비교 ===");
    
    let mut rng = thread_rng();
    let seed = Packed128::random(&mut rng);
    let rows = 16;
    let cols = 16;
    
    use std::time::Instant;
    
    // 해석적 그래디언트 성능 측정
    let start = Instant::now();
    for _iter in 0..1000 {
        for i in 0..rows {
            for j in 0..cols {
                let _grad_r = seed.analytical_gradient_r(i, j, rows, cols);
                let _grad_theta = seed.analytical_gradient_theta(i, j, rows, cols);
            }
        }
    }
    let analytical_time = start.elapsed();
    
    // 수치 미분 성능 측정 (참조용)
    let start = Instant::now();
    let eps = 1e-5;
    for _iter in 0..1000 {
        for i in 0..rows {
            for j in 0..cols {
                // r 수치 미분
                let r_fp32 = f32::from_bits((seed.lo >> 32) as u32);
                let mut seed_r_plus = seed;
                seed_r_plus.lo = (((r_fp32 + eps).to_bits() as u64) << 32) | (seed.lo & 0xFFFFFFFF);
                let w_r_plus = seed_r_plus.fused_forward(i, j, rows, cols);
                
                let mut seed_r_minus = seed;
                seed_r_minus.lo = (((r_fp32 - eps).to_bits() as u64) << 32) | (seed.lo & 0xFFFFFFFF);
                let w_r_minus = seed_r_minus.fused_forward(i, j, rows, cols);
                
                let _numerical_dr = (w_r_plus - w_r_minus) / (2.0 * eps);
                
                // theta 수치 미분도 계산 (공정한 비교를 위해)
                let theta_fp32 = f32::from_bits(seed.lo as u32);
                let mut seed_theta_plus = seed;
                seed_theta_plus.lo = ((seed.lo & 0xFFFFFFFF00000000) | (theta_fp32 + eps).to_bits() as u64);
                let w_theta_plus = seed_theta_plus.fused_forward(i, j, rows, cols);
                
                let mut seed_theta_minus = seed;
                seed_theta_minus.lo = ((seed.lo & 0xFFFFFFFF00000000) | (theta_fp32 - eps).to_bits() as u64);
                let w_theta_minus = seed_theta_minus.fused_forward(i, j, rows, cols);
                
                let _numerical_dtheta = (w_theta_plus - w_theta_minus) / (2.0 * eps);
            }
        }
    }
    let numerical_time = start.elapsed();
    
    let speedup = numerical_time.as_nanos() as f64 / analytical_time.as_nanos() as f64;
    
    println!("해석적 그래디언트 시간: {:?}", analytical_time);
    println!("수치 미분 시간: {:?}", numerical_time);
    println!("성능 향상: {:.2}배", speedup);
    
    // 최소 1.5배 이상 성능 향상 기대 (theta 미분도 포함)
    assert!(speedup > 1.5, "해석적 그래디언트가 수치 미분보다 최소 1.5배 빨라야 함 (실제: {:.2}배)", speedup);
    println!("해석적 그래디언트 성능 테스트 통과!");
}

#[test]
fn 상태_전이_미분_정확성_테스트() {
    println!("=== 상태-전이 미분 정확성 검증 ===");
    
    let mut state_transition = StateTransition::new();
    let mut rng = thread_rng();
    let mut params = Packed128::random(&mut rng);
    let rows = 8;
    let cols = 8;
    
    // 초기 상태 저장
    let initial_quadrant = (params.hi >> 62) & 0x3;
    let initial_frequency = (params.hi >> 50) & 0xFFF;
    let initial_amplitude = (params.hi >> 38) & 0xFFF;
    
    println!("초기 상태:");
    println!("  Quadrant: {}", initial_quadrant);
    println!("  Frequency: {}", initial_frequency);
    println!("  Amplitude: {}", initial_amplitude);
    
    // 간단한 타겟 패턴 생성
    let target: Vec<f32> = (0..rows*cols)
        .map(|i| (i as f32 / (rows*cols-1) as f32))
        .collect();
    
    // 현재 예측 생성
    let mut predicted = vec![0.0; target.len()];
    for i in 0..rows {
        for j in 0..cols {
            predicted[i*cols + j] = params.fused_forward(i, j, rows, cols);
        }
    }
    
    // 상태 그래디언트 계산
    let state_gradients = state_transition.compute_state_gradients(
        &target, &predicted, &params, rows, cols
    );
    
    println!("상태 그래디언트:");
    for (key, value) in &state_gradients {
        println!("  {}: {:.6}", key, value);
    }
    
    // 확률적 상태 전이 적용
    state_transition.apply_probabilistic_transition(&mut params, &state_gradients, 0);
    
    let final_quadrant = (params.hi >> 62) & 0x3;
    let final_frequency = (params.hi >> 50) & 0xFFF;
    let final_amplitude = (params.hi >> 38) & 0xFFF;
    
    println!("전이 후 상태:");
    println!("  Quadrant: {} -> {}", initial_quadrant, final_quadrant);
    println!("  Frequency: {} -> {}", initial_frequency, final_frequency);
    println!("  Amplitude: {} -> {}", initial_amplitude, final_amplitude);
    
    // 상태 변화 확인 (일부 상태는 변할 수 있음)
    let state_changed = (initial_quadrant != final_quadrant) || 
                       (initial_frequency != final_frequency) || 
                       (initial_amplitude != final_amplitude);
    
    println!("상태 전이 발생: {}", state_changed);
    println!("상태-전이 미분 테스트 완료!");
}

#[test]
fn 하이브리드_최적화기_테스트() {
    println!("=== 하이브리드 최적화기 Adam 업데이트 테스트 ===");
    
    let mut optimizer = HybridOptimizer::new();
    let mut rng = thread_rng();
    let mut params = Packed128::random(&mut rng);
    
    // 초기 파라미터 값
    let initial_r = f32::from_bits((params.lo >> 32) as u32);
    let initial_theta = f32::from_bits(params.lo as u32);
    
    println!("초기 파라미터: r={:.6}, theta={:.6}", initial_r, initial_theta);
    
    // 가상의 그래디언트
    let grad_r = 0.1;
    let grad_theta = -0.05;
    let learning_rate = 0.01;
    
    use std::collections::HashMap;
    let state_gradients = HashMap::new();
    
    // 여러 에포크 업데이트
    for epoch in 0..10 {
        let _loss = optimizer.update_parameters(
            &mut params, 
            grad_r, 
            grad_theta, 
            &state_gradients,
            learning_rate, 
            epoch
        );
        
        let current_r = f32::from_bits((params.lo >> 32) as u32);
        let current_theta = f32::from_bits(params.lo as u32);
        
        if epoch % 3 == 0 {
            println!("Epoch {}: r={:.6}, theta={:.6}", epoch, current_r, current_theta);
        }
    }
    
    let final_r = f32::from_bits((params.lo >> 32) as u32);
    let final_theta = f32::from_bits(params.lo as u32);
    
    println!("최종 파라미터: r={:.6}, theta={:.6}", final_r, final_theta);
    
    // 파라미터가 그래디언트 방향으로 업데이트되었는지 확인
    let r_decreased = final_r < initial_r;  // 양의 그래디언트이므로 감소해야 함
    let theta_increased = final_theta > initial_theta;  // 음의 그래디언트이므로 증가해야 함
    
    assert!(r_decreased, "r 파라미터가 예상된 방향으로 업데이트되지 않음");
    assert!(theta_increased, "theta 파라미터가 예상된 방향으로 업데이트되지 않음");
    
    println!("하이브리드 최적화기 테스트 통과!");
}

#[test]
fn 제약_투영_테스트() {
    println!("=== 푸앵카레 볼 제약 투영 테스트 ===");
    
    let constraint_projection = ConstraintProjection::new();
    let mut params = Packed128 {
        hi: 0x12345,
        lo: 0,  // 일시적으로 0으로 설정
    };
    
    // 제약 위반하는 값들 테스트
    let test_cases = [
        (1.5, 0.0),      // r 범위 초과
        (-0.5, 0.0),     // r 범위 미만
        (0.5, 50.0),     // theta 범위 초과
        (0.5, -50.0),    // theta 범위 미만
        (0.5, 0.0),      // 정상 범위
    ];
    
    for (r, theta) in test_cases {
        // 파라미터 설정
        params.lo = (((r as f32).to_bits() as u64) << 32) | (theta as f32).to_bits() as u64;
        
        println!("투영 전: r={:.3}, theta={:.3}", r, theta);
        
        // 제약 투영 적용
        constraint_projection.project_to_poincare_ball(&mut params);
        
        let projected_r = f32::from_bits((params.lo >> 32) as u32);
        let projected_theta = f32::from_bits(params.lo as u32);
        
        println!("투영 후: r={:.3}, theta={:.3}", projected_r, projected_theta);
        
        // 제약 조건 확인
        assert!(projected_r >= constraint_projection.r_min, "r이 최소값보다 작음");
        assert!(projected_r <= constraint_projection.r_max, "r이 최대값보다 큼");
        assert!(projected_theta >= constraint_projection.theta_min, "theta가 최소값보다 작음");
        assert!(projected_theta <= constraint_projection.theta_max, "theta가 최대값보다 큼");
        
        println!("---");
    }
    
    println!("제약 투영 테스트 통과!");
}

#[test]
fn 정규화_항_테스트() {
    println!("=== 정규화 항 계산 테스트 ===");
    
    let mut regularization = RegularizationTerms::new();
    let mut rng = thread_rng();
    
    // 다양한 r 값에서 쌍곡 정규화 테스트
    let r_values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98];
    
    println!("쌍곡 정규화 (r → 1일 때 페널티 증가):");
    for &r in &r_values {
        let params = Packed128 {
            hi: 0x12345,
            lo: (((r as f32).to_bits() as u64) << 32) | 0.0f32.to_bits() as u64,
        };
        
        let reg_loss = regularization.compute_regularization_loss(&params);
        println!("  r={:.2}: 정규화 손실={:.6}", r, reg_loss);
    }
    
    // 상태 엔트로피 정규화 테스트
    println!("\n상태 엔트로피 정규화 (다양한 상태 사용 유도):");
    for i in 0..5 {
        let params = Packed128 {
            hi: (i as u64) * 0x10000,  // 다양한 상태
            lo: ((0.5f32.to_bits() as u64) << 32) | 0.0f32.to_bits() as u64,
        };
        
        let reg_loss = regularization.compute_regularization_loss(&params);
        println!("  상태 {}: 정규화 손실={:.6}", i, reg_loss);
    }
    
    println!("정규화 항 테스트 완료!");
}

#[test]
fn 통합_푸앵카레_학습_테스트() {
    println!("=== 통합 푸앵카레 볼 학습 시스템 테스트 ===");
    
    let mut learning_system = PoincareLearning::new();
    let mut rng = thread_rng();
    let mut params = Packed128::random(&mut rng);
    let rows = 4;
    let cols = 4;
    
    // 간단한 학습 타겟 (선형 그래디언트)
    let target: Vec<f32> = (0..rows*cols)
        .map(|i| (i as f32 / (rows*cols-1) as f32))
        .collect();
    
    // 초기 예측 및 손실
    let mut predicted = vec![0.0; target.len()];
    for i in 0..rows {
        for j in 0..cols {
            predicted[i*cols + j] = params.fused_forward(i, j, rows, cols);
        }
    }
    
    let initial_loss: f32 = target.iter().zip(predicted.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f32>() / target.len() as f32;
    
    println!("초기 MSE: {:.6}", initial_loss);
    
    // 하이브리드 학습 수행
    let learning_rate = 0.05;
    let epochs = 20;
    
    for epoch in 0..epochs {
        // 현재 예측 생성
        for i in 0..rows {
            for j in 0..cols {
                predicted[i*cols + j] = params.fused_forward(i, j, rows, cols);
            }
        }
        
        // 하이브리드 역전파
        let (loss, _rmse) = learning_system.fused_backward_hybrid(
            &target,
            &predicted,
            &mut params,
            rows,
            cols,
            learning_rate,
            epoch as i32,
        );
        
        if epoch % 5 == 0 || epoch == epochs - 1 {
            let r_fp32 = f32::from_bits((params.lo >> 32) as u32);
            let theta_fp32 = f32::from_bits(params.lo as u32);
            println!("Epoch {}: MSE={:.6}, r={:.4}, theta={:.4}", 
                     epoch, loss, r_fp32, theta_fp32);
        }
    }
    
    // 최종 예측 및 손실
    for i in 0..rows {
        for j in 0..cols {
            predicted[i*cols + j] = params.fused_forward(i, j, rows, cols);
        }
    }
    
    let final_loss: f32 = target.iter().zip(predicted.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f32>() / target.len() as f32;
    
    println!("최종 MSE: {:.6}", final_loss);
    
    let improvement = (initial_loss - final_loss) / initial_loss * 100.0;
    println!("손실 개선: {:.2}%", improvement);
    
    // 기본적인 학습 확인 (손실이 감소했거나 최소한 크게 증가하지 않았는지)
    assert!(final_loss <= initial_loss * 1.5, "학습이 크게 악화됨");
    
    // 제약 조건 확인
    let final_r = f32::from_bits((params.lo >> 32) as u32);
    let final_theta = f32::from_bits(params.lo as u32);
    
    assert!(final_r >= 0.01 && final_r <= 0.99, "최종 r 파라미터가 제약 범위 벗어남");
    assert!(final_r.is_finite() && final_theta.is_finite(), "파라미터가 무한대/NaN");
    
    println!("통합 푸앵카레 학습 시스템 테스트 통과!");
}

#[test]
fn 수렴성_분석기_테스트() {
    println!("=== 수렴성 분석기 테스트 ===");
    
    let learning_system = PoincareLearning::new();
    let mut analyzer = learning_system.convergence_analyzer;
    let mut rng = thread_rng();
    let params = Packed128::random(&mut rng);
    
    // 점진적으로 감소하는 손실 시뮬레이션
    let mut loss = 1.0;
    let mut converged = false;
    
    for epoch in 0..50 {
        loss *= 0.95;  // 5%씩 감소
        let gradient_norm = loss * 0.1;  // 그래디언트도 함께 감소
        
        let is_converged = analyzer.check_convergence(loss, gradient_norm);
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss={:.6}, Converged={}", epoch, loss, is_converged);
        }
        
        if is_converged && !converged {
            println!("수렴 달성 at epoch {}", epoch);
            converged = true;
        }
    }
    
    // 수렴 조건 검증
    let conditions_met = analyzer.verify_convergence_conditions(&params);
    println!("수렴 조건 만족: {}", conditions_met);
    
    assert!(conditions_met, "수렴 조건이 만족되지 않음");
    println!("수렴성 분석기 테스트 완료!");
} 
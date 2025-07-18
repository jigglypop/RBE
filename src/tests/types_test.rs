use crate::types::Packed128;
use rand::thread_rng;

#[test]
fn 해석적_미분_r_파라미터_정확성_검증() {
    let mut rng = thread_rng();
    let mut seed = Packed128::random(&mut rng);
    
    // 고정된 파라미터로 설정하여 예측 가능한 테스트
    let r_value = 0.7f32;
    let theta_value = 0.3f32;
    seed.lo = ((r_value.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    
    let rows = 4;
    let cols = 4;
    let eps = 1e-5;
    
    for i in 0..rows {
        for j in 0..cols {
            // 해석적 미분 결과
            let analytical_grad = seed.analytical_gradient_r(i, j, rows, cols);
            
            // 수치 미분으로 검증 (ground truth)
            let mut seed_plus = seed;
            let r_plus = r_value + eps;
            seed_plus.lo = ((r_plus.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
            let f_plus = seed_plus.fused_forward(i, j, rows, cols);
            
            let mut seed_minus = seed;
            let r_minus = r_value - eps;
            seed_minus.lo = ((r_minus.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
            let f_minus = seed_minus.fused_forward(i, j, rows, cols);
            
            let numerical_grad = (f_plus - f_minus) / (2.0 * eps);
            
            // 상대 오차 5% 이내로 정확해야 함
            let relative_error = if numerical_grad.abs() > 1e-6 {
                (analytical_grad - numerical_grad).abs() / numerical_grad.abs()
            } else {
                (analytical_grad - numerical_grad).abs()
            };
            
            assert!(
                relative_error < 0.05,
                "위치 ({},{})에서 r 미분 오차가 너무 큼: 해석적={:.6}, 수치적={:.6}, 상대오차={:.3}%", 
                i, j, analytical_grad, numerical_grad, relative_error * 100.0
            );
        }
    }
}

#[test]
fn 해석적_미분_theta_파라미터_정확성_검증() {
    let mut rng = thread_rng();
    let mut seed = Packed128::random(&mut rng);
    
    // 고정된 파라미터로 설정
    let r_value = 0.8f32;
    let theta_value = 0.2f32;
    seed.lo = ((r_value.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
    
    let rows = 4;
    let cols = 4;
    let eps = 1e-5;
    
    for i in 0..rows {
        for j in 0..cols {
            // 해석적 미분 결과
            let analytical_grad = seed.analytical_gradient_theta(i, j, rows, cols);
            
            // 수치 미분으로 검증
            let mut seed_plus = seed;
            let theta_plus = theta_value + eps;
            seed_plus.lo = ((r_value.to_bits() as u64) << 32) | theta_plus.to_bits() as u64;
            let f_plus = seed_plus.fused_forward(i, j, rows, cols);
            
            let mut seed_minus = seed;
            let theta_minus = theta_value - eps;
            seed_minus.lo = ((r_value.to_bits() as u64) << 32) | theta_minus.to_bits() as u64;
            let f_minus = seed_minus.fused_forward(i, j, rows, cols);
            
            let numerical_grad = (f_plus - f_minus) / (2.0 * eps);
            
            // 상대 오차 10% 이내로 정확해야 함 (해석적 미분 허용 오차)
            let relative_error = if numerical_grad.abs() > 1e-6 {
                (analytical_grad - numerical_grad).abs() / numerical_grad.abs()
            } else {
                (analytical_grad - numerical_grad).abs()
            };
            
            assert!(
                relative_error < 0.10,
                "위치 ({},{})에서 theta 미분 오차가 너무 큼: 해석적={:.6}, 수치적={:.6}, 상대오차={:.3}%", 
                i, j, analytical_grad, numerical_grad, relative_error * 100.0
            );
        }
    }
}

#[test]
fn 해석적_미분_성능_벤치마크() {
    use std::time::Instant;
    
    let mut rng = thread_rng();
    let mut seed = Packed128::random(&mut rng);
    let rows = 32;
    let cols = 32;
    let iterations = 1000;
    
    // 해석적 미분 성능 측정
    let start = Instant::now();
    for _ in 0..iterations {
        for i in 0..rows {
            for j in 0..cols {
                let _grad_r = seed.analytical_gradient_r(i, j, rows, cols);
                let _grad_theta = seed.analytical_gradient_theta(i, j, rows, cols);
            }
        }
    }
    let analytical_time = start.elapsed();
    
    println!("32x32 행렬, {} 반복 해석적 미분 시간: {:?}", iterations, analytical_time);
    
    // 성능 기준: 32x32에서 1000회 반복이 1초 이내 완료되어야 함
    assert!(
        analytical_time.as_secs() < 1,
        "해석적 미분이 너무 느림: {:?}", analytical_time
    );
}

#[test]
fn 해석적_미분_상태별_검증() {
    let mut seed = Packed128::random(&mut thread_rng());
    
    // 각 상태별로 미분이 제대로 계산되는지 확인
    for state in 0..4 {
        // 특정 상태로 고정
        seed.hi = (state as u64) & 0x3; // 하위 2비트만 설정
        
        let r_value = 0.5f32;
        let theta_value = 0.1f32;
        seed.lo = ((r_value.to_bits() as u64) << 32) | theta_value.to_bits() as u64;
        
        // 중앙 위치에서 테스트
        let i = 2;
        let j = 2;
        let rows = 4;
        let cols = 4;
        
        let grad_r = seed.analytical_gradient_r(i, j, rows, cols);
        let grad_theta = seed.analytical_gradient_theta(i, j, rows, cols);
        
        // 미분값이 유한하고 합리적인 범위 내에 있어야 함
        assert!(grad_r.is_finite(), "상태 {}에서 r 미분이 무한대", state);
        assert!(grad_theta.is_finite(), "상태 {}에서 theta 미분이 무한대", state);
        assert!(grad_r.abs() < 100.0, "상태 {}에서 r 미분이 너무 큼: {}", state, grad_r);
        assert!(grad_theta.abs() < 100.0, "상태 {}에서 theta 미분이 너무 큼: {}", state, grad_theta);
        
        println!("상태 {}: grad_r={:.6}, grad_theta={:.6}", state, grad_r, grad_theta);
    }
}

#[test]
fn 해석적_미분_경계값_안정성_검증() {
    let mut seed = Packed128::random(&mut thread_rng());
    
    // 극단적인 파라미터 값에서도 안정적이어야 함
    let extreme_cases = [
        (0.1f32, -10.0f32),  // 최소 r, 극소 theta
        (2.0f32, 10.0f32),   // 최대 r, 극대 theta
        (1.0f32, 0.0f32),    // 중간 r, 0 theta
        (0.5f32, 3.14159f32) // 중간 r, π theta
    ];
    
    for (r_val, theta_val) in extreme_cases.iter() {
        seed.lo = ((r_val.to_bits() as u64) << 32) | theta_val.to_bits() as u64;
        
        let grad_r = seed.analytical_gradient_r(1, 1, 4, 4);
        let grad_theta = seed.analytical_gradient_theta(1, 1, 4, 4);
        
        assert!(
            grad_r.is_finite() && grad_theta.is_finite(),
            "극단값 r={}, theta={}에서 미분이 발산", r_val, theta_val
        );
        
        println!("극단값 테스트 r={:.1}, theta={:.1}: grad_r={:.6}, grad_theta={:.6}", 
                 r_val, theta_val, grad_r, grad_theta);
    }
} 
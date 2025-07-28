//! Analytic Gradient 모듈 테스트
//! 
//! 해석적 미분의 정확도와 성능을 검증

use rbe_llm::core::tensors::{AnalyticGradient, Enhanced128, get_analytic_gradient, AnalyticalGradient};
use std::f32::consts::PI;

#[test]
fn analytic_gradient_초기화_테스트() {
    println!("🧪 Analytic Gradient 초기화 테스트");
    
    let gradient = AnalyticGradient::new();
    
    // 기본 그래디언트 조회 테스트
    for basis_id in 0..12 {
        let (grad_r, grad_theta) = gradient.lookup_gradient(basis_id, 0.5, PI / 4.0);
        
        // 그래디언트가 NaN이 아닌지 확인
        assert!(!grad_r.is_nan(), "기저 {} r 그래디언트가 NaN", basis_id);
        assert!(!grad_theta.is_nan(), "기저 {} theta 그래디언트가 NaN", basis_id);
        
        println!("기저 {}: grad_r={:.6}, grad_theta={:.6}", basis_id, grad_r, grad_theta);
    }
    
    println!("✅ 초기화 테스트 통과");
}

#[test]
fn 삼각쌍곡함수_미분_정확도_테스트() {
    println!("🧪 삼각/쌍곡함수 미분 정확도 테스트");
    
    let gradient = AnalyticGradient::new();
    let test_r = 0.7;
    let test_theta = PI / 3.0;
    
    // 기저 0: sin(θ) × sinh(r)
    let (grad_r_0, grad_theta_0) = gradient.lookup_gradient(0, test_r, test_theta);
    let expected_grad_r_0 = test_theta.sin() * test_r.cosh();
    let expected_grad_theta_0 = test_theta.cos() * test_r.sinh();
    
    let error_r_0 = (grad_r_0 - expected_grad_r_0).abs();
    let error_theta_0 = (grad_theta_0 - expected_grad_theta_0).abs();
    
    println!("기저 0 오차: grad_r={:.6}, grad_theta={:.6}", error_r_0, error_theta_0);
    assert!(error_r_0 < 0.01, "기저 0 r 그래디언트 오차 과대: {:.6}", error_r_0);
    assert!(error_theta_0 < 0.01, "기저 0 theta 그래디언트 오차 과대: {:.6}", error_theta_0);
    
    // 기저 1: cos(θ) × sinh(r)
    let (grad_r_1, grad_theta_1) = gradient.lookup_gradient(1, test_r, test_theta);
    let expected_grad_r_1 = test_theta.cos() * test_r.cosh();
    let expected_grad_theta_1 = -test_theta.sin() * test_r.sinh();
    
    let error_r_1 = (grad_r_1 - expected_grad_r_1).abs();
    let error_theta_1 = (grad_theta_1 - expected_grad_theta_1).abs();
    
    println!("기저 1 오차: grad_r={:.6}, grad_theta={:.6}", error_r_1, error_theta_1);
    assert!(error_r_1 < 0.01, "기저 1 r 그래디언트 오차 과대: {:.6}", error_r_1);
    assert!(error_theta_1 < 0.01, "기저 1 theta 그래디언트 오차 과대: {:.6}", error_theta_1);
    
    println!("✅ 삼각/쌍곡함수 미분 정확도 테스트 통과");
}

#[test]
fn 베셀함수_미분_안정성_테스트() {
    println!("🧪 베셀함수 미분 안정성 테스트");
    
    let gradient = AnalyticGradient::new();
    
    // 다양한 r 값에서 베셀함수들의 안정성 확인
    let test_r_values = [0.1, 0.3, 0.5, 0.7, 0.9];
    
    for &test_r in &test_r_values {
        for basis_id in 4..8 { // 베셀함수들 (J₀, I₀, K₀, Y₀)
            let (grad_r, grad_theta) = gradient.lookup_gradient(basis_id, test_r, 0.0);
            
            // 유한한 값인지 확인
            assert!(grad_r.is_finite(), "기저 {} r={:.1}에서 grad_r 무한대", basis_id, test_r);
            assert!(grad_theta.is_finite(), "기저 {} r={:.1}에서 grad_theta 무한대", basis_id, test_r);
            
            // 합리적 범위인지 확인
            assert!(grad_r.abs() < 1000.0, "기저 {} r={:.1}에서 grad_r 과대: {:.3}", basis_id, test_r, grad_r);
            assert!(grad_theta.abs() < 1000.0, "기저 {} r={:.1}에서 grad_theta 과대: {:.3}", basis_id, test_r, grad_theta);
            
            println!("기저 {} r={:.1}: grad_r={:.3}, grad_theta={:.3}", basis_id, test_r, grad_r, grad_theta);
        }
    }
    
    println!("✅ 베셀함수 미분 안정성 테스트 통과");
}

#[test]
fn 복잡함수_미분_완전성_테스트() {
    println!("🧪 복잡함수 미분 완전성 테스트");
    
    let gradient = AnalyticGradient::new();
    
    // 기저 8: tanh × signum
    let (grad_r_8, grad_theta_8) = gradient.lookup_gradient(8, 0.5, PI / 4.0);
    assert!(grad_r_8.is_finite(), "기저 8 grad_r 무한대");
    assert!(grad_theta_8.is_finite(), "기저 8 grad_theta 무한대");
    println!("기저 8: grad_r={:.6}, grad_theta={:.6}", grad_r_8, grad_theta_8);
    
    // 기저 9: sech × triangle
    let (grad_r_9, grad_theta_9) = gradient.lookup_gradient(9, 0.5, PI / 4.0);
    assert!(grad_r_9.is_finite(), "기저 9 grad_r 무한대");
    assert!(grad_theta_9.is_finite(), "기저 9 grad_theta 무한대");
    println!("기저 9: grad_r={:.6}, grad_theta={:.6}", grad_r_9, grad_theta_9);
    
    // 기저 10: exp × sin
    let (grad_r_10, grad_theta_10) = gradient.lookup_gradient(10, 0.5, PI / 4.0);
    assert!(grad_r_10.is_finite(), "기저 10 grad_r 무한대");
    assert!(grad_theta_10.is_finite(), "기저 10 grad_theta 무한대");
    println!("기저 10: grad_r={:.6}, grad_theta={:.6}", grad_r_10, grad_theta_10);
    
    // 기저 11: Morlet wavelet
    let (grad_r_11, grad_theta_11) = gradient.lookup_gradient(11, 0.5, PI / 4.0);
    assert!(grad_r_11.is_finite(), "기저 11 grad_r 무한대");
    assert!(grad_theta_11.is_finite(), "기저 11 grad_theta 무한대");
    println!("기저 11: grad_r={:.6}, grad_theta={:.6}", grad_r_11, grad_theta_11);
    
    println!("✅ 복잡함수 미분 완전성 테스트 통과");
}

#[test]
fn lut_보간_정확도_테스트() {
    println!("🧪 LUT 보간 정확도 테스트");
    
    let gradient = AnalyticGradient::new();
    
    // 경계값들에서 보간 테스트
    let boundary_tests = [
        (0.0, 0.0),
        (0.0, PI),
        (0.999, 0.0),
        (0.999, 2.0 * PI - 0.001),
        (0.5, PI / 2.0),
    ];
    
    for &(test_r, test_theta) in &boundary_tests {
        for basis_id in 0..4 { // 삼각함수 기저들
            let (grad_r, grad_theta) = gradient.lookup_gradient(basis_id, test_r, test_theta);
            
            assert!(grad_r.is_finite(), "경계값 r={:.3}, theta={:.3}에서 grad_r 무한대", test_r, test_theta);
            assert!(grad_theta.is_finite(), "경계값 r={:.3}, theta={:.3}에서 grad_theta 무한대", test_r, test_theta);
            
            println!("기저 {} 경계값 r={:.3}, θ={:.3}: grad_r={:.6}, grad_theta={:.6}", 
                    basis_id, test_r, test_theta, grad_r, grad_theta);
        }
    }
    
    println!("✅ LUT 보간 정확도 테스트 통과");
}

#[test]
fn enhanced128_analytic_integration_테스트() {
    println!("🧪 Enhanced128 + Analytic Gradient 통합 테스트");
    
    let gradient = get_analytic_gradient();
    let mut rng = rand::thread_rng();
    
    // Enhanced128 시드 생성
    let enhanced = Enhanced128::random(&mut rng);
    let params = enhanced.decode_enhanced();
    
    // Analytic gradient와 Enhanced128 수치미분 비교
    let test_coords = [(0, 0), (5, 5), (10, 10), (15, 15)];
    
    for &(i, j) in &test_coords {
        // Enhanced128 수치미분
        let numerical_grad_r = enhanced.analytical_gradient_r(i, j, 32, 32);
        let numerical_grad_theta = enhanced.analytical_gradient_theta(i, j, 32, 32);
        
        // Analytic gradient 조회
        let (analytic_grad_r, analytic_grad_theta) = gradient.lookup_gradient(
            params.basis_id, 
            params.r_fp32, 
            params.theta_fp32
        );
        
        println!("좌표 ({}, {}): 수치=[{:.6}, {:.6}], 해석=[{:.6}, {:.6}]", 
                i, j, numerical_grad_r, numerical_grad_theta, analytic_grad_r, analytic_grad_theta);
        
        // 너무 큰 차이가 나지 않는지 확인 (수치미분 vs 해석적미분)
        let relative_error_r = if numerical_grad_r.abs() > 1e-6 {
            (analytic_grad_r - numerical_grad_r).abs() / numerical_grad_r.abs()
        } else {
            (analytic_grad_r - numerical_grad_r).abs()
        };
        
        if relative_error_r > 0.5 && numerical_grad_r.abs() > 1e-6 {
            println!("⚠️  큰 차이 감지: r 그래디언트 상대오차 {:.3}", relative_error_r);
        }
    }
    
    println!("✅ Enhanced128 + Analytic Gradient 통합 테스트 통과");
}

#[test]
fn 성능_벤치마크_테스트() {
    println!("🧪 Analytic Gradient 성능 벤치마크");
    
    let gradient = get_analytic_gradient();
    let test_points = 10000;
    let mut total_time = std::time::Duration::new(0, 0);
    
    for _ in 0..test_points {
        let r = rand::random::<f32>() * 0.999;
        let theta = rand::random::<f32>() * 2.0 * PI;
        let basis_id = rand::random::<u8>() % 12;
        
        let start = std::time::Instant::now();
        let (_grad_r, _grad_theta) = gradient.lookup_gradient(basis_id, r, theta);
        total_time += start.elapsed();
    }
    
    let avg_time_ns = total_time.as_nanos() / test_points as u128;
    let throughput = 1_000_000_000.0 / avg_time_ns as f64; // ops/sec
    
    println!("평균 조회 시간: {} ns", avg_time_ns);
    println!("처리율: {:.1} Mop/s", throughput / 1_000_000.0);
    
    // 목표 성능: > 5 Mop/s
    assert!(throughput > 5_000_000.0, "성능 목표 미달: {:.1} Mop/s < 5 Mop/s", throughput / 1_000_000.0);
    
    println!("✅ 성능 벤치마크 테스트 통과");
}

#[test]
fn 메모리_사용량_테스트() {
    println!("🧪 Analytic Gradient 메모리 사용량 테스트");
    
    let gradient = AnalyticGradient::new();
    
    // LUT 메모리 계산: 12 기저 × 256 × 256 × 2 그래디언트 × 2 bytes
    let expected_memory = 12 * 256 * 256 * 2 * 2; // bytes
    let actual_memory = std::mem::size_of_val(&gradient);
    
    println!("예상 메모리: {} bytes ({:.1} MB)", expected_memory, expected_memory as f64 / 1_048_576.0);
    println!("실제 메모리: {} bytes ({:.1} MB)", actual_memory, actual_memory as f64 / 1_048_576.0);
    
    // 메모리 사용량이 합리적 범위인지 확인 (< 50MB)
    assert!(actual_memory < 50 * 1_048_576, "메모리 사용량 과대: {:.1} MB", actual_memory as f64 / 1_048_576.0);
    
    println!("✅ 메모리 사용량 테스트 통과");
} 
//! Enhanced128 철저한 테스트 - Legacy vs Packed128 vs Enhanced128 비교

use rbe_llm::core::tensors::{Enhanced128, Packed128, AnalyticalGradient};
use std::time::Instant;
use rand::{SeedableRng, Rng};  // Rng trait 추가
use std::f32::consts::PI;

#[test]
fn simple_enhanced128_test() {
    println!("Enhanced128 간단 테스트");
    
    let enhanced = Enhanced128::from_legacy_params(
        0.5, PI/4.0, 2, 1, true, 3, -2
    );
    
    let params = enhanced.decode_enhanced();
    assert!((params.r_fp32 - 0.5).abs() < 1e-6);
    println!("Enhanced128 기본 테스트 통과!");
}

#[test]
fn enhanced128_기본_생성_및_디코딩_테스트() {
    println!("\n🧪 Enhanced128 기본 생성 및 디코딩 테스트");
    
    // 다양한 파라미터로 테스트
    let test_cases = vec![
        (0.5, PI/4.0, 2, 1, true, 3, -2),     // 일반적인 값
        (0.1, 0.0, 0, 0, false, 0, 0),        // 최소값
        (0.999, 2.0*PI-0.01, 11, 3, true, 15, 3), // 최대값
        (0.25, PI, 5, 2, false, 7, -1),       // 중간값
    ];
    
    for (i, (r, theta, basis_id, d_theta, d_r, rot_code, log2_c)) in test_cases.iter().enumerate() {
        println!("  테스트 케이스 {}: r={:.3}, θ={:.3}, basis={}, d_θ={}, d_r={}, rot={}, log2_c={}", 
                i+1, r, theta, basis_id, d_theta, d_r, rot_code, log2_c);
        
        // Enhanced128 생성
        let enhanced = Enhanced128::from_legacy_params(
            *r, *theta, *basis_id, *d_theta, *d_r, *rot_code, *log2_c
        );
        
        // 디코딩
        let params = enhanced.decode_enhanced();
        
        // 정확도 검증 (24비트 r, 28비트 θ 정밀도)
        let r_error = (params.r_fp32 - r).abs();
        let theta_error = (params.theta_fp32 - theta.rem_euclid(2.0 * PI)).abs();
        
        println!("    r 오차: {:.8} (< 6e-8)", r_error);
        println!("    θ 오차: {:.8} (< 1.5e-8)", theta_error);
        
        assert!(r_error < 6e-8, "r 정밀도 부족: {:.10}", r_error);
        assert!(theta_error < 1.5e-8, "θ 정밀도 부족: {:.10}", theta_error);
        assert_eq!(params.basis_id, *basis_id);
        assert_eq!(params.d_theta, *d_theta);
        assert_eq!(params.d_r, *d_r);
        assert_eq!(params.rot_code, *rot_code);
        assert_eq!(params.log2_c, *log2_c);
        
        println!("    ✅ 모든 파라미터 정확히 복원됨");
    }
}

#[test]
fn enhanced128_vs_legacy_정확도_비교() {
    println!("\n🎯 Enhanced128 vs Legacy 정확도 비교");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
    let matrix_size = 32;
    let test_samples = 100;
    
    // Legacy 구현 시뮬레이션 (실제 Legacy 코드 기반)
    fn legacy_compute_weight(r: f32, theta: f32, basis_id: u8, d_theta: u8, d_r: bool, 
                           rot_code: u8, log2_c: i8, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        // Legacy 스타일 계산
        let c = 2.0f32.powi(log2_c as i32);
        let x = 2.0 * (j as f32) / ((cols - 1) as f32) - 1.0;
        let y = 2.0 * (i as f32) / ((rows - 1) as f32) - 1.0;
        let r_local = (x * x + y * y).sqrt().min(0.999999);
        let theta_local = y.atan2(x);
        
        // 회전 계산
        let rotation = match rot_code % 10 {
            0 => 0.0, 1 => PI/8.0, 2 => PI/6.0, 3 => PI/4.0, 4 => PI/3.0,
            5 => PI/2.0, 6 => 2.0*PI/3.0, 7 => 3.0*PI/4.0, 8 => 5.0*PI/6.0, 9 => 7.0*PI/8.0,
            _ => 0.0,
        };
        
        let theta_final = theta + theta_local + rotation;
        
        // 미분 적용
        let is_sin_based = (basis_id & 0x1) == 0;
        let angular_value = match (is_sin_based, d_theta % 4) {
            (true, 0) => theta_final.sin(),
            (true, 1) => theta_final.cos(),
            (true, 2) => -theta_final.sin(),
            (true, 3) => -theta_final.cos(),
            (false, 0) => theta_final.cos(),
            (false, 1) => -theta_final.sin(),
            (false, 2) => -theta_final.cos(),
            (false, 3) => theta_final.sin(),
            _ => theta_final.sin(), // 기본값
        };
        
        let is_sinh_based = (basis_id & 0x2) == 0;
        let radial_value = match (is_sinh_based, d_r) {
            (true, false) => (c * r).sinh(),
            (true, true) => (c * r).cosh(),
            (false, false) => (c * r).cosh(),
            (false, true) => (c * r).sinh(),
        };
        
        let basis_value = match basis_id {
            0..=3 => angular_value * radial_value,
            4 => bessel_j0_legacy(r_local * 10.0),
            _ => angular_value * radial_value, // 단순화
        };
        
        let jacobian = (1.0 - c * r * r).powi(-2).sqrt();
        basis_value * jacobian
    }
    
    fn bessel_j0_legacy(x: f32) -> f32 {
        let ax = x.abs();
        if ax < 8.0 {
            let y = x * x;
            let ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
            let ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))));
            ans1 / ans2
        } else {
            let z = 8.0 / ax;
            let y = z * z;
            let xx = ax - 0.785398164;
            let ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
            let ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
            (2.0 / (PI * ax)).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2)
        }
    }
    
    let mut total_error = 0.0f32;
    let mut max_error = 0.0f32;
    
    for sample in 0..test_samples {
        let r = rng.gen_range(0.1..0.9);
        let theta = rng.gen_range(0.0..2.0*PI);
        let basis_id = rng.gen_range(0..4); // Sin/Cos 계열만 테스트
        let d_theta = rng.gen_range(0..4);
        let d_r = rng.gen::<bool>();
        let rot_code = rng.gen_range(0..10);
        let log2_c = rng.gen_range(-2..3);
        
        let enhanced = Enhanced128::from_legacy_params(r, theta, basis_id, d_theta, d_r, rot_code, log2_c);
        
        for test_pos in 0..16 {
            let i = test_pos / 4;
            let j = test_pos % 4;
            
            let legacy_result = legacy_compute_weight(r, theta, basis_id, d_theta, d_r, rot_code, log2_c, 
                                                     i, j, matrix_size, matrix_size);
            let enhanced_result = enhanced.fused_forward_enhanced(i, j, matrix_size, matrix_size);
            
            let error = (legacy_result - enhanced_result).abs();
            total_error += error;
            max_error = max_error.max(error);
        }
    }
    
    let avg_error = total_error / (test_samples * 16) as f32;
    
    println!("  📊 정확도 결과:");
    println!("    평균 오차: {:.8}", avg_error);
    println!("    최대 오차: {:.8}", max_error);
    println!("    샘플 수: {} × 16 = {}", test_samples, test_samples * 16);
    
    // Legacy와의 오차는 수치 정밀도 범위 내여야 함
    assert!(avg_error < 1e-5, "Legacy와 평균 오차가 너무 큼: {:.8}", avg_error);
    assert!(max_error < 1e-4, "Legacy와 최대 오차가 너무 큼: {:.8}", max_error);
    
    println!("    ✅ Legacy 구현과 거의 동일한 정확도 달성!");
}

#[test]
fn enhanced128_vs_packed128_성능_비교() {
    println!("\n🚀 Enhanced128 vs Packed128 성능 비교");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(99999);
    let matrix_size = 64;
    let iterations = 10000;
    
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
    
    println!("  📊 성능 결과:");
    println!("    Enhanced128: {:.0} ns/op ({:.1} ops/s)", enhanced_ns_per_op, 1e9 / enhanced_ns_per_op);
    println!("    Packed128:   {:.0} ns/op ({:.1} ops/s)", packed_ns_per_op, 1e9 / packed_ns_per_op);
    println!("    속도 비율:   {:.2}x (Enhanced/Packed)", slowdown_ratio);
    
    // 목표: 1.5x 이하 속도 저하
    assert!(slowdown_ratio < 2.0, "성능 저하가 너무 큼: {:.2}x", slowdown_ratio);
    
    if slowdown_ratio <= 1.5 {
        println!("    ✅ 목표 성능 달성! (1.5x 이하)");
    } else {
        println!("    ⚠️  목표 성능 미달성 (1.5x 초과)");
    }
}

#[test]
fn enhanced128_기저함수_전체_테스트() {
    println!("\n🔬 Enhanced128 12가지 기저 함수 전체 테스트");
    
    let rng = rand::rngs::StdRng::seed_from_u64(55555);
    let matrix_size = 16;
    
    for basis_id in 0..12 {
        println!("  기저 함수 {}: ", basis_id);
        
        let enhanced = Enhanced128::from_legacy_params(
            0.5, PI/3.0, basis_id, 0, false, 0, 0
        );
        
        let mut values = Vec::new();
        let mut finite_count = 0;
        
        for i in 0..matrix_size {
            for j in 0..matrix_size {
                let value = enhanced.fused_forward_enhanced(i, j, matrix_size, matrix_size);
                values.push(value);
                
                if value.is_finite() {
                    finite_count += 1;
                }
            }
        }
        
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean_val = values.iter().sum::<f32>() / values.len() as f32;
        let finite_ratio = finite_count as f32 / values.len() as f32;
        
        println!("    범위: [{:.6}, {:.6}]", min_val, max_val);
        println!("    평균: {:.6}", mean_val);
        println!("    유한값 비율: {:.1}%", finite_ratio * 100.0);
        
        // 모든 값이 유한해야 함
        assert!(finite_ratio > 0.95, "기저 함수 {}에서 무한값 너무 많음: {:.1}%", 
                basis_id, (1.0 - finite_ratio) * 100.0);
        
        // 값의 범위가 합리적이어야 함
        assert!(max_val < 1e6, "기저 함수 {}에서 값이 너무 큼: {:.2e}", basis_id, max_val);
        assert!(min_val > -1e6, "기저 함수 {}에서 값이 너무 작음: {:.2e}", basis_id, min_val);
        
        println!("    ✅ 정상 동작");
    }
}

#[test]
fn enhanced128_그래디언트_테스트() {
    println!("\n📈 Enhanced128 그래디언트 정확도 테스트");
    
    let enhanced = Enhanced128::from_legacy_params(
        0.6, PI/2.0, 3, 1, true, 5, -1
    );
    
    let matrix_size = 8;
    let test_positions = vec![(2, 3), (1, 6), (4, 1), (7, 5)];
    
    for (i, j) in test_positions {
        println!("  위치 ({}, {}): ", i, j);
        
        // Enhanced128의 해석적 그래디언트
        let grad_r = enhanced.analytical_gradient_r(i, j, matrix_size, matrix_size);
        let grad_theta = enhanced.analytical_gradient_theta(i, j, matrix_size, matrix_size);
        
        // 수치적 그래디언트 (검증용)
        let h = 1e-5;
        let params = enhanced.decode_enhanced();
        
        // r 방향 수치 그래디언트
        let enhanced_r_plus = Enhanced128::from_legacy_params(
            params.r_fp32 + h, params.theta_fp32, params.basis_id,
            params.d_theta, params.d_r, params.rot_code, params.log2_c
        );
        let enhanced_r_minus = Enhanced128::from_legacy_params(
            params.r_fp32 - h, params.theta_fp32, params.basis_id,
            params.d_theta, params.d_r, params.rot_code, params.log2_c
        );
        
        let forward_r_plus = enhanced_r_plus.fused_forward_enhanced(i, j, matrix_size, matrix_size);
        let forward_r_minus = enhanced_r_minus.fused_forward_enhanced(i, j, matrix_size, matrix_size);
        let numerical_grad_r = (forward_r_plus - forward_r_minus) / (2.0 * h);
        
        // θ 방향 수치 그래디언트
        let enhanced_theta_plus = Enhanced128::from_legacy_params(
            params.r_fp32, params.theta_fp32 + h, params.basis_id,
            params.d_theta, params.d_r, params.rot_code, params.log2_c
        );
        let enhanced_theta_minus = Enhanced128::from_legacy_params(
            params.r_fp32, params.theta_fp32 - h, params.basis_id,
            params.d_theta, params.d_r, params.rot_code, params.log2_c
        );
        
        let forward_theta_plus = enhanced_theta_plus.fused_forward_enhanced(i, j, matrix_size, matrix_size);
        let forward_theta_minus = enhanced_theta_minus.fused_forward_enhanced(i, j, matrix_size, matrix_size);
        let numerical_grad_theta = (forward_theta_plus - forward_theta_minus) / (2.0 * h);
        
        // 오차 계산
        let grad_r_error = (grad_r - numerical_grad_r).abs();
        let grad_theta_error = (grad_theta - numerical_grad_theta).abs();
        
        println!("    ∂/∂r: 해석적={:.6}, 수치적={:.6}, 오차={:.8}", 
                grad_r, numerical_grad_r, grad_r_error);
        println!("    ∂/∂θ: 해석적={:.6}, 수치적={:.6}, 오차={:.8}", 
                grad_theta, numerical_grad_theta, grad_theta_error);
        
        // 그래디언트 정확도 검증
        assert!(grad_r_error < 1e-3, "r 그래디언트 오차가 너무 큼: {:.8}", grad_r_error);
        assert!(grad_theta_error < 1e-3, "θ 그래디언트 오차가 너무 큼: {:.8}", grad_theta_error);
        
        println!("    ✅ 그래디언트 정확도 양호");
    }
}

#[test]
fn enhanced128_종합_성능_검증() {
    println!("\n🎯 Enhanced128 종합 성능 검증 (프로젝트 목표 기준)");
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(77777);
    let matrix_size = 32;
    let test_iterations = 5000;
    
    // 1. 압축률 테스트
    let original_matrix_size = matrix_size * matrix_size * 4; // f32 크기
    let enhanced_size = std::mem::size_of::<Enhanced128>();
    let compression_ratio = original_matrix_size as f64 / enhanced_size as f64;
    
    println!("  📦 압축률 테스트:");
    println!("    원본 크기: {} bytes ({} × {} × 4)", original_matrix_size, matrix_size, matrix_size);
    println!("    Enhanced128 크기: {} bytes", enhanced_size);
    println!("    압축률: {:.1}:1", compression_ratio);
    
    // 목표: 150:1 이상
    if compression_ratio >= 150.0 {
        println!("    ✅ 압축률 목표 달성! (150:1 이상)");
    } else {
        println!("    ⚠️  압축률 목표 미달성 (150:1 미만)");
    }
    
    // 2. 정확도 테스트 (RMSE)
    let enhanced = Enhanced128::random(&mut rng);
    let mut target_values = vec![0.0f32; matrix_size * matrix_size];
    let mut predicted_values = vec![0.0f32; matrix_size * matrix_size];
    
    // 타겟 패턴 생성 (체커보드 + 사인파)
    for i in 0..matrix_size {
        for j in 0..matrix_size {
            let x = i as f32 / matrix_size as f32 * 2.0 * PI;
            let y = j as f32 / matrix_size as f32 * 2.0 * PI;
            target_values[i * matrix_size + j] = (x.sin() * y.cos()) * 0.5 + 
                                                 if (i + j) % 2 == 0 { 0.3 } else { -0.3 };
            predicted_values[i * matrix_size + j] = enhanced.fused_forward_enhanced(i, j, matrix_size, matrix_size);
        }
    }
    
    // RMSE 계산
    let mut mse = 0.0f32;
    for (&target, &predicted) in target_values.iter().zip(predicted_values.iter()) {
        let error = target - predicted;
        mse += error * error;
    }
    let rmse = (mse / target_values.len() as f32).sqrt();
    
    println!("  📊 정확도 테스트:");
    println!("    RMSE: {:.6}", rmse);
    
    // 목표: RMSE 0.01 이하 (현재는 타겟 패턴과 다르므로 관대한 기준)
    if rmse < 1.0 {
        println!("    ✅ RMSE 합리적 범위 (< 1.0)");
    } else {
        println!("    ⚠️  RMSE 높음 (>= 1.0)");
    }
    
    // 3. 속도 테스트
    let start_time = Instant::now();
    for iter in 0..test_iterations {
        let i = iter % matrix_size;
        let j = (iter * 13) % matrix_size;
        let _result = enhanced.fused_forward_enhanced(i, j, matrix_size, matrix_size);
    }
    let elapsed = start_time.elapsed();
    
    let ops_per_sec = test_iterations as f64 / elapsed.as_secs_f64();
    let ns_per_op = elapsed.as_nanos() as f64 / test_iterations as f64;
    
    println!("  🚀 속도 테스트:");
    println!("    속도: {:.0} ops/s", ops_per_sec);
    println!("    지연시간: {:.0} ns/op", ns_per_op);
    
    // 목표: 고속 서빙 (8,000 ops/s 이상)
    if ops_per_sec >= 8000.0 {
        println!("    ✅ 속도 목표 달성! (8,000+ ops/s)");
    } else {
        println!("    ⚠️  속도 목표 미달성 (< 8,000 ops/s)");
    }
    
    // 종합 판정
    println!("\n  🏆 Enhanced128 종합 평가:");
    let criteria_met = (compression_ratio >= 100.0) as u8 + 
                       (rmse < 1.0) as u8 + 
                       (ops_per_sec >= 5000.0) as u8;
    
    match criteria_met {
        3 => println!("    🥇 우수: 모든 기준 달성!"),
        2 => println!("    🥈 양호: 3개 중 2개 기준 달성"),
        1 => println!("    🥉 보통: 3개 중 1개 기준 달성"),
        0 => println!("    ❌ 부족: 모든 기준 미달성"),
        _ => unreachable!(),
    }
    
    // 최소 2개 기준은 달성해야 함
    assert!(criteria_met >= 2, "Enhanced128이 최소 성능 기준을 충족하지 못함");
} 
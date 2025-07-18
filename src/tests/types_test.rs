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
    let seed = Packed128::random(&mut rng);
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

// ================================
// 1장: 푸앵카레 볼 기반 데이터 구조 테스트
// ================================

use crate::types::{PoincarePackedBit128, PoincareQuadrant};

#[test]
fn 푸앵카레_볼_128비트_인코딩_생성_테스트() {
    let poincare = PoincarePackedBit128::new(
        PoincareQuadrant::First,
        1000, // frequency
        2000, // amplitude
        42,   // basis function (6비트)
        0x12345678, // cordic sequence
        0.5,  // r_poincare
        1.0,  // theta_poincare
    );
    
    // 인코딩된 값들이 올바르게 저장되었는지 확인
    assert_eq!(poincare.get_quadrant(), PoincareQuadrant::First);
    assert_eq!(poincare.get_hyperbolic_frequency(), 1000);
    assert_eq!(poincare.get_geodesic_amplitude(), 2000);
    assert_eq!(poincare.get_basis_function_selector(), 42);
    assert_eq!(poincare.get_cordic_rotation_sequence(), 0x12345678);
    assert!((poincare.get_r_poincare() - 0.5).abs() < 1e-6);
    assert!((poincare.get_theta_poincare() - 1.0).abs() < 1e-6);
}

#[test]
fn 푸앵카레_사분면_인코딩_디코딩_테스트() {
    let quadrants = [
        PoincareQuadrant::First,
        PoincareQuadrant::Second,
        PoincareQuadrant::Third,
        PoincareQuadrant::Fourth,
    ];
    
    for original_quadrant in quadrants {
        let poincare = PoincarePackedBit128::new(
            original_quadrant,
            0, 0, 0, 0, 0.1, 0.1
        );
        
        let decoded_quadrant = poincare.get_quadrant();
        assert_eq!(original_quadrant, decoded_quadrant, 
                   "사분면 인코딩/디코딩 실패: {:?}", original_quadrant);
    }
}

#[test]
fn 쌍곡_주파수_12비트_인코딩_테스트() {
    let test_frequencies = [0, 1, 100, 2047, 4095]; // 12비트 최대값
    
    for freq in test_frequencies {
        let poincare = PoincarePackedBit128::new(
            PoincareQuadrant::First,
            freq, 0, 0, 0, 0.1, 0.1
        );
        
        assert_eq!(poincare.get_hyperbolic_frequency(), freq,
                   "쌍곡 주파수 인코딩 실패: {}", freq);
        
        // 실제 주파수 변환 테스트
        let real_freq = poincare.get_real_frequency(100.0);
        let expected = (freq as f32 / 4095.0) * 100.0;
        assert!((real_freq - expected).abs() < 1e-5,
                "실제 주파수 변환 실패: {} -> {}, 예상: {}", freq, real_freq, expected);
    }
}

#[test]
fn 측지선_진폭_12비트_인코딩_테스트() {
    let test_amplitudes = [0, 1, 500, 2048, 4095]; // 12비트 최대값
    
    for amp in test_amplitudes {
        let poincare = PoincarePackedBit128::new(
            PoincareQuadrant::First,
            0, amp, 0, 0, 0.1, 0.1
        );
        
        assert_eq!(poincare.get_geodesic_amplitude(), amp,
                   "측지선 진폭 인코딩 실패: {}", amp);
        
        // 실제 진폭 변환 테스트
        let real_amp = poincare.get_real_amplitude(50.0);
        let expected = (amp as f32 / 4095.0) * 50.0;
        assert!((real_amp - expected).abs() < 1e-5,
                "실제 진폭 변환 실패: {} -> {}, 예상: {}", amp, real_amp, expected);
    }
}

#[test]
fn 기저_함수_선택자_6비트_인코딩_테스트() {
    let test_selectors = [0, 1, 15, 31, 63]; // 6비트 최대값
    
    for selector in test_selectors {
        let poincare = PoincarePackedBit128::new(
            PoincareQuadrant::First,
            0, 0, selector, 0, 0.1, 0.1
        );
        
        assert_eq!(poincare.get_basis_function_selector(), selector,
                   "기저 함수 선택자 인코딩 실패: {}", selector);
    }
}

#[test]
fn 코딕_회전_시퀀스_32비트_인코딩_테스트() {
    let test_sequences = [0, 1, 0xFFFF, 0x12345678, 0xFFFFFFFF];
    
    for seq in test_sequences {
        let poincare = PoincarePackedBit128::new(
            PoincareQuadrant::First,
            0, 0, 0, seq, 0.1, 0.1
        );
        
        assert_eq!(poincare.get_cordic_rotation_sequence(), seq,
                   "CORDIC 회전 시퀀스 인코딩 실패: 0x{:08X}", seq);
    }
}

#[test]
fn 푸앵카레_반지름_float_인코딩_테스트() {
    let test_radii = [0.0, 0.1, 0.5, 0.9, 0.99999];
    
    for r in test_radii {
        let poincare = PoincarePackedBit128::new(
            PoincareQuadrant::First,
            0, 0, 0, 0, r, 0.1
        );
        
        let decoded_r = poincare.get_r_poincare();
        assert!((decoded_r - r).abs() < 1e-6,
                "푸앵카레 반지름 인코딩 실패: {} -> {}", r, decoded_r);
    }
}

#[test]
fn 푸앵카레_각도_float_인코딩_테스트() {
    use std::f32::consts::PI;
    let test_angles = [0.0, PI/4.0, PI/2.0, PI, 3.0*PI/2.0, 2.0*PI];
    
    for theta in test_angles {
        let poincare = PoincarePackedBit128::new(
            PoincareQuadrant::First,
            0, 0, 0, 0, 0.1, theta
        );
        
        let decoded_theta = poincare.get_theta_poincare();
        let normalized_theta = theta.rem_euclid(2.0 * PI);
        assert!((decoded_theta - normalized_theta).abs() < 1e-6,
                "푸앵카레 각도 인코딩 실패: {} -> {}, 정규화: {}", 
                theta, decoded_theta, normalized_theta);
    }
}

#[test]
fn 쌍곡거리_계산_테스트() {
    let test_cases = [
        (0.0, 0.0),           // 중심점
        (0.5, 0.549306),      // 중간
        (0.9, 1.472219),      // 경계 근처
        (0.99, 2.646653),     // 매우 경계 근처
    ];
    
    for (r, expected_distance) in test_cases {
        let poincare = PoincarePackedBit128::new(
            PoincareQuadrant::First,
            0, 0, 0, 0, r, 0.0
        );
        
        let computed_distance = poincare.compute_hyperbolic_distance();
        assert!((computed_distance - expected_distance).abs() < 0.001,
                "쌍곡거리 계산 실패: r={} -> {}, 예상: {}", 
                r, computed_distance, expected_distance);
    }
}

#[test]
fn 정보밀도_계산_테스트() {
    let test_cases = [
        (0.0, 1.0),           // 중심: 1.0
        (0.5, 1.777778),      // ρ(0.5) = 1/(1-0.25)² = 1.78
        (0.8, 7.716051),      // ρ(0.8) = 1/(1-0.64)² = 7.716051  
        (0.9, 27.777778),     // ρ(0.9) = 1/(1-0.81)² = 27.78
    ];
    
    for (r, expected_density) in test_cases {
        let poincare = PoincarePackedBit128::new(
            PoincareQuadrant::First,
            0, 0, 0, 0, r, 0.0
        );
        
        let computed_density = poincare.compute_information_density();
        assert!((computed_density - expected_density).abs() < 0.1,
                "정보밀도 계산 실패: r={} -> {}, 예상: {}", 
                r, computed_density, expected_density);
    }
}

#[test]
fn 쌍곡함수_사분면별_계산_테스트() {
    let input = 1.0f32;
    
    // 각 사분면별 함수 테스트
    let test_cases = [
        (PoincareQuadrant::First, input.sinh()),    // sinh(1.0) ≈ 1.175
        (PoincareQuadrant::Second, input.cosh()),   // cosh(1.0) ≈ 1.543
        (PoincareQuadrant::Third, input.tanh()),    // tanh(1.0) ≈ 0.762
        (PoincareQuadrant::Fourth, 1.0/(input.cosh()*input.cosh())), // sech²(1.0)
    ];
    
    for (quadrant, expected) in test_cases {
        let poincare = PoincarePackedBit128::new(
            quadrant, 0, 0, 0, 0, 0.1, 0.0
        );
        
        let computed = poincare.compute_hyperbolic_function(input);
        assert!((computed - expected).abs() < 1e-6,
                "쌍곡함수 계산 실패: {:?} -> {}, 예상: {}", 
                quadrant, computed, expected);
    }
}

#[test]
fn 랜덤_생성_유효성_테스트() {
    let mut rng = rand::thread_rng();
    
    // 100개의 랜덤 인스턴스 생성하여 모두 유효한지 확인
    for _ in 0..100 {
        let poincare = PoincarePackedBit128::random(&mut rng);
        
        assert!(poincare.is_valid_poincare(), 
                "랜덤 생성된 푸앵카레 볼이 유효하지 않음: r={}, theta={}", 
                poincare.get_r_poincare(), poincare.get_theta_poincare());
        
        // 각 필드가 올바른 범위에 있는지 확인
        assert!(poincare.get_hyperbolic_frequency() < 4096, 
                "쌍곡 주파수가 범위를 벗어남");
        assert!(poincare.get_geodesic_amplitude() < 4096,
                "측지선 진폭이 범위를 벗어남");
        assert!(poincare.get_basis_function_selector() < 64,
                "기저 함수 선택자가 범위를 벗어남");
        
        let r = poincare.get_r_poincare();
        assert!(r >= 0.0 && r < 1.0, "반지름이 푸앵카레 볼 범위를 벗어남: {}", r);
    }
}

#[test]
fn 푸앵카레_볼_경계_조건_테스트() {
    // 유효한 경우들
    let valid_cases = [
        (0.0, 0.0),
        (0.5, 1.0),
        (0.9, 2.0 * std::f32::consts::PI),
        (0.99999, 0.1),
    ];
    
    for (r, theta) in valid_cases {
        let poincare = PoincarePackedBit128::new(
            PoincareQuadrant::First,
            0, 0, 0, 0, r, theta
        );
        
        assert!(poincare.is_valid_poincare(),
                "유효한 케이스가 실패: r={}, theta={}", r, theta);
    }
    
    // 무효한 경우들 (r >= 1.0)
    let invalid_r_cases = [1.0, 1.1, 2.0];
    
    for r in invalid_r_cases {
        let poincare = PoincarePackedBit128::new(
            PoincareQuadrant::First,
            0, 0, 0, 0, r, 0.0
        );
        
        // r이 클램핑되어 0.99999로 제한되므로 여전히 유효해야 함
        assert!(poincare.is_valid_poincare(),
                "클램핑된 r이 여전히 유효해야 함: 원본 r={}, 클램핑된 r={}", 
                r, poincare.get_r_poincare());
        assert!(poincare.get_r_poincare() < 1.0,
                "r이 제대로 클램핑되지 않음");
    }
}

#[test]
fn 비트_필드_독립성_테스트() {
    // 각 비트 필드가 독립적으로 작동하는지 확인
    let base_poincare = PoincarePackedBit128::new(
        PoincareQuadrant::First,
        0, 0, 0, 0, 0.1, 0.1
    );
    
    // 사분면만 변경
    let quad_changed = PoincarePackedBit128::new(
        PoincareQuadrant::Third,
        0, 0, 0, 0, 0.1, 0.1
    );
    
    // 사분면만 다르고 나머지는 같아야 함
    assert_ne!(base_poincare.get_quadrant(), quad_changed.get_quadrant());
    assert_eq!(base_poincare.get_hyperbolic_frequency(), quad_changed.get_hyperbolic_frequency());
    assert_eq!(base_poincare.get_geodesic_amplitude(), quad_changed.get_geodesic_amplitude());
    assert_eq!(base_poincare.get_basis_function_selector(), quad_changed.get_basis_function_selector());
}

#[test]
fn 비트_마스킹_정확성_테스트() {
    // 최대값들로 모든 비트를 1로 설정하여 마스킹 테스트
    let max_poincare = PoincarePackedBit128::new(
        PoincareQuadrant::Fourth, // 11 (2비트)
        4095,  // 12비트 최대값
        4095,  // 12비트 최대값  
        63,    // 6비트 최대값
        0xFFFFFFFF, // 32비트 최대값
        0.99999, 1.0
    );
    
    // 각 필드가 정확히 추출되는지 확인
    assert_eq!(max_poincare.get_quadrant(), PoincareQuadrant::Fourth);
    assert_eq!(max_poincare.get_hyperbolic_frequency(), 4095);
    assert_eq!(max_poincare.get_geodesic_amplitude(), 4095);
    assert_eq!(max_poincare.get_basis_function_selector(), 63);
    assert_eq!(max_poincare.get_cordic_rotation_sequence(), 0xFFFFFFFF);
} 

// ================================
// 2장: 푸앵카레 인코딩 파이프라인 테스트
// ================================

use crate::encoder::{PoincareEncoder, FrequencyType};

#[test]
fn 푸앵카레_인코더_생성_및_기본_동작_테스트() {
    let mut encoder = PoincareEncoder::new(10);
    
    // 간단한 2x2 행렬로 기본 동작 테스트
    let test_matrix = vec![1.0, 0.0, 0.0, 1.0];
    let encoded = encoder.encode_matrix(&test_matrix, 2, 2);
    
    // 기본 검증
    assert!(encoded.is_valid_poincare(), "인코딩 결과가 푸앵카레 볼 조건을 만족하지 않음");
    
    let r = encoded.get_r_poincare();
    let theta = encoded.get_theta_poincare();
    assert!(r >= 0.01 && r <= 0.99, "r이 범위를 벗어남: {}", r);
    assert!(theta.is_finite(), "theta가 무한대: {}", theta);
    
    println!("기본 동작 테스트 성공: r={:.4}, theta={:.4}", r, theta);
}

#[test]
fn 다양한_크기_행렬_인코딩_테스트() {
    let mut encoder = PoincareEncoder::new(5);
    
    // 다양한 크기의 행렬 테스트
    let test_sizes = [(2, 2), (4, 4), (8, 8)];
    
    for (rows, cols) in test_sizes {
        // 체스판 패턴 생성
        let mut matrix = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                matrix[i * cols + j] = if (i + j) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        
        let encoded = encoder.encode_matrix(&matrix, rows, cols);
        
        assert!(encoded.is_valid_poincare(), 
                "{}x{} 행렬 인코딩 실패", rows, cols);
        
        let r = encoded.get_r_poincare();
        let theta = encoded.get_theta_poincare();
        assert!(r >= 0.01 && r <= 0.99, 
                "{}x{} 행렬에서 r이 범위를 벗어남: {}", rows, cols, r);
        assert!(theta.is_finite(), 
                "{}x{} 행렬에서 theta가 무한대: {}", rows, cols, theta);
        
        println!("{}x{} 행렬: r={:.4}, theta={:.4}, 사분면={:?}", 
                 rows, cols, r, theta, encoded.get_quadrant());
    }
}

#[test]
fn 특수_패턴_인코딩_테스트() {
    let mut encoder = PoincareEncoder::new(5);
    
    // 특수한 패턴들 테스트
    let test_patterns = [
        ("영행렬", vec![0.0, 0.0, 0.0, 0.0]),
        ("단위행렬", vec![1.0, 0.0, 0.0, 1.0]),
        ("전체 1", vec![1.0, 1.0, 1.0, 1.0]),
        ("음수 포함", vec![-1.0, 1.0, -1.0, 1.0]),
    ];
    
    for (pattern_name, matrix) in test_patterns {
        let encoded = encoder.encode_matrix(&matrix, 2, 2);
        
        assert!(encoded.is_valid_poincare(), 
                "{} 패턴 인코딩 실패", pattern_name);
        
        let r = encoded.get_r_poincare();
        let theta = encoded.get_theta_poincare();
        assert!(r >= 0.01 && r <= 0.99, 
                "{} 패턴에서 r이 범위를 벗어남: {}", pattern_name, r);
        assert!(theta.is_finite(), 
                "{} 패턴에서 theta가 무한대: {}", pattern_name, theta);
        
        println!("{}: r={:.4}, theta={:.4}, 사분면={:?}", 
                 pattern_name, r, theta, encoded.get_quadrant());
    }
}

#[test]
fn 인코딩_결과_일관성_테스트() {
    let mut encoder = PoincareEncoder::new(5);
    
    // 같은 패턴을 여러 번 인코딩했을 때 결과가 일관되는지 확인
    let matrix = vec![1.0, 0.5, 0.5, 0.0];
    
    let encoded1 = encoder.encode_matrix(&matrix, 2, 2);
    let encoded2 = encoder.encode_matrix(&matrix, 2, 2);
    
    // 같은 입력에 대해 같은 결과가 나와야 함
    assert!((encoded1.get_r_poincare() - encoded2.get_r_poincare()).abs() < 1e-6,
            "같은 입력에 대해 다른 r 값: {} vs {}", 
            encoded1.get_r_poincare(), encoded2.get_r_poincare());
    assert!((encoded1.get_theta_poincare() - encoded2.get_theta_poincare()).abs() < 1e-6,
            "같은 입력에 대해 다른 theta 값: {} vs {}", 
            encoded1.get_theta_poincare(), encoded2.get_theta_poincare());
    assert_eq!(encoded1.get_quadrant(), encoded2.get_quadrant(),
               "같은 입력에 대해 다른 사분면");
    
    println!("일관성 테스트 성공: 동일한 결과 확인됨");
} 
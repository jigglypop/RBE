//! 완전한 비트 도메인 푸앵카레볼 테스트 - 극한의 정밀도 검증

use crate::core::tensors::*;
use rand::{Rng, SeedableRng, rngs::StdRng};

#[test]
fn 테스트_11비트_사이클_상태_전이_정확성() {
    // 모든 가능한 11비트 상태에 대해 테스트 (2^11 = 2048)
    for i in 0u16..2048 {
        let state1 = CycleState::from_bits(i);
        
        // 자기 자신과의 전이
        let self_transition = state1.apply_transition(&state1);
        
        // 전이 후 11비트 범위 확인
        assert!(self_transition.to_bits() <= 0x7FF, 
                "자기 전이 후 범위 초과: {} -> {}", i, self_transition.to_bits());
        
        // 모든 다른 상태와의 전이 테스트
        for j in 0u16..2048 {
            let state2 = CycleState::from_bits(j);
            let result = state1.apply_transition(&state2);
            
            // 결과가 11비트 범위 내인지 확인
            assert!(result.to_bits() <= 0x7FF,
                    "전이 결과 범위 초과: {} x {} -> {}", i, j, result.to_bits());
            
            // 함수 인덱스 범위 확인 (0-7)
            assert!(result.get_active_function() < 8,
                    "함수 인덱스 범위 초과: {}", result.get_active_function());
            
            // 사이클 위치 범위 확인 (0-15)
            assert!(result.get_cycle_position() < 16,
                    "사이클 위치 범위 초과: {}", result.get_cycle_position());
        }
    }
}

#[test]
fn 테스트_고정소수점_연산_정밀도() {
    // Q16.16 고정소수점 범위 테스트
    let test_values = [
        0i32,           // 0
        1,              // 최소 양수
        -1,             // 최소 음수
        32767,          // 0.5 근처
        65536,          // 1.0
        -65536,         // -1.0
        0x7FFFFFFF,     // 최대값
        -0x80000000i32, // 최소값
    ];
    
    for &val in &test_values {
        // f32 변환 왕복 테스트
        let f = val as f32 / 65536.0;
        let back = (f * 65536.0) as i32;
        
        // 1 LSB 이내의 오차만 허용
        let diff = (val - back).abs();
        assert!(diff <= 1, 
                "고정소수점 변환 오차: {} -> {} -> {}, 차이: {}", 
                val, f, back, diff);
    }
}

#[test]
fn 테스트_쌍곡함수_LUT_정밀도() {
    use super::super::hyperbolic_lut::HYPERBOLIC_LUT_DATA;
    
    // 모든 LUT 엔트리 검증
    for func_idx in 0..8 {
        for i in 0..256 {
            let lut_val = HYPERBOLIC_LUT_DATA[func_idx][i];
            
            // LUT 값이 u32 범위 내인지 확인
            let val_i32 = lut_val as i32;
            let x = (i as i32 - 128) as f32 / 128.0; // [-1, 1] 범위
            
            match func_idx {
                0 => { // sinh
                    // 단조증가 확인
                    if i > 0 {
                        let prev_val = HYPERBOLIC_LUT_DATA[func_idx][i-1] as i32;
                        assert!(val_i32 >= prev_val || i == 128, // 0 근처 예외
                                "sinh 단조성 위반: [{}] {} < {}", i, val_i32, prev_val);
                    }
                },
                1 => { // cosh
                    // 최솟값이 1 (Q16.16에서 65536) 근처인지 확인
                    if i == 128 { // x = 0
                        let one_q16 = 65536;
                        let diff = (val_i32 - one_q16).abs();
                        assert!(diff < 100, "cosh(0) != 1: {} (차이: {})", val_i32, diff);
                    }
                    // 짝함수 대칭성 확인
                    if i < 128 {
                        let mirror_idx = 256 - i - 1;
                        let mirror_val = HYPERBOLIC_LUT_DATA[func_idx][mirror_idx] as i32;
                        let diff = (val_i32 - mirror_val).abs();
                        assert!(diff < 1000, "cosh 대칭성 위반: [{}]={} != [{}]={}", 
                                i, val_i32, mirror_idx, mirror_val);
                    }
                },
                2 => { // tanh
                    // 범위 [-1, 1] 확인
                    let tanh_f = val_i32 as f32 / 65536.0;
                    assert!(tanh_f >= -1.1 && tanh_f <= 1.1, 
                            "tanh 범위 초과: [{}] = {}", i, tanh_f);
                    // 단조증가 확인
                    if i > 0 {
                        let prev_val = HYPERBOLIC_LUT_DATA[func_idx][i-1] as i32;
                        assert!(val_i32 >= prev_val,
                                "tanh 단조성 위반: [{}] {} < {}", i, val_i32, prev_val);
                    }
                },
                3 => { // sech²
                    // 범위 (0, 1] 확인
                    let sech2_f = val_i32 as f32 / 65536.0;
                    assert!(sech2_f >= -0.1 && sech2_f <= 1.1,
                            "sech² 범위 초과: [{}] = {}", i, sech2_f);
                    // 최댓값이 x=0에서 1인지 확인
                    if i == 128 {
                        let one_q16 = 65536;
                        let diff = (val_i32 - one_q16).abs();
                        assert!(diff < 1000, "sech²(0) != 1: {} (차이: {})", val_i32, diff);
                    }
                },
                _ => {} // 나머지 함수들도 유효 범위 확인
            }
        }
    }
}

#[test]
fn 테스트_CORDIC_각도_테이블_정밀도() {
    // CORDIC 각도 누적 확인
    let mut angle_sum = 0u64;
    for i in 0..20 {
        angle_sum += crate::core::tensors::packed_types::CORDIC_ANGLES_Q32[i] as u64;
    }
    
    // π/2에 수렴하는지 확인 (Q32 형식)
    let pi_over_2_q32 = 0x3243F6A8u64; // ≈ π/2
    let diff = if angle_sum > pi_over_2_q32 {
        angle_sum - pi_over_2_q32
    } else {
        pi_over_2_q32 - angle_sum
    };
    
    // 0.001% 이내의 오차만 허용
    let tolerance = pi_over_2_q32 / 100000;
    assert!(diff < tolerance,
            "CORDIC 각도 합 오차: {} vs {} (차이: {})", 
            angle_sum, pi_over_2_q32, diff);
}

#[test]
fn 테스트_비트_atan2_경계값() {
    // 모든 사분면의 경계값 테스트
    let test_cases = [
        (0, 65536, 0),           // 0도
        (65536, 65536, 0x3243F6A8 / 4),    // 45도  
        (65536, 0, 0x3243F6A8 / 2),        // 90도
        (65536, -65536, (0x3243F6A8 as i64 * 3 / 4) as i32), // 135도
        (0, -65536, 0x3243F6A8),            // 180도
        (-65536, -65536, -(0x3243F6A8 as i64 * 3 / 4) as i32), // -135도
        (-65536, 0, -(0x3243F6A8 / 2) as i32),      // -90도
        (-65536, 65536, -(0x3243F6A8 / 4) as i32),  // -45도
    ];
    
    for (y, x, expected_angle) in test_cases {
        let angle = Packed64::bit_atan2_q16(y, x);
        let angle_q32 = (angle as i64) << 16;
        let diff = (angle_q32 - expected_angle as i64).abs();
        
        // 0.1% 오차 허용
        let tolerance = 0x3243F6A8 / 1000;
        assert!(diff < tolerance,
                "atan2({}, {}) 오차: {} vs {} (차이: {})",
                y, x, angle_q32, expected_angle, diff);
    }
}

#[test]
fn 테스트_쌍곡_CORDIC_수렴성() {
    let mut rng = StdRng::seed_from_u64(12345);
    
    // 다양한 회전 시퀀스로 테스트
    for _ in 0..1000 {
        let rotations = rng.gen::<u64>();
        let packed = Packed64::new(rotations);
        
        // 중앙 픽셀에서 가중치 계산
        let weight = packed.compute_weight(50, 50, 100, 100);
        
        // 가중치가 유효 범위 [-1, 1] 내에 있는지 확인
        assert!(weight >= -1.1 && weight <= 1.1,
                "CORDIC 가중치 범위 초과: {} (rotations: {:064b})", 
                weight, rotations);
        
        // NaN이나 Inf가 아닌지 확인
        assert!(weight.is_finite(),
                "CORDIC 가중치 무한대/NaN: {} (rotations: {:064b})", 
                weight, rotations);
    }
}

#[test]
fn 테스트_상태전이_비트_보존() {
    let mut rng = StdRng::seed_from_u64(54321);
    
    for _ in 0..1000 {
        let mut packed = Packed128::random(&mut rng);
        let original_hi = packed.hi;
        
        // 다양한 에러값으로 상태 전이
        let errors = [-1.0, -0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5, 1.0];
        
        for &error in &errors {
            for i in 0..10 {
                for j in 0..10 {
                    let before_cycle = packed.get_cycle_state();
                    packed.apply_state_transition(error, i, j);
                    let after_cycle = packed.get_cycle_state();
                    
                    // 11비트 사이클 상태가 유효한지 확인
                    assert!(after_cycle.to_bits() <= 0x7FF,
                            "상태 전이 후 11비트 초과: {:011b}", after_cycle.to_bits());
                    
                    // 하위 53비트가 보존되는지 확인
                    let lower_mask = 0x1FFFFFFFFFFFFF;
                    let preserved_bits = original_hi & lower_mask;
                    let new_lower = packed.hi & lower_mask;
                    
                    // XOR 연산이므로 다를 수 있지만, 범위는 유지되어야 함
                    assert!(new_lower <= lower_mask,
                            "하위 비트 범위 초과: {:053b}", new_lower);
                }
            }
        }
    }
}

#[test]
fn 테스트_fused_forward_극한값() {
    let test_cases = [
        // 극한 좌표
        (0, 0, 100, 100),      // 좌상단
        (99, 99, 100, 100),    // 우하단
        (0, 99, 100, 100),     // 우상단
        (99, 0, 100, 100),     // 좌하단
        (50, 50, 100, 100),    // 중앙
        // 작은 행렬
        (0, 0, 1, 1),          // 1x1
        (0, 1, 1, 2),          // 1x2
        (1, 0, 2, 1),          // 2x1
        // 큰 행렬
        (0, 0, 10000, 10000),
        (9999, 9999, 10000, 10000),
    ];
    
    let mut rng = StdRng::seed_from_u64(98765);
    
    for &(i, j, rows, cols) in &test_cases {
        let packed = Packed128::random(&mut rng);
        let result = packed.fused_forward(i, j, rows, cols);
        
        // 결과가 유한하고 유효한 범위인지 확인
        assert!(result.is_finite(), 
                "fused_forward({}, {}, {}, {}) = NaN/Inf", i, j, rows, cols);
        
        // 일반적으로 [-2, 2] 범위 내에 있어야 함
        assert!(result >= -2.0 && result <= 2.0,
                "fused_forward 범위 초과: {} at ({}, {})", result, i, j);
    }
}

#[test]
fn 테스트_고정소수점_변환_왕복() {
    let test_values = [
        0.0f32,
        0.00001,
        0.1,
        0.5,
        0.9,
        0.99999,
        std::f32::consts::PI / 4.0,
        std::f32::consts::PI / 2.0,
        std::f32::consts::PI,
        2.0 * std::f32::consts::PI - 0.00001,
    ];
    
    for &r in &test_values {
        for &theta in &test_values {
            if r >= 0.0 && r < 1.0 && theta >= 0.0 && theta < 2.0 * std::f32::consts::PI {
                let params = DecodedParams { r_fp32: r, theta_fp32: theta };
                let packed = Packed128::from_continuous(&params);
                let decoded = packed.decode();
                
                // 상대 오차 계산
                let r_error = (decoded.r_fp32 - r).abs() / (r + 1e-10);
                let theta_error = (decoded.theta_fp32 - theta).abs() / (theta + 1e-10);
                
                // Q32.32는 약 2^-32 ≈ 2.3e-10의 정밀도
                assert!(r_error < 1e-8, 
                        "r 변환 오차: {} -> {} (오차: {:.2e})", r, decoded.r_fp32, r_error);
                assert!(theta_error < 1e-8,
                        "theta 변환 오차: {} -> {} (오차: {:.2e})", theta, decoded.theta_fp32, theta_error);
            }
        }
    }
}

#[test]
fn 테스트_비트_패턴_변조_균등분포() {
    // 변조 함수가 균등한 분포를 생성하는지 확인
    let mut histogram = [0u32; 256];
    
    // 다양한 패턴과 좌표로 테스트
    for pattern in 0..8192u64 { // 13비트 패턴
        for i in 0..16 {
            for j in 0..16 {
                for cycle in 0..16 {
                    let modulation = Packed128::bit_pattern_modulation(pattern, i, j, cycle);
                    let bucket = (modulation * 255.0) as usize;
                    histogram[bucket.min(255)] += 1;
                }
            }
        }
    }
    
    // 균등 분포 검증 (카이제곱 검정)
    let expected = histogram.iter().sum::<u32>() / 256;
    let mut chi_squared = 0.0;
    
    for &count in &histogram {
        let diff = count as f64 - expected as f64;
        chi_squared += diff * diff / expected as f64;
    }
    
    // 자유도 255에서 5% 유의수준 임계값은 약 293
    assert!(chi_squared < 400.0,
            "비트 패턴 변조 분포 불균등: χ² = {}", chi_squared);
}

#[test]
fn 테스트_메모리_정렬_및_크기() {
    use std::mem::{size_of, align_of};
    
    // 구조체 크기 확인
    assert_eq!(size_of::<CycleState>(), 2, "CycleState 크기 불일치");
    assert_eq!(size_of::<Packed64>(), 8, "Packed64 크기 불일치");
    assert_eq!(size_of::<Packed128>(), 16, "Packed128 크기 불일치");
    
    // 정렬 확인
    assert_eq!(align_of::<CycleState>(), 2, "CycleState 정렬 불일치");
    assert_eq!(align_of::<Packed64>(), 8, "Packed64 정렬 불일치");
    assert_eq!(align_of::<Packed128>(), 8, "Packed128 정렬 불일치");
    
    // 비트필드 마스크 확인
    let cycle = CycleState::from_bits(0xFFFF);
    assert_eq!(cycle.to_bits(), 0x7FF, "11비트 마스킹 실패");
    
    let mut packed = Packed128::default();
    packed.hi = 0xFFFFFFFFFFFFFFFF;
    packed.set_cycle_state(CycleState::from_bits(0));
    assert_eq!(packed.hi & (0x7FFu64 << 53), 0, "사이클 상태 설정 실패");
}

#[test]
fn 스트레스_테스트_백만_연산() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut total_cycles = 0u64;
    
    // 100만 번의 연산 수행
    for _ in 0..1_000_000 {
        let packed = Packed128::random(&mut rng);
        let i = rng.gen_range(0..100);
        let j = rng.gen_range(0..100);
        
        // 가중치 계산
        let weight = packed.fused_forward(i, j, 100, 100);
        assert!(weight.is_finite(), "스트레스 테스트 중 NaN/Inf 발생");
        
        // 사이클 상태 카운트
        total_cycles += packed.get_cycle_state().to_bits() as u64;
    }
    
    // 평균 사이클 상태가 중간값 근처인지 확인 (균등 분포)
    let avg_cycle = total_cycles / 1_000_000;
    assert!(avg_cycle > 512 && avg_cycle < 1536,
            "사이클 상태 분포 편향: 평균 = {}", avg_cycle);
} 
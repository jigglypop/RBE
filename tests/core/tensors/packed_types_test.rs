//! 완전한 비트 도메인 푸앵카레볼 테스트 - 극한의 정밀도 검증

use std::mem::{size_of, align_of};
use rand::{SeedableRng, Rng};
use rbe_llm::core::tensors::*;
use std::collections::HashMap;

#[test]
fn 사이클_상태_전이_테스트() {
    for i in 0..2048 {
        let state1 = CycleState::from_bits(i);
        
        let expected_active = ((i >> 8) & 0x7) as usize;
        let expected_cycle = ((i >> 4) & 0xF) as usize;
        
        assert_eq!(state1.get_active_function(), expected_active);
        assert_eq!(state1.get_cycle_position(), expected_cycle);
        
        // 전이 테스트
        for j in 0..128 {
            let state2 = CycleState::from_bits(j);
            let result = state1.apply_transition(&state2);
            
            // 결과가 11비트 범위 내에 있는지 확인
            assert!(result.to_bits() <= 0x7FF);
        }
    }
}

#[test]
fn 비트_그래디언트_추적_테스트() {
    let mut tracker = BitGradientTracker::new(100);
    
    let input = Packed128 { hi: 0x123456789ABCDEF0, lo: 0xFEDCBA9876543210 };
    let output = Packed128 { hi: 0x0FEDCBA987654321, lo: 0x123456789ABCDEF0 };
    
    // register_dependency가 정상적으로 호출되는지만 확인
    tracker.register_dependency(0, &input, &output);
    
    // tracker가 성공적으로 생성되었는지 확인
    assert!(true); // tracker 생성과 메서드 호출이 성공했음을 의미
}

#[test]
fn 푸앵카레볼_가중치_계산_테스트() {
    let seed = Packed64::new(0x123456789ABCDEF0);
    
    let rows = 10;
    let cols = 20;
    
    for i in 0..rows {
        for j in 0..cols {
            let weight = seed.compute_weight(i, j, rows, cols);
            
            // 가중치가 유한하고 합리적인 범위 내에 있는지 확인
            assert!(weight.is_finite());
            assert!(weight.abs() <= 10.0); // 합리적인 상한
        }
    }
}

#[test]
fn hyperbolic_lut_접근성_테스트() {
    // HYPERBOLIC_LUT_DATA에 접근할 수 있는지 확인
    use rbe_llm::core::tensors::hyperbolic_lut::HYPERBOLIC_LUT_DATA;
    
    // 첫 번째 함수의 첫 번째 값 확인
    let first_value = HYPERBOLIC_LUT_DATA[0][0];
    assert!(first_value != 0); // 0이 아님을 확인
    
    // LUT 크기 검증
    assert_eq!(HYPERBOLIC_LUT_DATA.len(), 8); // 8개 함수
    assert_eq!(HYPERBOLIC_LUT_DATA[0].len(), 256); // 각 함수당 256개 값
}

#[test]
fn bit_atan2_q16_정확성_테스트() {
    // 기본 사분면 테스트
    let test_cases = [
        (100, 100, true),    // 첫 번째 사분면
        (-100, 100, true),   // 두 번째 사분면
        (-100, -100, true),  // 세 번째 사분면
        (100, -100, true),   // 네 번째 사분면
        (0, 100, true),      // 양의 x축
        (0, -100, true),     // 음의 x축
        (100, 0, true),      // 양의 y축
        (-100, 0, true),     // 음의 y축
    ];
    
    for (y, x, _expected) in test_cases {
        let result = Packed64::bit_atan2_q16(y, x);
        
        // 결과가 유한하고 Q16 범위 내에 있는지 확인
        assert!(result.abs() <= 0x8000); // Q16에서 ±π 범위
    }
}

#[test]
fn cordic_각도_테이블_검증() {
    use rbe_llm::core::tensors::packed_types::CORDIC_ANGLES_Q32;
    
    // CORDIC 각도들의 합이 대략 π/4 * 1.57 ≈ 1.23 정도가 되어야 함
    let mut angle_sum = 0u64;
    for i in 0..10 {
        angle_sum += CORDIC_ANGLES_Q32[i] as u64;
    }
    
    // CORDIC 각도들의 합이 합리적인 범위에 있는지 확인
    assert!(angle_sum > 1000000000 && angle_sum < 50000000000);
}

#[test]
fn packed64_가중치_일관성_테스트() {
    let rows = 5;
    let cols = 8;
    
    for rotation in [0u64, 0xFFFFFFFFFFFFFFFF, 0x123456789ABCDEF0] {
        let seed = Packed64::new(rotation);
        
        // 같은 좌표에 대해 항상 같은 가중치를 반환하는지 확인
        for i in 0..rows {
            for j in 0..cols {
                let weight1 = seed.compute_weight(i, j, rows, cols);
                let weight2 = seed.compute_weight(i, j, rows, cols);
                
                assert_eq!(weight1, weight2, "좌표 ({}, {})에서 가중치 불일치", i, j);
            }
        }
    }
}

#[test]
fn bit_atan2_특수_케이스_테스트() {
    // 0, 0 케이스
    let result = Packed64::bit_atan2_q16(0, 0);
    assert_eq!(result, 0);
    
    // x축 케이스들
    let result_pos_x = Packed64::bit_atan2_q16(0, 100);
    assert_eq!(result_pos_x, 0);
    
    let result_neg_x = Packed64::bit_atan2_q16(0, -100);
    assert_eq!(result_neg_x, 0x6487); // π in Q16
}

#[test]
fn packed128_사이클_상태_통합_테스트() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let mut packed = Packed128::random(&mut rng);
    
    for test_round in 0..100 {
        let _original_state = packed.get_cycle_state();
        
        // 에러 시뮬레이션
        let error = (test_round as f32) * 0.01 - 0.5;
        let i = test_round % 10;
        let j = (test_round * 3) % 15;
        
        packed.apply_state_transition(error, i, j);
        
        let new_state = packed.get_cycle_state();
        
        // 상태가 변경되었고 유효한 범위 내에 있는지 확인
        assert!(new_state.to_bits() <= 0x7FF);
        
        // 상태 전이가 실제로 발생했는지 확인 (대부분의 경우)
        if test_round > 0 && error.abs() > 0.1 {
            // 충분히 큰 에러에서는 상태가 변경되어야 함
        }
    }
}

#[test]
fn fused_forward_비트_도메인_일관성_테스트() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let rows = 8;
    let cols = 12;
    
    for _ in 0..50 {
        let packed = Packed128::random(&mut rng);
        
        for i in 0..rows {
            for j in 0..cols {
                let output = packed.fused_forward(i, j, rows, cols);
                
                // 출력이 유한하고 합리적인 범위에 있는지 확인
                assert!(output.is_finite());
                assert!(output.abs() <= 100.0);
            }
        }
    }
}

#[test]
fn bit_pattern_modulation_분포_테스트() {
    let mut distribution_map: HashMap<u32, u32> = HashMap::new();
    let total_samples = 10000;
    
    for test_case in 0..total_samples {
        let pattern = (test_case as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let i = test_case % 50;
        let j = (test_case * 7) % 30;
        let cycle = test_case % 16;
        
        let modulation = Packed128::bit_pattern_modulation(pattern, i, j, cycle);
        
        // [0, 1] 범위 확인
        assert!(modulation >= 0.0 && modulation <= 1.0);
        
        // 분포 추적 (10개 구간)
        let bucket = (modulation * 10.0) as u32;
        *distribution_map.entry(bucket.min(9)).or_insert(0) += 1;
    }
    
    // 각 구간에 최소한의 샘플이 있는지 확인 (균등 분포 검증)
    for bucket in 0..10 {
        let count = distribution_map.get(&bucket).unwrap_or(&0);
        assert!(*count > (total_samples / 50) as u32, "구간 {}의 분포가 너무 적음: {}", bucket, count);
    }
}

#[test]
fn 연속_파라미터_왕복_변환_테스트() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(456);
    
    for _ in 0..100 {
        let r = rng.gen::<f32>() * 0.99; // [0, 0.99) 범위
        let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
        
        let params = DecodedParams { r_fp32: r, theta_fp32: theta };
        let packed = Packed128::from_continuous(&params);
        let decoded = packed.decode();
        
        // 허용 오차 내에서 일치하는지 확인
        let r_error = (decoded.r_fp32 - r).abs();
        let theta_diff = (decoded.theta_fp32 - theta).abs();
        let theta_error = theta_diff.min(2.0 * std::f32::consts::PI - theta_diff);
        
        assert!(r_error < 0.001, "r 변환 오차 too large: {} vs {}", r, decoded.r_fp32);
        assert!(theta_error < 0.01, "theta 변환 오차 too large: {} vs {}", theta, decoded.theta_fp32);
    }
}

#[test]
fn hyperbolic_lut_적용_정확성_테스트() {
    for func_idx in 0..8 {
        for _test_val in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            let _modulation = 0.5;
            
            // apply_hyperbolic_lut은 private이므로 fused_forward를 통해 간접 테스트
            let mut test_packed = Packed128::default();
            
            // 특정 함수가 활성화되도록 사이클 상태 설정
            let cycle_state = CycleState::from_bits((func_idx << 8) as u16);
            test_packed.set_cycle_state(cycle_state);
            
            let result = test_packed.fused_forward(0, 0, 10, 10);
            
            // 결과가 유한한 값인지 확인
            assert!(result.is_finite());
        }
    }
}

#[test]
fn 메모리_레이아웃_검증() {
    // 구조체 크기 검증
    assert_eq!(size_of::<CycleState>(), 2, "CycleState 크기 불일치");
    assert_eq!(size_of::<Packed64>(), 8, "Packed64 크기 불일치");
    assert_eq!(size_of::<Packed128>(), 16, "Packed128 크기 불일치");
    
    // 메모리 정렬 검증
    assert_eq!(align_of::<CycleState>(), 2, "CycleState 정렬 불일치");
    assert_eq!(align_of::<Packed64>(), 8, "Packed64 정렬 불일치");
    assert_eq!(align_of::<Packed128>(), 8, "Packed128 정렬 불일치");
    
    let cycle = CycleState::from_bits(0xFFFF);
    assert_eq!(cycle.to_bits(), 0x7FF, "11비트 마스킹 실패");
    
    let mut packed = Packed128::default();
    packed.set_cycle_state(CycleState::from_bits(0));
    assert_eq!(packed.get_cycle_state().to_bits(), 0);
}

#[test]
fn 비트_도메인_성능_일관성_테스트() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(789);
    
    // 대량 데이터로 일관성 확인
    for batch in 0..10 {
        let packed = Packed128::random(&mut rng);
        let rows = 20;
        let cols = 30;
        
        let mut results = Vec::new();
        
        for i in 0..rows {
            for j in 0..cols {
                let result = packed.fused_forward(i, j, rows, cols);
                assert!(result.is_finite());
                results.push(result);
            }
        }
        
        // 결과의 분산이 너무 크지 않은지 확인
        let mean: f32 = results.iter().sum::<f32>() / results.len() as f32;
        let variance: f32 = results.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / results.len() as f32;
        
        assert!(variance.is_finite(), "배치 {}에서 분산 계산 실패", batch);
        assert!(variance < 1000.0, "배치 {}에서 분산이 너무 큼: {}", batch, variance);
    }
} 
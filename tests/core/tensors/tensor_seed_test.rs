use rbe_llm::core::tensors::*;
use rand::SeedableRng;

#[test]
fn 테스트_크기_정보() {
    assert_eq!(std::mem::size_of::<Packed64>(), 8);  // 64 bits
    assert_eq!(std::mem::size_of::<Packed128>(), 16); // 128 bits
}

#[test]
fn 테스트_packed64_기본_동작() {
    let seed = Packed64::new(0x123456789ABCDEF0);
    
    // 기본 가중치 계산
    let weight1 = seed.compute_weight(0, 0, 10, 10);
    let weight2 = seed.compute_weight(5, 5, 10, 10);
    let weight3 = seed.compute_weight(9, 9, 10, 10);
    
    // 가중치가 finite한지 확인
    assert!(weight1.is_finite());
    assert!(weight2.is_finite());
    assert!(weight3.is_finite());
    
    // 재현성 확인
    let weight1_again = seed.compute_weight(0, 0, 10, 10);
    assert_eq!(weight1, weight1_again);
}

#[test]
fn 테스트_packed128_기본_동작() {
    let mut seed = Packed128::default();
    
    // 사이클 상태 설정
    let cycle_state = CycleState::from_bits(0x100);
    seed.set_cycle_state(cycle_state);
    
    // 설정 확인
    let retrieved_state = seed.get_cycle_state();
    assert_eq!(retrieved_state.to_bits(), 0x100);
    
    // fused_forward 테스트
    let result = seed.fused_forward(0, 0, 10, 10);
    assert!(result.is_finite());
}

#[test]
fn 테스트_64bit_to_128bit_변환() {
    let mut seed = Packed64::new(0);
    let original = 0x123456789ABCDEF0u64;
    seed.rotations = original;
    
    let seed128 = Packed128 { hi: original, lo: 0 };
    
    // 같은 데이터로부터 유사한 동작을 보이는지 확인
    let weight64 = seed.compute_weight(5, 5, 10, 10);
    let weight128 = seed128.fused_forward(5, 5, 10, 10);
    
    // 둘 다 finite해야 함
    assert!(weight64.is_finite());
    assert!(weight128.is_finite());
    
    // 다를 수 있지만 모두 유효한 범위여야 함
    assert!(weight64.abs() < 10.0);
    assert!(weight128.abs() < 10.0);
}

#[test]
fn 테스트_random_seed_다양성() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    let seed64 = Packed64::new(0x123456789ABCDEF0);
    
    // 다양한 위치에서 가중치 계산
    let weights: Vec<f32> = (0..10).map(|i| {
        seed64.compute_weight(i, i, 10, 10)
    }).collect();
    
    // 모든 가중치가 finite해야 함
    for &w in &weights {
        assert!(w.is_finite());
    }
    
    let seed128 = Packed128::random(&mut rng);
    let decoded = seed128.decode();
    
    // decode 결과 확인
    assert!(decoded.r_fp32 >= 0.0 && decoded.r_fp32 < 1.0);
    assert!(decoded.theta_fp32 >= 0.0 && decoded.theta_fp32 < 2.0 * std::f32::consts::PI);
    
    let new_params = DecodedParams {
        r_fp32: 0.5,
        theta_fp32: std::f32::consts::PI,
    };
    let seed_from_params = Packed128::from_continuous(&new_params);
    let redecoded = seed_from_params.decode();
    
    // 왕복 변환 정확성 확인 (허용 오차 내)
    assert!((redecoded.r_fp32 - 0.5).abs() < 0.01);
    assert!((redecoded.theta_fp32 - std::f32::consts::PI).abs() < 0.01);
}

#[test]
fn 테스트_cordic_가중치_계산() {
    let seed = Packed64::new(0x123456789ABCDEF0);
    
    // 다양한 행렬 크기에서 테스트
    for rows in [5, 10, 20] {
        for cols in [5, 10, 20] {
            for i in 0..rows.min(5) {
                for j in 0..cols.min(5) {
                    let weight = seed.compute_weight(i, j, rows, cols);
                    assert!(weight.is_finite(), "({}, {}) in {}x{}", i, j, rows, cols);
                    assert!(weight.abs() < 10.0, "weight too large: {}", weight);
                }
            }
        }
    }
}

#[test]
fn 테스트_state_transition() {
    let mut seed = Packed128::default();
    
    // 초기 상태
    let _initial_state = seed.get_cycle_state();
    
    // 상태 전이 적용
    seed.apply_state_transition(0.1, 0, 0);
    seed.apply_state_transition(-0.1, 1, 1);
    seed.apply_state_transition(0.5, 2, 2);
    
    // 상태가 변경되었는지 확인 (변경될 수도 있고 안 될 수도 있음)
    let final_state = seed.get_cycle_state();
    assert!(final_state.to_bits() <= 0x7FF); // 11비트 범위 내
}

#[test]
fn 테스트_packed128_다양성() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    
    let seed1 = Packed128::random(&mut rng);
    let seed2 = Packed128::random(&mut rng);
    
    // 두 시드가 다른지 확인 (매우 높은 확률로)
    assert_ne!(seed1.hi, seed2.hi);
    assert_ne!(seed1.lo, seed2.lo);
    
    // 둘 다 유효한 결과를 생성하는지 확인
    let result1 = seed1.fused_forward(0, 0, 10, 10);
    let result2 = seed2.fused_forward(0, 0, 10, 10);
    
    assert!(result1.is_finite());
    assert!(result2.is_finite());
} 
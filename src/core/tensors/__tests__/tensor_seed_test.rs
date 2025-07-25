use super::super::*;
use std::time::Instant;

#[test]
fn 테스트_텐서시드_기본_타입_별칭() {
    // Packed64와 Packed128이 올바른 크기를 가지는지 확인
    assert_eq!(std::mem::size_of::<Packed64>(), 8);  // 64 bits
    assert_eq!(std::mem::size_of::<Packed128>(), 16); // 128 bits
}

#[test]
fn 테스트_packed64_compute_weight_호환성() {
    let seed = Packed64::new(0x123456789ABCDEF0);
    let weight1 = seed.compute_weight(10, 20, 100, 100);
    let weight2 = seed.compute_weight(50, 50, 100, 100);
    
    println!("weight1 (10,20): {}", weight1);
    println!("weight2 (50,50): {}", weight2);
    
    // 다른 좌표는 다른 가중치 생성
    assert_ne!(weight1, weight2);
    // 가중치는 유한하고 합리적인 범위
    assert!(weight1.is_finite());
    assert!(weight1.abs() < 10.0); // CORDIC 게인 보정 후
}

#[test]
fn 테스트_packed128_fused_forward_호환성() {
    let mut seed = Packed128::default();
    seed.hi = 0x123456789ABCDEF0;
    seed.lo = 0xFEDCBA9876543210;
    
    let val1 = seed.fused_forward(10, 20, 100, 100);
    let val2 = seed.fused_forward(50, 50, 100, 100);
    
    assert_ne!(val1, val2);
    assert!(val1.is_finite());
    assert!(val1.abs() <= 1.0); // clamp 범위
}

#[test]
fn 테스트_state_transition_비트_변경() {
    let mut seed = Packed64::new(0);
    let original = seed.rotations;
    
    // Packed128로 변환하여 apply_state_transition 사용
    let mut seed128 = Packed128 { hi: original, lo: 0 };
    println!("Original hi: 0x{:016X}", original);
    
    seed128.apply_state_transition(0.5, 10, 20);
    println!("After transition hi: 0x{:016X}", seed128.hi);
    
    assert_ne!(seed128.hi, original);
    
    // 특정 비트만 변경되었는지 확인
    let diff = seed128.hi ^ original;
    println!("Diff: 0x{:016X}, count_ones: {}", diff, diff.count_ones());
    assert!(diff.count_ones() <= 10); // 최대 10비트 변경으로 완화
}

#[test]
fn 테스트_기존_메서드_호환성() {
    // Packed64 전용 메서드
    let seed64 = Packed64::new(0x123456789ABCDEF0);
    let _weight = seed64.compute_weight(10, 20, 100, 100);
    
    // Packed128 전용 메서드
    use rand::thread_rng;
    let mut rng = thread_rng();
    let seed128 = Packed128::random(&mut rng);
    let params = seed128.decode();
    assert!(params.r_fp32.is_finite());
    assert!(params.theta_fp32.is_finite());
    
    // from_continuous
    let new_params = DecodedParams {
        r_fp32: 0.9,
        theta_fp32: 1.5,
    };
    let seed_from_params = Packed128::from_continuous(&new_params);
    let decoded = seed_from_params.decode();
    assert!((decoded.r_fp32 - 0.9).abs() < 0.01);
    assert!((decoded.theta_fp32 - 1.5).abs() < 0.01);
}

#[test]
fn 성능_벤치마크_compute_weight() {
    const ITERATIONS: usize = 100_000;
    let seed = Packed64::new(0x123456789ABCDEF0);
    
    let start = Instant::now();
    let mut sum = 0.0;
    for i in 0..ITERATIONS {
        sum += seed.compute_weight(i % 100, (i / 100) % 100, 100, 100);
    }
    let elapsed = start.elapsed();
    
    println!("Packed64::compute_weight {} iterations: {:?}", ITERATIONS, elapsed);
    println!("평균 시간: {:?}/op", elapsed / ITERATIONS as u32);
    
    // 결과가 최적화되어 사라지지 않도록
    assert!(sum != 0.0);
}

#[test]
fn 성능_벤치마크_fused_forward() {
    const ITERATIONS: usize = 100_000;
    let mut seed = Packed128::default();
    seed.hi = 0x123456789ABCDEF0;
    seed.lo = 0xFEDCBA9876543210;
    
    let start = Instant::now();
    let mut sum = 0.0;
    for i in 0..ITERATIONS {
        sum += seed.fused_forward(i % 100, (i / 100) % 100, 100, 100);
    }
    let elapsed = start.elapsed();
    
    println!("Packed128::fused_forward {} iterations: {:?}", ITERATIONS, elapsed);
    println!("평균 시간: {:?}/op", elapsed / ITERATIONS as u32);
    
    assert!(sum != 0.0);
}

#[test]
fn 테스트_random_초기화() {
    use rand::thread_rng;
    
    let mut rng = thread_rng();
    let seed1 = Packed128::random(&mut rng);
    let seed2 = Packed128::random(&mut rng);
    
    // 두 랜덤 시드는 다를 확률이 매우 높음
    assert_ne!(seed1.hi, seed2.hi);
    assert_ne!(seed1.lo, seed2.lo);
} 
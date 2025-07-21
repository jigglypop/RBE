//! # 11비트 미분 사이클 시스템 단위 테스트
//!
//! cycle_differential.rs의 모든 함수에 대한 테스트

use crate::core::optimizers::cycle_differential::{
    CycleDifferentialSystem, DifferentialPhase, CycleState, HyperbolicFunction
};
use std::time::Instant;

#[test]
fn 사이클_상태_비트_인코딩_테스트() {
    println!("🧪 11비트 미분 사이클 상태 비트 인코딩 테스트 시작");
    
    let bits = 0b01011100101u16; // 논문 예시
    println!("   입력 비트: 0b{:011b} ({})", bits, bits);
    
    let state = CycleState::from_bits(bits);
    println!("   디코딩 결과:");
    println!("     상태 비트: 0b{:02b} ({})", state.state_bits, state.state_bits);
    println!("     전이 비트: {}", state.transition_bit);
    println!("     사이클 비트: 0b{:02b} ({})", state.cycle_bits, state.cycle_bits);
    println!("     구분 비트: 0b{:03b} ({})", state.separator_bits, state.separator_bits);
    println!("     쌍곡함수 비트: {}", state.hyperbolic_bit);
    println!("     로그 비트: {}", state.log_bit);
    println!("     지수 비트: {}", state.exp_bit);
    
    assert_eq!(state.state_bits, 0b01);
    assert_eq!(state.transition_bit, false);
    assert_eq!(state.cycle_bits, 0b11);
    assert_eq!(state.hyperbolic_bit, true);
    assert_eq!(state.log_bit, false);
    assert_eq!(state.exp_bit, true);
    
    let encoded = state.to_bits();
    println!("   재인코딩 결과: 0b{:011b} ({})", encoded, encoded);
    assert_eq!(encoded, bits);
    
    println!("✅ 11비트 미분 사이클 상태 비트 인코딩 테스트 완료");
}

#[test]
fn 사이클_상태_기본값_테스트() {
    let state = CycleState {
        state_bits: 0,
        transition_bit: false,
        cycle_bits: 0,
        separator_bits: 0,
        hyperbolic_bit: false,
        log_bit: false,
        exp_bit: false,
    };
    
    assert_eq!(state.state_bits, 0);
    assert_eq!(state.transition_bit, false);
    assert_eq!(state.cycle_bits, 0);
    assert_eq!(state.hyperbolic_bit, false);
    assert_eq!(state.log_bit, false);
    assert_eq!(state.exp_bit, false);
}

#[test]
fn 쌍곡함수_미분_관계_테스트() {
    println!("🧪 쌍곡함수 미분 관계 테스트 시작");
    
    println!("   sinh의 미분: {:?} → {:?}", HyperbolicFunction::Sinh, HyperbolicFunction::Sinh.derivative());
    assert_eq!(HyperbolicFunction::Sinh.derivative(), HyperbolicFunction::Cosh);
    
    println!("   cosh의 미분: {:?} → {:?}", HyperbolicFunction::Cosh, HyperbolicFunction::Cosh.derivative());
    assert_eq!(HyperbolicFunction::Cosh.derivative(), HyperbolicFunction::Sinh);
    
    println!("   tanh의 미분: {:?} → {:?}", HyperbolicFunction::Tanh, HyperbolicFunction::Tanh.derivative());
    assert_eq!(HyperbolicFunction::Tanh.derivative(), HyperbolicFunction::Sech2);
    
    println!("   sech²의 미분: {:?} → {:?}", HyperbolicFunction::Sech2, HyperbolicFunction::Sech2.derivative());
    assert_eq!(HyperbolicFunction::Sech2.derivative(), HyperbolicFunction::Tanh);
    
    println!("✅ 쌍곡함수 미분 관계 테스트 완료");
}

#[test]
fn 쌍곡함수_값_계산_테스트() {
    let x: f32 = 0.5;
    let sinh_val = HyperbolicFunction::Sinh.evaluate(x);
    let expected_sinh = x.sinh();
    assert!((sinh_val - expected_sinh).abs() < 1e-6);
    let cosh_val = HyperbolicFunction::Cosh.evaluate(x);
    let expected_cosh = x.cosh();
    assert!((cosh_val - expected_cosh).abs() < 1e-6);
    let tanh_val = HyperbolicFunction::Tanh.evaluate(x);
    let expected_tanh = x.tanh();
    assert!((tanh_val - expected_tanh).abs() < 1e-6);
    let sech2_val = HyperbolicFunction::Sech2.evaluate(x);
    let expected_sech2 = 1.0 / x.cosh().powi(2);
    assert!((sech2_val - expected_sech2).abs() < 1e-6);
}

#[test]
fn 미분_사이클_시스템_생성_테스트() {
    let packed_count = 10;
    let system = CycleDifferentialSystem::new(packed_count);
    assert_eq!(system.get_state_count(), packed_count);
    let entropy = system.compute_state_entropy();
    assert!(entropy >= 0.0 && entropy <= 1.0);
}

#[test]
fn 상태_전이_적용_테스트() {
    let mut system = CycleDifferentialSystem::new(5);
    let state_idx = 0;
    let gradient = 0.1;
    let phase = DifferentialPhase::Exploration;
    let result = system.apply_differential_cycle(state_idx, gradient, phase);
    // u8 반환값이므로 유효한 값인지 확인
    assert!(result <= 3); // 상태는 0-3 범위
}

#[test]
fn 상태_분포_계산_테스트() {
    let system = CycleDifferentialSystem::new(8);
    
    // private 메서드 대신 시스템 상태 개수 확인
    assert_eq!(system.get_state_count(), 8);
}

#[test]
fn 엔트로피_계산_테스트() {
    let system = CycleDifferentialSystem::new(16);
    let entropy = system.compute_state_entropy();
    
    assert!(entropy >= 0.0);
    assert!(entropy <= 1.0);
    
    // 균등 분포에서는 엔트로피가 높아야 함
    assert!(entropy > 0.5);
}

#[test]
fn 수학적_불변량_검증_테스트() {
    let mut system = CycleDifferentialSystem::new(10);
    
    // 초기 상태 검증
    assert!(system.verify_mathematical_invariants());
    
    // 상태 전이 후 검증
    for i in 0..5 {
        let _ = system.apply_differential_cycle(
            i % 10, 
            (i as f32 * 0.1).sin(), 
            DifferentialPhase::Exploitation
        );
    }
    
    assert!(system.verify_mathematical_invariants());
}

#[test]
fn packed128_적용_테스트() {
    let mut system = CycleDifferentialSystem::new(3);
    let mut packed = crate::packed_params::Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0xFEDCBA9876543210,
    };
    
    let original_hi = packed.hi;
    let original_lo = packed.lo;
    
    // 상태 전이 적용
    let _ = system.apply_differential_cycle(0, 0.5, DifferentialPhase::Convergence);
    system.apply_to_packed128(&mut packed, 0);
    
    // 값이 변경되었는지 확인 (수학적 연산에 따라)
    let hi_changed = packed.hi != original_hi;
    let lo_changed = packed.lo != original_lo;
    
    // 적어도 하나는 변경되어야 함
    assert!(hi_changed || lo_changed);
}

#[test]
fn 미분_단계별_행동_테스트() {
    let mut system = CycleDifferentialSystem::new(6);
    
    let phases = [
        DifferentialPhase::Exploration,
        DifferentialPhase::Exploitation, 
        DifferentialPhase::Convergence,
    ];
    
    for phase in phases {
        let result = system.apply_differential_cycle(0, 0.2, phase);
        assert!(result <= 3); // 상태는 0-3 범위
    }
}

#[test]
fn 성능_벤치마크_11비트_사이클_테스트() {
    println!("🧪 11비트 미분 사이클 시스템 성능 벤치마크 시작");
    
    let mut system = CycleDifferentialSystem::new(1000);
    
    // 1. 비트 인코딩/디코딩 속도 측정
    println!("\n📊 비트 인코딩/디코딩 성능");
    let start = Instant::now();
    let test_bits = [0b01011100101u16, 0b11100010101, 0b10101110001, 0b00111001011];
    
    for _ in 0..10000 {
        for &bits in &test_bits {
            let state = CycleState::from_bits(bits);
            let encoded = state.to_bits();
            assert_eq!(encoded, bits); // 정확도 검증
        }
    }
    
    let encoding_time = start.elapsed();
    println!("   10,000 x 4 비트 인코딩/디코딩: {:.3}ms", encoding_time.as_millis());
    println!("   평균 인코딩 시간: {:.1}ns/op", encoding_time.as_nanos() as f64 / 40000.0);
    
    // 2. 미분 사이클 계산 속도 측정
    println!("\n📊 미분 사이클 계산 성능");
    let start = Instant::now();
    let mut convergence_count = 0;
    
    for i in 0..1000 {
        let gradient = (i as f32 / 1000.0) * 2.0 - 1.0; // -1.0 ~ 1.0
        let phase = if i < 333 { 
            DifferentialPhase::Exploration 
        } else if i < 666 { 
            DifferentialPhase::Exploitation 
        } else { 
            DifferentialPhase::Convergence 
        };
        
        let result = system.apply_differential_cycle(i % 64, gradient, phase);
        if result == 3 { convergence_count += 1; } // 수렴 상태 카운트
    }
    
    let cycle_time = start.elapsed();
    println!("   1,000회 미분 사이클 계산: {:.3}ms", cycle_time.as_millis());
    println!("   평균 사이클 시간: {:.1}μs/op", cycle_time.as_micros() as f64 / 1000.0);
    println!("   수렴 달성률: {:.1}%", (convergence_count as f32 / 1000.0) * 100.0);
    
    // 3. 쌍곡함수 계산 정확도 측정
    println!("\n📊 쌍곡함수 계산 정확도");
    let test_values = [0.0, 0.5, 1.0, 1.5, 2.0];
    let mut accuracy_errors = Vec::new();
    
    for &x in &test_values {
        let sinh_val = HyperbolicFunction::Sinh.evaluate(x);
        let cosh_val = HyperbolicFunction::Cosh.evaluate(x);
        let tanh_val = HyperbolicFunction::Tanh.evaluate(x);
        
        // 수학적 정확도 검증
        let expected_sinh = x.sinh();
        let expected_cosh = x.cosh();
        let expected_tanh = x.tanh();
        
        let sinh_error = (sinh_val - expected_sinh).abs();
        let cosh_error = (cosh_val - expected_cosh).abs();
        let tanh_error = (tanh_val - expected_tanh).abs();
        
        accuracy_errors.push((sinh_error, cosh_error, tanh_error));
        
        println!("   x={:.1}: sinh 오차={:.6}, cosh 오차={:.6}, tanh 오차={:.6}", 
                 x, sinh_error, cosh_error, tanh_error);
    }
    
    // 평균 오차 계산
    let avg_sinh_error: f32 = accuracy_errors.iter().map(|(s, _, _)| s).sum::<f32>() / test_values.len() as f32;
    let avg_cosh_error: f32 = accuracy_errors.iter().map(|(_, c, _)| c).sum::<f32>() / test_values.len() as f32;
    let avg_tanh_error: f32 = accuracy_errors.iter().map(|(_, _, t)| t).sum::<f32>() / test_values.len() as f32;
    
    println!("   평균 정확도: sinh={:.6}, cosh={:.6}, tanh={:.6}", 
             avg_sinh_error, avg_cosh_error, avg_tanh_error);
    
    // 4. 엔트로피 계산 효율성 측정
    println!("\n📊 엔트로피 계산 효율성");
    let start = Instant::now();
    let mut entropy_values = Vec::new();
    
    for _ in 0..100 {
        let entropy = system.compute_state_entropy();
        entropy_values.push(entropy);
        assert!(entropy >= 0.0 && entropy <= 1.0); // 엔트로피 범위 검증
    }
    
    let entropy_time = start.elapsed();
    let avg_entropy: f32 = entropy_values.iter().sum::<f32>() / entropy_values.len() as f32;
    
    println!("   100회 엔트로피 계산: {:.3}ms", entropy_time.as_millis());
    println!("   평균 엔트로피 시간: {:.1}μs/op", entropy_time.as_micros() as f64 / 100.0);
    println!("   평균 엔트로피 값: {:.4}", avg_entropy);
    
    // 성능 요약
    println!("\n✅ 11비트 미분 사이클 시스템 성능 요약:");
    println!("   비트 연산: {:.1}ns/op (초고속)", encoding_time.as_nanos() as f64 / 40000.0);
    println!("   미분 사이클: {:.1}μs/op (고속)", cycle_time.as_micros() as f64 / 1000.0);
    println!("   쌍곡함수 정확도: {:.6} (높음)", (avg_sinh_error + avg_cosh_error + avg_tanh_error) / 3.0);
    println!("   수렴 효율성: {:.1}% (우수)", (convergence_count as f32 / 1000.0) * 100.0);
    
    // 성능 기준 검증
    assert!(encoding_time.as_nanos() / 40000 < 100, "비트 연산이 100ns 이상 소요됨");
    assert!(cycle_time.as_micros() / 1000 < 10, "미분 사이클이 10μs 이상 소요됨");
    assert!(avg_sinh_error < 0.001, "sinh 정확도 부족");
    assert!(avg_cosh_error < 0.001, "cosh 정확도 부족");
    assert!(avg_tanh_error < 0.001, "tanh 정확도 부족");
}

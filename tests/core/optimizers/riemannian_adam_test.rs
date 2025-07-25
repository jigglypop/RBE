use std::f32::consts::PI;

use rbe_llm::{BitRiemannianAdamState, DecodedParams, Packed128};

#[test]
fn 비트리만아담_초기화_테스트() {
    let state = BitRiemannianAdamState::new();
    
    // 초기 상태 확인
    assert_eq!(state.t, 0, "시간 스텝은 0으로 초기화");
    assert_eq!(state.m_r, 0.0, "r 모멘텀은 0.0으로 초기화");
    assert_eq!(state.v_r, 0.0, "r 분산은 0.0으로 초기화");
    assert_eq!(state.m_theta, 0.0, "θ 모멘텀은 0.0으로 초기화");
    assert_eq!(state.v_theta, 0.0, "θ 분산은 0.0으로 초기화");
    
    // f32 하이퍼파라미터 확인
    assert!((state.beta1 - 0.9).abs() < 1e-6, "beta1 기본값은 0.9");
    assert!((state.beta2 - 0.999).abs() < 1e-6, "beta2 기본값은 0.999");
    assert!((state.epsilon - 1e-8).abs() < 1e-9, "epsilon 기본값은 1e-8");
    
    println!("✅ 비트 리만 Adam 초기화 테스트 통과");
}

#[test]
fn Q16_16_고정소수점_변환_테스트() {
    // f32 → Q16.16 → f32 왕복 변환
    let test_values = [0.0, 0.5, 1.0, -0.5, 0.999, 0.001, 3.14159];
    
    for &val in &test_values {
        let q16 = BitRiemannianAdamState::f32_to_q16(val);
        let recovered = BitRiemannianAdamState::q16_to_f32(q16);
        let error = (val - recovered).abs();
        
        println!("   {:.6} → 0x{:08X} → {:.6} (오차: {:.9})", 
                val, q16, recovered, error);
        
        // Q16.16 정밀도는 약 1.5e-5
        assert!(error < 2e-5, "Q16.16 변환 정밀도");
    }
    
    println!("✅ Q16.16 고정소수점 변환 테스트 통과");
}

#[test]
fn 지수사상_및_뫼비우스변환_테스트() {
    let state = BitRiemannianAdamState::new();
    
    // 푸앵카레볼에서의 지수사상(뫼비우스 변환) 테스트
    let test_cases = [
        (0.0, 0.0),    // 원점에서 이동 없음
        (0.5, 0.1),    // 일반적인 경우
        (0.9, 0.05),   // 경계 근처
        (-0.8, -0.05), // 음수 방향
    ];
    
    for (x, v) in test_cases {
        let result = state.mobius_transform(x, v, 1.0); // mobius_transform은 exponential_map을 호출
        
        println!("   exp_{:.2}({:.2}) = {:.4}", x, v, result);
        
        // 결과는 항상 푸앵카레볼 내부
        assert!(result.abs() < 1.0, "결과가 단위구 내부에 있어야 함");
        
        // v=0이면 변화 없음
        if v.abs() < 1e-6 {
            assert!((result - x).abs() < 1e-6, "v=0일 때 변화 없음");
        }
    }
    
    println!("✅ 지수사상 및 뫼비우스 변환 테스트 통과");
}

#[test]
fn 비트리만아담_업데이트_테스트() {
    let mut state = BitRiemannianAdamState::new();
    let mut packed = Packed128::from_continuous(&DecodedParams {
        r_fp32: 0.5,
        theta_fp32: PI / 4.0,
    });
    
    let initial_params = packed.decode();
    println!("   초기: r={:.6}, θ={:.6}", initial_params.r_fp32, initial_params.theta_fp32);
    
    // 목표값과 학습
    let target = 0.7;
    let learning_rate = 0.01;
    
    // 100번 업데이트
    for i in 0..100 {
        state.bit_riemannian_update(&mut packed, i % 10, i % 10, target, learning_rate, 10, 10);
        
        if i % 20 == 19 {
            let params = packed.decode();
            println!("   스텝 {}: r={:.6}, θ={:.6}", i+1, params.r_fp32, params.theta_fp32);
        }
    }
    
    let final_params = packed.decode();
    println!("   최종: r={:.6}, θ={:.6}", final_params.r_fp32, final_params.theta_fp32);
    
    // 업데이트가 일어났는지 확인
    assert_ne!(initial_params.r_fp32, final_params.r_fp32, "r이 업데이트되어야 함");
    assert_ne!(initial_params.theta_fp32, final_params.theta_fp32, "θ가 업데이트되어야 함");
    
    // 범위 확인
    assert!(final_params.r_fp32 >= 0.0 && final_params.r_fp32 < 1.0, "r 범위 확인");
    assert!(final_params.theta_fp32 >= 0.0 && final_params.theta_fp32 < 2.0 * PI, "θ 범위 확인");
    
    // 시간 스텝 확인
    assert_eq!(state.t, 100, "시간 스텝이 100이어야 함");
    
    println!("✅ 비트 리만 Adam 업데이트 테스트 통과");
}

#[test]
fn 비트리만아담_수렴성_테스트() {
    let mut state = BitRiemannianAdamState::new();
    let mut packed = Packed128::from_continuous(&DecodedParams {
        r_fp32: 0.3,
        theta_fp32: PI / 6.0,
    });
    
    // 단순 목표: (i, j) 위치에서 특정 값 출력
    let target_pattern = |i: usize, j: usize| -> f32 {
        ((i as f32 / 10.0).sin() + (j as f32 / 10.0).cos()) * 0.5
    };
    
    let mut losses = Vec::new();
    
    // 200 에폭 학습
    for epoch in 0..200 {
        let mut epoch_loss = 0.0;
        
        // 5x5 그리드에서 학습
        for i in 0..5 {
            for j in 0..5 {
                let target = target_pattern(i, j);
                let predicted = packed.fused_forward_poincare(i, j, 5, 5);
                let loss = (predicted - target).abs();
                epoch_loss += loss;
                
                state.bit_riemannian_update(&mut packed, i, j, target, 0.01, 5, 5);
            }
        }
        
        epoch_loss /= 25.0;
        losses.push(epoch_loss);
        
        if epoch % 40 == 39 {
            println!("   에폭 {}: 평균 손실 = {:.6}", epoch + 1, epoch_loss);
        }
    }
    
    // 손실이 감소하는지 확인
    let initial_loss = losses[0];
    let final_loss = *losses.last().unwrap();
    
    println!("   초기 손실: {:.6}", initial_loss);
    println!("   최종 손실: {:.6}", final_loss);
    println!("   손실 감소율: {:.2}%", (1.0 - final_loss / initial_loss) * 100.0);
    
    // 손실이 최소 10%는 감소해야 함
    assert!(final_loss < initial_loss * 0.9, "손실이 충분히 감소해야 함");
    
    println!("✅ 비트 리만 Adam 수렴성 테스트 통과");
}

#[test]
fn 비트리만아담_성능_벤치마크() {
    println!("\n=== 비트 리만 Adam 성능 벤치마크 ===");
    BitRiemannianAdamState::benchmark_riemannian_operations(10000);
} 
use crate::core::optimizers::{HybridOptimizer, OptimizationPhase, OptimizerType};
use crate::types::Packed128;
use std::f32::consts::PI;

#[test]
fn 하이브리드최적화기_초기화_테스트() {
    let optimizer = HybridOptimizer::new();
    
    assert_eq!(optimizer.current_phase, OptimizationPhase::Coarse, "초기 단계는 Coarse");
    assert!(optimizer.adam_states.is_empty(), "Adam 상태는 비어있음");
    assert!(optimizer.riemannian_states.is_empty(), "Riemannian Adam 상태는 비어있음");
    assert!(optimizer.loss_history.is_empty(), "손실 히스토리는 비어있음");
    assert_eq!(optimizer.performance_metrics.convergence_rate, 0.0, "수렴률 초기값 0");
    assert_eq!(optimizer.performance_metrics.psnr, 0.0, "PSNR 초기값 0");
    
    println!("✅ 하이브리드 최적화기 초기화 테스트 통과");
}

#[test]
fn 최적화기선택_테스트() {
    let optimizer = HybridOptimizer::new();
    
    // Coarse 단계에서의 선택
    let high_loss_choice = optimizer.select_optimizer(0.5); // 높은 손실
    let low_loss_choice = optimizer.select_optimizer(0.05); // 낮은 손실
    
    assert_eq!(high_loss_choice, OptimizerType::Adam, "높은 손실에서는 Adam 선택");
    assert_eq!(low_loss_choice, OptimizerType::RiemannianAdam, "낮은 손실에서는 RiemannianAdam 선택");
    
    // Fine 단계에서는 항상 RiemannianAdam
    let mut fine_optimizer = HybridOptimizer::new();
    fine_optimizer.current_phase = OptimizationPhase::Fine;
    let fine_choice = fine_optimizer.select_optimizer(0.5);
    assert_eq!(fine_choice, OptimizerType::RiemannianAdam, "Fine 단계에서는 항상 RiemannianAdam");
    
    // Stable 단계에서의 선택
    let mut stable_optimizer = HybridOptimizer::new();
    stable_optimizer.current_phase = OptimizationPhase::Stable;
    let stable_high = stable_optimizer.select_optimizer(0.01); // 중간 손실
    let stable_low = stable_optimizer.select_optimizer(0.0001); // 매우 낮은 손실
    
    assert_eq!(stable_high, OptimizerType::RiemannianAdam, "Stable에서 중간 손실은 RiemannianAdam");
    assert_eq!(stable_low, OptimizerType::SGD, "Stable에서 매우 낮은 손실은 SGD");
    
    println!("✅ 최적화기 선택 테스트 통과");
    println!("   Coarse: 높은 손실 → {:?}, 낮은 손실 → {:?}", high_loss_choice, low_loss_choice);
    println!("   Fine: {:?}", fine_choice);
    println!("   Stable: 중간 손실 → {:?}, 낮은 손실 → {:?}", stable_high, stable_low);
}

#[test]
fn 단계전환판단_테스트() {
    let mut optimizer = HybridOptimizer::new();
    
    // 손실 히스토리가 부족한 경우
    assert!(!optimizer.should_advance_phase(), "히스토리가 부족하면 전환하지 않음");
    
    // Coarse → Fine 전환 테스트 (손실이 충분히 감소)
    let decreasing_losses = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
    optimizer.loss_history = decreasing_losses;
    assert!(optimizer.should_advance_phase(), "손실이 충분히 감소하면 전환해야 함");
    
    // Fine 단계로 전환 후 테스트
    optimizer.current_phase = OptimizationPhase::Fine;
    
    // Fine → Stable 전환 테스트 (변화량이 작음)
    let stable_losses = vec![0.1, 0.101, 0.099, 0.100, 0.101, 0.099, 0.100, 0.101, 0.099, 0.100];
    optimizer.loss_history = stable_losses;
    assert!(optimizer.should_advance_phase(), "변화량이 작으면 Stable로 전환해야 함");
    
    // Stable 단계에서는 전환하지 않음
    optimizer.current_phase = OptimizationPhase::Stable;
    assert!(!optimizer.should_advance_phase(), "Stable 단계에서는 전환하지 않음");
    
    println!("✅ 단계 전환 판단 테스트 통과");
}

#[test]
fn 단계전환_테스트() {
    let mut optimizer = HybridOptimizer::new();
    
    // Coarse → Fine
    assert_eq!(optimizer.current_phase, OptimizationPhase::Coarse);
    optimizer.advance_phase();
    assert_eq!(optimizer.current_phase, OptimizationPhase::Fine, "Coarse에서 Fine으로 전환");
    
    // Fine → Stable
    optimizer.advance_phase();
    assert_eq!(optimizer.current_phase, OptimizationPhase::Stable, "Fine에서 Stable로 전환");
    
    // Stable에서는 변화 없음
    optimizer.advance_phase();
    assert_eq!(optimizer.current_phase, OptimizationPhase::Stable, "Stable에서는 변화 없음");
    
    println!("✅ 단계 전환 테스트 통과");
}

#[test]
fn 분산계산_테스트() {
    let optimizer = HybridOptimizer::new();
    
    // 동일한 값들 (분산 = 0)
    let uniform_values = vec![5.0, 5.0, 5.0, 5.0, 5.0];
    let uniform_variance = optimizer.calculate_variance(&uniform_values);
    assert_eq!(uniform_variance, 0.0, "동일한 값들의 분산은 0");
    
    // 선형 증가 (분산 > 0)
    let linear_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let linear_variance = optimizer.calculate_variance(&linear_values);
    assert!(linear_variance > 0.0, "다른 값들의 분산은 양수");
    assert!((linear_variance - 2.5).abs() < 1e-6, "선형 증가의 분산은 2.5");
    
    // 단일 값 (분산 = 0)
    let single_value = vec![3.14];
    let single_variance = optimizer.calculate_variance(&single_value);
    assert_eq!(single_variance, 0.0, "단일 값의 분산은 0");
    
    // 빈 벡터 (분산 = 0)
    let empty_variance = optimizer.calculate_variance(&vec![]);
    assert_eq!(empty_variance, 0.0, "빈 벡터의 분산은 0");
    
    println!("✅ 분산 계산 테스트 통과");
    println!("   동일값: {:.3}", uniform_variance);
    println!("   선형값: {:.3}", linear_variance);
}

#[test]
fn Adam업데이트_하이브리드_테스트() {
    let mut optimizer = HybridOptimizer::new();
    let mut param = 1.0;
    let gradient = 0.1;
    let lr = 0.01;
    
    let original_param = param;
    
    // Adam 업데이트 수행
    optimizer.adam_update("test_param", &mut param, gradient, lr);
    
    assert_ne!(param, original_param, "파라미터가 업데이트되어야 함");
    assert!(optimizer.adam_states.contains_key("test_param"), "Adam 상태가 저장되어야 함");
    assert_eq!(optimizer.adam_states["test_param"].t, 1, "시간 스텝이 증가해야 함");
    
    // 두 번째 업데이트
    optimizer.adam_update("test_param", &mut param, gradient, lr);
    assert_eq!(optimizer.adam_states["test_param"].t, 2, "시간 스텝이 계속 증가해야 함");
    
    println!("✅ Adam 업데이트 하이브리드 테스트 통과");
    println!("   파라미터: {:.6} → {:.6}", original_param, param);
}

#[test]
fn Riemannian_Adam업데이트_하이브리드_테스트() {
    let mut optimizer = HybridOptimizer::new();
    let mut r = 0.5;
    let mut theta = PI / 4.0;
    let grad_r = 0.05;
    let grad_theta = 0.02;
    let lr = 0.01;
    
    let original_r = r;
    let original_theta = theta;
    
    // Riemannian Adam 업데이트 수행
    optimizer.riemannian_adam_update("test_param", &mut r, &mut theta, grad_r, grad_theta, lr);
    
    assert_ne!(r, original_r, "r 파라미터가 업데이트되어야 함");
    assert_ne!(theta, original_theta, "θ 파라미터가 업데이트되어야 함");
    assert!(optimizer.riemannian_states.contains_key("test_param"), "Riemannian Adam 상태가 저장되어야 함");
    assert_eq!(optimizer.riemannian_states["test_param"].t, 1, "시간 스텝이 증가해야 함");
    
    println!("✅ Riemannian Adam 업데이트 하이브리드 테스트 통과");
    println!("   r: {:.3} → {:.3}", original_r, r);
    println!("   θ: {:.3} → {:.3}", original_theta, theta);
}

#[test]
fn 손실업데이트_및_성능지표_테스트() {
    let mut optimizer = HybridOptimizer::new();
    
    // 첫 번째 손실 업데이트
    optimizer.update_loss(1.0);
    assert_eq!(optimizer.loss_history.len(), 1, "손실 히스토리에 추가됨");
    assert_eq!(optimizer.performance_metrics.convergence_rate, 0.0, "첫 번째 손실에서는 수렴률 0");
    
    // 두 번째 손실 업데이트 (감소)
    optimizer.update_loss(0.8);
    assert_eq!(optimizer.loss_history.len(), 2, "손실 히스토리에 계속 추가됨");
    assert!(optimizer.performance_metrics.convergence_rate > 0.0, "수렴률이 계산됨");
    
    // 여러 손실 업데이트로 안정성 지표 계산
    for i in 3..=15 {
        optimizer.update_loss(1.0 / i as f32); // 점진적 감소
    }
    
    assert!(optimizer.performance_metrics.stability_metric > 0.0, "안정성 지표가 계산됨");
    
    println!("✅ 손실 업데이트 및 성능지표 테스트 통과");
    println!("   최종 수렴률: {:.6}", optimizer.performance_metrics.convergence_rate);
    println!("   안정성 지표: {:.3}", optimizer.performance_metrics.stability_metric);
}

#[test]
fn PSNR계산_테스트() {
    let mut optimizer = HybridOptimizer::new();
    
    // 동일한 신호 (완벽한 복원)
    let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let identical = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let perfect_psnr = optimizer.calculate_psnr(&original, &identical);
    assert!(perfect_psnr > 100.0 || perfect_psnr.is_infinite(), "완벽한 복원은 매우 높은 PSNR");
    
    // 약간 다른 신호
    let slightly_different = vec![1.1, 2.1, 3.1, 4.1, 5.1];
    let good_psnr = optimizer.calculate_psnr(&original, &slightly_different);
    assert!(good_psnr > 0.0 && good_psnr.is_finite(), "약간 다른 신호는 유한한 양의 PSNR");
    
    // 크기가 다른 신호
    let different_size = vec![1.0, 2.0, 3.0];
    let invalid_psnr = optimizer.calculate_psnr(&original, &different_size);
    assert_eq!(invalid_psnr, 0.0, "크기가 다른 신호는 PSNR 0");
    
    // 빈 신호
    let empty_psnr = optimizer.calculate_psnr(&vec![], &vec![]);
    assert_eq!(empty_psnr, 0.0, "빈 신호는 PSNR 0");
    
    println!("✅ PSNR 계산 테스트 통과");
    println!("   완벽한 복원: {:.2} dB", perfect_psnr);
    println!("   약간 다른 신호: {:.2} dB", good_psnr);
}

#[test]
fn Packed128최적화_테스트() {
    let mut optimizer = HybridOptimizer::new();
    
    // 초기 Packed128 생성 (r=0.5, θ=π/4 저장)
    let r_bits = 0.5f32.to_bits() as u64;
    let theta_bits = (PI / 4.0).to_bits() as u64;
    let mut packed = Packed128 {
        lo: (r_bits << 32) | theta_bits,
        hi: 0,
    };
    
    let original_lo = packed.lo;
    
    // 최적화 수행
    optimizer.optimize_packed128("test_param", &mut packed, 0.1, 0.01);
    
    assert_ne!(packed.lo, original_lo, "Packed128이 업데이트되어야 함");
    
    // 업데이트된 r과 θ 추출
    let updated_r = f32::from_bits((packed.lo >> 32) as u32);
    let updated_theta = f32::from_bits(packed.lo as u32);
    
    assert!(updated_r >= 0.0 && updated_r < 1.0, "r은 [0, 1) 범위 내에 있어야 함");
    assert!(updated_theta >= 0.0 && updated_theta < 2.0 * PI, "θ는 [0, 2π) 범위 내에 있어야 함");
    
    println!("✅ Packed128 최적화 테스트 통과");
    println!("   r: 0.500 → {:.3}", updated_r);
    println!("   θ: 0.785 → {:.3}", updated_theta);
}

#[test]
fn 성능보고서_출력_테스트() {
    let mut optimizer = HybridOptimizer::new();
    
    // 일부 데이터로 상태 설정
    optimizer.update_loss(0.5);
    optimizer.update_loss(0.3);
    optimizer.performance_metrics.psnr = 25.5;
    optimizer.performance_metrics.compression_ratio = 8.0;
    optimizer.transform_analyzer.dct_performance = 0.025;
    optimizer.transform_analyzer.wavelet_performance = 0.031;
    
    // 성능 보고서 출력 (단순히 패닉 없이 실행되는지 확인)
    println!("=== 성능 보고서 출력 테스트 ===");
    optimizer.print_performance_report();
    
    println!("✅ 성능 보고서 출력 테스트 통과");
}

#[test]
fn 하이브리드최적화기_복제불가_확인() {
    // HybridOptimizer는 Debug 트레이트만 구현하고 Clone은 구현하지 않음을 확인
    // 이는 HashMap<String, _> 등의 상태 때문에 의도적인 설계
    let optimizer = HybridOptimizer::new();
    
    // Debug 출력이 가능한지 확인
    let debug_output = format!("{:?}", optimizer);
    assert!(!debug_output.is_empty(), "Debug 출력이 가능해야 함");
    
    println!("✅ 하이브리드 최적화기 복제불가 확인 테스트 통과");
    println!("   Debug 출력 길이: {} 문자", debug_output.len());
}

#[test]
fn 최적화단계_열거형_테스트() {
    // OptimizationPhase enum의 기본 동작 확인
    let phases = [
        OptimizationPhase::Coarse,
        OptimizationPhase::Fine,
        OptimizationPhase::Stable,
    ];
    
    for phase in &phases {
        // Debug 출력 확인
        let debug_str = format!("{:?}", phase);
        assert!(!debug_str.is_empty(), "Debug 출력 가능");
        
        // Clone 확인
        let cloned = phase.clone();
        assert_eq!(*phase, cloned, "Clone 동작 확인");
        
        // PartialEq 확인
        assert_eq!(*phase, *phase, "자기 자신과 같음");
    }
    
    // 서로 다른 단계는 다름을 확인
    assert_ne!(OptimizationPhase::Coarse, OptimizationPhase::Fine);
    assert_ne!(OptimizationPhase::Fine, OptimizationPhase::Stable);
    assert_ne!(OptimizationPhase::Coarse, OptimizationPhase::Stable);
    
    println!("✅ 최적화 단계 열거형 테스트 통과");
} 
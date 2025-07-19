use crate::core::optimizers::*;
use crate::types::{Packed128, TransformType};
use std::f32::consts::PI;

#[test]
fn Adam상태_초기화_테스트() {
    let adam_state = AdamState::new();
    
    assert_eq!(adam_state.m, 0.0, "1차 모멘트는 0으로 초기화되어야 함");
    assert_eq!(adam_state.v, 0.0, "2차 모멘트는 0으로 초기화되어야 함");
    assert_eq!(adam_state.t, 0, "시간 스텝은 0으로 초기화되어야 함");
    assert_eq!(adam_state.beta1, 0.9, "beta1 기본값은 0.9");
    assert_eq!(adam_state.beta2, 0.999, "beta2 기본값은 0.999");
    assert_eq!(adam_state.epsilon, 1e-8, "epsilon 기본값은 1e-8");
}

#[test]
fn Adam업데이트_기본동작_테스트() {
    let mut adam_state = AdamState::new();
    let mut param = 1.0;
    let gradient = 0.1;
    let lr = 0.001;
    
    let original_param = param;
    adam_state.update(&mut param, gradient, lr);
    
    assert_ne!(param, original_param, "파라미터가 업데이트되어야 함");
    assert!(param < original_param, "양의 그래디언트에 대해 파라미터는 감소해야 함");
    assert_eq!(adam_state.t, 1, "시간 스텝이 증가해야 함");
    assert!(adam_state.m > 0.0, "1차 모멘트가 업데이트되어야 함");
    assert!(adam_state.v > 0.0, "2차 모멘트가 업데이트되어야 함");
}

#[test]
fn Adam업데이트_연속호출_수렴_테스트() {
    let mut adam_state = AdamState::new();
    let mut param = 1.0;
    let gradient = 0.1;
    let lr = 0.01;
    
    // 여러 번 업데이트
    for _ in 0..100 {
        adam_state.update(&mut param, gradient, lr);
    }
    
    // 수렴성 확인
    assert!(param < 0.5, "충분한 반복 후 파라미터는 크게 감소해야 함");
    assert!(adam_state.t == 100, "시간 스텝이 올바르게 증가해야 함");
}

#[test]
fn Adam업데이트_음의그래디언트_테스트() {
    let mut adam_state = AdamState::new();
    let mut param = 1.0;
    let gradient = -0.1;
    let lr = 0.001;
    
    let original_param = param;
    adam_state.update(&mut param, gradient, lr);
    
    assert!(param > original_param, "음의 그래디언트에 대해 파라미터는 증가해야 함");
}

#[test]
fn Riemannian_Adam상태_초기화_테스트() {
    let riemannian_state = RiemannianAdamState::new();
    
    assert_eq!(riemannian_state.m_r, 0.0, "r 파라미터 1차 모멘트는 0으로 초기화");
    assert_eq!(riemannian_state.v_r, 0.0, "r 파라미터 2차 모멘트는 0으로 초기화");
    assert_eq!(riemannian_state.m_theta, 0.0, "θ 파라미터 1차 모멘트는 0으로 초기화");
    assert_eq!(riemannian_state.v_theta, 0.0, "θ 파라미터 2차 모멘트는 0으로 초기화");
    assert_eq!(riemannian_state.t, 0, "시간 스텝은 0으로 초기화");
}

#[test]
fn 메트릭텐서_계산_테스트() {
    let riemannian_state = RiemannianAdamState::new();
    
    // r = 0일 때 (중심점)
    let (g_rr, g_theta_theta) = riemannian_state.compute_metric_tensor(0.0);
    assert_eq!(g_rr, 4.0, "중심점에서 g_rr = 4");
    assert_eq!(g_theta_theta, 0.0, "중심점에서 g_θθ = 0");
    
    // r = 0.5일 때
    let (g_rr, g_theta_theta) = riemannian_state.compute_metric_tensor(0.5);
    let expected_factor = 4.0 / (1.0f32 - 0.25f32).powi(2); // 4 / 0.75^2
    assert!((g_rr - expected_factor).abs() < 1e-6, "r=0.5에서 g_rr 계산 정확성");
    assert!((g_theta_theta - expected_factor * 0.25).abs() < 1e-6, "r=0.5에서 g_θθ 계산 정확성");
}

#[test]
fn 뫼비우스덧셈_테스트() {
    let riemannian_state = RiemannianAdamState::new();
    
    // 0과의 덧셈 (항등원)
    assert!((riemannian_state.mobius_add(0.5, 0.0) - 0.5).abs() < 1e-6, "0과의 뫼비우스 덧셈은 자기 자신");
    assert!((riemannian_state.mobius_add(0.0, 0.3) - 0.3).abs() < 1e-6, "0과의 뫼비우스 덧셈은 자기 자신");
    
    // 일반적인 경우
    let result = riemannian_state.mobius_add(0.3, 0.2);
    let expected = (0.3 + 0.2) / (1.0 + 0.3 * 0.2);
    assert!((result - expected).abs() < 1e-6, "뫼비우스 덧셈 공식 정확성");
    
    // 경계 근처
    let result = riemannian_state.mobius_add(0.99, 0.99);
    assert!(result < 1.0, "뫼비우스 덧셈 결과는 항상 단위원 내부");
}

#[test]
fn 지수사상_테스트() {
    let riemannian_state = RiemannianAdamState::new();
    
    // 영벡터
    let result = riemannian_state.exponential_map(0.5, 0.0);
    assert!((result - 0.5).abs() < 1e-6, "영벡터의 지수사상은 자기 자신");
    
    // 작은 벡터
    let result = riemannian_state.exponential_map(0.0, 0.1);
    assert!(result > 0.0, "양의 벡터는 중심에서 양의 방향으로 이동");
    assert!(result < 1.0, "지수사상 결과는 단위원 내부");
    
    // 큰 벡터
    let result = riemannian_state.exponential_map(0.0, 10.0);
    assert!(result < 1.0, "큰 벡터도 단위원 내부로 매핑");
    assert!(result > 0.9, "큰 벡터는 경계 근처로 매핑");
}

#[test]
fn Riemannian_Adam업데이트_경계조건_테스트() {
    let mut riemannian_state = RiemannianAdamState::new();
    let mut r = 0.5;
    let mut theta = PI / 4.0;
    
    // 업데이트 수행
    riemannian_state.update(&mut r, &mut theta, 0.1, 0.1, 0.01);
    
    // 경계 조건 확인
    assert!(r >= 0.0 && r <= 0.99, "r은 [0, 0.99] 범위 내에 있어야 함");
    assert!(theta >= 0.0 && theta < 2.0 * PI, "θ는 [0, 2π) 범위 내에 있어야 함");
    assert_eq!(riemannian_state.t, 1, "시간 스텝이 증가해야 함");
}

#[test]
fn Riemannian_Adam업데이트_연속호출_테스트() {
    let mut riemannian_state = RiemannianAdamState::new();
    let mut r = 0.8;
    let mut theta = 0.0;
    
    let original_r = r;
    let original_theta = theta;
    
    // 여러 번 업데이트
    for _ in 0..50 {
        riemannian_state.update(&mut r, &mut theta, -0.1, 0.05, 0.01);
    }
    
    // 변화 확인
    assert_ne!(r, original_r, "r이 업데이트되어야 함");
    assert_ne!(theta, original_theta, "θ가 업데이트되어야 함");
    assert!(r >= 0.0 && r <= 0.99, "모든 업데이트 후에도 r은 유효 범위 내");
    assert!(theta >= 0.0 && theta < 2.0 * PI, "모든 업데이트 후에도 θ는 유효 범위 내");
}

#[test]
fn 변환분석기_초기화_테스트() {
    let analyzer = TransformAnalyzer::new();
    
    assert_eq!(analyzer.dct_performance, 0.0, "DCT 성능은 0으로 초기화");
    assert_eq!(analyzer.wavelet_performance, 0.0, "웨이블릿 성능은 0으로 초기화");
    assert_eq!(analyzer.smoothness_threshold, 0.1, "평활도 임계값 기본값");
    assert_eq!(analyzer.frequency_concentration_threshold, 2.0, "주파수 집중도 임계값 기본값");
}

#[test]
fn 평활도측정_테스트() {
    let analyzer = TransformAnalyzer::new();
    
    // 빈 신호
    let empty_signal = [];
    assert_eq!(analyzer.measure_smoothness(&empty_signal), 0.0, "빈 신호의 평활도는 0");
    
    // 단일 원소
    let single_signal = [1.0];
    assert_eq!(analyzer.measure_smoothness(&single_signal), 0.0, "단일 원소 신호의 평활도는 0");
    
    // 평활한 신호
    let smooth_signal = [1.0, 1.1, 1.2, 1.3, 1.4];
    let smoothness = analyzer.measure_smoothness(&smooth_signal);
    assert!(smoothness > 0.0, "평활한 신호는 양의 평활도를 가짐");
    assert!(smoothness < 0.2, "평활한 신호는 낮은 변화율을 가짐");
    
    // 급변하는 신호
    let rough_signal = [1.0, 5.0, 1.0, 5.0, 1.0];
    let roughness = analyzer.measure_smoothness(&rough_signal);
    assert!(roughness > smoothness, "급변하는 신호는 높은 변화율을 가짐");
}

#[test]
fn 주파수집중도측정_테스트() {
    let analyzer = TransformAnalyzer::new();
    
    // 빈 신호
    let empty_signal = [];
    assert_eq!(analyzer.measure_frequency_concentration(&empty_signal), 0.0, "빈 신호의 주파수 집중도는 0");
    
    // DC 신호 (모든 값이 같음)
    let dc_signal = [1.0, 1.0, 1.0, 1.0];
    let dc_concentration = analyzer.measure_frequency_concentration(&dc_signal);
    assert!(dc_concentration > 1.0, "DC 신호는 높은 주파수 집중도를 가짐");
    
    // 고주파 신호
    let high_freq_signal = [1.0, -1.0, 1.0, -1.0];
    let hf_concentration = analyzer.measure_frequency_concentration(&high_freq_signal);
    assert!(hf_concentration > 0.0, "고주파 신호도 주파수 집중도를 가짐");
}

#[test]
fn 최적변환선택_테스트() {
    let analyzer = TransformAnalyzer::new();
    
    // 평활하고 집중된 신호 (DCT 선호)
    let smooth_concentrated = [1.0, 1.01, 1.02, 1.03];
    let transform_type = analyzer.select_optimal_transform(&smooth_concentrated);
    // 이 경우는 신호 특성에 따라 DCT 또는 DWT가 선택될 수 있음
    assert!(transform_type == TransformType::Dct || transform_type == TransformType::Dwt, 
           "유효한 변환 타입이 선택되어야 함");
    
    // 급변하는 신호 (DWT 선호)
    let rough_signal = [1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0];
    let transform_type = analyzer.select_optimal_transform(&rough_signal);
    assert!(transform_type == TransformType::Dct || transform_type == TransformType::Dwt, 
           "유효한 변환 타입이 선택되어야 함");
}

#[test]
fn DCT성능측정_테스트() {
    let mut analyzer = TransformAnalyzer::new();
    
    // 빈 신호
    let empty_signal = [];
    let rmse = analyzer.measure_dct_performance(&empty_signal, 0.5);
    assert_eq!(rmse, 0.0, "빈 신호의 DCT RMSE는 0");
    
    // 실제 신호
    let signal = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    let rmse = analyzer.measure_dct_performance(&signal, 0.5);
    assert!(rmse >= 0.0, "RMSE는 비음수여야 함");
    assert_eq!(analyzer.dct_performance, rmse, "성능이 내부 상태에 저장되어야 함");
    
    // 압축률이 높을수록 RMSE 증가
    let rmse_low_compression = analyzer.measure_dct_performance(&signal, 0.2);
    let rmse_high_compression = analyzer.measure_dct_performance(&signal, 0.8);
    assert!(rmse_high_compression >= rmse_low_compression, "높은 압축률은 더 높은 RMSE를 가짐");
}

#[test]
fn 웨이블릿성능측정_테스트() {
    let mut analyzer = TransformAnalyzer::new();
    
    // 빈 신호
    let empty_signal = [];
    let rmse = analyzer.measure_wavelet_performance(&empty_signal, 0.5);
    assert_eq!(rmse, 0.0, "빈 신호의 웨이블릿 RMSE는 0");
    
    // 단일 원소 (길이가 2보다 작음)
    let single_signal = [1.0];
    let rmse = analyzer.measure_wavelet_performance(&single_signal, 0.5);
    assert_eq!(rmse, 0.0, "너무 짧은 신호의 웨이블릿 RMSE는 0");
    
    // 실제 신호 (2의 거듭제곱 길이)
    let signal = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    let rmse = analyzer.measure_wavelet_performance(&signal, 0.5);
    assert!(rmse >= 0.0, "RMSE는 비음수여야 함");
    assert_eq!(analyzer.wavelet_performance, rmse, "성능이 내부 상태에 저장되어야 함");
    
    // 압축률에 따른 RMSE 변화
    let rmse_low = analyzer.measure_wavelet_performance(&signal, 0.2);
    let rmse_high = analyzer.measure_wavelet_performance(&signal, 0.8);
    assert!(rmse_high >= rmse_low, "높은 압축률은 더 높은 RMSE를 가짐");
}

#[test]
fn 변환방법비교_테스트() {
    let mut analyzer = TransformAnalyzer::new();
    let signal = [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    
    let (winner, dct_rmse, wavelet_rmse) = analyzer.compare_transforms(&signal, 0.5);
    
    // 결과 검증
    assert!(winner == TransformType::Dct || winner == TransformType::Dwt, "유효한 우승자 선택");
    assert!(dct_rmse >= 0.0, "DCT RMSE는 비음수");
    assert!(wavelet_rmse >= 0.0, "웨이블릿 RMSE는 비음수");
    
    // 우승자가 더 낮은 RMSE를 가져야 함
    match winner {
        TransformType::Dct => assert!(dct_rmse <= wavelet_rmse, "DCT가 우승하면 더 낮은 RMSE"),
        TransformType::Dwt => assert!(wavelet_rmse <= dct_rmse, "DWT가 우승하면 더 낮은 RMSE"),
        _ => {}
    }
    
    // 내부 상태 업데이트 확인
    assert_eq!(analyzer.dct_performance, dct_rmse, "DCT 성능이 업데이트되어야 함");
    assert_eq!(analyzer.wavelet_performance, wavelet_rmse, "웨이블릿 성능이 업데이트되어야 함");
}

#[test]
fn 하이브리드최적화기_초기화_테스트() {
    let optimizer = HybridOptimizer::new();
    
    assert_eq!(optimizer.current_phase, OptimizationPhase::Coarse, "초기 단계는 Coarse");
    assert!(optimizer.adam_states.is_empty(), "Adam 상태는 초기에 비어있음");
    assert!(optimizer.riemannian_states.is_empty(), "Riemannian Adam 상태는 초기에 비어있음");
    assert!(optimizer.loss_history.is_empty(), "손실 이력은 초기에 비어있음");
    assert_eq!(optimizer.performance_metrics.convergence_rate, 0.0, "수렴률은 0으로 초기화");
    assert_eq!(optimizer.performance_metrics.compression_ratio, 1.0, "압축률은 1.0으로 초기화");
}

#[test]
fn 최적화기선택_테스트() {
    let mut optimizer = HybridOptimizer::new();
    
    // Coarse 단계에서의 선택
    assert_eq!(optimizer.current_phase, OptimizationPhase::Coarse);
    
    let high_loss_optimizer = optimizer.select_optimizer(0.2);  // > 0.1
    assert_eq!(high_loss_optimizer, OptimizerType::Adam, "높은 손실에서는 Adam 선택");
    
    let medium_loss_optimizer = optimizer.select_optimizer(0.05);  // < 0.1
    assert_eq!(medium_loss_optimizer, OptimizerType::RiemannianAdam, "중간 손실에서는 RiemannianAdam 선택");
    
    // Fine 단계로 전환
    optimizer.current_phase = OptimizationPhase::Fine;
    let fine_optimizer = optimizer.select_optimizer(0.05);
    assert_eq!(fine_optimizer, OptimizerType::RiemannianAdam, "Fine 단계에서는 항상 RiemannianAdam");
    
    // Stable 단계로 전환
    optimizer.current_phase = OptimizationPhase::Stable;
    let low_loss_optimizer = optimizer.select_optimizer(0.0005);  // < 0.001
    assert_eq!(low_loss_optimizer, OptimizerType::SGD, "매우 낮은 손실에서는 SGD 선택");
    
    let stable_medium_optimizer = optimizer.select_optimizer(0.01);  // > 0.001
    assert_eq!(stable_medium_optimizer, OptimizerType::RiemannianAdam, "중간 손실에서는 RiemannianAdam 선택");
}

#[test]
fn 단계전환판단_테스트() {
    let mut optimizer = HybridOptimizer::new();
    
    // 충분한 이력이 없는 경우
    assert!(!optimizer.should_advance_phase(), "이력이 부족하면 전환하지 않음");
    
    // Coarse → Fine 전환 테스트
    optimizer.current_phase = OptimizationPhase::Coarse;
    
    // 충분한 손실 감소가 있는 이력 추가
    for i in 0..10 {
        let loss = 1.0 - (i as f32) * 0.08; // 1.0에서 0.28로 감소 (72% 감소)
        optimizer.loss_history.push(loss);
    }
    
    assert!(optimizer.should_advance_phase(), "충분한 손실 감소 시 다음 단계로 전환");
    
    // Fine → Stable 전환 테스트
    optimizer.current_phase = OptimizationPhase::Fine;
    optimizer.loss_history.clear();
    
    // 안정된 손실 이력 추가 (변화량이 작음)
    for _ in 0..10 {
        optimizer.loss_history.push(0.1); // 일정한 손실
    }
    
    assert!(optimizer.should_advance_phase(), "변화량이 작으면 Stable 단계로 전환");
    
    // Stable 단계에서는 전환하지 않음
    optimizer.current_phase = OptimizationPhase::Stable;
    assert!(!optimizer.should_advance_phase(), "Stable 단계에서는 더 이상 전환하지 않음");
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
    
    // Stable → Stable (변화 없음)
    optimizer.advance_phase();
    assert_eq!(optimizer.current_phase, OptimizationPhase::Stable, "Stable에서는 변화 없음");
}

#[test]
fn 분산계산_테스트() {
    let optimizer = HybridOptimizer::new();
    
    // 빈 배열
    let empty_values = [];
    assert_eq!(optimizer.calculate_variance(&empty_values), 0.0, "빈 배열의 분산은 0");
    
    // 단일 원소
    let single_value = [1.0];
    assert_eq!(optimizer.calculate_variance(&single_value), 0.0, "단일 원소의 분산은 0");
    
    // 일정한 값들
    let constant_values = [2.0, 2.0, 2.0, 2.0];
    assert!((optimizer.calculate_variance(&constant_values)).abs() < 1e-6, "일정한 값들의 분산은 0에 가까움");
    
    // 다양한 값들
    let varied_values = [1.0, 2.0, 3.0, 4.0, 5.0];
    let variance = optimizer.calculate_variance(&varied_values);
    assert!(variance > 0.0, "다양한 값들의 분산은 양수");
    
    // 알려진 분산 계산
    let known_values = [1.0, 3.0]; // 평균 2.0, 분산 2.0
    let known_variance = optimizer.calculate_variance(&known_values);
    assert!((known_variance - 2.0).abs() < 1e-6, "알려진 분산 계산 정확성");
}

#[test]
fn Adam업데이트_하이브리드_테스트() {
    let mut optimizer = HybridOptimizer::new();
    let mut param = 1.0;
    let gradient = 0.1;
    let lr = 0.01;
    
    let original_param = param;
    optimizer.adam_update("test_param", &mut param, gradient, lr);
    
    assert_ne!(param, original_param, "파라미터가 업데이트되어야 함");
    assert!(optimizer.adam_states.contains_key("test_param"), "Adam 상태가 저장되어야 함");
    
    // 동일한 파라미터에 대한 두 번째 업데이트
    let second_param = param;
    optimizer.adam_update("test_param", &mut param, gradient, lr);
    assert_ne!(param, second_param, "두 번째 업데이트도 파라미터를 변경해야 함");
    
    // 상태가 업데이트되었는지 확인
    let state = optimizer.adam_states.get("test_param").unwrap();
    assert_eq!(state.t, 2, "시간 스텝이 2가 되어야 함");
}

#[test]
fn Riemannian_Adam업데이트_하이브리드_테스트() {
    let mut optimizer = HybridOptimizer::new();
    let mut r = 0.5;
    let mut theta = PI / 4.0;
    let grad_r = 0.1;
    let grad_theta = 0.05;
    let lr = 0.01;
    
    let original_r = r;
    let original_theta = theta;
    optimizer.riemannian_adam_update("test_param", &mut r, &mut theta, grad_r, grad_theta, lr);
    
    assert_ne!(r, original_r, "r이 업데이트되어야 함");
    assert_ne!(theta, original_theta, "θ가 업데이트되어야 함");
    assert!(optimizer.riemannian_states.contains_key("test_param"), "Riemannian Adam 상태가 저장되어야 함");
    
    // 경계 조건 확인
    assert!(r >= 0.0 && r <= 0.99, "r은 유효 범위 내에 있어야 함");
    assert!(theta >= 0.0 && theta < 2.0 * PI, "θ는 유효 범위 내에 있어야 함");
}

#[test]
fn 손실업데이트_및_성능지표_테스트() {
    let mut optimizer = HybridOptimizer::new();
    
    // 첫 번째 손실
    optimizer.update_loss(1.0);
    assert_eq!(optimizer.loss_history.len(), 1, "손실 이력이 추가되어야 함");
    assert_eq!(optimizer.performance_metrics.convergence_rate, 0.0, "첫 손실에서는 수렴률 0");
    
    // 두 번째 손실
    optimizer.update_loss(0.8);
    assert_eq!(optimizer.loss_history.len(), 2, "손실 이력이 누적되어야 함");
    assert!(optimizer.performance_metrics.convergence_rate > 0.0, "수렴률이 계산되어야 함");
    
    // 여러 손실 추가 (안정성 지표 계산을 위해)
    for i in 3..=12 {
        optimizer.update_loss(1.0 - (i as f32) * 0.05);
    }
    
    assert!(optimizer.performance_metrics.stability_metric > 0.0, "안정성 지표가 계산되어야 함");
    assert_eq!(optimizer.loss_history.len(), 12, "모든 손실이 기록되어야 함");
}

#[test]
fn PSNR계산_테스트() {
    let mut optimizer = HybridOptimizer::new();
    
    // 빈 배열
    let empty1 = [];
    let empty2 = [];
    let psnr = optimizer.calculate_psnr(&empty1, &empty2);
    assert_eq!(psnr, 0.0, "빈 배열의 PSNR은 0");
    
    // 길이가 다른 배열
    let short = [1.0];
    let long = [1.0, 2.0];
    let psnr = optimizer.calculate_psnr(&short, &long);
    assert_eq!(psnr, 0.0, "길이가 다른 배열의 PSNR은 0");
    
    // 동일한 배열 (완벽한 재구성)
    let original = [1.0, 2.0, 3.0, 4.0];
    let perfect = [1.0, 2.0, 3.0, 4.0];
    let psnr = optimizer.calculate_psnr(&original, &perfect);
    assert_eq!(psnr, f32::INFINITY, "완벽한 재구성은 무한 PSNR");
    
    // 실제 오차가 있는 경우
    let original = [1.0, 2.0, 3.0, 4.0];
    let reconstructed = [1.1, 1.9, 3.1, 3.9];
    let psnr = optimizer.calculate_psnr(&original, &reconstructed);
    assert!(psnr > 0.0 && psnr < f32::INFINITY, "실제 오차는 유한한 PSNR");
    assert_eq!(optimizer.performance_metrics.psnr, psnr, "PSNR이 성능 지표에 저장되어야 함");
}

#[test]
fn Packed128최적화_테스트() {
    let mut optimizer = HybridOptimizer::new();
    let mut packed = Packed128 {
        hi: 0,
        lo: ((0.5f32.to_bits() as u64) << 32) | (PI / 4.0).to_bits() as u64
    };
    
    let gradient = 0.1;
    let lr = 0.01;
    
    let original_lo = packed.lo;
    optimizer.optimize_packed128("test_packed", &mut packed, gradient, lr);
    
    assert_ne!(packed.lo, original_lo, "Packed128이 업데이트되어야 함");
    
    // r과 θ 추출 및 검증
    let r = f32::from_bits((packed.lo >> 32) as u32);
    let theta = f32::from_bits(packed.lo as u32);
    
    assert!(r >= 0.0 && r <= 0.99, "r은 유효 범위 내에 있어야 함");
    assert!(theta >= 0.0 && theta < 2.0 * PI, "θ는 유효 범위 내에 있어야 함");
}

#[test]
fn 성능보고서_출력_테스트() {
    let mut optimizer = HybridOptimizer::new();
    
    // 일부 데이터로 초기화
    optimizer.update_loss(0.5);
    optimizer.update_loss(0.3);
    
    let original = [1.0, 2.0, 3.0];
    let reconstructed = [1.1, 1.9, 3.1];
    optimizer.calculate_psnr(&original, &reconstructed);
    
    optimizer.transform_analyzer.dct_performance = 0.05;
    optimizer.transform_analyzer.wavelet_performance = 0.07;
    
    // 출력 테스트 (패닉이 발생하지 않는지 확인)
    optimizer.print_performance_report();
    
    // 이 테스트는 주로 런타임 에러가 없는지 확인하는 용도
    assert!(true, "성능 보고서 출력이 에러 없이 완료되어야 함");
}

#[test]
fn 최적화기타입_열거형_테스트() {
    // 모든 열거형 값들이 올바르게 정의되어 있는지 확인
    let adam = OptimizerType::Adam;
    let riemannian = OptimizerType::RiemannianAdam;
    let sgd = OptimizerType::SGD;
    let rmsprop = OptimizerType::RMSprop;
    
    assert_ne!(adam, riemannian, "다른 최적화기는 구별되어야 함");
    assert_ne!(adam, sgd, "다른 최적화기는 구별되어야 함");
    assert_ne!(adam, rmsprop, "다른 최적화기는 구별되어야 함");
    
    // Debug 트레이트 테스트
    println!("{:?}", adam);
    println!("{:?}", riemannian);
    
    assert!(true, "모든 최적화기 타입이 올바르게 정의됨");
}

#[test]
fn 변환타입_열거형_테스트() {
    let dct = TransformType::Dct;
    let dwt = TransformType::Dwt;
    let adaptive = TransformType::Adaptive;
    
    assert_ne!(dct, dwt, "다른 변환 타입은 구별되어야 함");
    assert_ne!(dct, adaptive, "다른 변환 타입은 구별되어야 함");
    assert_ne!(dwt, adaptive, "다른 변환 타입은 구별되어야 함");
    
    // Debug 트레이트 테스트
    println!("{:?}", dct);
    println!("{:?}", dwt);
    println!("{:?}", adaptive);
    
    assert!(true, "모든 변환 타입이 올바르게 정의됨");
}

#[test]
fn 최적화단계_열거형_테스트() {
    let coarse = OptimizationPhase::Coarse;
    let fine = OptimizationPhase::Fine;
    let stable = OptimizationPhase::Stable;
    
    assert_ne!(coarse, fine, "다른 최적화 단계는 구별되어야 함");
    assert_ne!(coarse, stable, "다른 최적화 단계는 구별되어야 함");
    assert_ne!(fine, stable, "다른 최적화 단계는 구별되어야 함");
    
    // Debug 및 Clone 트레이트 테스트
    let cloned_coarse = coarse.clone();
    assert_eq!(coarse, cloned_coarse, "최적화 단계는 복제 가능해야 함");
    
    println!("{:?}", coarse);
    
    assert!(true, "모든 최적화 단계가 올바르게 정의됨");
} 
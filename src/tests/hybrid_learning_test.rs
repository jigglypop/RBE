use crate::layer::{
    HybridPoincareRBESystem, SystemConfiguration, LearningParameters, 
    LossWeights, HardwareConfiguration, ConvergenceStatus, LossComponents
};
use crate::matrix::QualityLevel;
use crate::types::Packed128;
use std::collections::HashMap;
use std::time::Instant;

/// 7.1 통합 시스템 아키텍처 테스트
#[test]
fn test_system_architecture() {
    println!("=== 하이브리드 푸앵카레 RBE 시스템 아키텍처 테스트 ===");
    
    // 시스템 구성 설정
    let config = SystemConfiguration {
        layer_sizes: vec![(784, 256), (256, 128), (128, 10)],
        compression_ratio: 1000.0,
        quality_level: QualityLevel::High,
        learning_params: LearningParameters::default(),
        hardware_config: HardwareConfiguration::default(),
    };
    
    // 시스템 초기화
    let system = HybridPoincareRBESystem::new(config);
    
    // 기본 구조 검증
    assert_eq!(system.layers.len(), 3, "레이어 개수가 올바르지 않음");
    assert_eq!(system.layers[0].input_dim, 784, "첫 번째 레이어 입력 차원 오류");
    assert_eq!(system.layers[0].output_dim, 256, "첫 번째 레이어 출력 차원 오류");
    assert_eq!(system.layers[2].output_dim, 10, "마지막 레이어 출력 차원 오류");
    
    // 수렴 상태 초기값 확인
    assert_eq!(system.learning_state.convergence_status, ConvergenceStatus::Training);
    assert_eq!(system.learning_state.current_epoch, 0);
    
    println!("시스템 아키텍처:");
    for (i, layer) in system.layers.iter().enumerate() {
        println!("  레이어 {}: {}→{}", i, layer.input_dim, layer.output_dim);
    }
    
    println!("하드웨어 구성:");
    println!("  CPU 스레드: {}", system.config.hardware_config.num_cpu_threads);
    println!("  메모리 풀: {:.1} MB", system.config.hardware_config.memory_pool_size as f32 / 1024.0 / 1024.0);
    
    println!("시스템 아키텍처 테스트 성공!");
}

/// 7.2.3 멀티모달 손실 함수 테스트
#[test]
fn test_multimodal_loss_function() {
    println!("=== 멀티모달 손실 함수 테스트 ===");
    
    let system = HybridPoincareRBESystem::new(SystemConfiguration::default());
    
    // 테스트 데이터 생성
    let predictions = vec![0.8, 0.1, 0.05, 0.05];
    let targets = vec![1.0, 0.0, 0.0, 0.0];
    
    // 푸앵카레 파라미터 생성
    let mut poincare_params = Vec::new();
    for i in 0..4 {
        let r = 0.5 + (i as f32 * 0.1);
        let theta = i as f32 * 0.5;
        let packed = Packed128 {
            hi: i as u64,
            lo: ((r.to_bits() as u64) << 32) | theta.to_bits() as u64,
        };
        poincare_params.push(packed);
    }
    
    // 상태 사용 통계
    let mut state_usage = HashMap::new();
    state_usage.insert(0, 10);
    state_usage.insert(1, 15);
    state_usage.insert(2, 8);
    state_usage.insert(3, 12);
    
    // 잔차 데이터
    let residuals = vec![0.01, -0.02, 0.015, -0.008, 0.003];
    
    // 멀티모달 손실 계산
    let (total_loss, loss_components) = system.compute_multimodal_loss(
        &predictions,
        &targets,
        &poincare_params,
        &state_usage,
        &residuals
    );
    
    println!("손실 구성요소:");
    println!("  데이터 손실: {:.6}", loss_components.data_loss);
    println!("  푸앵카레 정규화: {:.6}", loss_components.poincare_loss);
    println!("  상태 분포 균형: {:.6}", loss_components.state_loss);
    println!("  희소성 손실: {:.6}", loss_components.sparsity_loss);
    println!("  총 손실: {:.6}", total_loss);
    
    // 손실 값들이 합리적인 범위인지 확인
    assert!(loss_components.data_loss >= 0.0, "데이터 손실이 음수");
    assert!(loss_components.poincare_loss >= 0.0, "푸앵카레 손실이 음수");
    assert!(loss_components.state_loss >= 0.0, "상태 손실이 음수");
    assert!(loss_components.sparsity_loss >= 0.0, "희소성 손실이 음수");
    assert!(total_loss >= 0.0, "총 손실이 음수");
    
    // 총 손실이 구성요소들의 가중합인지 확인
    let weights = &system.config.learning_params.loss_weights;
    let expected_total = weights.data_loss_weight * loss_components.data_loss
        + weights.poincare_regularization_weight * loss_components.poincare_loss
        + weights.state_balance_weight * loss_components.state_loss
        + weights.sparsity_weight * loss_components.sparsity_loss;
    
    let diff = (total_loss - expected_total).abs();
    assert!(diff < 1e-6, "총 손실 계산 오류: 차이 = {}", diff);
    
    println!("멀티모달 손실 함수 테스트 성공!");
}

/// 7.4 순전파/역전파 테스트
#[test]
fn test_forward_backward() {
    println!("=== 하이브리드 순전파/역전파 테스트 ===");
    
    let mut system = HybridPoincareRBESystem::new(SystemConfiguration {
        layer_sizes: vec![(4, 3), (3, 2)],
        compression_ratio: 100.0,
        quality_level: QualityLevel::Medium,
        learning_params: LearningParameters::default(),
        hardware_config: HardwareConfiguration::default(),
    });
    
    // 입력 데이터
    let input = vec![1.0, 0.5, -0.3, 0.8];
    
    // 순전파 테스트
    let start_time = Instant::now();
    let output = system.forward(&input);
    let forward_time = start_time.elapsed();
    
    println!("순전파 결과:");
    println!("  입력: {:?}", input);
    println!("  출력: {:?}", output);
    println!("  순전파 시간: {:.2} μs", forward_time.as_micros());
    
    // 출력 크기 확인
    assert_eq!(output.len(), 2, "출력 크기가 올바르지 않음");
    
    // 성능 모니터링 확인
    assert!(system.performance_monitor.computation_time.forward_time_us > 0, 
            "순전파 시간이 기록되지 않음");
    
    // 역전파 테스트
    let loss_gradient = vec![0.1, -0.05];
    
    let start_time = Instant::now();
    system.backward(&loss_gradient, 0.01);
    let backward_time = start_time.elapsed();
    
    println!("역전파 결과:");
    println!("  손실 그래디언트: {:?}", loss_gradient);
    println!("  역전파 시간: {:.2} μs", backward_time.as_micros());
    
    // 성능 모니터링 확인
    assert!(system.performance_monitor.computation_time.backward_time_us > 0,
            "역전파 시간이 기록되지 않음");
    
    println!("순전파/역전파 테스트 성공!");
}

/// 7.3 성능 모니터링 테스트
#[test]
fn test_performance_monitoring() {
    println!("=== 성능 모니터링 테스트 ===");
    
    let mut system = HybridPoincareRBESystem::new(SystemConfiguration::default());
    
    // 초기 성능 지표 확인
    let initial_memory = system.performance_monitor.memory_usage.current_usage;
    let initial_accuracy = system.performance_monitor.quality_metrics.accuracy;
    
    // 시뮬레이션된 학습 진행
    for epoch in 0..5 {
        system.learning_state.current_epoch = epoch;
        
        // 가상의 손실 구성요소 생성 (점진적 개선)
        let loss_components = LossComponents {
            data_loss: 1.0 - (epoch as f32 * 0.1),
            poincare_loss: 0.01 - (epoch as f32 * 0.001),
            state_loss: 0.1 - (epoch as f32 * 0.01),
            sparsity_loss: 0.05 - (epoch as f32 * 0.005),
            total_loss: 1.16 - (epoch as f32 * 0.116),
        };
        
        let learning_rate = 0.001 * (0.9_f32).powi(epoch as i32);
        
        // 학습 상태 업데이트
        system.update_learning_state(loss_components, learning_rate);
        
        // 성능 지표 업데이트 (시뮬레이션)
        system.performance_monitor.quality_metrics.accuracy = epoch as f32 * 0.15 + 0.7;
        system.performance_monitor.quality_metrics.loss = loss_components.total_loss;
        system.performance_monitor.memory_usage.current_usage = 1024 * 1024; // 1MB
        system.performance_monitor.energy_efficiency.power_consumption = 50.0 + epoch as f32;
    }
    
    // 학습 히스토리 확인
    assert_eq!(system.learning_state.loss_history.len(), 5, "손실 히스토리 길이 오류");
    assert_eq!(system.learning_state.learning_rate_history.len(), 5, "학습률 히스토리 길이 오류");
    
    // 손실 감소 확인
    let first_loss = system.learning_state.loss_history[0].total_loss;
    let last_loss = system.learning_state.loss_history[4].total_loss;
    assert!(last_loss < first_loss, "손실이 감소하지 않음: {} -> {}", first_loss, last_loss);
    
    // 학습률 감소 확인 (지수적 감소)
    let first_lr = system.learning_state.learning_rate_history[0];
    let last_lr = system.learning_state.learning_rate_history[4];
    assert!(last_lr < first_lr, "학습률이 감소하지 않음: {} -> {}", first_lr, last_lr);
    
    println!("학습 진행 상황:");
    for (i, (loss, lr)) in system.learning_state.loss_history.iter()
        .zip(system.learning_state.learning_rate_history.iter())
        .enumerate() {
        println!("  에포크 {}: 손실={:.4}, 학습률={:.6}", i, loss.total_loss, lr);
    }
    
    // 성능 보고서 출력
    system.print_performance_report();
    
    println!("성능 모니터링 테스트 성공!");
}

/// 7.4.2 수렴 상태 판단 테스트
#[test]
fn test_convergence_detection() {
    println!("=== 수렴 상태 판단 테스트 ===");
    
    let mut system = HybridPoincareRBESystem::new(SystemConfiguration::default());
    
    // 수렴 시나리오 테스트 - 점진적 개선
    println!("시나리오 1: 점진적 수렴");
    for i in 0..20 {
        let loss = LossComponents {
            data_loss: 1.0 * (-0.2 * i as f32).exp(),
            poincare_loss: 0.01,
            state_loss: 0.001,
            sparsity_loss: 0.0001,
            total_loss: 1.0111 * (-0.2 * i as f32).exp(),
        };
        
        system.update_learning_state(loss, 0.001);
        
        if i >= 15 {
            println!("  에포크 {}: 상태={:?}, 손실={:.6}", 
                     i, system.learning_state.convergence_status, loss.total_loss);
        }
    }
    
    // 수렴 상태가 올바르게 감지되었는지 확인
    assert_eq!(system.learning_state.convergence_status, ConvergenceStatus::Converged,
               "수렴 상태가 감지되지 않음");
    
    // 발산 시나리오 테스트
    println!("\n시나리오 2: 발산");
    system.learning_state.loss_history.clear();
    system.learning_state.convergence_status = ConvergenceStatus::Training;
    
    for i in 0..15 {
        let loss = LossComponents {
            data_loss: 0.5 + 0.1 * i as f32,
            poincare_loss: 0.01,
            state_loss: 0.001,
            sparsity_loss: 0.0001,
            total_loss: 0.5111 + 0.1 * i as f32,
        };
        
        system.update_learning_state(loss, 0.001);
        
        if i >= 10 {
            println!("  에포크 {}: 상태={:?}, 손실={:.6}", 
                     i, system.learning_state.convergence_status, loss.total_loss);
        }
    }
    
    // 발산 상태가 감지되었는지 확인
    assert_eq!(system.learning_state.convergence_status, ConvergenceStatus::Diverged,
               "발산 상태가 감지되지 않음: {:?}", system.learning_state.convergence_status);
    
    println!("수렴 상태 판단 테스트 성공!");
}

/// 7.5 대규모 시스템 확장성 테스트
#[test]
fn test_system_scalability() {
    println!("=== 시스템 확장성 테스트 ===");
    
    let configurations = vec![
        ("작은 시스템", vec![(10, 5), (5, 2)]),
        ("중간 시스템", vec![(100, 50), (50, 25), (25, 10)]),
        ("큰 시스템", vec![(784, 512), (512, 256), (256, 128), (128, 10)]),
    ];
    
    for (name, layer_sizes) in configurations {
        println!("\n{} 테스트:", name);
        
        let config = SystemConfiguration {
            layer_sizes: layer_sizes.clone(),
            compression_ratio: 500.0,
            quality_level: QualityLevel::Medium,
            learning_params: LearningParameters::default(),
            hardware_config: HardwareConfiguration::default(),
        };
        
        let start_time = Instant::now();
        let mut system = HybridPoincareRBESystem::new(config);
        let init_time = start_time.elapsed();
        
        // 파라미터 개수 계산
        let total_params: usize = layer_sizes.iter()
            .map(|(input, output)| input * output)
            .sum();
        
        // 압축된 파라미터 개수 (추정)
        let compressed_params = total_params / 500; // 압축률 500:1
        
        println!("  레이어 구조: {:?}", layer_sizes);
        println!("  초기화 시간: {:.2} ms", init_time.as_secs_f64() * 1000.0);
        println!("  원본 파라미터: {}", total_params);
        println!("  압축된 파라미터: {}", compressed_params);
        println!("  메모리 절약률: {:.1}%", (1.0 - compressed_params as f32 / total_params as f32) * 100.0);
        
        // 간단한 순전파 테스트
        let input_size = layer_sizes[0].0;
        let input = vec![0.1; input_size];
        
        let start_time = Instant::now();
        let output = system.forward(&input);
        let forward_time = start_time.elapsed();
        
        println!("  순전파 시간: {:.2} μs", forward_time.as_micros());
        println!("  출력 크기: {}", output.len());
        
        // 성능 기준 확인
        assert!(init_time.as_millis() < 1000, "초기화 시간이 너무 오래 걸림");
        assert!(forward_time.as_millis() < 100, "순전파 시간이 너무 오래 걸림");
        assert_eq!(output.len(), layer_sizes.last().unwrap().1, "출력 크기 오류");
    }
    
    println!("\n시스템 확장성 테스트 성공!");
}

/// 손실 함수 가중치 조정 테스트
#[test]
fn test_loss_weight_adjustment() {
    println!("=== 손실 함수 가중치 조정 테스트 ===");
    
    let test_cases = vec![
        ("균형 설정", LossWeights {
            data_loss_weight: 1.0,
            poincare_regularization_weight: 0.01,
            state_balance_weight: 0.001,
            sparsity_weight: 0.0001,
        }),
        ("데이터 중심", LossWeights {
            data_loss_weight: 10.0,
            poincare_regularization_weight: 0.001,
            state_balance_weight: 0.0001,
            sparsity_weight: 0.00001,
        }),
        ("정규화 중심", LossWeights {
            data_loss_weight: 1.0,
            poincare_regularization_weight: 1.0,
            state_balance_weight: 0.1,
            sparsity_weight: 0.1,
        }),
    ];
    
    let predictions = vec![0.7, 0.2, 0.1];
    let targets = vec![1.0, 0.0, 0.0];
    let poincare_params = vec![
        Packed128 { hi: 0, lo: ((0.9_f32.to_bits() as u64) << 32) | 0_u64 }
    ];
    let mut state_usage = HashMap::new();
    state_usage.insert(0, 1);
    let residuals = vec![0.1, -0.05, 0.02];
    
    for (name, weights) in test_cases {
        println!("\n{} 테스트:", name);
        
        let config = SystemConfiguration {
            learning_params: LearningParameters {
                loss_weights: weights.clone(),
                ..LearningParameters::default()
            },
            ..SystemConfiguration::default()
        };
        
        let system = HybridPoincareRBESystem::new(config);
        
        let (total_loss, components) = system.compute_multimodal_loss(
            &predictions,
            &targets,
            &poincare_params,
            &state_usage,
            &residuals
        );
        
        println!("  가중치: data={:.3}, poincare={:.3}, state={:.3}, sparsity={:.3}",
                 weights.data_loss_weight,
                 weights.poincare_regularization_weight,
                 weights.state_balance_weight,
                 weights.sparsity_weight);
        
        println!("  손실: data={:.4}, poincare={:.4}, state={:.4}, sparsity={:.4}, total={:.4}",
                 components.data_loss,
                 components.poincare_loss,
                 components.state_loss,
                 components.sparsity_loss,
                 total_loss);
        
        // 가중치가 손실에 올바르게 반영되었는지 확인
        let expected_total = weights.data_loss_weight * components.data_loss
            + weights.poincare_regularization_weight * components.poincare_loss
            + weights.state_balance_weight * components.state_loss
            + weights.sparsity_weight * components.sparsity_loss;
        
        let diff = (total_loss - expected_total).abs();
        assert!(diff < 1e-6, "손실 가중치 계산 오류: 차이 = {}", diff);
    }
    
    println!("\n손실 함수 가중치 조정 테스트 성공!");
} 
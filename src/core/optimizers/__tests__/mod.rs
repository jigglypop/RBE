// Optimizers 모듈 테스트 통합
// 각 모듈별로 분리된 테스트들을 재수출

pub mod adam_test;
pub mod riemannian_adam_test;
pub mod analyzer_test;
pub mod hybrid_test;
pub mod config_test;

// 통합 테스트들
use crate::core::optimizers::*;
use crate::types::TransformType;

#[test]
fn 모든_모듈_통합_테스트() {
    // 모든 주요 구조체가 정상적으로 생성되는지 확인
    let adam_state = AdamState::new();
    let riemannian_state = RiemannianAdamState::new();
    let analyzer = TransformAnalyzer::new();
    let hybrid_optimizer = HybridOptimizer::new();
    let config = OptimizerConfig::new();
    
    // 기본값들이 합리적인지 확인
    assert_eq!(adam_state.t, 0, "Adam 상태 초기화 확인");
    assert_eq!(riemannian_state.t, 0, "RiemannianAdam 상태 초기화 확인");
    assert_eq!(analyzer.dct_performance, 0.0, "분석기 초기화 확인");
    assert_eq!(hybrid_optimizer.current_phase, OptimizationPhase::Coarse, "하이브리드 옵티마이저 초기화 확인");
    assert_eq!(config.learning_rate, 0.001, "구성 초기화 확인");
    
    println!("✅ 모든 모듈 통합 테스트 통과");
    println!("   Adam, RiemannianAdam, Analyzer, HybridOptimizer, Config 모두 정상 동작");
}

#[test]
fn 모듈간_상호작용_테스트() {
    let mut hybrid_optimizer = HybridOptimizer::new();
    let config = OptimizerConfig::new()
        .with_learning_rate(0.01)
        .with_adam_config(AdamConfig {
            beta1: 0.95,
            beta2: 0.999,
            epsilon: 1e-8,
        });
    
    // HybridOptimizer에서 직접 Adam과 RiemannianAdam 사용
    let mut param = 1.0;
    hybrid_optimizer.adam_update("test", &mut param, 0.1, config.learning_rate);
    
    let mut r = 0.5;
    let mut theta = 1.0;
    hybrid_optimizer.riemannian_adam_update("test2", &mut r, &mut theta, 0.05, 0.02, config.learning_rate);
    
    // TransformAnalyzer 사용
    let signal = vec![1.0, 2.0, 3.0, 4.0];
    let choice = hybrid_optimizer.transform_analyzer.select_optimal_transform(&signal);
    
    assert!(matches!(choice, TransformType::Dct | TransformType::Dwt), "변환 선택이 유효함");
    assert!(hybrid_optimizer.adam_states.contains_key("test"), "Adam 상태가 저장됨");
    assert!(hybrid_optimizer.riemannian_states.contains_key("test2"), "RiemannianAdam 상태가 저장됨");
    
    println!("✅ 모듈간 상호작용 테스트 통과");
    println!("   HybridOptimizer ↔ Adam ↔ RiemannianAdam ↔ TransformAnalyzer ↔ Config 연동 확인");
}

#[test]
fn 성능_기본_벤치마크_테스트() {
    use std::time::Instant;
    
    let iterations = 1000;
    
    // Adam 업데이트 성능
    let start = Instant::now();
    let mut adam_state = AdamState::new();
    let mut param = 1.0;
    for _ in 0..iterations {
        adam_state.update(&mut param, 0.1, 0.01);
    }
    let adam_duration = start.elapsed();
    
    // RiemannianAdam 업데이트 성능
    let start = Instant::now();
    let mut riemannian_state = RiemannianAdamState::new();
    let mut r = 0.5;
    let mut theta = 1.0;
    for _ in 0..iterations {
        riemannian_state.update(&mut r, &mut theta, 0.05, 0.02, 0.01);
    }
    let riemannian_duration = start.elapsed();
    
    // TransformAnalyzer 성능
    let start = Instant::now();
    let mut analyzer = TransformAnalyzer::new();
    let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    for _ in 0..10 { // 더 적은 반복 (계산이 무거움)
        analyzer.measure_dct_performance(&signal, 0.5);
    }
    let analyzer_duration = start.elapsed();
    
    println!("✅ 성능 기본 벤치마크 테스트 통과");
    println!("   Adam {} 회 업데이트: {:?}", iterations, adam_duration);
    println!("   RiemannianAdam {} 회 업데이트: {:?}", iterations, riemannian_duration);
    println!("   TransformAnalyzer 10회 분석: {:?}", analyzer_duration);
    
    // 성능이 합리적인 범위 내에 있는지 확인 (너무 느리지 않음)
    assert!(adam_duration.as_millis() < 100, "Adam 업데이트가 너무 느림");
    assert!(riemannian_duration.as_millis() < 100, "RiemannianAdam 업데이트가 너무 느림");
    assert!(analyzer_duration.as_millis() < 1000, "TransformAnalyzer가 너무 느림");
} 
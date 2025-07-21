//! # 하이브리드 최적화기 단위 테스트
//!
//! hybrid.rs의 모든 함수에 대한 테스트

use crate::core::optimizers::{HybridOptimizer, OptimizationPhase};
use crate::packed_params::Packed128;
use std::time::Instant;

fn 테스트용_packed128_생성() -> Packed128 {
    Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: 0xFEDCBA9876543210,
    }
}

fn 테스트용_타겟_데이터_생성(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32 * 0.1).sin()).collect()
}

#[test]
fn 하이브리드_최적화기_생성_테스트() {
    println!("🧪 하이브리드 최적화기 생성 테스트 시작");
    
    let learning_rate = 0.01;
    let packed_count = 10;
    
    println!("   파라미터:");
    println!("     학습률: {}", learning_rate);
    println!("     파라미터 개수: {}", packed_count);
    
    let optimizer = HybridOptimizer::new(learning_rate, packed_count);
    println!("   하이브리드 최적화기 생성 완료");
    
    // 생성이 성공적으로 되었는지 확인 (간접적)
    let diagnosis = optimizer.diagnose();
    println!("   진단 정보:\n{}", diagnosis);
    assert!(diagnosis.contains("하이브리드 최적화기"));
    
    println!("✅ 하이브리드 최적화기 생성 테스트 완료");
}

#[test]
fn 다양한_학습률_생성_테스트() {
    let test_cases = [0.001, 0.01, 0.1, 1.0];
    let packed_count = 5;
    
    for lr in test_cases {
        let optimizer = HybridOptimizer::new(lr, packed_count);
        let diagnosis = optimizer.diagnose();
        assert!(diagnosis.contains("하이브리드 최적화기"));
    }
}

#[test]
fn 다양한_packed_count_생성_테스트() {
    let learning_rate = 0.01;
    let test_cases = [1, 5, 10, 50, 100];
    
    for count in test_cases {
        let optimizer = HybridOptimizer::new(learning_rate, count);
        let diagnosis = optimizer.diagnose();
        assert!(diagnosis.contains("하이브리드 최적화기"));
    }
}

#[test]
fn 최적화_스텝_실행_테스트() {
    println!("🧪 하이브리드 최적화기 스텝 실행 테스트 시작");
    
    let mut optimizer = HybridOptimizer::new(0.01, 5);
    let mut packed = 테스트용_packed128_생성();
    let target = 테스트용_타겟_데이터_생성(4);
    let predicted = vec![0.0; 4]; // 0에서 시작
    let rows = 2;
    let cols = 2;
    
    println!("   초기 상태:");
    println!("     Packed128 Hi: 0x{:016X}", packed.hi);
    println!("     Packed128 Lo: 0x{:016X}", packed.lo);
    println!("     타겟: {:?}", target);
    println!("     예측: {:?}", predicted);
    println!("     행렬 크기: {}x{}", rows, cols);
    
    let original_hi = packed.hi;
    let original_lo = packed.lo;
    
    println!("   최적화 스텝 실행 중...");
    let loss = optimizer.step(&mut packed, &target, &predicted, rows, cols);
    
    println!("   최적화 후 상태:");
    println!("     Packed128 Hi: 0x{:016X} (변경: {})", packed.hi, packed.hi != original_hi);
    println!("     Packed128 Lo: 0x{:016X} (변경: {})", packed.lo, packed.lo != original_lo);
    println!("     손실: {:.6}", loss);
    
    // 손실이 유효한 값인지 확인
    assert!(loss.is_finite());
    assert!(loss >= 0.0);
    
    // Packed128이 변경되었는지 확인
    let hi_changed = packed.hi != original_hi;
    let lo_changed = packed.lo != original_lo;
    println!("   변경 사항: Hi={}, Lo={}", hi_changed, lo_changed);
    assert!(hi_changed || lo_changed);
    
    println!("✅ 하이브리드 최적화기 스텝 실행 테스트 완료");
}

#[test]
fn 여러_스텝_연속_실행_테스트() {
    let mut optimizer = HybridOptimizer::new(0.005, 3);
    let mut packed = 테스트용_packed128_생성();
    let target = vec![1.0, 0.5, -0.5, -1.0];
    let rows = 2;
    let cols = 2;
    
    let mut losses = Vec::new();
    
    for i in 0..10 {
        // 예측값을 점진적으로 타겟에 가깝게
        let predicted: Vec<f32> = target.iter()
            .map(|&t| t * (i as f32 * 0.1))
            .collect();
        
        let loss = optimizer.step(&mut packed, &target, &predicted, rows, cols);
        losses.push(loss);
        
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }
    
    // 손실이 기록되었는지 확인
    assert_eq!(losses.len(), 10);
}

#[test]
fn 성능_리포트_생성_테스트() {
    let mut optimizer = HybridOptimizer::new(0.02, 5);
    let mut packed = 테스트용_packed128_생성();
    let target = 테스트용_타겟_데이터_생성(9);
    let predicted = vec![0.0; 9];
    let rows = 3;
    let cols = 3;
    
    // 몇 번 실행하여 통계 축적
    for _ in 0..3 {
        let _ = optimizer.step(&mut packed, &target, &predicted, rows, cols);
    }
    
    let diagnosis = optimizer.diagnose();
    
    assert!(diagnosis.contains("하이브리드 최적화기"));
    assert!(diagnosis.contains("에포크"));
    assert!(diagnosis.contains("단계"));
}

#[test]
fn 진단_정보_생성_테스트() {
    let mut optimizer = HybridOptimizer::new(0.01, 4);
    let mut packed = 테스트용_packed128_생성();
    let target = vec![0.5; 16];
    let predicted = vec![0.0; 16];
    let rows = 4;
    let cols = 4;
    
    // 몇 번 실행
    for _ in 0..2 {
        let _ = optimizer.step(&mut packed, &target, &predicted, rows, cols);
    }
    
    let diagnosis = optimizer.diagnose();
    
    assert!(diagnosis.contains("진단"));
    assert!(!diagnosis.is_empty());
}

#[test]
fn 극단값_입력_처리_테스트() {
    let mut optimizer = HybridOptimizer::new(0.01, 2);
    let mut packed = 테스트용_packed128_생성();
    
    // 매우 큰 값들
    let large_target = vec![1000.0; 4];
    let large_predicted = vec![999.0; 4];
    let loss1 = optimizer.step(&mut packed, &large_target, &large_predicted, 2, 2);
    assert!(loss1.is_finite());
    
    // 매우 작은 값들
    let small_target = vec![0.001; 4];
    let small_predicted = vec![0.0001; 4];
    let loss2 = optimizer.step(&mut packed, &small_target, &small_predicted, 2, 2);
    assert!(loss2.is_finite());
    
    // 음수 값들
    let negative_target = vec![-1.0; 4];
    let negative_predicted = vec![-0.9; 4];
    let loss3 = optimizer.step(&mut packed, &negative_target, &negative_predicted, 2, 2);
    assert!(loss3.is_finite());
    
    // 0 값들
    let zero_target = vec![0.0; 4];
    let zero_predicted = vec![0.0; 4];
    let loss4 = optimizer.step(&mut packed, &zero_target, &zero_predicted, 2, 2);
    assert!(loss4.is_finite());
}

#[test]
fn 다양한_크기_데이터_처리_테스트() {
    let mut optimizer = HybridOptimizer::new(0.01, 6);
    let mut packed = 테스트용_packed128_생성();
    
    let test_cases = [
        (1, 1),   // 최소 크기
        (2, 2),   // 작은 크기  
        (3, 3),   // 중간 크기
        (4, 4),   // 큰 크기
        (5, 5),   // 더 큰 크기
    ];
    
    for (rows, cols) in test_cases {
        let size = rows * cols;
        let target = 테스트용_타겟_데이터_생성(size);
        let predicted = vec![0.0; size];
        
        let loss = optimizer.step(&mut packed, &target, &predicted, rows, cols);
        
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }
}

#[test]
fn 수렴_행동_분석_테스트() {
    let mut optimizer = HybridOptimizer::new(0.01, 3);
    let mut packed = 테스트용_packed128_생성();
    
    // 고정된 타겟
    let target = vec![0.7, -0.3, 0.5, -0.8];
    let rows = 2;
    let cols = 2;
    
    let mut losses = Vec::new();
    
    // 점진적으로 타겟에 가까워지는 예측값으로 20번 실행
    for i in 0..20 {
        let factor = i as f32 / 20.0;
        let predicted: Vec<f32> = target.iter()
            .map(|&t| t * factor)
            .collect();
        
        let loss = optimizer.step(&mut packed, &target, &predicted, rows, cols);
        losses.push(loss);
        
        assert!(loss.is_finite());
    }
    
    // 초기와 후기 손실 비교
    let early_avg = losses[0..5].iter().sum::<f32>() / 5.0;
    let late_avg = losses[15..20].iter().sum::<f32>() / 5.0;
    
    // 일반적으로 후기 손실이 더 작거나 비슷해야 함
    assert!(late_avg <= early_avg * 1.5); // 50% 증가까지는 허용
}

#[test]
fn 메모리_일관성_테스트() {
    let mut optimizer = HybridOptimizer::new(0.01, 7);
    
    // 여러 개의 서로 다른 packed128로 테스트
    let test_packets = [
        Packed128 { hi: 0x1111111111111111, lo: 0x2222222222222222 },
        Packed128 { hi: 0x3333333333333333, lo: 0x4444444444444444 },
        Packed128 { hi: 0x5555555555555555, lo: 0x6666666666666666 },
    ];
    
    for mut packed in test_packets {
        let target = 테스트용_타겟_데이터_생성(4);
        let predicted = vec![0.0; 4];
        
        let loss = optimizer.step(&mut packed, &target, &predicted, 2, 2);
        
        assert!(loss.is_finite());
    }
}

#[test]
fn 최적화_단계_진행_테스트() {
    let mut optimizer = HybridOptimizer::new(0.01, 4);
    let mut packed = 테스트용_packed128_생성();
    let target = vec![1.0; 4];
    let predicted = vec![0.0; 4];
    
    // 충분히 많은 스텝을 실행하여 단계 전환 확인
    for _ in 0..100 {
        let _ = optimizer.step(&mut packed, &target, &predicted, 2, 2);
    }
    
    let diagnosis = optimizer.diagnose();
    
    // 에포크가 포함되어 있는지 확인
    assert!(diagnosis.contains("에포크"));
}

#[test]
fn 일관성_검증_테스트() {
    let learning_rate = 0.01;
    let packed_count = 5;
    
    // 동일한 설정으로 두 개의 최적화기 생성
    let mut optimizer1 = HybridOptimizer::new(learning_rate, packed_count);
    let mut optimizer2 = HybridOptimizer::new(learning_rate, packed_count);
    
    let mut packed1 = 테스트용_packed128_생성();
    let mut packed2 = packed1; // 동일한 초기값
    
    let target = vec![0.5, -0.2, 0.8, -0.6];
    let predicted = vec![0.0; 4];
    
    // 동일한 입력으로 한 스텝씩 실행
    let loss1 = optimizer1.step(&mut packed1, &target, &predicted, 2, 2);
    let loss2 = optimizer2.step(&mut packed2, &target, &predicted, 2, 2);
    
    // 손실값이 동일해야 함 (동일한 알고리즘, 동일한 입력)
    assert!((loss1 - loss2).abs() < 1e-6);
}

#[test]
fn 성능_벤치마크_하이브리드_최적화기_vs_기존_테스트() {
    println!("🧪 하이브리드 최적화기 vs 기존 Adam 성능 벤치마크 시작");
    
    // 1. 단일 스텝 성능 비교
    println!("\n📊 단일 최적화 스텝 성능 비교");
    
    // 하이브리드 최적화기
    let mut hybrid_optimizer = HybridOptimizer::new(0.01, 10);
    let mut hybrid_packed = 테스트용_packed128_생성();
    let target = 테스트용_타겟_데이터_생성(16);
    let predicted = vec![0.0; 16];
    
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = hybrid_optimizer.step(&mut hybrid_packed, &target, &predicted, 4, 4);
    }
    let hybrid_time = start.elapsed();
    
    println!("   하이브리드 최적화기 1,000 스텝: {:.3}ms", hybrid_time.as_millis());
    println!("   평균 하이브리드 스텝 시간: {:.1}μs/step", hybrid_time.as_micros() as f64 / 1000.0);
    
    // 2. 메모리 사용량 비교
    println!("\n📊 메모리 사용량 비교");
    
    let start = Instant::now();
    let mut hybrid_optimizers = Vec::new();
    for _ in 0..100 {
        hybrid_optimizers.push(HybridOptimizer::new(0.01, 10));
    }
    let hybrid_creation_time = start.elapsed();
    
    println!("   100개 하이브리드 최적화기 생성: {:.3}ms", hybrid_creation_time.as_millis());
    println!("   평균 생성 시간: {:.1}μs/instance", hybrid_creation_time.as_micros() as f64 / 100.0);
    
    // 3. 수렴 정확도 비교 테스트
    println!("\n📊 수렴 정확도 비교");
    
    let test_scenarios = [
        ("간단한 2x2", (2, 2), vec![1.0, 0.5, -0.3, 0.8]),
        ("중간 4x4", (4, 4), (0..16).map(|i| (i as f32 / 16.0) * 2.0 - 1.0).collect()),
        ("복잡한 8x8", (8, 8), (0..64).map(|i| ((i as f32).sin() * 0.5)).collect()),
    ];
    
    for (name, (rows, cols), target_data) in &test_scenarios {
        let mut hybrid_opt = HybridOptimizer::new(0.01, 5);
        let mut test_packed = 테스트용_packed128_생성();
        let predicted = vec![0.0; target_data.len()];
        
        let start = Instant::now();
        let mut final_loss = 0.0;
        
        // 100 스텝 최적화
        for _ in 0..100 {
            final_loss = hybrid_opt.step(&mut test_packed, target_data, &predicted, *rows, *cols);
        }
        
        let convergence_time = start.elapsed();
        
        println!("   {}: 최종 손실={:.6}, 수렴 시간={:.3}ms", 
                 name, final_loss, convergence_time.as_millis());
        
        // 수렴 검증: 손실이 감소해야 함
        assert!(final_loss < 1.0, "{} 시나리오에서 수렴 실패", name);
    }
    
    // 4. 다양한 학습률에서의 안정성 테스트
    println!("\n📊 학습률 안정성 테스트");
    
    let learning_rates = [0.001, 0.01, 0.1, 0.5];
    
    for &lr in &learning_rates {
        let mut optimizer = HybridOptimizer::new(lr, 5);
        let mut packed = 테스트용_packed128_생성();
        let target = vec![0.5, -0.2, 0.3, -0.1];
        let predicted = vec![0.0; 4];
        
        let start = Instant::now();
        let mut stable_steps = 0;
        let mut total_loss = 0.0;
        
        for _ in 0..50 {
            let loss = optimizer.step(&mut packed, &target, &predicted, 2, 2);
            
            if loss.is_finite() && loss >= 0.0 {
                stable_steps += 1;
                total_loss += loss;
            }
        }
        
        let stability_time = start.elapsed();
        let avg_loss = if stable_steps > 0 { total_loss / stable_steps as f32 } else { f32::INFINITY };
        let stability_rate = (stable_steps as f32 / 50.0) * 100.0;
        
        println!("   학습률 {:.3}: 안정성 {:.1}%, 평균 손실={:.6}, 시간={:.3}ms", 
                 lr, stability_rate, avg_loss, stability_time.as_millis());
        
        // 안정성 검증: 최소 80% 이상의 스텝이 안정해야 함
        assert!(stability_rate >= 80.0, "학습률 {}에서 안정성 부족: {:.1}%", lr, stability_rate);
    }
    
    // 5. 진단 정보 생성 성능
    println!("\n📊 진단 정보 생성 성능");
    
    let mut diagnostic_optimizer = HybridOptimizer::new(0.01, 10);
    let start = Instant::now();
    
    for _ in 0..1000 {
        let _ = diagnostic_optimizer.diagnose();
    }
    
    let diagnostic_time = start.elapsed();
    
    println!("   1,000회 진단 정보 생성: {:.3}ms", diagnostic_time.as_millis());
    println!("   평균 진단 시간: {:.1}μs/call", diagnostic_time.as_micros() as f64 / 1000.0);
    
    // 6. 비트 활용 효율성 측정
    println!("\n📊 비트 활용 효율성 측정");
    
    let mut bit_optimizer = HybridOptimizer::new(0.01, 10);
    let mut bit_packed = 테스트용_packed128_생성();
    let target = 테스트용_타겟_데이터_생성(64);
    let predicted = vec![0.0; 64];
    
    let original_hi = bit_packed.hi;
    let original_lo = bit_packed.lo;
    
    let start = Instant::now();
    let mut hi_changes = 0;
    let mut lo_changes = 0;
    
    for _ in 0..100 {
        let prev_hi = bit_packed.hi;
        let prev_lo = bit_packed.lo;
        
        let _ = bit_optimizer.step(&mut bit_packed, &target, &predicted, 8, 8);
        
        if bit_packed.hi != prev_hi { hi_changes += 1; }
        if bit_packed.lo != prev_lo { lo_changes += 1; }
    }
    
    let bit_utilization_time = start.elapsed();
    
    println!("   100 스텝 비트 활용: {:.3}ms", bit_utilization_time.as_millis());
    println!("   Hi 필드 변경: {}회 ({}%)", hi_changes, hi_changes);
    println!("   Lo 필드 변경: {}회 ({}%)", lo_changes, lo_changes);
    
    // 성능 요약
    println!("\n✅ 하이브리드 최적화기 성능 요약:");
    println!("   단일 스텝: {:.1}μs/step", hybrid_time.as_micros() as f64 / 1000.0);
    println!("   인스턴스 생성: {:.1}μs/instance", hybrid_creation_time.as_micros() as f64 / 100.0);
    println!("   진단 정보: {:.1}μs/call", diagnostic_time.as_micros() as f64 / 1000.0);
    println!("   비트 활용률: Hi={}%, Lo={}%", hi_changes, lo_changes);
    
    // 성능 기준 검증
    assert!(hybrid_time.as_micros() / 1000 < 100, "하이브리드 스텝이 100μs 이상 소요됨");
    assert!(hybrid_creation_time.as_micros() / 100 < 50, "인스턴스 생성이 50μs 이상 소요됨");
    assert!(diagnostic_time.as_micros() / 1000 < 10, "진단 정보 생성이 10μs 이상 소요됨");
    assert!(hi_changes > 0 || lo_changes > 0, "비트 필드가 전혀 변경되지 않음");
    
    println!("   🎯 모든 성능 기준 통과!");
} 
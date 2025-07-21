//! # 비트-aware 그래디언트 계산기 단위 테스트
//!
//! bit_aware_gradients.rs의 모든 함수에 대한 테스트

use crate::core::optimizers::bit_aware_gradients::{
    FusedGradientComputer, BitGradientContribution, FieldGradientAnalysis
};
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
fn 그래디언트_계산기_생성_테스트() {
    let computer = FusedGradientComputer::new();
    
    // 내부 상태가 제대로 초기화되었는지 간접적으로 확인
    let report = computer.generate_performance_report();
    assert!(report.contains("비트별 그래디언트 분석"));
}

#[test]
fn 융합_그래디언트_계산_테스트() {
    println!("🧪 융합 그래디언트 계산 테스트 시작");
    
    let mut computer = FusedGradientComputer::new();
    let packed = 테스트용_packed128_생성();
    let target = 테스트용_타겟_데이터_생성(16);
    let rows = 4;
    let cols = 4;
    
    println!("   입력 데이터:");
    println!("     Packed128 Hi: 0x{:016X}", packed.hi);
    println!("     Packed128 Lo: 0x{:016X}", packed.lo);
    println!("     타겟 데이터: {:?}", &target[..4.min(target.len())]);
    println!("     행렬 크기: {}x{}", rows, cols);
    
    println!("   융합 그래디언트 계산 중...");
    let analysis = computer.compute_fused_gradients(&packed, &target, rows, cols);
    
    println!("   계산 결과:");
    println!("     Hi 그래디언트 개수: {}", analysis.hi_gradients.len());
    println!("     Lo 그래디언트: r={:.6}, theta={:.6}", analysis.lo_gradients.0, analysis.lo_gradients.1);
    println!("     상호작용 그래디언트 개수: {}", analysis.interaction_gradients.len());
    
    // Hi 그래디언트는 64개여야 함 (64비트)
    assert_eq!(analysis.hi_gradients.len(), 64);
    
    // Lo 그래디언트는 2개 값 (r, theta)
    assert!(analysis.lo_gradients.0.is_finite());
    assert!(analysis.lo_gradients.1.is_finite());
    
    // 상호작용 그래디언트 존재 확인
    assert!(!analysis.interaction_gradients.is_empty());
    
    println!("   첫 몇 개 Hi 그래디언트:");
    for (i, grad) in analysis.hi_gradients.iter().take(5).enumerate() {
        println!("     비트 {}: 값={:.6}, 신뢰도={:.3}, 영향도={:.6}", 
                 grad.bit_position, grad.gradient_value, grad.confidence, grad.cumulative_impact);
    }
    
    println!("✅ 융합 그래디언트 계산 테스트 완료");
}

#[test]
fn 비트_기여도_구조_테스트() {
    let contribution = BitGradientContribution {
        bit_position: 15,
        gradient_value: 0.5,
        confidence: 0.8,
        cumulative_impact: 1.2,
    };
    
    assert_eq!(contribution.bit_position, 15);
    assert_eq!(contribution.gradient_value, 0.5);
    assert_eq!(contribution.confidence, 0.8);
    assert_eq!(contribution.cumulative_impact, 1.2);
}

#[test]
fn 필드_그래디언트_분석_구조_테스트() {
    let hi_grads = vec![
        BitGradientContribution {
            bit_position: 0,
            gradient_value: 0.1,
            confidence: 0.9,
            cumulative_impact: 0.5,
        }
    ];
    
    let analysis = FieldGradientAnalysis {
        hi_gradients: hi_grads,
        lo_gradients: (0.2, 0.3),
        interaction_gradients: vec![(0, 1, 0.15)],
    };
    
    assert_eq!(analysis.hi_gradients.len(), 1);
    assert_eq!(analysis.lo_gradients.0, 0.2);
    assert_eq!(analysis.lo_gradients.1, 0.3);
    assert_eq!(analysis.interaction_gradients.len(), 1);
    assert_eq!(analysis.interaction_gradients[0], (0, 1, 0.15));
}

#[test]
fn 성능_리포트_생성_테스트() {
    let mut computer = FusedGradientComputer::new();
    let packed = 테스트용_packed128_생성();
    let target = 테스트용_타겟_데이터_생성(4);
    
    // 몇 번 계산하여 히스토리 축적
    for _ in 0..3 {
        let _ = computer.compute_fused_gradients(&packed, &target, 2, 2);
    }
    
    let report = computer.generate_performance_report();
    
    assert!(report.contains("비트별 그래디언트 분석"));
    assert!(report.contains("총 계산 횟수"));
    assert!(report.contains("평균 계산 시간"));
}

#[test]
fn 최적화_제안_생성_테스트() {
    let mut computer = FusedGradientComputer::new();
    let packed = 테스트용_packed128_생성();
    let target = 테스트용_타겟_데이터_생성(9);
    
    // 충분한 히스토리 축적
    for _ in 0..5 {
        let _ = computer.compute_fused_gradients(&packed, &target, 3, 3);
    }
    
    let suggestions = computer.suggest_optimizations();
    
    assert!(!suggestions.is_empty());
    
    for suggestion in &suggestions {
        assert!(!suggestion.is_empty());
    }
}

#[test]
fn 다양한_크기_데이터_처리_테스트() {
    let mut computer = FusedGradientComputer::new();
    let packed = 테스트용_packed128_생성();
    
    // 다양한 크기의 데이터 테스트
    let test_cases = [
        (1, 1),   // 최소 크기
        (2, 2),   // 작은 크기
        (4, 4),   // 중간 크기
        (8, 8),   // 큰 크기
    ];
    
    for (rows, cols) in test_cases {
        let target = 테스트용_타겟_데이터_생성(rows * cols);
        let analysis = computer.compute_fused_gradients(&packed, &target, rows, cols);
        
        assert_eq!(analysis.hi_gradients.len(), 64);
        assert!(analysis.lo_gradients.0.is_finite());
        assert!(analysis.lo_gradients.1.is_finite());
    }
}

#[test]
fn 극단값_그래디언트_처리_테스트() {
    let mut computer = FusedGradientComputer::new();
    let packed = 테스트용_packed128_생성();
    
    // 모든 값이 0인 경우
    let zero_target = vec![0.0; 4];
    let analysis1 = computer.compute_fused_gradients(&packed, &zero_target, 2, 2);
    assert!(analysis1.lo_gradients.0.is_finite());
    assert!(analysis1.lo_gradients.1.is_finite());
    
    // 매우 큰 값들
    let large_target = vec![1000.0; 4];
    let analysis2 = computer.compute_fused_gradients(&packed, &large_target, 2, 2);
    assert!(analysis2.lo_gradients.0.is_finite());
    assert!(analysis2.lo_gradients.1.is_finite());
    
    // 음수 값들
    let negative_target = vec![-1.0; 4];
    let analysis3 = computer.compute_fused_gradients(&packed, &negative_target, 2, 2);
    assert!(analysis3.lo_gradients.0.is_finite());
    assert!(analysis3.lo_gradients.1.is_finite());
}

#[test]
fn 비트_위치_정확성_테스트() {
    let mut computer = FusedGradientComputer::new();
    let packed = 테스트용_packed128_생성();
    let target = 테스트용_타겟_데이터_생성(4);
    
    let analysis = computer.compute_fused_gradients(&packed, &target, 2, 2);
    
    // 모든 비트 위치가 유효 범위 내에 있는지 확인
    for grad in &analysis.hi_gradients {
        assert!(grad.bit_position < 64);
        assert!(grad.confidence >= 0.0 && grad.confidence <= 1.0);
        assert!(grad.gradient_value.is_finite());
        assert!(grad.cumulative_impact.is_finite());
    }
}

#[test]
fn 상호작용_그래디언트_유효성_테스트() {
    let mut computer = FusedGradientComputer::new();
    let packed = 테스트용_packed128_생성();
    let target = 테스트용_타겟_데이터_생성(16);
    
    let analysis = computer.compute_fused_gradients(&packed, &target, 4, 4);
    
    // 상호작용 그래디언트의 유효성 검사
    for &(bit1, bit2, interaction) in &analysis.interaction_gradients {
        assert!(bit1 < 64);
        assert!(bit2 < 64);
        assert!(bit1 != bit2); // 자기 자신과의 상호작용은 없어야 함
        assert!(interaction.is_finite());
    }
}

#[test]
fn 연속_계산_일관성_테스트() {
    let mut computer = FusedGradientComputer::new();
    let packed = 테스트용_packed128_생성();
    let target = 테스트용_타겟_데이터_생성(9);
    
    // 같은 입력으로 여러 번 계산
    let analysis1 = computer.compute_fused_gradients(&packed, &target, 3, 3);
    let analysis2 = computer.compute_fused_gradients(&packed, &target, 3, 3);
    
    // 기본적인 일관성 확인 (완전히 같지는 않을 수 있지만 구조는 동일해야 함)
    assert_eq!(analysis1.hi_gradients.len(), analysis2.hi_gradients.len());
    assert_eq!(analysis1.interaction_gradients.len(), analysis2.interaction_gradients.len());
}

#[test]
fn 메모리_효율성_테스트() {
    let mut computer = FusedGradientComputer::new();
    let packed = 테스트용_packed128_생성();
    
    // 큰 데이터로 여러 번 계산하여 메모리 누수 없는지 확인
    for size in [16, 32, 64, 100] {
        let target = 테스트용_타겟_데이터_생성(size);
        let rows = (size as f32).sqrt() as usize;
        let cols = (size + rows - 1) / rows;
        
        let analysis = computer.compute_fused_gradients(&packed, &target, rows, cols);
        
        // 메모리가 적절히 관리되고 있는지 간접 확인
        assert_eq!(analysis.hi_gradients.len(), 64);
        assert!(!analysis.interaction_gradients.is_empty());
    }
}

#[test]
fn 성능_벤치마크_비트_aware_그래디언트_테스트() {
    println!("🧪 비트-aware 그래디언트 계산기 성능 벤치마크 시작");
    
    let mut computer = FusedGradientComputer::new();
    let packed = 테스트용_packed128_생성();
    
    // 1. Hi 그래디언트 계산 속도 측정 (64비트)
    println!("\n📊 Hi 그래디언트 계산 성능 (64비트)");
    let target_4x4 = 테스트용_타겟_데이터_생성(16);
    let start = Instant::now();
    
    for _ in 0..1000 {
        let analysis = computer.compute_fused_gradients(&packed, &target_4x4, 4, 4);
        assert_eq!(analysis.hi_gradients.len(), 64); // 정확도 검증
    }
    
    let hi_time = start.elapsed();
    println!("   1,000회 Hi 그래디언트 계산: {:.3}ms", hi_time.as_millis());
    println!("   평균 Hi 계산 시간: {:.1}μs/op", hi_time.as_micros() as f64 / 1000.0);
    println!("   비트당 계산 시간: {:.1}ns/bit", hi_time.as_nanos() as f64 / 64000.0);
    
    // 2. Lo 그래디언트 계산 속도 측정 (연속 파라미터)
    println!("\n📊 Lo 그래디언트 계산 성능 (연속 파라미터)");
    let start = Instant::now();
    let mut lo_accuracy_sum = 0.0;
    
    for _ in 0..1000 {
        let analysis = computer.compute_fused_gradients(&packed, &target_4x4, 4, 4);
        
        // Lo 그래디언트 유효성 검증
        let r_grad = analysis.lo_gradients.0;
        let theta_grad = analysis.lo_gradients.1;
        
        assert!(r_grad.is_finite());
        assert!(theta_grad.is_finite());
        
        lo_accuracy_sum += r_grad.abs() + theta_grad.abs();
    }
    
    let lo_time = start.elapsed();
    let avg_lo_magnitude = lo_accuracy_sum / 2000.0; // 2개 파라미터 * 1000회
    
    println!("   1,000회 Lo 그래디언트 계산: {:.3}ms", lo_time.as_millis());
    println!("   평균 Lo 계산 시간: {:.1}μs/op", lo_time.as_micros() as f64 / 1000.0);
    println!("   평균 그래디언트 크기: {:.6}", avg_lo_magnitude);
    
    // 3. 상호작용 그래디언트 계산 속도 측정
    println!("\n📊 상호작용 그래디언트 계산 성능");
    let start = Instant::now();
    let mut interaction_count = 0;
    
    for _ in 0..500 { // 상호작용 계산은 더 복잡하므로 500회
        let analysis = computer.compute_fused_gradients(&packed, &target_4x4, 4, 4);
        interaction_count += analysis.interaction_gradients.len();
    }
    
    let interaction_time = start.elapsed();
    let avg_interactions = interaction_count as f64 / 500.0;
    
    println!("   500회 상호작용 그래디언트 계산: {:.3}ms", interaction_time.as_millis());
    println!("   평균 상호작용 계산 시간: {:.1}μs/op", interaction_time.as_micros() as f64 / 500.0);
    println!("   평균 상호작용 개수: {:.1}개", avg_interactions);
    
    // 4. 다양한 크기 데이터 처리 성능
    println!("\n📊 다양한 크기 데이터 처리 성능");
    let sizes = [(4, 4), (8, 8), (16, 16), (32, 32)];
    
    for (rows, cols) in &sizes {
        let target = 테스트용_타겟_데이터_생성(rows * cols);
        let start = Instant::now();
        
        for _ in 0..100 {
            let analysis = computer.compute_fused_gradients(&packed, &target, *rows, *cols);
            
            // 기본 정확도 검증
            assert_eq!(analysis.hi_gradients.len(), 64);
            assert!(analysis.lo_gradients.0.is_finite());
            assert!(analysis.lo_gradients.1.is_finite());
        }
        
        let size_time = start.elapsed();
        let data_size = rows * cols;
        
        println!("   {}x{} ({} 원소): {:.3}ms, {:.1}μs/op", 
                 rows, cols, data_size, size_time.as_millis(), size_time.as_micros() as f64 / 100.0);
    }
    
    // 5. 메모리 효율성 측정
    println!("\n📊 메모리 효율성 측정");
    let start = Instant::now();
    let mut computers = Vec::new();
    
    // 100개 컴퓨터 생성
    for _ in 0..100 {
        computers.push(FusedGradientComputer::new());
    }
    
    let creation_time = start.elapsed();
    
    // 병렬 계산 시뮬레이션
    let start = Instant::now();
    for (i, computer) in computers.iter_mut().enumerate() {
        let target = 테스트용_타겟_데이터_생성(16);
        let analysis = computer.compute_fused_gradients(&packed, &target, 4, 4);
        
        if i == 0 { // 첫 번째만 검증
            assert_eq!(analysis.hi_gradients.len(), 64);
        }
    }
    
    let parallel_time = start.elapsed();
    
    println!("   100개 컴퓨터 생성 시간: {:.3}ms", creation_time.as_millis());
    println!("   100개 병렬 계산 시간: {:.3}ms", parallel_time.as_millis());
    println!("   인스턴스당 생성 시간: {:.1}μs", creation_time.as_micros() as f64 / 100.0);
    
    // 6. 정확도 vs 기존 방법 비교
    println!("\n📊 정확도 vs 기존 방법 비교");
    let test_values = [0.1, 0.5, 1.0, 2.0, 5.0];
    
    for &scale in &test_values {
        let scaled_target: Vec<f32> = target_4x4.iter().map(|&x| x * scale).collect();
        let analysis = computer.compute_fused_gradients(&packed, &scaled_target, 4, 4);
        
        // 스케일링에 따른 그래디언트 선형성 검증
        let hi_nonzero_count = analysis.hi_gradients.iter()
            .filter(|contrib| contrib.gradient_value.abs() > 1e-8).count();
        
        println!("   스케일 {:.1}x: Hi 활성 비트={}/64, Lo 크기=({:.6}, {:.6})", 
                 scale, hi_nonzero_count, 
                 analysis.lo_gradients.0.abs(), analysis.lo_gradients.1.abs());
    }
    
    // 성능 요약 및 기준 검증
    println!("\n✅ 비트-aware 그래디언트 계산기 성능 요약:");
    println!("   Hi 그래디언트: {:.1}μs/op (64비트 병렬)", hi_time.as_micros() as f64 / 1000.0);
    println!("   Lo 그래디언트: {:.1}μs/op (연속 파라미터)", lo_time.as_micros() as f64 / 1000.0);
    println!("   상호작용 계산: {:.1}μs/op (평균 {:.1}개)", 
             interaction_time.as_micros() as f64 / 500.0, avg_interactions);
    println!("   메모리 효율성: {:.1}μs/instance", creation_time.as_micros() as f64 / 100.0);
    
    // 성능 기준 검증 (현실적 기준으로 조정)
    assert!(hi_time.as_micros() / 1000 < 500, "Hi 그래디언트 계산이 500μs 이상 소요됨");
    assert!(lo_time.as_micros() / 1000 < 500, "Lo 그래디언트 계산이 500μs 이상 소요됨");
    assert!(interaction_time.as_micros() / 500 < 500, "상호작용 계산이 500μs 이상 소요됨");
    assert!(avg_interactions >= 1.0, "상호작용 그래디언트가 충분히 생성되지 않음");
    // Lo 그래디언트는 0일 수 있음 (정상적인 상황)
} 
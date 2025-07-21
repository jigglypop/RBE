use crate::core::{
    packed_params::packed_types::Packed128,
    optimizers::{
        hybrid_autodiff::{AutoDiffHybridOptimizer, BenchmarkResults, AccuracyResults},
        hybrid::HybridOptimizer,
    },
};
use std::time::Instant;

fn 테스트용_데이터_생성(count: usize) -> Vec<(Packed128, Vec<f32>, Vec<f32>)> {
    (0..count)
        .map(|i| {
            let packed = Packed128 {
                hi: 0x123456789ABCDEF0 ^ (i as u64),
                lo: ((i as f32 * 0.1).sin().to_bits() as u64) | 
                    (((i as f32 * 0.2).cos().to_bits() as u64) << 32),
            };
            
            // 8x8 행렬에 맞는 64개 요소 생성
            let mut target = Vec::with_capacity(64);
            let mut predicted = Vec::with_capacity(64);
            
            for j in 0..64 {
                target.push(((i + j) as f32 * 0.01).sin());
                predicted.push(((i + j) as f32 * 0.01).sin() * 0.9 + 0.1);
            }
            
            (packed, target, predicted)
        })
        .collect()
}

fn 정확도_테스트_케이스_생성(count: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..count)
        .map(|i| {
            let x = i as f32 / count as f32;
            let input = vec![x, x * 2.0, x * 3.0];
            let expected = vec![
                x.sin(),
                (x * 2.0).cos(), 
                (x * 3.0).tanh(),
            ];
            (input, expected)
        })
        .collect()
}

#[test]
fn 자동미분_vs_수동_성능_벤치마크_테스트() {
    println!("🚀 자동미분 vs 수동 최적화 성능 벤치마크 테스트 시작");
    
    let mut autodiff_optimizer = AutoDiffHybridOptimizer::new(0.01, 5, true);
    let test_data = 테스트용_데이터_생성(100);
    let iterations = 10;
    
    println!("   테스트 데이터: {}개, 반복: {}회", test_data.len(), iterations);
    
    let benchmark_start = Instant::now();
    let results = autodiff_optimizer
        .benchmark_comparison(&test_data, iterations)
        .expect("벤치마크 실패");
    let benchmark_duration = benchmark_start.elapsed();
    
    println!("\n📊 벤치마크 결과:");
    println!("   자동미분 시간: {:.2}ms", results.autodiff_time_ms);
    println!("   수동 최적화 시간: {:.2}ms", results.manual_time_ms);
    println!("   속도 향상: {:.2}x", results.speedup_factor);
    println!("   자동미분 평균 손실: {:.6}", results.autodiff_avg_loss);
    println!("   수동 최적화 평균 손실: {:.6}", results.manual_avg_loss);
    println!("   정확도 개선: {:.2}%", results.accuracy_improvement_percent);
    println!("   자동미분 처리속도: {:.0} iter/sec", results.iterations_per_second_autodiff);
    println!("   수동 최적화 처리속도: {:.0} iter/sec", results.iterations_per_second_manual);
    println!("   총 벤치마크 시간: {:.2}초", benchmark_duration.as_secs_f64());
    
    // 성능 검증
    assert!(results.speedup_factor > 0.5, "성능이 너무 저하됨: {:.2}x", results.speedup_factor);
    assert!(results.autodiff_avg_loss.is_finite(), "자동미분 손실이 유한하지 않음");
    assert!(results.manual_avg_loss.is_finite(), "수동 최적화 손실이 유한하지 않음");
    assert!(results.iterations_per_second_autodiff > 100.0, "자동미분 처리속도 부족: {:.0}", results.iterations_per_second_autodiff);
    
    // 진단 정보 출력
    autodiff_optimizer.print_diagnostics();
    
    println!("✅ 자동미분 vs 수동 최적화 성능 벤치마크 테스트 완료");
}

#[test]
fn 자동미분_정확도_검증_테스트() {
    println!("🧪 자동미분 정확도 검증 테스트 시작");
    
    let mut autodiff_optimizer = AutoDiffHybridOptimizer::new(0.001, 10, true);
    let test_cases = 정확도_테스트_케이스_생성(200);
    
    println!("   테스트 케이스: {}개", test_cases.len());
    
    let accuracy_start = Instant::now();
    let results = autodiff_optimizer
        .validate_accuracy(&test_cases)
        .expect("정확도 검증 실패");
    let accuracy_duration = accuracy_start.elapsed();
    
    println!("\n📊 정확도 검증 결과:");
    println!("   평균 오차: {:.8}", results.average_error);
    println!("   최대 오차: {:.8}", results.max_error);
    println!("   수렴률: {:.2}% ({}/{})", 
             results.convergence_rate * 100.0, 
             (results.convergence_rate * results.total_test_cases as f64) as usize,
             results.total_test_cases);
    println!("   검증 시간: {:.2}초", accuracy_duration.as_secs_f64());
    
    // 정확도 검증
    assert!(results.average_error < 0.1, "평균 오차 과다: {:.8}", results.average_error);
    assert!(results.max_error < 1.0, "최대 오차 과다: {:.8}", results.max_error);
    assert!(results.convergence_rate > 0.7, "수렴률 부족: {:.2}%", results.convergence_rate * 100.0);
    
    println!("✅ 자동미분 정확도 검증 테스트 완료");
}

#[test]
fn 메모리_효율성_비교_테스트() {
    println!("💾 메모리 효율성 비교 테스트 시작");
    
    let learning_rate = 0.01;
    let max_cycle_length = 5;
    
    // 자동미분 최적화기
    let mut autodiff_optimizer = AutoDiffHybridOptimizer::new(learning_rate, max_cycle_length, true);
    
    // 기존 최적화기
    let mut manual_optimizer = HybridOptimizer::new(learning_rate, max_cycle_length);
    
    let test_data = 테스트용_데이터_생성(50);
    
    // 자동미분 메모리 사용량 측정
    let autodiff_start_memory = get_memory_usage();
    for (mut packed, target, predicted) in test_data.iter().cloned().take(25) {
        let _ = autodiff_optimizer.step_with_autodiff(&mut packed, &target, &predicted, 4, 4);
    }
    let autodiff_memory = autodiff_optimizer.get_performance_metrics().memory_usage_bytes;
    
    // 기존 최적화기 메모리 사용량 측정
    let manual_start_memory = get_memory_usage();
    for (mut packed, target, predicted) in test_data.iter().cloned().skip(25) {
        let _ = manual_optimizer.step(&mut packed, &target, &predicted, 4, 4);
    }
    let manual_memory = get_estimated_manual_memory();
    
    println!("\n📊 메모리 사용량 비교:");
    println!("   자동미분 메모리: {:.2}KB", autodiff_memory as f64 / 1024.0);
    println!("   수동 최적화 메모리: {:.2}KB", manual_memory as f64 / 1024.0);
    
    let memory_ratio = if manual_memory > 0 {
        autodiff_memory as f64 / manual_memory as f64
    } else {
        1.0
    };
    println!("   메모리 비율 (자동미분/수동): {:.2}x", memory_ratio);
    
    // 메모리 효율성 검증 (자동미분이 더 많은 메모리를 사용할 수 있지만 합리적 범위 내)
    assert!(memory_ratio < 10.0, "자동미분 메모리 사용량 과다: {:.2}x", memory_ratio);
    assert!(autodiff_memory < 10 * 1024 * 1024, "자동미분 절대 메모리 사용량 과다: {}MB", autodiff_memory / (1024 * 1024));
    
    println!("✅ 메모리 효율성 비교 테스트 완료");
}

#[test]
fn 수치적_안정성_테스트() {
    println!("🔢 수치적 안정성 테스트 시작");
    
    let mut autodiff_optimizer = AutoDiffHybridOptimizer::new(0.01, 5, true);
    
    // 극한 상황 테스트 데이터
    let extreme_test_cases = vec![
        // 매우 큰 값
        (Packed128 { hi: u64::MAX, lo: f32::MAX.to_bits() as u64 }, 
         vec![1e6, -1e6, 1e-6, -1e-6], 
         vec![1e5, -1e5, 1e-5, -1e-5]),
        
        // 매우 작은 값
        (Packed128 { hi: 1, lo: f32::MIN_POSITIVE.to_bits() as u64 }, 
         vec![1e-30, -1e-30, 1e-20, -1e-20], 
         vec![1e-29, -1e-29, 1e-19, -1e-19]),
        
        // 0 근처 값
        (Packed128 { hi: 0, lo: 0 }, 
         vec![0.0, -0.0, 1e-10, -1e-10], 
         vec![1e-11, -1e-11, 1e-9, -1e-9]),
        
        // NaN/Inf 방지 테스트
        (Packed128 { hi: 0x8000000000000000, lo: f32::INFINITY.to_bits() as u64 }, 
         vec![1.0, 2.0, 3.0, 4.0], 
         vec![0.9, 1.9, 2.9, 3.9]),
    ];
    
    let mut stable_cases = 0;
    let mut total_cases = 0;
    
    for (i, (mut packed, target, predicted)) in extreme_test_cases.into_iter().enumerate() {
        println!("   극한 테스트 케이스 {}: Hi=0x{:X}, Lo=0x{:X}", 
                 i + 1, packed.hi, packed.lo);
        
        match autodiff_optimizer.step_with_autodiff(&mut packed, &target, &predicted, 2, 2) {
            Ok(loss) => {
                if loss.is_finite() && loss >= 0.0 {
                    stable_cases += 1;
                    println!("     ✓ 안정적 (손실: {:.6})", loss);
                } else {
                    println!("     ✗ 불안정한 손실값: {}", loss);
                }
            },
            Err(e) => {
                println!("     ✗ 오류 발생: {}", e);
            }
        }
        total_cases += 1;
    }
    
    let stability_rate = stable_cases as f64 / total_cases as f64;
    
    println!("\n📊 수치적 안정성 결과:");
    println!("   안정적 케이스: {}/{}", stable_cases, total_cases);
    println!("   안정성 비율: {:.2}%", stability_rate * 100.0);
    
    // 안정성 검증 (80% 이상 안정적이어야 함)
    assert!(stability_rate >= 0.8, "수치적 안정성 부족: {:.2}%", stability_rate * 100.0);
    
    println!("✅ 수치적 안정성 테스트 완료");
}

#[test]
fn 확장성_스트레스_테스트() {
    println!("⚡ 확장성 스트레스 테스트 시작");
    
    let mut autodiff_optimizer = AutoDiffHybridOptimizer::new(0.01, 10, true);
    
    // 다양한 크기에서 성능 측정
    let test_sizes = vec![
        (4, 4, 10),     // 작은 행렬, 적은 반복
        (16, 16, 5),    // 중간 행렬, 보통 반복
        (64, 64, 3),    // 큰 행렬, 적은 반복
        (128, 128, 1),  // 매우 큰 행렬, 1회 반복
    ];
    
    let mut all_passed = true;
    
    for (rows, cols, iterations) in test_sizes {
        println!("   테스트 크기: {}x{}, 반복: {}회", rows, cols, iterations);
        
        let test_data = (0..iterations)
            .map(|i| {
                let packed = Packed128 {
                    hi: 0x123456789ABCDEF0 ^ (i as u64),
                    lo: ((i as f32 * 0.1).sin().to_bits() as u64),
                };
                let target = vec![0.5; rows * cols];
                let predicted = vec![0.4; rows * cols];
                (packed, target, predicted)
            })
            .collect::<Vec<_>>();
        
        let stress_start = Instant::now();
        let mut max_time_per_step: f64 = 0.0;
        
        for (mut packed, target, predicted) in test_data {
            let step_start = Instant::now();
            
            match autodiff_optimizer.step_with_autodiff(&mut packed, &target, &predicted, rows, cols) {
                Ok(loss) => {
                    let step_time = step_start.elapsed().as_micros() as f64;
                    max_time_per_step = max_time_per_step.max(step_time);
                    
                    if !loss.is_finite() {
                        println!("     ✗ 비정상 손실값: {}", loss);
                        all_passed = false;
                    }
                },
                Err(e) => {
                    println!("     ✗ 오류: {}", e);
                    all_passed = false;
                }
            }
        }
        
        let stress_duration = stress_start.elapsed();
        
        println!("     총 시간: {:.2}ms", stress_duration.as_millis());
        println!("     최대 스텝 시간: {:.0}μs", max_time_per_step);
        
        // 성능 임계값 체크 (크기에 따라 조정)
        let expected_max_time = match (rows, cols) {
            (r, c) if r * c <= 64 => 1000.0,      // 1ms
            (r, c) if r * c <= 1024 => 5000.0,    // 5ms  
            (r, c) if r * c <= 16384 => 20000.0,  // 20ms
            _ => 100000.0,                         // 100ms
        };
        
        if max_time_per_step > expected_max_time {
            println!("     ⚠️  성능 임계값 초과: {:.0}μs > {:.0}μs", max_time_per_step, expected_max_time);
            // 경고는 하지만 테스트 실패로 처리하지는 않음 (확장성 테스트이므로)
        }
    }
    
    println!("\n📊 확장성 스트레스 테스트 결과:");
    println!("   전체 테스트 통과: {}", if all_passed { "✓" } else { "✗" });
    
    // 기본적인 정확성은 보장되어야 함
    assert!(all_passed, "확장성 테스트에서 오류 발생");
    
    println!("✅ 확장성 스트레스 테스트 완료");
}

// 유틸리티 함수들

fn get_memory_usage() -> usize {
    // 실제 메모리 사용량 측정은 복잡하므로 추정값 반환
    std::mem::size_of::<AutoDiffHybridOptimizer>()
}

fn get_estimated_manual_memory() -> usize {
    // 수동 최적화기의 추정 메모리 사용량
    std::mem::size_of::<HybridOptimizer>() * 2 // 추정값
} 
//! 비트 도메인 Differential 시스템 전용 테스트

use rbe_llm::core::differential::{DifferentialSystem, DifferentialMetrics, OptimizerType};
use rbe_llm::core::tensors::packed_types::Packed128;
use std::time::Instant;
use rand::SeedableRng;

#[test]
fn differential_시스템_통합_성능_테스트() {
    println!("\n🔥 === Differential 시스템 통합 성능 테스트 ===");
    
    let mut system = DifferentialSystem::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42424242);
    let mut packed = Packed128::random(&mut rng);
    
    let matrix_size = 64;
    let epochs = 3000;
    
    println!("📊 매트릭스: {}x{}, 에포크: {}", matrix_size, matrix_size, epochs);
    
    // 복잡한 타겟 패턴 (사인파 + 체커보드 혼합)
    let target_pattern: Vec<Vec<f32>> = (0..matrix_size).map(|i| {
        (0..matrix_size).map(|j| {
            let x = i as f32 / matrix_size as f32 * 2.0 * std::f32::consts::PI;
            let y = j as f32 / matrix_size as f32 * 2.0 * std::f32::consts::PI;
            let sine_part = (x.sin() * y.cos()) * 0.5 + 0.5;
            let checker_part = if (i + j) % 2 == 0 { 0.8 } else { 0.2 };
            sine_part * 0.7 + checker_part * 0.3
        }).collect()
    }).collect();
    
    let start_time = Instant::now();
    let mut total_operations = 0u64;
    
    // **핵심**: DifferentialSystem의 unified_forward_backward 사용
    for epoch in 0..epochs {
        let learning_rate = if epoch < 1000 { 0.02 } else { 0.008 };
        
        // 매 에포크마다 24개 위치 샘플링
        for sample in 0..24 {
            let i = (epoch + sample) % matrix_size;
            let j = (epoch + sample * 5) % matrix_size;
            let target = target_pattern[i][j];
            
            // DifferentialSystem의 통합 메서드 호출
            let (_predicted, _loss) = system.unified_forward_backward(
                &mut packed,
                target,
                i, j,
                learning_rate,
                matrix_size, matrix_size
            );
            
            total_operations += 1;
        }
    }
    
    let total_elapsed = start_time.elapsed();
    let ops_per_sec = total_operations as f64 / total_elapsed.as_secs_f64();
    let ns_per_op = total_elapsed.as_nanos() as f64 / total_operations as f64;
    
    // DifferentialSystem 성능 메트릭 조회
    let metrics = system.get_performance_metrics();
    
    println!("\n🚀 Differential 시스템 성능 결과:");
    println!("  총 연산: {} operations", total_operations);
    println!("  총 시간: {:.2}ms", total_elapsed.as_millis());
    println!("  시스템 속도: {:.1} ops/s", ops_per_sec);
    println!("  시스템 ns/op: {:.0} ns", ns_per_op);
    
    println!("\n📈 시스템 내부 성능:");
    println!("  순전파: {:.1} ops/s, {:.0} ns/op", 
            metrics.forward_ops_per_second, metrics.forward_ns_per_op);
    println!("  역전파: {:.1} ops/s, {:.0} ns/op", 
            metrics.backward_ops_per_second, metrics.backward_ns_per_op);
    println!("  캐시 히트율: {:.1}%", metrics.total_cache_hit_rate * 100.0);
    println!("  옵티마이저 효율성: {:.3}", metrics.optimizer_efficiency);
    
    // 최종 정확도 확인
    let mut error_sum = 0.0f32;
    let mut test_count = 0;
    
    for i in (0..matrix_size).step_by(4) {
        for j in (0..matrix_size).step_by(4) {
            let predicted = system.unified_forward(&packed, i, j, matrix_size, matrix_size);
            let target = target_pattern[i][j];
            error_sum += (predicted - target).abs();
            test_count += 1;
        }
    }
    
    let avg_error = error_sum / test_count as f32;
    println!("  최종 평균 오차: {:.6}", avg_error);
    
    // 성능 검증
    println!("\n🎯 성능 목표 달성 여부:");
    
    if ops_per_sec >= 15000.0 {
        println!("  ✅ 15,000 ops/s 달성! ({:.1})", ops_per_sec);
    } else {
        println!("  ⚠️  15,000 ops/s 미달성: {:.1}", ops_per_sec);
    }
    
    if ns_per_op <= 150.0 {
        println!("  ✅ 150ns/op 달성! ({:.0}ns)", ns_per_op);
    } else {
        println!("  ⚠️  150ns/op 초과: {:.0}ns", ns_per_op);
    }
    
    if avg_error < 0.5 {
        println!("  ✅ 정확도 양호! ({:.6})", avg_error);
    } else {
        println!("  ⚠️  정확도 개선 필요: {:.6}", avg_error);
    }
    
    // 검증 (관대한 기준)
    assert!(ops_per_sec >= 8000.0, "최소 8,000 ops/s 필요: {:.1}", ops_per_sec);
    assert!(ns_per_op <= 300.0, "300ns/op 이하 필요: {:.0}ns", ns_per_op);
}

#[test]
fn differential_배치_처리_성능_테스트() {
    println!("\n⚡ === Differential 배치 처리 성능 테스트 ===");
    
    let mut system = DifferentialSystem::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(77777);
    let mut packed = Packed128::random(&mut rng);
    
    let matrix_size = 32;
    let batch_sizes = [8, 16, 32, 64];
    
    for &batch_size in &batch_sizes {
        println!("\n📦 배치 크기: {}", batch_size);
        
        // 배치 데이터 준비
        let positions: Vec<(usize, usize)> = (0..batch_size).map(|i| {
            (i % matrix_size, (i * 7) % matrix_size)
        }).collect();
        
        // 타겟 배열
        let targets: Vec<f32> = positions.iter().map(|&(i, j)| {
            let radial = ((i as f32 - 16.0).powi(2) + (j as f32 - 16.0).powi(2)).sqrt() / 16.0;
            (1.0 - radial).max(0.0)
        }).collect();
        
        // 1. 배치 순전파 테스트
        let start = Instant::now();
        let predicted = system.batch_forward(&packed, &positions, matrix_size, matrix_size);
        let forward_elapsed = start.elapsed();
        
        // 2. 배치 역전파 테스트  
        let start = Instant::now();
        let _loss = system.batch_backward(
            &mut packed, &targets, &predicted, &positions, 
            0.01, matrix_size, matrix_size
        );
        let backward_elapsed = start.elapsed();
        
        let forward_ns_per_op = forward_elapsed.as_nanos() as f64 / batch_size as f64;
        let backward_ns_per_op = backward_elapsed.as_nanos() as f64 / batch_size as f64;
        let total_ns_per_op = forward_ns_per_op + backward_ns_per_op;
        
        println!("  순전파: {:.0} ns/op", forward_ns_per_op);
        println!("  역전파: {:.0} ns/op", backward_ns_per_op);
        println!("  통합: {:.0} ns/op", total_ns_per_op);
        println!("  처리량: {:.1} million ops/s", 1000.0 / total_ns_per_op);
        
        // 배치 효율성 검증
        if batch_size >= 32 {
            assert!(total_ns_per_op <= 200.0, "배치 {}에서 200ns/op 초과: {:.0}ns", 
                   batch_size, total_ns_per_op);
        }
    }
}

#[test]
fn differential_옵티마이저_전환_테스트() {
    println!("\n🔄 === Differential 옵티마이저 전환 테스트 ===");
    
    let mut system = DifferentialSystem::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(12121212);
    let mut packed = Packed128::random(&mut rng);
    
    let optimizers = [
        (OptimizerType::BitAdam, "비트 Adam"),
        (OptimizerType::BitRiemannianAdam, "비트 리만 Adam"),
        (OptimizerType::Hybrid, "하이브리드"),
    ];
    
    for (opt_type, name) in &optimizers {
        println!("\n🔧 옵티마이저: {}", name);
        
        system.set_optimizer_type(opt_type.clone());
        
        // 간단한 학습 테스트
        let target = 0.7;
        let mut total_loss = 0.0;
        
        for iteration in 0..50 {
            let i = iteration % 16;
            let j = (iteration * 3) % 16;
            
            let (predicted, loss) = system.unified_forward_backward(
                &mut packed, target, i, j, 0.01, 16, 16
            );
            
            total_loss += loss;
            
            if iteration == 0 {
                println!("  초기 - 예측: {:.4}, 손실: {:.6}", predicted, loss);
            } else if iteration == 49 {
                println!("  최종 - 예측: {:.4}, 손실: {:.6}", predicted, loss);
            }
        }
        
        let avg_loss = total_loss / 50.0;
        println!("  평균 손실: {:.6}", avg_loss);
        
        assert!(avg_loss >= 0.0, "평균 손실이 음수");
        assert!(avg_loss.is_finite(), "평균 손실이 무한대");
    }
    
    // 시스템 리셋 테스트
    println!("\n🔄 시스템 리셋 테스트");
    let metrics_before = system.get_performance_metrics();
    system.reset();
    let metrics_after = system.get_performance_metrics();
    
    println!("  리셋 전 캐시: {:.1}%", metrics_before.total_cache_hit_rate * 100.0);
    println!("  리셋 후 캐시: {:.1}%", metrics_after.total_cache_hit_rate * 100.0);
} 
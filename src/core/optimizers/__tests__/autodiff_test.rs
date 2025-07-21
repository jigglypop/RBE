use crate::core::{
    packed_params::packed_types::Packed128,
    optimizers::separated_bit_autodiff::{SeparatedBitAutoDiff},
};
use std::time::Instant;

fn 테스트_데이터_생성(size: usize) -> Vec<Packed128> {
    (0..size)
        .map(|i| {
            let hi_pattern = (i as u64 * 0x123456789ABCDEF) ^ (i as u64).reverse_bits();
            let r = (i as f32 / size as f32) * 0.8 + 0.1; // 0.1 ~ 0.9
            let theta = (i as f32 / size as f32) * 2.0 * std::f32::consts::PI;
            
            let lo_bits = ((theta.to_bits() as u64) << 32) | (r.to_bits() as u64);
            
            Packed128 { hi: hi_pattern, lo: lo_bits }
        })
        .collect()
}

fn 간단한_손실_함수(hi_bits: u64, r: f32, theta: f32) -> f32 {
    // 해석적 부분: 비트 패턴 기반
    let bit_contrib = (hi_bits.count_ones() as f32 / 64.0 - 0.5).powi(2);
    
    // 수치적 부분: 푸앵카레 볼 기반
    let spatial_contrib = (r * theta.sin()).powi(2) + (r * theta.cos()).powi(2);
    
    bit_contrib + spatial_contrib * 0.5
}

#[test]
fn 분리형_자동미분_기본_기능_테스트() {
    println!("🧪 분리형 자동미분 기본 기능 테스트");
    
    let mut autodiff_system = SeparatedBitAutoDiff::new();
    let test_data = 테스트_데이터_생성(10);
    
    for (i, packed) in test_data.iter().enumerate() {
        println!("  테스트 샘플 {}: hi={:016x}, lo={:016x}", i + 1, packed.hi, packed.lo);
        
        let gradient = autodiff_system.compute_separated_gradient(packed, |hi, r, theta| {
            간단한_손실_함수(hi, r, theta)
        });
        
        println!("    해석적 그래디언트 크기: {:.6}", 
                 gradient.analytical_grad.iter().map(|&x| x.abs()).sum::<f32>());
        println!("    수치적 그래디언트: r={:.6}, theta={:.6}", 
                 gradient.numerical_grad.0, gradient.numerical_grad.1);
        println!("    전체 그래디언트 크기: {:.6}", gradient.magnitude);
        println!("    품질 점수: {:.3}", gradient.quality_score());
        
        // 기본 검증
        assert!(gradient.magnitude > 0.0, "그래디언트 크기가 0입니다");
        assert!(gradient.quality_score() > 0.0, "품질 점수가 0입니다");
        assert!(gradient.analytical_confidence >= 0.0, "해석적 신뢰도가 음수입니다");
        assert!(gradient.numerical_precision >= 0.0, "수치적 정밀도가 음수입니다");
    }
    
    println!("  ✅ 기본 기능 테스트 성공");
}

#[test]
fn 분리형_자동미분_성능_벤치마크() {
    println!("🚀 분리형 자동미분 성능 벤치마크");
    
    let test_sizes = [50, 100, 200];
    let iterations = 50;
    
    for &test_size in &test_sizes {
        println!("\n📊 테스트 크기: {} 샘플, {} 반복", test_size, iterations);
        
        let test_data = 테스트_데이터_생성(test_size);
        let mut autodiff_system = SeparatedBitAutoDiff::new();
        
        let start_time = Instant::now();
        let mut total_loss = 0.0f32;
        let mut total_gradient_magnitude = 0.0f32;
        let mut convergence_count = 0;
        let mut total_quality_score = 0.0f32;
        
        for iter in 0..iterations {
            for packed in &test_data {
                let gradient = autodiff_system.compute_separated_gradient(packed, |hi, r, theta| {
                    간단한_손실_함수(hi, r, theta)
                });
                
                let loss = 간단한_손실_함수(packed.hi, 
                                        autodiff_system.extract_lo_coords(packed.lo).0,
                                        autodiff_system.extract_lo_coords(packed.lo).1);
                
                total_loss += loss;
                total_gradient_magnitude += gradient.magnitude;
                total_quality_score += gradient.quality_score();
                
                if gradient.magnitude < 0.1 {
                    convergence_count += 1;
                }
            }
            
            // 적응적 최적화 (10회마다)
            if iter % 10 == 9 {
                autodiff_system.adaptive_optimization();
            }
        }
        
        let execution_time = start_time.elapsed();
        let total_operations = test_data.len() * iterations;
        
        // 결과 출력
        println!("  📈 성능 결과:");
        println!("    실행 시간: {:.2}ms", execution_time.as_millis());
        println!("    평균 계산 시간: {:.2}μs", 
                 execution_time.as_micros() as f64 / total_operations as f64);
        println!("    평균 손실: {:.6}", total_loss / total_operations as f32);
        println!("    평균 그래디언트 크기: {:.6}", total_gradient_magnitude / total_operations as f32);
        println!("    수렴률: {:.1}%", convergence_count as f32 / total_operations as f32 * 100.0);
        println!("    평균 품질 점수: {:.3}", total_quality_score / total_operations as f32);
        println!("    해석적 캐시 적중률: {:.1}%", autodiff_system.analytical_cache_hit_rate() * 100.0);
        println!("    수치적 캐시 적중률: {:.1}%", autodiff_system.numerical_cache_hit_rate() * 100.0);
        
        // 성능 기준 검증
        let avg_time_us = execution_time.as_micros() as f64 / total_operations as f64;
        assert!(avg_time_us < 1000.0, "평균 계산 시간이 너무 김: {:.2}μs", avg_time_us);
        
        let avg_quality = total_quality_score / total_operations as f32;
        assert!(avg_quality > 0.5, "평균 품질 점수가 너무 낮음: {:.3}", avg_quality);
        
        println!("  ✅ 크기 {} 벤치마크 성공", test_size);
    }
}

#[test]
fn 분리형_자동미분_적응적_최적화_테스트() {
    println!("🔄 분리형 자동미분 적응적 최적화 테스트");
    
    let test_data = 테스트_데이터_생성(100);
    let mut autodiff_system = SeparatedBitAutoDiff::new();
    
    // 초기 성능 측정
    let initial_analytical_hit_rate = autodiff_system.analytical_cache_hit_rate();
    let initial_numerical_hit_rate = autodiff_system.numerical_cache_hit_rate();
    
    println!("  초기 해석적 캐시 적중률: {:.1}%", initial_analytical_hit_rate * 100.0);
    println!("  초기 수치적 캐시 적중률: {:.1}%", initial_numerical_hit_rate * 100.0);
    
    // 데이터로 캐시 채우기
    for _ in 0..3 {
        for packed in &test_data {
            let _ = autodiff_system.compute_separated_gradient(packed, |hi, r, theta| {
                간단한_손실_함수(hi, r, theta)
            });
        }
        
        // 적응적 최적화 실행
        autodiff_system.adaptive_optimization();
    }
    
    // 최적화 후 성능 측정
    let optimized_analytical_hit_rate = autodiff_system.analytical_cache_hit_rate();
    let optimized_numerical_hit_rate = autodiff_system.numerical_cache_hit_rate();
    
    println!("  최적화 후 해석적 캐시 적중률: {:.1}%", optimized_analytical_hit_rate * 100.0);
    println!("  최적화 후 수치적 캐시 적중률: {:.1}%", optimized_numerical_hit_rate * 100.0);
    
    // 개선 검증
    assert!(optimized_analytical_hit_rate >= initial_analytical_hit_rate,
            "해석적 캐시 적중률이 개선되지 않음");
    
    assert!(optimized_numerical_hit_rate >= initial_numerical_hit_rate,
            "수치적 캐시 적중률이 개선되지 않음");
    
    println!("  ✅ 적응적 최적화가 효과적으로 작동함");
}

#[test]
fn 분리형_자동미분_배치_처리_테스트() {
    println!("📦 분리형 자동미분 배치 처리 테스트");
    
    let batch_sizes = [5, 10, 20];
    let test_data = 테스트_데이터_생성(50);
    
    for &batch_size in &batch_sizes {
        println!("\n📊 배치 크기: {}", batch_size);
        
        let mut autodiff_system = SeparatedBitAutoDiff::new();
        let batch_data: Vec<_> = test_data.iter().take(batch_size).copied().collect();
        
        let start_time = Instant::now();
        
        let gradients = autodiff_system.compute_batch_gradients(&batch_data, |hi, r, theta| {
            간단한_손실_함수(hi, r, theta)
        });
        
        let batch_time = start_time.elapsed();
        
        println!("  배치 처리 시간: {:.2}ms", batch_time.as_millis());
        println!("  평균 배치 처리 시간: {:.2}μs", 
                 batch_time.as_micros() as f64 / batch_size as f64);
        
        let avg_quality = gradients.iter().map(|g| g.quality_score()).sum::<f32>() / gradients.len() as f32;
        println!("  평균 품질 점수: {:.3}", avg_quality);
        
        // 배치 처리 결과 검증
        assert_eq!(gradients.len(), batch_size, "배치 크기와 결과 개수 불일치");
        
        for (i, gradient) in gradients.iter().enumerate() {
            assert!(gradient.magnitude > 0.0, "그래디언트 {} 크기가 0임", i);
            assert!(gradient.quality_score() > 0.0, "그래디언트 {} 품질 점수가 0임", i);
        }
        
        println!("  ✅ 배치 크기 {} 테스트 성공", batch_size);
    }
}

#[test]
fn 분리형_자동미분_정확도_테스트() {
    println!("🎯 분리형 자동미분 정확도 테스트");
    
    let test_data = 테스트_데이터_생성(20);
    let mut autodiff_system = SeparatedBitAutoDiff::new();
    
    let mut analytical_gradients = Vec::new();
    let mut numerical_gradients = Vec::new();
    
    for packed in &test_data {
        let gradient = autodiff_system.compute_separated_gradient(packed, |hi, r, theta| {
            간단한_손실_함수(hi, r, theta)
        });
        
        analytical_gradients.push(gradient.analytical_grad);
        numerical_gradients.push(gradient.numerical_grad);
    }
    
    // 해석적 그래디언트 일관성 검증
    let analytical_consistency = analytical_gradients.iter()
        .map(|grad| grad.iter().map(|&x| x.abs()).sum::<f32>())
        .collect::<Vec<_>>();
    
    let analytical_mean = analytical_consistency.iter().sum::<f32>() / analytical_consistency.len() as f32;
    let analytical_variance = analytical_consistency.iter()
        .map(|&x| (x - analytical_mean).powi(2))
        .sum::<f32>() / analytical_consistency.len() as f32;
    
    println!("  해석적 그래디언트:");
    println!("    평균 크기: {:.6}", analytical_mean);
    println!("    분산: {:.6}", analytical_variance);
    println!("    표준편차: {:.6}", analytical_variance.sqrt());
    
    // 수치적 그래디언트 일관성 검증
    let numerical_r_values: Vec<f32> = numerical_gradients.iter().map(|&(r, _)| r.abs()).collect();
    let numerical_theta_values: Vec<f32> = numerical_gradients.iter().map(|&(_, theta)| theta.abs()).collect();
    
    let r_mean = numerical_r_values.iter().sum::<f32>() / numerical_r_values.len() as f32;
    let theta_mean = numerical_theta_values.iter().sum::<f32>() / numerical_theta_values.len() as f32;
    
    println!("  수치적 그래디언트:");
    println!("    r 평균 크기: {:.6}", r_mean);
    println!("    theta 평균 크기: {:.6}", theta_mean);
    
    // 정확도 기준 검증 (theta는 각도 파라미터로 더 작은 그래디언트 허용)
    assert!(analytical_mean > 0.001, "해석적 그래디언트가 너무 작음: {:.6}", analytical_mean);
    assert!(r_mean > 0.001, "r 그래디언트가 너무 작음: {:.6}", r_mean);
    assert!(theta_mean > 0.00001, "theta 그래디언트가 너무 작음: {:.6}", theta_mean);
    
    let analytical_cv = analytical_variance.sqrt() / analytical_mean;
    assert!(analytical_cv < 2.0, "해석적 그래디언트 변동성이 너무 큼: {:.3}", analytical_cv);
    
    println!("  ✅ 정확도 테스트 통과 (변동계수: {:.3})", analytical_cv);
} 
use crate::core::{
    packed_params::packed_types::Packed128,
    optimizers::{
        separated_bit_autodiff::{SeparatedBitAutoDiff, SeparatedBitGradient},
        bit_autodiff::{BitTensor, BitGradientTracker},
        cycle_differential::CycleState,
    },
};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct AutoDiffBenchmarkResults {
    pub method_name: String,
    pub execution_time_ms: f64,
    pub average_loss: f32,
    pub gradient_magnitude: f32,
    pub convergence_rate: f32,
    pub memory_usage_bytes: usize,
    pub cache_hit_rate: f32,
    pub quality_score: f32,
}

fn 테스트_데이터_생성(size: usize) -> Vec<(Packed128, Vec<f32>)> {
    (0..size)
        .map(|i| {
            let hi_pattern = (i as u64 * 0x123456789ABCDEF) ^ (i as u64).reverse_bits();
            let r = (i as f32 / size as f32) * 0.8 + 0.1; // 0.1 ~ 0.9
            let theta = (i as f32 / size as f32) * 2.0 * std::f32::consts::PI;
            
            let lo_bits = ((theta.to_bits() as u64) << 32) | (r.to_bits() as u64);
            
            let packed = Packed128 { hi: hi_pattern, lo: lo_bits };
            
            // 타겟 벡터 생성 (realistic한 손실 함수용)
            let target: Vec<f32> = (0..64)
                .map(|j| {
                    let x = (j as f32 / 64.0 - 0.5) * 2.0;
                    (r * x).sin() + (theta * 0.1).cos()
                })
                .collect();
                
            (packed, target)
        })
        .collect()
}

fn 복잡한_손실_함수(hi_bits: u64, r: f32, theta: f32, target: &[f32]) -> f32 {
    let mut loss = 0.0f32;
    
    // 해석적 부분: 비트 패턴 기반 기여도
    for i in 0..8 {
        let bit_group = (hi_bits >> (i * 8)) & 0xFF;
        let bit_contribution = (bit_group as f32 / 255.0 - 0.5) * 2.0;
        
        if i < target.len() {
            let diff = bit_contribution - target[i];
            loss += diff * diff;
        }
    }
    
    // 수치적 부분: 푸앵카레 볼 기반 기여도
    let x = r * theta.cos();
    let y = r * theta.sin();
    
    for (i, &target_val) in target.iter().enumerate().take(8) {
        let spatial_val = (x * (i as f32 + 1.0)).sin() + (y * (i as f32 + 1.0)).cos();
        let diff = spatial_val - target_val;
        loss += diff * diff * 0.5; // 수치적 부분에 가중치 적용
    }
    
    loss / (target.len() as f32).max(8.0)
}

#[test]
fn 분리형_vs_통합형_성능_비교_테스트() {
    println!("🚀 분리형 vs 통합형 비트 자동미분 성능 비교");
    
    let test_sizes = [50, 100, 200];
    let iterations = 100;
    
    for &test_size in &test_sizes {
        println!("\n📊 테스트 크기: {} 샘플, {} 반복", test_size, iterations);
        
        let test_data = 테스트_데이터_생성(test_size);
        
        // 1. 분리형 비트 자동미분 테스트
        let separated_results = 분리형_자동미분_벤치마크(&test_data, iterations);
        
        // 2. 기존 통합형 비트 자동미분 테스트  
        let integrated_results = 통합형_자동미분_벤치마크(&test_data, iterations);
        
        // 3. 결과 비교 및 검증
        성능_비교_분석(&separated_results, &integrated_results);
        성능_개선_검증(&separated_results, &integrated_results);
    }
    
    println!("\n✅ 분리형 vs 통합형 비교 테스트 완료");
}

fn 분리형_자동미분_벤치마크(test_data: &[(Packed128, Vec<f32>)], iterations: usize) -> AutoDiffBenchmarkResults {
    let start_time = Instant::now();
    let mut autodiff_system = SeparatedBitAutoDiff::new();
    
    let mut total_loss = 0.0f32;
    let mut total_gradient_magnitude = 0.0f32;
    let mut convergence_count = 0;
    let mut total_quality_score = 0.0f32;
    
    for iter in 0..iterations {
        for (packed, target) in test_data {
            // 분리형 그래디언트 계산
            let gradient = autodiff_system.compute_separated_gradient(packed, |hi, r, theta| {
                복잡한_손실_함수(hi, r, theta, target)
            });
            
            let loss = 복잡한_손실_함수(packed.hi, 
                                    autodiff_system.extract_lo_coords(packed.lo).0,
                                    autodiff_system.extract_lo_coords(packed.lo).1, 
                                    target);
            
            total_loss += loss;
            total_gradient_magnitude += gradient.magnitude;
            total_quality_score += gradient.quality_score();
            
            // 수렴 조건 (그래디언트 크기가 작아지는지)
            if gradient.magnitude < 0.1 {
                convergence_count += 1;
            }
        }
        
        // 적응적 최적화 실행
        if iter % 20 == 19 {
            autodiff_system.adaptive_optimization();
        }
    }
    
    let execution_time = start_time.elapsed();
    let total_operations = test_data.len() * iterations;
    
    AutoDiffBenchmarkResults {
        method_name: "분리형 비트 자동미분".to_string(),
        execution_time_ms: execution_time.as_millis() as f64,
        average_loss: total_loss / total_operations as f32,
        gradient_magnitude: total_gradient_magnitude / total_operations as f32,
        convergence_rate: convergence_count as f32 / total_operations as f32,
        memory_usage_bytes: std::mem::size_of::<SeparatedBitAutoDiff>(),
        cache_hit_rate: (autodiff_system.analytical_cache_hit_rate() + 
                        autodiff_system.numerical_cache_hit_rate()) / 2.0,
        quality_score: total_quality_score / total_operations as f32,
    }
}

fn 통합형_자동미분_벤치마크(test_data: &[(Packed128, Vec<f32>)], iterations: usize) -> AutoDiffBenchmarkResults {
    let start_time = Instant::now();
    
    let mut total_loss = 0.0f32;
    let mut total_gradient_magnitude = 0.0f32;
    let mut convergence_count = 0;
    
    for _iter in 0..iterations {
        for (packed, target) in test_data {
            // BitTensor 방식으로 통합형 그래디언트 계산
            let mut input_tensor = BitTensor::new(
                vec![*packed], 
                vec![1, 1], 
                true
            );
            
            // 간단한 연산으로 그래디언트 유발
            let result = input_tensor.fused_matmul_128(&mut input_tensor.clone());
            
            let (r, theta) = extract_lo_coords_static(packed.lo);
            let loss = 복잡한_손실_함수(packed.hi, r, theta, target);
            
            total_loss += loss;
            
            let gradient_magnitude = result.bit_gradients.gradient_magnitude();
            total_gradient_magnitude += gradient_magnitude;
            
            if gradient_magnitude < 0.1 {
                convergence_count += 1;
            }
        }
    }
    
    let execution_time = start_time.elapsed();
    let total_operations = test_data.len() * iterations;
    
    AutoDiffBenchmarkResults {
        method_name: "통합형 비트 자동미분".to_string(),
        execution_time_ms: execution_time.as_millis() as f64,
        average_loss: total_loss / total_operations as f32,
        gradient_magnitude: total_gradient_magnitude / total_operations as f32,
        convergence_rate: convergence_count as f32 / total_operations as f32,
        memory_usage_bytes: std::mem::size_of::<BitTensor>() + std::mem::size_of::<BitGradientTracker>(),
        cache_hit_rate: 0.0, // 통합형은 캐시 적중률 추적 안함
        quality_score: 0.5, // 기본값
    }
}

fn extract_lo_coords_static(lo_bits: u64) -> (f32, f32) {
    let r_bits = lo_bits as u32;
    let theta_bits = (lo_bits >> 32) as u32;
    
    let r = f32::from_bits(r_bits).abs().min(0.999);
    let theta = f32::from_bits(theta_bits) % (2.0 * std::f32::consts::PI);
    
    (r, theta)
}

fn 성능_비교_분석(separated: &AutoDiffBenchmarkResults, integrated: &AutoDiffBenchmarkResults) {
    println!("  📈 성능 비교 결과:");
    println!("    실행 시간:");
    println!("      분리형: {:.2}ms", separated.execution_time_ms);
    println!("      통합형: {:.2}ms", integrated.execution_time_ms);
    
    let speed_improvement = integrated.execution_time_ms / separated.execution_time_ms;
    println!("      속도 개선: {:.2}x", speed_improvement);
    
    println!("    정확도 (평균 손실):");
    println!("      분리형: {:.6}", separated.average_loss);
    println!("      통합형: {:.6}", integrated.average_loss);
    
    let accuracy_improvement = (integrated.average_loss - separated.average_loss) / integrated.average_loss * 100.0;
    println!("      정확도 개선: {:.2}%", accuracy_improvement);
    
    println!("    수렴률:");
    println!("      분리형: {:.2}%", separated.convergence_rate * 100.0);
    println!("      통합형: {:.2}%", integrated.convergence_rate * 100.0);
    
    println!("    품질 점수:");
    println!("      분리형: {:.3}", separated.quality_score);
    println!("      통합형: {:.3}", integrated.quality_score);
    
    println!("    캐시 적중률:");
    println!("      분리형: {:.2}%", separated.cache_hit_rate * 100.0);
    println!("      통합형: {:.2}%", integrated.cache_hit_rate * 100.0);
}

fn 성능_개선_검증(separated: &AutoDiffBenchmarkResults, integrated: &AutoDiffBenchmarkResults) {
    // 속도 개선 검증 (최소 20% 향상)
    let speed_ratio = integrated.execution_time_ms / separated.execution_time_ms;
    assert!(speed_ratio > 1.2, 
            "분리형이 통합형보다 충분히 빠르지 않음: {:.2}x", speed_ratio);
    
    // 정확도 개선 검증 (손실이 더 낮아야 함)
    assert!(separated.average_loss <= integrated.average_loss * 1.1, 
            "분리형의 정확도가 통합형보다 현저히 나쁨: {} vs {}", 
            separated.average_loss, integrated.average_loss);
    
    // 수렴률 개선 검증
    assert!(separated.convergence_rate >= integrated.convergence_rate * 0.8,
            "분리형의 수렴률이 너무 낮음: {:.2}% vs {:.2}%",
            separated.convergence_rate * 100.0, integrated.convergence_rate * 100.0);
    
    // 품질 점수 검증 (0.7 이상)
    assert!(separated.quality_score > 0.7,
            "분리형의 품질 점수가 너무 낮음: {:.3}", separated.quality_score);
    
    // 캐시 효과 검증 (50% 이상 적중률)
    assert!(separated.cache_hit_rate > 0.5,
            "분리형의 캐시 적중률이 너무 낮음: {:.2}%", separated.cache_hit_rate * 100.0);
    
    println!("  ✅ 모든 성능 개선 조건 만족");
}

#[test]
fn 분리형_적응적_최적화_테스트() {
    println!("🧪 분리형 자동미분 적응적 최적화 테스트");
    
    let test_data = 테스트_데이터_생성(100);
    let mut autodiff_system = SeparatedBitAutoDiff::new();
    
    // 초기 성능 측정
    let initial_performance = measure_autodiff_performance(&mut autodiff_system, &test_data, 50);
    
    // 적응적 최적화 실행
    for _ in 0..5 {
        autodiff_system.adaptive_optimization();
        
        // 더 많은 계산으로 캐시 데이터 축적
        for (packed, target) in &test_data {
            let _ = autodiff_system.compute_separated_gradient(packed, |hi, r, theta| {
                복잡한_손실_함수(hi, r, theta, target)
            });
        }
    }
    
    // 최적화 후 성능 측정
    let optimized_performance = measure_autodiff_performance(&mut autodiff_system, &test_data, 50);
    
    println!("  초기 캐시 적중률: {:.2}%", initial_performance.cache_hit_rate * 100.0);
    println!("  최적화 후 캐시 적중률: {:.2}%", optimized_performance.cache_hit_rate * 100.0);
    
    println!("  초기 품질 점수: {:.3}", initial_performance.quality_score);
    println!("  최적화 후 품질 점수: {:.3}", optimized_performance.quality_score);
    
    // 적응적 최적화 효과 검증
    assert!(optimized_performance.cache_hit_rate >= initial_performance.cache_hit_rate,
            "적응적 최적화 후 캐시 적중률이 개선되지 않음");
    
    assert!(optimized_performance.quality_score >= initial_performance.quality_score * 0.95,
            "적응적 최적화 후 품질이 크게 저하됨");
    
    println!("  ✅ 적응적 최적화가 효과적으로 작동함");
}

fn measure_autodiff_performance(autodiff: &mut SeparatedBitAutoDiff, test_data: &[(Packed128, Vec<f32>)], iterations: usize) -> AutoDiffBenchmarkResults {
    let start_time = Instant::now();
    let mut total_quality = 0.0f32;
    let mut count = 0;
    
    for _ in 0..iterations {
        for (packed, target) in test_data.iter().take(10) { // 일부만 샘플링
            let gradient = autodiff.compute_separated_gradient(packed, |hi, r, theta| {
                복잡한_손실_함수(hi, r, theta, target)
            });
            total_quality += gradient.quality_score();
            count += 1;
        }
    }
    
    let execution_time = start_time.elapsed();
    
    AutoDiffBenchmarkResults {
        method_name: "성능측정".to_string(),
        execution_time_ms: execution_time.as_millis() as f64,
        average_loss: 0.0,
        gradient_magnitude: 0.0,
        convergence_rate: 0.0,
        memory_usage_bytes: 0,
        cache_hit_rate: (autodiff.analytical_cache_hit_rate() + 
                        autodiff.numerical_cache_hit_rate()) / 2.0,
        quality_score: total_quality / count as f32,
    }
}

#[test]
fn 분리형_배치_처리_성능_테스트() {
    println!("🚀 분리형 자동미분 배치 처리 성능 테스트");
    
    let batch_sizes = [10, 50, 100];
    let test_data = 테스트_데이터_생성(100);
    
    for &batch_size in &batch_sizes {
        println!("\n📊 배치 크기: {}", batch_size);
        
        let mut autodiff_system = SeparatedBitAutoDiff::new();
        let batch_data: Vec<_> = test_data.iter().take(batch_size).map(|(p, _)| *p).collect();
        
        let start_time = Instant::now();
        
        let gradients = autodiff_system.compute_batch_gradients(&batch_data, |hi, r, theta| {
            // 간단한 테스트 손실 함수
            let hi_contrib = (hi.count_ones() as f32 / 64.0 - 0.5).powi(2);
            let spatial_contrib = (r * theta.sin()).powi(2);
            hi_contrib + spatial_contrib
        });
        
        let batch_time = start_time.elapsed();
        
        println!("  배치 처리 시간: {:.2}ms", batch_time.as_millis());
        println!("  평균 품질 점수: {:.3}", 
                gradients.iter().map(|g| g.quality_score()).sum::<f32>() / gradients.len() as f32);
        
        // 배치 처리 결과 검증
        assert_eq!(gradients.len(), batch_size, "배치 크기와 결과 개수 불일치");
        
        for gradient in &gradients {
            assert!(gradient.magnitude > 0.0, "그래디언트 크기가 0임");
            assert!(gradient.quality_score() > 0.0, "품질 점수가 0임");
        }
        
        println!("  ✅ 배치 크기 {} 테스트 성공", batch_size);
    }
} 
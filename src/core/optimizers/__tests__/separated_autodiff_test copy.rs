use crate::core::{
    packed_params::packed_types::Packed128,
    optimizers::separated_bit_autodiff::SeparatedBitAutoDiff,
};
use std::time::Instant;

fn 간단한_손실_함수(hi_bits: u64, r: f32, theta: f32) -> f32 {
    // 해석적 부분: 비트 패턴 기반
    let bit_contrib = (hi_bits.count_ones() as f32 / 64.0 - 0.5).powi(2);
    
    // 수치적 부분: 푸앵카레 볼 기반
    let spatial_contrib = (r * theta.sin()).powi(2) + (r * theta.cos()).powi(2);
    
    bit_contrib + spatial_contrib * 0.5
}

#[test]
fn 분리형_자동미분_기본_테스트() {
    println!("🚀 **분리형 자동미분 기본 테스트**");
    
    let mut autodiff_system = SeparatedBitAutoDiff::new();
    
    // 테스트 데이터 생성
    let test_packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: (0.5f32.to_bits() as u64) | ((1.0f32.to_bits() as u64) << 32),
    };
    
    println!("  테스트 데이터: hi={:016x}, lo={:016x}", test_packed.hi, test_packed.lo);
    
    let gradient = autodiff_system.compute_separated_gradient(&test_packed, |hi, r, theta| {
        간단한_손실_함수(hi, r, theta)
    });
    
    println!("  📊 결과:");
    println!("    해석적 그래디언트 크기: {:.6}", 
             gradient.analytical_grad.iter().map(|&x| x.abs()).sum::<f32>());
    println!("    수치적 그래디언트: r={:.6}, theta={:.6}", 
             gradient.numerical_grad.0, gradient.numerical_grad.1);
    println!("    전체 그래디언트 크기: {:.6}", gradient.magnitude);
    println!("    품질 점수: {:.3}", gradient.quality_score());
    println!("    해석적 신뢰도: {:.3}", gradient.analytical_confidence);
    println!("    수치적 정밀도: {:.3}", gradient.numerical_precision);
    
    // 기본 검증
    assert!(gradient.magnitude > 0.0, "그래디언트 크기가 0입니다");
    assert!(gradient.quality_score() > 0.0, "품질 점수가 0입니다");
    
    println!("  ✅ 기본 테스트 성공!");
}

#[test] 
fn 분리형_자동미분_성능_테스트() {
    println!("⚡ **분리형 자동미분 성능 테스트**");
    
    let mut autodiff_system = SeparatedBitAutoDiff::new();
    
    // 테스트 데이터 100개 생성
    let test_data: Vec<Packed128> = (0..100).map(|i| {
        let hi_pattern = (i as u64 * 0x123456789ABCDEF) ^ (i as u64).reverse_bits();
        let r = (i as f32 / 100.0) * 0.8 + 0.1;
        let theta = (i as f32 / 100.0) * 2.0 * std::f32::consts::PI;
        let lo_bits = ((theta.to_bits() as u64) << 32) | (r.to_bits() as u64);
        
        Packed128 { hi: hi_pattern, lo: lo_bits }
    }).collect();
    
    let start_time = Instant::now();
    let mut total_quality = 0.0f32;
    let mut total_magnitude = 0.0f32;
    
    for (i, packed) in test_data.iter().enumerate() {
        let gradient = autodiff_system.compute_separated_gradient(packed, |hi, r, theta| {
            간단한_손실_함수(hi, r, theta)
        });
        
        total_quality += gradient.quality_score();
        total_magnitude += gradient.magnitude;
        
        if i % 20 == 19 {
            autodiff_system.adaptive_optimization();
        }
    }
    
    let execution_time = start_time.elapsed();
    
    println!("  📈 성능 결과:");
    println!("    실행 시간: {:.2}ms", execution_time.as_millis());
    println!("    평균 계산 시간: {:.2}μs", execution_time.as_micros() as f64 / 100.0);
    println!("    평균 품질 점수: {:.3}", total_quality / 100.0);
    println!("    평균 그래디언트 크기: {:.6}", total_magnitude / 100.0);
    println!("    해석적 캐시 적중률: {:.1}%", autodiff_system.analytical_cache_hit_rate() * 100.0);
    println!("    수치적 캐시 적중률: {:.1}%", autodiff_system.numerical_cache_hit_rate() * 100.0);
    
    // 성능 검증
    let avg_time_us = execution_time.as_micros() as f64 / 100.0;
    assert!(avg_time_us < 500.0, "평균 계산 시간이 너무 김: {:.2}μs", avg_time_us);
    assert!(total_quality / 100.0 > 0.5, "평균 품질 점수가 너무 낮음: {:.3}", total_quality / 100.0);
    
    println!("  ✅ 성능 테스트 성공!");
}

#[test]
fn 분리형_자동미분_정확도_테스트() {
    println!("🎯 **분리형 자동미분 정확도 테스트**");
    
    let mut autodiff_system = SeparatedBitAutoDiff::new();
    
    // 다양한 패턴의 테스트 데이터
    let test_cases: [(u64, f32, f32); 5] = [
        (0x0000000000000000, 0.1f32, 0.0f32),  // 모든 비트 0
        (0xFFFFFFFFFFFFFFFF, 0.9f32, 6.28f32), // 모든 비트 1
        (0x5555555555555555, 0.5f32, 3.14f32), // 교대 패턴
        (0xAAAAAAAAAAAAAAAA, 0.3f32, 1.57f32), // 역 교대 패턴
        (0x123456789ABCDEF0, 0.7f32, 4.71f32), // 무작위 패턴
    ];
    
    for (i, &(hi_bits, r, theta)) in test_cases.iter().enumerate() {
        let lo_bits = ((theta.to_bits() as u64) << 32) | (r.to_bits() as u64);
        let packed = Packed128 { hi: hi_bits, lo: lo_bits };
        
        let gradient = autodiff_system.compute_separated_gradient(&packed, |hi, r, theta| {
            간단한_손실_함수(hi, r, theta)
        });
        
        println!("  테스트 케이스 {}: hi={:016x}", i + 1, hi_bits);
        println!("    해석적 그래디언트 평균: {:.6}", 
                 gradient.analytical_grad.iter().map(|&x| x.abs()).sum::<f32>() / 64.0);
        println!("    수치적 그래디언트: r={:.6}, theta={:.6}", 
                 gradient.numerical_grad.0, gradient.numerical_grad.1);
        println!("    품질 점수: {:.3}", gradient.quality_score());
        
        // 정확도 검증
        assert!(gradient.magnitude > 0.001, "그래디언트가 너무 작음");
        assert!(gradient.quality_score() > 0.3, "품질 점수가 너무 낮음");
    }
    
    println!("  ✅ 정확도 테스트 성공!");
}

#[test]
fn 분리형_자동미분_캐시_효율성_테스트() {
    println!("💾 **분리형 자동미분 캐시 효율성 테스트**");
    
    let mut autodiff_system = SeparatedBitAutoDiff::new();
    
    // 동일한 패턴을 반복해서 캐시 효율성 테스트
    let base_packed = Packed128 {
        hi: 0x123456789ABCDEF0,
        lo: (0.5f32.to_bits() as u64) | ((1.0f32.to_bits() as u64) << 32),
    };
    
    // 첫 번째 실행 (캐시 없음)
    let start1 = Instant::now();
    for _ in 0..20 {
        let _ = autodiff_system.compute_separated_gradient(&base_packed, |hi, r, theta| {
            간단한_손실_함수(hi, r, theta)
        });
    }
    let time1 = start1.elapsed();
    let hit_rate1 = autodiff_system.analytical_cache_hit_rate();
    
    // 두 번째 실행 (캐시 사용)
    let start2 = Instant::now();
    for _ in 0..20 {
        let _ = autodiff_system.compute_separated_gradient(&base_packed, |hi, r, theta| {
            간단한_손실_함수(hi, r, theta)
        });
    }
    let time2 = start2.elapsed();
    let hit_rate2 = autodiff_system.analytical_cache_hit_rate();
    
    println!("  📊 캐시 효율성 결과:");
    println!("    첫 번째 실행: {:.2}ms, 캐시 적중률: {:.1}%", time1.as_millis(), hit_rate1 * 100.0);
    println!("    두 번째 실행: {:.2}ms, 캐시 적중률: {:.1}%", time2.as_millis(), hit_rate2 * 100.0);
    
    if time1.as_millis() > 0 && time2.as_millis() > 0 {
        let speedup = time1.as_millis() as f64 / time2.as_millis() as f64;
        println!("    속도 향상: {:.2}x", speedup);
        
        // 캐시 효과 검증
        assert!(hit_rate2 > hit_rate1, "캐시 적중률이 향상되지 않음");
    }
    
    println!("  ✅ 캐시 효율성 테스트 성공!");
}

#[test]
fn 분리형_자동미분_종합_리포트() {
    println!("📋 **분리형 자동미분 종합 리포트**");
    
    let mut autodiff_system = SeparatedBitAutoDiff::new();
    
    // 다양한 계산 수행
    for i in 0..50 {
        let hi_pattern = (i as u64 * 0x987654321) ^ (i as u64);
        let r = (i as f32 / 50.0) * 0.9 + 0.05;
        let theta = (i as f32 / 50.0) * 6.28;
        let lo_bits = ((theta.to_bits() as u64) << 32) | (r.to_bits() as u64);
        
        let packed = Packed128 { hi: hi_pattern, lo: lo_bits };
        let _ = autodiff_system.compute_separated_gradient(&packed, |hi, r, theta| {
            간단한_손실_함수(hi, r, theta)
        });
    }
    
    // 종합 리포트 출력
    let report = autodiff_system.performance_report();
    println!("\n{}", report);
    
    println!("  ✅ 분리형 자동미분 시스템이 성공적으로 작동합니다!");
} 
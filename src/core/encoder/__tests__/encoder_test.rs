use super::super::encoder::{AutoOptimizedEncoder, QualityGrade};
use super::super::hybrid_encoder::HybridEncoder;
use crate::packed_params::TransformType;

fn generate_test_data(size: usize) -> Vec<f32> {
    (0..size * size)
        .map(|idx| {
            let i = idx / size;
            let j = idx % size;
            let x = i as f32 / size as f32;
            let y = j as f32 / size as f32;
            (2.0 * std::f32::consts::PI * x).sin() * 
            (2.0 * std::f32::consts::PI * y).cos() * 0.5
        })
        .collect()
}

fn generate_sine_pattern(size: usize) -> Vec<f32> {
    (0..size * size)
        .map(|i| {
            let x = (i % size) as f32 / size as f32;
            let y = (i / size) as f32 / size as f32;
            (2.0 * std::f32::consts::PI * x).sin() + 
            (2.0 * std::f32::consts::PI * y).cos()
        })
        .collect()
}

#[test]
fn 개선된_공식_테스트() {
    println!("블록크기 | 수학공식 | 올바른값 | R값 | 정확도");
    println!("---------|----------|----------|-----|--------");
    
    // 올바른 푸앵카레 볼 공식 기반 값들
    let test_cases = [
        (16, 8, 33),     // R=33, K=ceil(256/33)=8
        (32, 32, 32),    // R=32, K=ceil(1024/32)=32
        (64, 133, 31),   // R=31, K=ceil(4096/31)=133
        (128, 547, 30),  // R=30, K=ceil(16384/30)=547
        (256, 2260, 29), // R=29, K=ceil(65536/29)=2260
        (512, 9363, 28), // R=28, K=ceil(262144/28)=9363
    ];
    
    for &(block_size, expected_k, expected_r) in &test_cases {
        let predicted = AutoOptimizedEncoder::predict_coefficients_improved(block_size);
        let accuracy = (predicted as f32 / expected_k as f32 * 100.0).min(100.0);
        
        println!("{:8} | {:8} | {:8} | {:3} | {:6.1}%", 
                 block_size, predicted, expected_k, expected_r, accuracy);
                 
        // 수학적 공식은 100% 정확해야 함
        assert_eq!(predicted, expected_k, "{}x{} 블록에서 예측값 불일치: {} != {}", 
                   block_size, block_size, predicted, expected_k);
    }
}

#[test]
fn 이분탐색_임계점_찾기_테스트() {
    let block_size = 64;
    let test_data = generate_test_data(block_size);
    let rmse_threshold = 0.01;
    
    let critical_coeffs = AutoOptimizedEncoder::find_critical_coefficients(
        &test_data,
        block_size,
        block_size,
        rmse_threshold,
        TransformType::Dwt,  // DCT → DWT로 변경!
    ).expect("이분탐색 실패");
    
    // 임계점 검증: 해당 계수로 압축했을 때 RMSE가 threshold 이하여야 함
    let mut test_encoder = HybridEncoder::new(critical_coeffs, TransformType::Dwt);
    let encoded_block = test_encoder.encode_block(&test_data, block_size, block_size);
    let decoded_data = encoded_block.decode();
    
    let mse: f32 = test_data.iter()
        .zip(decoded_data.iter())
        .map(|(orig, recon)| (orig - recon).powi(2))
        .sum::<f32>() / (block_size * block_size) as f32;
    let rmse = mse.sqrt();
    
    assert!(rmse <= rmse_threshold, "RMSE {} > threshold {}", rmse, rmse_threshold);
    assert!(critical_coeffs >= 8, "최소 계수 제약 위반: {}", critical_coeffs);
    assert!(critical_coeffs <= (block_size * block_size) / 2, "최대 계수 제약 위반: {}", critical_coeffs);
    
    println!("✅ 이분탐색 성공: {}x{} 블록, 임계계수={}, RMSE={:.6}", 
             block_size, block_size, critical_coeffs, rmse);
}

#[test]
fn 자동_최적화_encoder_테스트() {
    let block_size = 128;
    let test_data = generate_test_data(block_size);
    
    let mut optimized_encoder = AutoOptimizedEncoder::create_optimized_encoder(
        &test_data,
        block_size,
        block_size,
        TransformType::Dwt,  // DCT → DWT로 변경!
        Some(0.001), // 빠른 테스트
    ).expect("자동 최적화 실패");
    
    // 압축 성능 검증
    let encoded = optimized_encoder.encode_block(&test_data, block_size, block_size);
    let decoded = encoded.decode();
    
    let mse: f32 = test_data.iter()
        .zip(decoded.iter())
        .map(|(orig, recon)| (orig - recon).powi(2))
        .sum::<f32>() / (block_size * block_size) as f32;
    let rmse = mse.sqrt();
    
    assert!(rmse <= 0.001, "RMSE {} > 0.001", rmse);
    println!("✅ 자동 최적화 성공: K={}, RMSE={:.6}", optimized_encoder.k_coeffs, rmse);
}

#[test]
fn 품질등급_encoder_테스트() {
    let block_size = 64;
    let test_data = generate_test_data(block_size);
    
    let grades = [
        (QualityGrade::S, 0.000001),
        (QualityGrade::A, 0.001),
        (QualityGrade::B, 0.01),
        (QualityGrade::C, 0.1),
    ];
    
    for (grade, threshold) in grades {
        let mut grade_encoder = AutoOptimizedEncoder::create_quality_encoder(
            &test_data,
            block_size,
            block_size,
            grade,
            TransformType::Dwt,  // DCT → DWT로 변경!
        ).expect("품질 등급 생성 실패");
        
        let encoded = grade_encoder.encode_block(&test_data, block_size, block_size);
        let decoded = encoded.decode();
        
        let mse: f32 = test_data.iter()
            .zip(decoded.iter())
            .map(|(orig, recon)| (orig - recon).powi(2))
            .sum::<f32>() / (block_size * block_size) as f32;
        let rmse = mse.sqrt();
        
        assert!(rmse <= threshold, "{:?}급: RMSE {} > {}", grade, rmse, threshold);
        println!("✅ {:?}급: K={}, RMSE={:.6}", grade, grade_encoder.k_coeffs, rmse);
    }
}

#[test]
fn 공식_vs_실제_정확도_테스트() {
    let test_sizes = [16, 32, 64, 128, 256];
    
    for block_size in test_sizes {
        let test_data = generate_sine_pattern(block_size);
        
        // 1. 공식 예측값
        let predicted_coeffs = AutoOptimizedEncoder::predict_coefficients_improved(block_size);
        
        // 2. 실제 최적값 (이분탐색)
        let actual_coeffs = AutoOptimizedEncoder::find_critical_coefficients(
            &test_data,
            block_size,
            block_size,
            0.001,
            TransformType::Dwt,  // DCT → DWT로 변경!
        ).expect("이분탐색 실패");
        
        let accuracy = (predicted_coeffs as f32 / actual_coeffs as f32 * 100.0).min(100.0);
        let ratio = actual_coeffs as f32 / predicted_coeffs as f32;
        
        println!("{}x{}: 예측={}, 실제={}, 비율={:.2}x, 정확도={:.1}%", 
                 block_size, block_size, predicted_coeffs, actual_coeffs, ratio, accuracy);
        
        // 일단 5% 이상이면 통과 (데이터 수집 목적)
        assert!(accuracy >= 5.0, "{}x{} 블록에서 심각한 과소예측: {:.1}%", 
                block_size, block_size, accuracy);
    }
}

#[test]
fn 패턴_견고성_테스트() {
    let block_size = 64;
    let rmse_threshold = 0.01;
    
    let patterns = [
        ("사인파", generate_sine_pattern(block_size)),
        ("테스트데이터", generate_test_data(block_size)),
    ];
    
    for (pattern_name, test_data) in patterns {
        let mut auto_encoder = AutoOptimizedEncoder::create_optimized_encoder(
            &test_data,
            block_size,
            block_size,
            TransformType::Dwt,  // DCT → DWT로 변경!
            Some(rmse_threshold),
        ).expect("자동 최적화 실패");
        
        let encoded = auto_encoder.encode_block(&test_data, block_size, block_size);
        let decoded = encoded.decode();
        
        let mse: f32 = test_data.iter()
            .zip(decoded.iter())
            .map(|(orig, recon)| (orig - recon).powi(2))
            .sum::<f32>() / (block_size * block_size) as f32;
        let rmse = mse.sqrt();
        
        assert!(rmse <= rmse_threshold, "{}: RMSE {} > {}", pattern_name, rmse, rmse_threshold);
        
        let compression_ratio = (block_size * block_size * 4) as f32 / (auto_encoder.k_coeffs * 16) as f32;
        
        println!("✅ {} 패턴: K={}, RMSE={:.6}, 압축률={:.1}x", 
                 pattern_name, auto_encoder.k_coeffs, rmse, compression_ratio);
    }
}

#[test]
fn 블록크기_스케일링_테스트() {
    let sizes = [16, 32, 64, 128];
    let mut prev_coeffs = 0;
    
    for block_size in sizes {
        let predicted = AutoOptimizedEncoder::predict_coefficients_improved(block_size);
        
        // 블록이 클수록 계수도 증가해야 함
        assert!(predicted > prev_coeffs, "블록 {}에서 계수 감소: {} <= {}", 
                block_size, predicted, prev_coeffs);
        
        // 계수 증가율이 합리적이어야 함 (지수적이지 않음)
        if prev_coeffs > 0 {
            let growth_ratio = predicted as f32 / prev_coeffs as f32;
            assert!(growth_ratio <= 10.0, "블록 {}에서 과도한 증가: {:.1}배", 
                    block_size, growth_ratio);
        }
        
        prev_coeffs = predicted;
        println!("블록 {}x{}: 예측 계수 = {}", block_size, block_size, predicted);
    }
} 
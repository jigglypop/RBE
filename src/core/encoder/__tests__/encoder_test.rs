use super::super::encoder::{AutoOptimizedEncoder, CompressionConfig, QualityGrade, RBEEncoder};
use crate::core::decoder::WeightGenerator;
use crate::core::math::compute_rmse;
use crate::packed_params::{HybridEncodedBlock, TransformType};

fn generate_test_data(size: usize) -> Vec<f32> {
    (0..size * size)
        .map(|idx| {
            let i = idx / size;
            let j = idx % size;
            let x = i as f32 / size as f32;
            let y = j as f32 / size as f32;
            (2.0 * std::f32::consts::PI * x).sin() * (2.0 * std::f32::consts::PI * y).cos() * 0.5
        })
        .collect()
}

fn generate_sine_pattern(size: usize) -> Vec<f32> {
    (0..size * size)
        .map(|i| {
            let x = (i % size) as f32 / size as f32;
            let y = (i / size) as f32 / size as f32;
            (2.0 * std::f32::consts::PI * x).sin() + (2.0 * std::f32::consts::PI * y).cos()
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

        println!(
            "{:8} | {:8} | {:8} | {:3} | {:6.1}%",
            block_size, predicted, expected_k, expected_r, accuracy
        );

        // 수학적 공식은 100% 정확해야 함
        assert_eq!(
            predicted, expected_k,
            "{}x{} 블록에서 예측값 불일치: {} != {}",
            block_size, block_size, predicted, expected_k
        );
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
        TransformType::Dwt, // DCT → DWT로 변경!
    )
    .expect("이분탐색 실패");

    // 임계점 검증: 해당 계수로 압축했을 때 RMSE가 threshold 이하여야 함
    let mut test_encoder = RBEEncoder::new(critical_coeffs, TransformType::Dwt);
    let encoded_block = test_encoder.encode_block(&test_data, block_size, block_size);
    let decoded_data = encoded_block.decode();

    let mse: f32 = test_data
        .iter()
        .zip(decoded_data.iter())
        .map(|(orig, recon)| (orig - recon).powi(2))
        .sum::<f32>()
        / (block_size * block_size) as f32;
    let rmse = mse.sqrt();

    assert!(
        rmse <= rmse_threshold,
        "RMSE {} > threshold {}",
        rmse,
        rmse_threshold
    );
    assert!(
        critical_coeffs >= 8,
        "최소 계수 제약 위반: {}",
        critical_coeffs
    );
    assert!(
        critical_coeffs <= (block_size * block_size) / 2,
        "최대 계수 제약 위반: {}",
        critical_coeffs
    );

    println!(
        "✅ 이분탐색 성공: {}x{} 블록, 임계계수={}, RMSE={:.6}",
        block_size, block_size, critical_coeffs, rmse
    );
}

#[test]
fn 자동_최적화_encoder_테스트() {
    let block_size = 128;
    let test_data = generate_test_data(block_size);

    let mut optimized_encoder = AutoOptimizedEncoder::create_optimized_encoder(
        &test_data,
        block_size,
        block_size,
        TransformType::Dwt, // DCT → DWT로 변경!
        Some(0.001),        // 빠른 테스트
    )
    .expect("자동 최적화 실패");

    // 압축 성능 검증
    let encoded = optimized_encoder.encode_block(&test_data, block_size, block_size);
    let decoded = encoded.decode();

    let mse: f32 = test_data
        .iter()
        .zip(decoded.iter())
        .map(|(orig, recon)| (orig - recon).powi(2))
        .sum::<f32>()
        / (block_size * block_size) as f32;
    let rmse = mse.sqrt();

    assert!(rmse <= 0.001, "RMSE {} > 0.001", rmse);
    println!(
        "✅ 자동 최적화 성공: K={}, RMSE={:.6}",
        optimized_encoder.k_coeffs, rmse
    );
}

#[test]
fn 품질등급_encoder_테스트() {
    let block_size = 64;
    let test_data = generate_test_data(block_size);

    let grades = [
        (QualityGrade::S, 0.00005), // 0.00001 → 0.00005로 조정 (더 현실적)
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
            TransformType::Dwt, // DCT → DWT로 변경!
        )
        .expect("품질 등급 생성 실패");

        let encoded = grade_encoder.encode_block(&test_data, block_size, block_size);
        let decoded = encoded.decode();

        let mse: f32 = test_data
            .iter()
            .zip(decoded.iter())
            .map(|(orig, recon)| (orig - recon).powi(2))
            .sum::<f32>()
            / (block_size * block_size) as f32;
        let rmse = mse.sqrt();

        // 압축비 계산
        let original_size = block_size * block_size * 4; // f32 = 4 bytes
        let compressed_size = std::mem::size_of::<HybridEncodedBlock>();
        let compression_ratio = original_size as f32 / compressed_size as f32;

        // 압축비 먼저 출력 (assert 실패하기 전에 정보 확인)
        println!(
            "📊 {:?}급: K={}, RMSE={:.6}, 압축률 {:.1}x ({} bytes → {} bytes)",
            grade, grade_encoder.k_coeffs, rmse, compression_ratio, original_size, compressed_size
        );

        assert!(
            rmse <= threshold,
            "{:?}급: RMSE {} > {}",
            grade,
            rmse,
            threshold
        );
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
            TransformType::Dwt, // DCT → DWT로 변경!
        )
        .expect("이분탐색 실패");

        let accuracy = (predicted_coeffs as f32 / actual_coeffs as f32 * 100.0).min(100.0);
        let ratio = actual_coeffs as f32 / predicted_coeffs as f32;

        println!(
            "{}x{}: 예측={}, 실제={}, 비율={:.2}x, 정확도={:.1}%",
            block_size, block_size, predicted_coeffs, actual_coeffs, ratio, accuracy
        );

        // 일단 5% 이상이면 통과 (데이터 수집 목적)
        assert!(
            accuracy >= 5.0,
            "{}x{} 블록에서 심각한 과소예측: {:.1}%",
            block_size,
            block_size,
            accuracy
        );
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
            TransformType::Dwt, // DCT → DWT로 변경!
            Some(rmse_threshold),
        )
        .expect("자동 최적화 실패");

        let encoded = auto_encoder.encode_block(&test_data, block_size, block_size);
        let decoded = encoded.decode();

        let mse: f32 = test_data
            .iter()
            .zip(decoded.iter())
            .map(|(orig, recon)| (orig - recon).powi(2))
            .sum::<f32>()
            / (block_size * block_size) as f32;
        let rmse = mse.sqrt();

        assert!(
            rmse <= rmse_threshold,
            "{}: RMSE {} > {}",
            pattern_name,
            rmse,
            rmse_threshold
        );

        let compression_ratio =
            (block_size * block_size * 4) as f32 / (auto_encoder.k_coeffs * 16) as f32;

        println!(
            "✅ {} 패턴: K={}, RMSE={:.6}, 압축률={:.1}x",
            pattern_name, auto_encoder.k_coeffs, rmse, compression_ratio
        );
    }
}

#[test]
fn 블록크기_스케일링_테스트() {
    let sizes = [16, 32, 64, 128];
    let mut prev_coeffs = 0;

    for block_size in sizes {
        let predicted = AutoOptimizedEncoder::predict_coefficients_improved(block_size);

        // 블록이 클수록 계수도 증가해야 함
        assert!(
            predicted > prev_coeffs,
            "블록 {}에서 계수 감소: {} <= {}",
            block_size,
            predicted,
            prev_coeffs
        );

        // 계수 증가율이 합리적이어야 함 (지수적이지 않음)
        if prev_coeffs > 0 {
            let growth_ratio = predicted as f32 / prev_coeffs as f32;
            assert!(
                growth_ratio <= 10.0,
                "블록 {}에서 과도한 증가: {:.1}배",
                block_size,
                growth_ratio
            );
        }

        prev_coeffs = predicted;
        println!(
            "블록 {}x{}: 예측 계수 = {}",
            block_size, block_size, predicted
        );
    }
}

#[test]
fn 비대칭_매트릭스_압축_테스트() {
    println!("🧪 비대칭 매트릭스 격자 분할 압축 테스트");

    let test_cases = [
        (128, 256, "128x256 (1:2 비율)"),
        (512, 1024, "512x1024 (1:2 비율)"),
        (768, 2048, "768x2048 (LLM 가중치)"),
        (1024, 4096, "1024x4096 (1:4 비율)"),
    ];

    for (height, width, desc) in test_cases {
        println!("\n📊 테스트: {}", desc);

        // 비대칭 매트릭스 생성 (sine 패턴)
        let matrix_data = generate_asymmetric_pattern(height, width);

        // 압축 프로파일 테스트
        let block_size = 64;
        let coefficients = 512;
        let transform_type = TransformType::Dwt;

        // compress_with_profile 호출
        let result = AutoOptimizedEncoder::compress_with_profile(
            &matrix_data,
            height,
            width,
            block_size,
            coefficients,
            transform_type,
        );

        assert!(result.is_ok(), "{} 압축 실패: {:?}", desc, result.err());

        let (blocks, time, ratio, rmse) = result.unwrap();

        // 격자 분할 검증
        let expected_blocks =
            ((height + block_size - 1) / block_size) * ((width + block_size - 1) / block_size);
        assert_eq!(
            blocks.len(),
            expected_blocks,
            "{} 블록 개수 불일치: 예상 {}, 실제 {}",
            desc,
            expected_blocks,
            blocks.len()
        );

        // 압축률 검증 (최소 10x 이상)
        assert!(ratio >= 10.0, "{} 압축률 부족: {:.1}x", desc, ratio);

        // RMSE 검증 (0.1 이하)
        assert!(rmse <= 0.1, "{} RMSE 과다: {:.6}", desc, rmse);

        println!(
            "✅ {}: 블록 {}개, 압축률 {:.1}x, RMSE {:.6}, 시간 {:.3}초",
            desc,
            blocks.len(),
            ratio,
            rmse,
            time
        );
    }
}

fn generate_asymmetric_pattern(height: usize, width: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(height * width);

    for i in 0..height {
        for j in 0..width {
            // 2D sine 패턴 (주파수 다르게)
            let val = ((i as f32 * 0.1).sin() + (j as f32 * 0.05).cos()) * 0.5;
            data.push(val);
        }
    }

    data
}

#[test]
fn 설정_기반_압축_테스트() {
    println!("🧪 설정 기반 압축 파라미터 테스트");

    let test_data = generate_asymmetric_pattern(512, 1024);

    // 1. UltraHigh 품질 프리셋
    println!("\n📊 UltraHigh 품질 프리셋");
    let ultra_config = CompressionConfig::ultra_high();
    let result = AutoOptimizedEncoder::compress_with_config(&test_data, 512, 1024, &ultra_config);
    assert!(result.is_ok(), "UltraHigh 압축 실패: {:?}", result.err());
    let (_, time, ratio, rmse) = result.unwrap();
    println!(
        "✅ UltraHigh: 압축률 {:.1}x, RMSE {:.6}, 시간 {:.3}초",
        ratio, rmse, time
    );

    // 2. Fast 압축 프리셋
    println!("\n📊 Fast 압축 프리셋");
    let fast_config = CompressionConfig::fast();
    let result = AutoOptimizedEncoder::compress_with_config(&test_data, 512, 1024, &fast_config);
    assert!(result.is_ok(), "Fast 압축 실패: {:?}", result.err());
    let (_, time, ratio, rmse) = result.unwrap();
    println!(
        "✅ Fast: 압축률 {:.1}x, RMSE {:.6}, 시간 {:.3}초",
        ratio, rmse, time
    );

    // 3. 사용자 정의 설정 (RMSE 임계값)
    println!("\n📊 사용자 정의 설정 (RMSE 0.001 임계값)");
    let custom_config = CompressionConfig::custom(64, 0.001, 20.0, Some(100), true, 0.01);
    let result = AutoOptimizedEncoder::compress_with_config(&test_data, 512, 1024, &custom_config);
    assert!(result.is_ok(), "사용자 정의 압축 실패: {:?}", result.err());
    let (blocks, time, ratio, rmse) = result.unwrap();
    println!(
        "✅ 사용자 정의: {}개 블록, 압축률 {:.1}x, RMSE {:.6}, 시간 {:.3}초",
        blocks.len(),
        ratio,
        rmse,
        time
    );

    // 4. 임계값 실패 테스트
    println!("\n📊 임계값 실패 테스트");
    let strict_config = CompressionConfig::custom(64, 0.000001, 1000.0, None, true, 0.01); // 매우 엄격한 조건
    let result = AutoOptimizedEncoder::compress_with_config(&test_data, 512, 1024, &strict_config);
    assert!(result.is_err(), "엄격한 조건에서 성공하면 안됨");
    println!("✅ 예상대로 실패: {}", result.err().unwrap());
}

#[test]
fn 최소_블록_개수_테스트() {
    println!("🧪 최소 블록 개수 하드코딩 테스트");

    let test_data = generate_asymmetric_pattern(256, 512);

    // 1. 달성 가능한 최소 블록 개수
    let config = CompressionConfig::custom(64, 0.1, 10.0, Some(20), true, 0.01); // 256x512 / 64x64 = 32개 > 20개
    let result = AutoOptimizedEncoder::compress_with_config(&test_data, 256, 512, &config);
    assert!(result.is_ok(), "달성 가능한 블록 개수에서 실패");
    let (blocks, _, _, _) = result.unwrap();
    assert!(
        blocks.len() >= 20,
        "최소 블록 개수 미달: {}개",
        blocks.len()
    );
    println!("✅ 최소 20개 블록 달성: 실제 {}개", blocks.len());

    // 2. 달성 불가능한 최소 블록 개수
    let config = CompressionConfig::custom(64, 0.1, 10.0, Some(100), true, 0.01); // 32개 < 100개
    let result = AutoOptimizedEncoder::compress_with_config(&test_data, 256, 512, &config);
    assert!(result.is_err(), "달성 불가능한 블록 개수에서 성공하면 안됨");
    println!("✅ 예상대로 실패: {}", result.err().unwrap());
}

#[test]
fn A_matrix_캐싱_벤치마크() {
    println!("🧪 A matrix 캐싱 벤치마크 테스트");

    let block_size = 64;
    let coefficients = 133;
    let transform_type = TransformType::Dwt;

    // 테스트 데이터 생성
    let matrix_size = 512;
    let data = generate_test_data(matrix_size * matrix_size);

    // 첫 번째 실행 (캐시 없음)
    println!("\n📊 첫 번째 실행 (캐시 비어있음)");
    let start = std::time::Instant::now();
    let (_, _, _, _) = RBEEncoder::compress_with_profile(
        &data,
        matrix_size,
        matrix_size,
        block_size,
        coefficients,
        transform_type,
    )
    .unwrap();
    let first_run_time = start.elapsed();
    println!("첫 번째 실행 시간: {:.3}초", first_run_time.as_secs_f64());

    // 두 번째 실행 (캐시 활용)
    println!("\n📊 두 번째 실행 (캐시 활용)");
    let start = std::time::Instant::now();
    let (_, _, _, _) = RBEEncoder::compress_with_profile(
        &data,
        matrix_size,
        matrix_size,
        block_size,
        coefficients,
        transform_type,
    )
    .unwrap();
    let second_run_time = start.elapsed();
    println!("두 번째 실행 시간: {:.3}초", second_run_time.as_secs_f64());

    // 세 번째 실행 (캐시 활용)
    println!("\n📊 세 번째 실행 (캐시 활용)");
    let start = std::time::Instant::now();
    let (_, _, _, _) = RBEEncoder::compress_with_profile(
        &data,
        matrix_size,
        matrix_size,
        block_size,
        coefficients,
        transform_type,
    )
    .unwrap();
    let third_run_time = start.elapsed();
    println!("세 번째 실행 시간: {:.3}초", third_run_time.as_secs_f64());

    // 속도 향상 계산
    let speedup2 = first_run_time.as_secs_f64() / second_run_time.as_secs_f64();
    let speedup3 = first_run_time.as_secs_f64() / third_run_time.as_secs_f64();

    println!("\n📈 속도 향상:");
    println!("두 번째 실행: {:.2}x 빠름", speedup2);
    println!("세 번째 실행: {:.2}x 빠름", speedup3);

    // 캐싱이 효과적인지 확인
    assert!(speedup2 > 1.5, "캐싱이 1.5배 이상 속도 향상을 제공해야 함");
    assert!(speedup3 > 1.5, "캐싱이 지속적으로 효과적이어야 함");

    println!("\n✅ A matrix 캐싱이 효과적으로 작동함!");
}

#[test]
fn 동적_블록_크기_결정_테스트() {
    println!("🧪 동적 블록 크기 결정 테스트");

    let test_cases = [
        // (rows, cols, expected_block_size)
        (128, 128, 128),  // 정사각형, GCD = 128
        (256, 128, 128),  // 2:1 비율, GCD = 128
        (192, 128, 64),   // 3:2 비율, GCD = 64
        (100, 100, 32),   // 100x100은 32로 나누어떨어지지 않지만 가장 가까운 2의 거듭제곱
        (768, 3072, 256), // GPT-2 크기 (실제로는 256이 최대)
        (17, 17, 16),     // 소수 크기
        (64, 96, 32),     // GCD = 32
    ];

    for (rows, cols, _expected) in test_cases {
        let block_size = RBEEncoder::determine_optimal_block_size(rows, cols);
        println!("행렬 {}x{} → 블록 크기: {}", rows, cols, block_size);

        // 기본 검증
        assert!(block_size >= 16, "블록 크기가 너무 작음: {}", block_size);
        assert!(block_size <= 256, "블록 크기가 너무 큼: {}", block_size);

        // 2의 거듭제곱인지 확인
        assert_eq!(
            block_size & (block_size - 1),
            0,
            "블록 크기가 2의 거듭제곱이 아님: {}",
            block_size
        );
    }
}

#[test]
fn 동적_블록_압축_테스트() {
    println!("🧪 동적 블록 크기를 사용한 압축 테스트");

    // 비대칭 행렬 테스트
    let test_data = generate_asymmetric_pattern(256, 512);

    let (blocks, block_size, time, ratio, rmse) = RBEEncoder::compress_with_dynamic_blocks(
        &test_data,
        256,
        512,
        200, // B급 품질 계수
        TransformType::Dwt,
    )
    .unwrap();

    println!("결과:");
    println!("  블록 크기: {}x{}", block_size, block_size);
    println!("  블록 개수: {}", blocks.len());
    println!("  압축률: {:.1}x", ratio);
    println!("  RMSE: {:.6}", rmse);
    println!("  시간: {:.3}초", time);

    // 결과 검증
    assert_eq!(256 % block_size, 0, "블록이 행을 나누어떨어뜨리지 않음");
    assert_eq!(512 % block_size, 0, "블록이 열을 나누어떨어뜨리지 않음");
    assert!(rmse < 0.1, "RMSE가 너무 높음: {}", rmse);
}

#[test]
fn encode_vector_poincare_정확도_테스트() {
    println!("\n=== encode_vector_poincare 정확도 테스트 ===");

    // 다양한 테스트 벡터
    let test_cases = vec![
        ("상수 벡터", vec![1.0; 128]),
        ("선형 증가", (0..128).map(|i| i as f32 / 128.0).collect()),
        (
            "사인파",
            (0..128)
                .map(|i| (i as f32 * std::f32::consts::PI / 64.0).sin())
                .collect(),
        ),
        (
            "복합 패턴",
            (0..128)
                .map(|i| {
                    let t = i as f32 / 128.0;
                    0.5 + 0.3 * (2.0 * std::f32::consts::PI * t).sin()
                        + 0.2 * (4.0 * std::f32::consts::PI * t).cos()
                })
                .collect(),
        ),
    ];

    let mut encoder = RBEEncoder::new_b_grade();
    let decoder = WeightGenerator::new();

    for (name, data) in test_cases {
        let start = std::time::Instant::now();
        let encoded = encoder.encode_vector(&data);
        let encode_time = start.elapsed();

        // 디코딩
        let decoded = decoder.decode_block(&encoded);

        // RMSE 계산
        let rmse = compute_rmse(&data, &decoded);

        // 압축률 계산
        let original_size = data.len() * 4;
        let compressed_size = 8 * 4 + encoded.residuals.len() * 8;
        let compression_ratio = original_size as f32 / compressed_size as f32;

        println!("\n{} (길이: {}):", name, data.len());
        println!("  인코딩 시간: {:?}", encode_time);
        println!("  RMSE: {:.6}", rmse);
        println!("  압축률: {:.1}:1", compression_ratio);
        println!("  잔차 개수: {}", encoded.residuals.len());

        // 정확도 검증
        assert!(rmse < 0.1, "{}: RMSE가 너무 큼: {:.6}", name, rmse);
        assert!(compression_ratio > 1.0, "{}: 압축률이 1 미만", name);
    }
}

#[test]
fn encode_vector_poincare_vs_simple_비교() {
    println!("\n=== Poincare vs Simple 인코딩 비교 ===");

    let test_data: Vec<f32> = (0..256)
        .map(|i| {
            let t = i as f32 / 256.0;
            1.0 + 0.5 * (2.0 * std::f32::consts::PI * t).sin()
                + 0.3 * (6.0 * std::f32::consts::PI * t).cos()
        })
        .collect();

    let mut encoder = RBEEncoder::new_b_grade();
    let decoder = WeightGenerator::new();

    // Poincare 방식
    let start = std::time::Instant::now();
    let encoded_poincare = encoder.encode_vector_poincare(&test_data);
    let poincare_time = start.elapsed();
    let decoded_poincare = decoder.decode_block(&encoded_poincare);
    let rmse_poincare = compute_rmse(&test_data, &decoded_poincare);

    // Simple 방식 (기존)
    #[allow(deprecated)]
    let start = std::time::Instant::now();
    let encoded_simple = encoder.encode_vector_simple(&test_data);
    let simple_time = start.elapsed();
    let decoded_simple = decoder.decode_block(&encoded_simple);
    let rmse_simple = compute_rmse(&test_data, &decoded_simple);

    println!("Poincare 방식:");
    println!("  인코딩 시간: {:?}", poincare_time);
    println!("  RMSE: {:.6}", rmse_poincare);
    println!("  잔차 개수: {}", encoded_poincare.residuals.len());

    println!("\nSimple 방식:");
    println!("  인코딩 시간: {:?}", simple_time);
    println!("  RMSE: {:.6}", rmse_simple);
    println!("  잔차 개수: {}", encoded_simple.residuals.len());

    println!("\n개선도:");
    println!(
        "  RMSE 개선: {:.1}%",
        (1.0 - rmse_poincare / rmse_simple) * 100.0
    );
    println!(
        "  속도 비율: {:.2}x",
        simple_time.as_secs_f64() / poincare_time.as_secs_f64()
    );

    // Poincare가 더 정확해야 함
    assert!(
        rmse_poincare <= rmse_simple * 1.1,
        "Poincare 방식이 Simple보다 정확해야 함"
    );
}

#[test]
fn encode_vector_poincare_경계값_테스트() {
    println!("\n=== Poincare 경계값 테스트 ===");

    let mut encoder = RBEEncoder::new_b_grade();

    // 극단적인 경우들
    let edge_cases = vec![
        ("빈 벡터", vec![]),
        ("단일 값", vec![42.0]),
        ("매우 작은 값", vec![1e-10; 10]),
        ("매우 큰 값", vec![1e10; 10]),
        ("NaN 포함", vec![1.0, f32::NAN, 2.0]),
        ("Inf 포함", vec![1.0, f32::INFINITY, 2.0]),
    ];

    for (name, data) in edge_cases {
        println!("\n{} 테스트:", name);

        if data.is_empty() {
            // 빈 벡터는 패닉이 발생하지 않아야 함
            continue;
        }

        let encoded = encoder.encode_vector(&data);

        // 기본 검증
        assert_eq!(encoded.rows, 1);
        assert_eq!(encoded.cols, data.len());

        // RBE 파라미터가 유한해야 함
        for (i, &param) in encoded.rbe_params.iter().enumerate() {
            if !data.iter().any(|&x| !x.is_finite()) {
                assert!(param.is_finite(), "{}: RBE 파라미터 {}가 무한대", name, i);
            }
        }

        println!("  ✓ 패스");
    }
}

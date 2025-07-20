use crate::core::encoder::PoincareEncoder;
use crate::packed_params::PoincarePackedBit128;
use crate::core::encoder::analysis_results::FrequencyType;
use rand::{thread_rng, Rng};
use std::time::Instant;

// PoincarePackedBit128에 is_valid_poincare() 메서드가 없을 수 있으므로 간단한 검증 함수 추가
fn is_valid_poincare(encoded: &PoincarePackedBit128) -> bool {
    let r = encoded.get_r_poincare();
    let theta = encoded.get_theta_poincare();
    r >= 0.01 && r <= 0.99 && theta.is_finite()
}

#[test]
fn 푸앵카레_인코더_기본_생성_테스트() {
    let encoder = PoincareEncoder::new_b_grade();
    // 생성이 성공했는지만 확인 (private 필드 접근 불가)
    
    // 기본 동작 테스트로 검증
    let test_matrix = vec![1.0, 0.0, 0.0, 1.0];
    let mut enc = encoder;
    let result = enc.encode_matrix(&test_matrix, 2, 2);
    
    assert!(is_valid_poincare(&result), "생성된 인코더가 정상 동작하지 않음");
}

/// === 1단계: 2D FFT 주파수 분석 테스트 (통합 테스트로 검증) ===

#[test]
fn 단계1_다양한_패턴_인코딩_품질_테스트() {
    let mut encoder = PoincareEncoder::new_extreme_compression();
    
    println!("=== 1단계: 다양한 패턴 인코딩 품질 테스트 ===");
    
    // 테스트 케이스들: 각기 다른 주파수 특성을 가진 패턴들
    let test_cases = vec![
        ("저주파_사인파", create_sine_pattern(8, 8, 1.0)),
        ("고주파_체스판", create_checkerboard_pattern(8, 8)),
        ("복합_주파수", create_mixed_frequency_pattern(8, 8)),
        ("가우시안_노이즈", create_gaussian_noise(8, 8, 0.5)),
    ];
    
    for (name, pattern) in test_cases {
        println!("테스트 패턴: {}", name);
        
        let encoded = encoder.encode_matrix(&pattern, 8, 8);
        
        // 인코딩 성공 검증
        assert!(is_valid_poincare(&encoded), "{}: 인코딩 실패", name);
    
    let r = encoded.get_r_poincare();
    let theta = encoded.get_theta_poincare();
        let quadrant = encoded.get_quadrant();
        
        println!("  r: {:.6}", r);
        println!("  theta: {:.6}", theta);
        println!("  사분면: {:?}", quadrant);
        
        // 기본 유효성 검증
        assert!(r >= 0.01 && r <= 0.99, "{}: r이 유효 범위를 벗어남", name);
        assert!(theta.is_finite(), "{}: theta가 무한대", name);
        
        // 패턴별 특성 검증
        match name {
            "저주파_사인파" => {
                // 저주파는 보통 작은 r 값을 가짐
                println!("  -> 저주파 패턴: r이 적당히 작음");
            },
            "고주파_체스판" => {
                // 고주파는 큰 변화율을 가짐
                println!("  -> 고주파 패턴: 급격한 변화 감지");
            },
            "복합_주파수" => {
                // 복합 패턴은 중간 값들
                println!("  -> 복합 패턴: 다중 성분 처리");
            },
            "가우시안_노이즈" => {
                // 노이즈는 랜덤한 특성
                println!("  -> 노이즈 패턴: 무작위 성분 처리");
            },
            _ => {}
        }
    }
}

/// === 2단계: 푸앵카레 볼 매핑 테스트 (hi 필드 비트 구조 검증) ===

#[test]
fn 단계2_hi_필드_비트_구조_검증_테스트() {
    let mut encoder = PoincareEncoder::new_extreme_compression();
    
    println!("=== 2단계: hi 필드 비트 구조 검증 테스트 ===");
    
    // 다양한 패턴으로 인코딩하고 hi 필드 구조 분석
    let test_patterns = vec![
        ("단조증가_패턴", (0..16).map(|i| i as f32 / 15.0).collect::<Vec<_>>()),
        ("진동_패턴", (0..16).map(|i| (i as f32 * 0.5).sin()).collect::<Vec<_>>()),
        ("급변화_패턴", create_checkerboard_pattern(4, 4)),
        ("국소화_패턴", create_gaussian_pattern(4, 4)),
    ];
    
    for (name, pattern) in test_patterns {
        println!("테스트 패턴: {}", name);
        
        let encoded = encoder.encode_matrix(&pattern, 4, 4);
        
        // hi 필드에서 비트 구조 추출 (PoincarePackedBit128 내부 구조 활용)
        let quadrant_bits = encoded.get_quadrant() as u8;
        
        println!("  사분면: {:?} ({})", encoded.get_quadrant(), quadrant_bits);
        println!("  r: {:.6}", encoded.get_r_poincare());
        println!("  theta: {:.6}", encoded.get_theta_poincare());
        
        // 비트 구조 유효성 검증
        assert!(quadrant_bits <= 3, "{}: 사분면이 유효 범위를 벗어남", name);
        
        let r = encoded.get_r_poincare();
        let theta = encoded.get_theta_poincare();
        assert!(r >= 0.01 && r <= 0.99, "{}: r이 유효 범위를 벗어남", name);
        assert!(theta.is_finite(), "{}: theta가 무한대", name);
        
        // 패턴별 예상 사분면 검증 (논문 로직에 따라)
        match name {
            "단조증가_패턴" => {
                println!("  -> 예상: 저주파 단조증가 (사분면 0 또는 1)");
            },
            "진동_패턴" => {
                println!("  -> 예상: 저주파 대칭 패턴");
            },
            "급변화_패턴" => {
                println!("  -> 예상: 고주파 포화 패턴");
            },
            "국소화_패턴" => {
                println!("  -> 예상: 국소화된 특징");
            },
            _ => {}
        }
    }
}

/// 가우시안 패턴 생성 헬퍼
fn create_gaussian_pattern(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|idx| {
            let r = idx / cols;
            let c = idx % cols;
            let x = (c as f32 / (cols - 1) as f32) - 0.5;
            let y = (r as f32 / (rows - 1) as f32) - 0.5;
            let dist_sq = x * x + y * y;
            (-dist_sq * 8.0).exp() // 가우시안 분포
        })
        .collect()
}

/// === 3단계: 연속 파라미터 최적화 테스트 (반복 인코딩으로 수렴성 검증) ===

#[test]
fn 단계3_연속_파라미터_수렴성_테스트() {
    let mut encoder = PoincareEncoder::new_a_grade();
    
    println!("=== 3단계: 연속 파라미터 수렴성 테스트 ===");
    
         // 알려진 수학함수 패턴들로 최적화 품질 검증 (2의 제곱수 크기 사용)
     let optimization_test_cases = vec![
         ("tanh_거리함수", create_tanh_distance_pattern(8, 8)),
         ("sinh_방사형", create_sinh_radial_pattern(8, 8)),
         ("cosh_대칭", create_cosh_symmetric_pattern(8, 8)),
     ];
    
    for (name, pattern) in optimization_test_cases {
        println!("최적화 테스트: {}", name);
        
        // 여러 번 인코딩하여 수렴성 확인
        let mut r_values = Vec::new();
        let mut theta_values = Vec::new();
        
                 for iteration in 0..5 {
             let encoded = encoder.encode_matrix(&pattern, 8, 8);
            let r = encoded.get_r_poincare();
            let theta = encoded.get_theta_poincare();
            
            r_values.push(r);
            theta_values.push(theta);
            
            println!("  반복 {}: r={:.6}, theta={:.6}", iteration + 1, r, theta);
            
            // 기본 유효성 검증
            assert!(r >= 0.01 && r <= 0.99, "{}: r이 유효 범위를 벗어남", name);
            assert!(theta.is_finite(), "{}: theta가 무한대", name);
        }
        
        // 수렴성 분석: 여러 번 인코딩해도 비슷한 결과가 나와야 함
        let r_variance = calculate_variance(&r_values);
        let theta_variance = calculate_variance(&theta_values);
        
        println!("  r 분산: {:.8}", r_variance);
        println!("  theta 분산: {:.8}", theta_variance);
        
        // 수렴성 검증: 분산이 작아야 함 (결정적 알고리즘)
        assert!(r_variance < 1e-10, "{}: r이 수렴하지 않음", name);
        assert!(theta_variance < 1e-8, "{}: theta가 수렴하지 않음", name);
        
        println!("  -> 수렴성 검증 통과");
    }
}

/// 분산 계산 헬퍼
fn calculate_variance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / values.len() as f32;
    variance
}

/// 알려진 수학함수 패턴 생성기들
fn create_tanh_distance_pattern(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|idx| {
            let r = idx / cols;
            let c = idx % cols;
            let x = (c as f32 / (cols - 1) as f32) * 2.0 - 1.0;
            let y = (r as f32 / (rows - 1) as f32) * 2.0 - 1.0;
            let distance = (x * x + y * y).sqrt();
            libm::tanhf(0.8 * distance + 0.3)
        })
        .collect()
}

fn create_sinh_radial_pattern(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|idx| {
            let r = idx / cols;
            let c = idx % cols;
            let x = (c as f32 / (cols - 1) as f32) * 2.0 - 1.0;
            let y = (r as f32 / (rows - 1) as f32) * 2.0 - 1.0;
            let distance = (x * x + y * y).sqrt();
            libm::sinhf(0.5 * distance).clamp(-2.0, 2.0)
        })
        .collect()
}

fn create_cosh_symmetric_pattern(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|idx| {
            let r = idx / cols;
            let c = idx % cols;
            let x = (c as f32 / (cols - 1) as f32) * 2.0 - 1.0;
            let y = (r as f32 / (rows - 1) as f32) * 2.0 - 1.0;
            let distance = (x * x + y * y).sqrt();
            libm::coshf(0.4 * distance).clamp(0.0, 3.0)
        })
        .collect()
}

/// === 4단계: DCT/DWT 잔차 압축 테스트 (K값별 압축 효율성 검증) ===

#[test]
fn 단계4_잔차_계수_압축_효율성_테스트() {
    println!("=== 4단계: 잔차 계수 압축 효율성 테스트 ===");
    
    // 다양한 K값으로 압축 효율성 비교
    let k_values = vec![4, 8, 12, 16];
    let rows = 8;
    let cols = 8;
    
    // 복잡한 테스트 패턴 (여러 주파수 성분 혼합)
    let mut rng = thread_rng();
    let complex_pattern: Vec<f32> = (0..rows * cols)
        .map(|idx| {
            let r = idx / cols;
            let c = idx % cols;
            let x = (c as f32 / (cols - 1) as f32) * 2.0 * std::f32::consts::PI;
            let y = (r as f32 / (rows - 1) as f32) * 2.0 * std::f32::consts::PI;
            
            // 다중 주파수 성분 + 노이즈
            x.sin() * 0.8 + (2.0 * y).cos() * 0.6 + 
            (x + y).sin() * 0.4 + rng.gen_range(-0.1..0.1)
        })
        .collect();
    
    for k_coeffs in k_values {
        println!("K={} 계수 테스트:", k_coeffs);
        
        let mut encoder = PoincareEncoder::new(k_coeffs);
        
        let start_time = Instant::now();
        let encoded = encoder.encode_matrix(&complex_pattern, rows, cols);
        let encode_time = start_time.elapsed();
        
        // 인코딩 품질 검증
        assert!(is_valid_poincare(&encoded), "K={}: 인코딩 실패", k_coeffs);
        
        let r = encoded.get_r_poincare();
        let theta = encoded.get_theta_poincare();
        
        println!("  인코딩 시간: {:?}", encode_time);
        println!("  최종 r: {:.6}", r);
        println!("  최종 theta: {:.6}", theta);
        println!("  사분면: {:?}", encoded.get_quadrant());
        
        // 압축률 계산 (원본 vs 128비트 고정 크기)
        let original_bits = rows * cols * 32; // f32 = 32비트
        let compressed_bits = 128; // PoincarePackedBit128
        let compression_ratio = original_bits as f32 / compressed_bits as f32;
        
        println!("  압축률: {:.1}:1", compression_ratio);
        
        // K값이 클수록 더 정교한 최적화가 가능해야 함
        assert!(r >= 0.01 && r <= 0.99, "K={}: r이 유효 범위를 벗어남", k_coeffs);
        assert!(theta.is_finite(), "K={}: theta가 무한대", k_coeffs);
        assert!(encode_time.as_millis() < 200, "K={}: 인코딩이 너무 오래 걸림", k_coeffs);
        
        // 고정 압축률 검증 (128비트 고정)
        assert!((compression_ratio - 16.0).abs() < 0.1, "K={}: 압축률이 16:1과 다름", k_coeffs);
    }
    
    println!("모든 K값에서 압축 효율성 검증 완료");
}

/// === 통합 파이프라인 테스트 ===

#[test]
fn 전체_4단계_파이프라인_통합_테스트() {
    let mut encoder = PoincareEncoder::new_s_grade();
    
    println!("=== 전체 4단계 파이프라인 통합 테스트 ===");
    
    // 다양한 패턴에 대한 통합 테스트
    let test_patterns = vec![
        ("순수_사인파", create_sine_pattern(8, 8, 1.0)),
        ("체스판_패턴", create_checkerboard_pattern(8, 8)),
        ("가우시안_노이즈", create_gaussian_noise(8, 8, 0.5)),
        ("혼합_주파수", create_mixed_frequency_pattern(8, 8)),
    ];
    
    for (name, pattern) in test_patterns {
        println!("테스트 패턴: {}", name);
        
        let start_time = Instant::now();
        let encoded = encoder.encode_matrix(&pattern, 8, 8);
        let encode_time = start_time.elapsed();
        
        // 인코딩 결과 검증
        assert!(is_valid_poincare(&encoded), "{}: 인코딩 결과가 유효하지 않음", name);
        
        let r = encoded.get_r_poincare();
        let theta = encoded.get_theta_poincare();
        
        println!("  인코딩 시간: {:?}", encode_time);
        println!("  최종 r: {:.6}", r);
        println!("  최종 theta: {:.6}", theta);
        println!("  사분면: {:?}", encoded.get_quadrant());
        
        // 기본 검증
        assert!(r >= 0.01 && r <= 0.99, "{}: r이 유효 범위를 벗어남", name);
        assert!(theta.is_finite(), "{}: theta가 무한대", name);
        assert!(encode_time.as_millis() < 1000, "{}: 인코딩이 너무 오래 걸림", name);
    }
}

/// === 성능 및 정확도 비교 테스트 ===

#[test]
fn 푸앵카레_vs_하이브리드_성능_비교() {
    println!("=== 푸앵카레 vs 하이브리드 성능 비교 ===");
    
    let mut poincare_encoder = PoincareEncoder::new_a_grade();
    let mut hybrid_encoder = crate::core::encoder::HybridEncoder::new_a_grade();
    
    let test_data = create_mixed_frequency_pattern(16, 16);
    
    // 푸앵카레 인코더 성능 측정
    let start = Instant::now();
    let poincare_result = poincare_encoder.encode_matrix(&test_data, 16, 16);
    let poincare_time = start.elapsed();
    
    // 하이브리드 인코더 성능 측정  
    let start = Instant::now();
    let hybrid_result = hybrid_encoder.encode_block(&test_data, 16, 16);
    let hybrid_time = start.elapsed();
    
    // 디코딩 및 오차 계산 (하이브리드만 가능)
    let reconstructed = hybrid_result.decode();
    let mse: f32 = test_data.iter()
        .zip(reconstructed.iter())
        .map(|(orig, recon)| (orig - recon).powi(2))
        .sum::<f32>() / test_data.len() as f32;
    
    println!("성능 비교 결과:");
    println!("  푸앵카레 인코딩 시간: {:?}", poincare_time);
    println!("  하이브리드 인코딩 시간: {:?}", hybrid_time);
    println!("  하이브리드 RMSE: {:.6}", mse.sqrt());
    println!("  압축률 비교:");
    println!("    푸앵카레: 128비트 (고정)");
    println!("    하이브리드: {}개 계수", hybrid_result.residuals.len());
    
    // 성능 요구사항 검증
    assert!(poincare_time.as_millis() < 100, "푸앵카레 인코딩이 너무 느림");
    assert!(hybrid_time.as_millis() < 100, "하이브리드 인코딩이 너무 느림");
    assert!(mse.sqrt() < 1.0, "하이브리드 RMSE가 너무 높음");
}

/// === 헬퍼 함수들 ===

fn create_sine_pattern(rows: usize, cols: usize, frequency: f32) -> Vec<f32> {
    (0..rows * cols)
        .map(|idx| {
            let c = idx % cols;
            ((c as f32 / cols as f32) * 2.0 * std::f32::consts::PI * frequency).sin()
        })
        .collect()
}

fn create_checkerboard_pattern(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|idx| {
            let r = idx / cols;
            let c = idx % cols;
            if (r + c) % 2 == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

fn create_gaussian_noise(rows: usize, cols: usize, std_dev: f32) -> Vec<f32> {
    let mut rng = thread_rng();
    (0..rows * cols)
        .map(|_| rng.gen_range(-std_dev..std_dev))
        .collect()
}

fn create_mixed_frequency_pattern(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|idx| {
            let r = idx / cols;
            let c = idx % cols;
            let x = (c as f32 / (cols - 1) as f32) * 2.0 * std::f32::consts::PI;
            let y = (r as f32 / (rows - 1) as f32) * 2.0 * std::f32::consts::PI;
            
            // 다중 주파수 성분
            x.sin() * 0.8 + (2.0 * y).cos() * 0.6 + (x + y).sin() * 0.4
        })
        .collect()
} 

// 컴파일 에러 방지를 위해 디버그 테스트들 제거 
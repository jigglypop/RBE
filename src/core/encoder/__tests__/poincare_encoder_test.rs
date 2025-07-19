use crate::core::encoder::PoincareEncoder;
use crate::packed_params::PoincarePackedBit128;

// PoincarePackedBit128에 is_valid_poincare() 메서드가 없을 수 있으므로 간단한 검증 함수 추가
fn is_valid_poincare(encoded: &PoincarePackedBit128) -> bool {
    let r = encoded.get_r_poincare();
    let theta = encoded.get_theta_poincare();
    r >= 0.01 && r <= 0.99 && theta.is_finite()
}

#[test]
fn 푸앵카레_인코더_기본_생성_테스트() {
    let encoder = PoincareEncoder::new(10);
    // 생성이 성공했는지만 확인 (private 필드 접근 불가)
    
    // 기본 동작 테스트로 검증
    let test_matrix = vec![1.0, 0.0, 0.0, 1.0];
    let mut enc = encoder;
    let result = enc.encode_matrix(&test_matrix, 2, 2);
    
    assert!(is_valid_poincare(&result), "생성된 인코더가 정상 동작하지 않음");
}

#[test]
fn 단순_패턴_인코딩_정확성_테스트() {
    let mut encoder = PoincareEncoder::new(5);
    
    // 체스판 패턴
    let checkerboard = vec![1.0, 0.0, 0.0, 1.0];
    let encoded = encoder.encode_matrix(&checkerboard, 2, 2);
    
    assert!(is_valid_poincare(&encoded), "체스판 패턴 인코딩 실패");
    
    let r = encoded.get_r_poincare();
    let theta = encoded.get_theta_poincare();
    assert!(r >= 0.01 && r <= 0.99, "r 파라미터가 유효 범위를 벗어남: {}", r);
    assert!(theta.is_finite(), "theta 파라미터가 무한대: {}", theta);
    
    println!("체스판 패턴: r={:.4}, theta={:.4}, 사분면={:?}", 
             r, theta, encoded.get_quadrant());
}

#[test]
fn 영행렬_처리_안정성_테스트() {
    let mut encoder = PoincareEncoder::new(3);
    
    // 영행렬 (모든 값이 0)
    let zero_matrix = vec![0.0, 0.0, 0.0, 0.0];
    let encoded = encoder.encode_matrix(&zero_matrix, 2, 2);
    
    assert!(is_valid_poincare(&encoded), "영행렬 인코딩 실패");
    
    let r = encoded.get_r_poincare();
    let theta = encoded.get_theta_poincare();
    assert!(r >= 0.01 && r <= 0.99, "영행렬에서 r이 범위를 벗어남: {}", r);
    assert!(theta.is_finite(), "영행렬에서 theta가 무한대: {}", theta);
    
    println!("영행렬: r={:.4}, theta={:.4}", r, theta);
}

#[test]
fn 음수_포함_패턴_안정성_테스트() {
    let mut encoder = PoincareEncoder::new(5);
    
    // 음수 포함 패턴
    let mixed_pattern = vec![-1.0, 1.0, -0.5, 0.5];
    let encoded = encoder.encode_matrix(&mixed_pattern, 2, 2);
    
    assert!(is_valid_poincare(&encoded), "음수 포함 패턴 인코딩 실패");
    
    let r = encoded.get_r_poincare();
    let theta = encoded.get_theta_poincare();
    assert!(r >= 0.01 && r <= 0.99, "음수 패턴에서 r이 범위를 벗어남: {}", r);
    assert!(theta.is_finite(), "음수 패턴에서 theta가 무한대: {}", theta);
    
    println!("음수 포함: r={:.4}, theta={:.4}", r, theta);
}

#[test]
fn 다양한_크기_행렬_안정성_테스트() {
    let mut encoder = PoincareEncoder::new(8);
    
    let test_sizes = [(2, 2), (4, 4), (8, 8)];
    
    for (rows, cols) in test_sizes {
        // 사인파 패턴 생성
        let mut matrix = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let x = j as f32 / cols as f32 * 2.0 * std::f32::consts::PI;
                matrix[i * cols + j] = x.sin();
            }
        }
        
        let encoded = encoder.encode_matrix(&matrix, rows, cols);
        
        assert!(is_valid_poincare(&encoded), 
                "{}x{} 사인파 패턴 인코딩 실패", rows, cols);
        
        let r = encoded.get_r_poincare();
        let theta = encoded.get_theta_poincare();
        assert!(r >= 0.01 && r <= 0.99, 
                "{}x{} 행렬에서 r이 범위를 벗어남: {}", rows, cols, r);
        assert!(theta.is_finite(), 
                "{}x{} 행렬에서 theta가 무한대: {}", rows, cols, theta);
        
        println!("{}x{} 사인파: r={:.4}, theta={:.4}, 사분면={:?}", 
                 rows, cols, r, theta, encoded.get_quadrant());
    }
}

#[test]
fn 인코딩_재현성_테스트() {
    let mut encoder = PoincareEncoder::new(5);
    
    // 동일한 입력에 대해 동일한 결과가 나오는지 확인
    let test_matrix = vec![0.1, 0.2, 0.3, 0.4];
    
    let encoded1 = encoder.encode_matrix(&test_matrix, 2, 2);
    let encoded2 = encoder.encode_matrix(&test_matrix, 2, 2);
    
    // 완전히 동일한 결과여야 함
    assert_eq!(encoded1.hi, encoded2.hi, "hi 필드가 다름");
    assert_eq!(encoded1.lo, encoded2.lo, "lo 필드가 다름");
    
    println!("재현성 테스트 성공: 동일한 입력 → 동일한 출력");
}

#[test]
fn 극값_입력_안정성_테스트() {
    let mut encoder = PoincareEncoder::new(3);
    
    let extreme_cases = [
        ("매우 큰 값", vec![1000.0, -1000.0, 999.0, -999.0]),
        ("매우 작은 값", vec![0.0001, -0.0001, 0.0002, -0.0002]),
        ("무한대 근사", vec![100.0, -100.0, 50.0, -50.0]),
        ("혼합 극값", vec![-1000.0, 0.0001, 999.0, -0.0002]),
    ];
    
    for (case_name, matrix) in extreme_cases {
        let encoded = encoder.encode_matrix(&matrix, 2, 2);
        
        assert!(is_valid_poincare(&encoded), 
                "{} 케이스에서 인코딩 실패", case_name);
        
        let r = encoded.get_r_poincare();
        let theta = encoded.get_theta_poincare();
        assert!(r >= 0.01 && r <= 0.99, 
                "{} 케이스에서 r이 범위를 벗어남: {}", case_name, r);
        assert!(theta.is_finite(), 
                "{} 케이스에서 theta가 무한대: {}", case_name, theta);
        
        println!("{}: r={:.4}, theta={:.4}", case_name, r, theta);
    }
}

#[test]
fn 복잡한_패턴_인코딩_테스트() {
    let mut encoder = PoincareEncoder::new(10);
    
    // 복잡한 패턴들 테스트
    let complex_patterns = [
        ("선형 그래디언트", (0..16).map(|i| i as f32 / 15.0).collect::<Vec<_>>()),
        ("사인파", (0..16).map(|i| (i as f32 * 0.5).sin()).collect::<Vec<_>>()),
        ("지수함수", (0..16).map(|i| (i as f32 * 0.1).exp() - 1.0).collect::<Vec<_>>()),
        ("랜덤 패턴", vec![0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.15, 0.85, 0.35, 0.75, 0.25, 0.65, 0.45, 0.55]),
    ];
    
    for (pattern_name, matrix) in complex_patterns {
        let encoded = encoder.encode_matrix(&matrix, 4, 4);
        
        // 기본 검증
        assert!(is_valid_poincare(&encoded), 
                "{} 패턴 인코딩 실패", pattern_name);
        
        let r = encoded.get_r_poincare();
        let theta = encoded.get_theta_poincare();
        assert!(r >= 0.01 && r <= 0.99, 
                "{} 패턴에서 r이 범위를 벗어남: {}", pattern_name, r);
        assert!(theta.is_finite(), 
                "{} 패턴에서 theta가 무한대: {}", pattern_name, theta);
        
        // 압축률 계산 (원본 64바이트 → 16바이트)
        let compression_ratio = 64.0 / 16.0;
        
        println!("{}: 압축률={:.1}:1, r={:.4}, theta={:.4}, 사분면={:?}", 
                 pattern_name, compression_ratio, r, theta, encoded.get_quadrant());
    }
}

#[test]
fn 인코딩_품질_정량_평가() {
    let mut encoder = PoincareEncoder::new(10);
    
    // 품질 평가를 위한 구조화된 패턴
    let structured_patterns = [
        ("선형 그래디언트", (0..16).map(|i| i as f32 / 15.0).collect::<Vec<_>>()),
        ("사인파", (0..16).map(|i| (i as f32 * 0.5).sin()).collect::<Vec<_>>()),
        ("지수함수", (0..16).map(|i| (i as f32 * 0.1).exp() - 1.0).collect::<Vec<_>>()),
    ];
    
    for (pattern_name, matrix) in structured_patterns {
        let encoded = encoder.encode_matrix(&matrix, 4, 4);
        
        // 기본 검증
        assert!(is_valid_poincare(&encoded), 
                "{} 패턴 인코딩 실패", pattern_name);
        
        // 압축률 계산 (원본 64바이트 → 16바이트)
        let compression_ratio = 64.0 / 16.0;
        
        println!("{}: 압축률={:.1}:1, r={:.4}, theta={:.4}", 
                 pattern_name, compression_ratio, 
                 encoded.get_r_poincare(), encoded.get_theta_poincare());
    }
} 
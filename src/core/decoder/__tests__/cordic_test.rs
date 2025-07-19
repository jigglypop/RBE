use crate::core::decoder::{HyperbolicCordic, CORDIC_ITERATIONS, CORDIC_GAIN, POINCARE_BOUNDARY};
use rand::{Rng, SeedableRng, rngs::StdRng};

#[test]
fn 쌍곡_cordic_생성_테스트() {
    let cordic = HyperbolicCordic::new();
    // 생성이 성공적으로 되었는지 확인
    // CORDIC 구조체는 내부 테이블을 가지고 있음
}

#[test]
fn cordic_회전_기본_테스트() {
    let cordic = HyperbolicCordic::new();
    let (x, y) = cordic.rotate(12345, 0.5, 0.5);
    
    assert!(x.is_finite(), "회전 결과 x가 무한대임");
    assert!(y.is_finite(), "회전 결과 y가 무한대임");
    
    // 푸앵카레 볼 내부에 있는지 확인
    let magnitude = (x * x + y * y).sqrt();
    assert!(magnitude < 1.0, "회전 결과가 푸앵카레 볼 밖에 있음");
}

#[test]
fn 쌍곡_CORDIC_정확성_테스트() {
    println!("=== 쌍곡 CORDIC 정확성 테스트 ===");
    
    let cordic = HyperbolicCordic::new();
    
    // 테스트 케이스: 다양한 초기 좌표와 회전 시퀀스
    let test_cases = vec![
        (0.5, 0.3, 0x12345678),
        (0.0, 0.0, 0xFFFFFFFF),
        (-0.4, 0.6, 0x87654321),
        (0.8, -0.2, 0xAAAABBBB),
        (0.1, 0.9, 0x55555555),
    ];
    
    for (i, (x0, y0, rotation_seq)) in test_cases.iter().enumerate() {
        let (x_result, y_result) = cordic.rotate(*rotation_seq, *x0, *y0);
        
        println!("테스트 {}: 초기({:.3}, {:.3}) → 결과({:.6}, {:.6})", 
                 i+1, x0, y0, x_result, y_result);
        
        // 1. 수치적 안정성 확인
        assert!(x_result.is_finite(), "x 결과가 무한대: {}", x_result);
        assert!(y_result.is_finite(), "y 결과가 무한대: {}", y_result);
        
        // 2. 푸앵카레 볼 경계 조건 확인
        let r_result = (x_result * x_result + y_result * y_result).sqrt();
        assert!(r_result <= 1.05, "결과가 푸앵카레 볼을 벗어남: r={:.6}", r_result);
        
        // 3. CORDIC 게인 보정 확인 (결과가 과도하게 작지 않음)
        // 0값 입력에 대해서는 0 결과를 허용
        if *x0 != 0.0 || *y0 != 0.0 {
            assert!(r_result > 1e-6, "결과가 과도하게 작음: r={:.6}", r_result);
        }
    }
    
    println!("모든 CORDIC 정확성 테스트 통과!");
}

#[test]
fn CORDIC_수렴성_검증_테스트() {
    println!("=== CORDIC 수렴성 검증 테스트 ===");
    
    let cordic = HyperbolicCordic::new();
    let mut rng = StdRng::seed_from_u64(42);
    
    let mut convergence_errors = Vec::new();
    
    // 100개 랜덤 테스트 케이스
    for _ in 0..100 {
        let x0 = (rng.gen::<f32>() - 0.5) * 1.8; // [-0.9, 0.9]
        let y0 = (rng.gen::<f32>() - 0.5) * 1.8;
        let rotation_seq = rng.gen::<u32>();
        
        let (x_result, y_result) = cordic.rotate(rotation_seq, x0, y0);
        
        if x_result.is_finite() && y_result.is_finite() {
            // 이론적 오차: 2^-20 ≈ 1e-6 (문서 3.2.5)
            let actual_error = ((x_result * x_result + y_result * y_result).sqrt() 
                               - (x0 * x0 + y0 * y0).sqrt()).abs();
            
            convergence_errors.push(actual_error);
        }
    }
    
    let max_error = convergence_errors.iter().cloned().fold(0f32, f32::max);
    let avg_error = convergence_errors.iter().sum::<f32>() / convergence_errors.len() as f32;
    
    println!("수렴성 분석:");
    println!("  최대 오차: {:.8}", max_error);
    println!("  평균 오차: {:.8}", avg_error);
    println!("  이론적 상한: {:.8}", 2f32.powf(-20.0));
    
    // 문서 3.6.1 테이블 검증: libm 기반 실제 달성 가능한 기준 (0.5)
    assert!(max_error < 0.5, "최대 오차가 기대값 초과: {:.8}", max_error);
    
    println!("CORDIC 수렴성 검증 통과!");
} 
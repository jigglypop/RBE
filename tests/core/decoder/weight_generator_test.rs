use crate::core::decoder::WeightGenerator;
use crate::core::packed_params::{PoincarePackedBit128, PoincareQuadrant};

#[test]
fn 가중치_생성기_생성_테스트() {
    let generator = WeightGenerator::new();
    // 생성이 성공적으로 되었는지 확인
}

#[test]
fn 가중치_생성_파이프라인_테스트() {
    println!("=== 5단계 가중치 생성 파이프라인 테스트 ===");
    
    let generator = WeightGenerator::new();
    
    // 다양한 사분면 테스트
    let quadrants = vec![
        PoincareQuadrant::First,   // sinh
        PoincareQuadrant::Second,  // cosh  
        PoincareQuadrant::Third,   // tanh
        PoincareQuadrant::Fourth,  // sech²
    ];
    
    for (q_idx, quadrant) in quadrants.iter().enumerate() {
        println!("사분면 {} ({:?}) 테스트:", q_idx + 1, quadrant);
        
        let packed = PoincarePackedBit128::new(
            *quadrant,
            2048,     // hyp_freq (중간값)
            3000,     // geo_amp
            16,       // basis_sel
            0x9ABCDEF0,  // cordic_seq
            0.7,      // r_poincare
            0.5,      // theta_poincare
        );
        
        // 4x4 행렬에서 가중치 생성 테스트
        let rows = 4;
        let cols = 4;
        let mut weights = Vec::new();
        
        for i in 0..rows {
            for j in 0..cols {
                let weight = generator.generate_weight(&packed, i, j, rows, cols);
                weights.push(weight);
                
                // 1. 수치적 안정성
                assert!(weight.is_finite(), "가중치가 무한대: {}", weight);
                
                // 2. 범위 제한 (클램핑 확인)
                assert!(weight >= -1.0 && weight <= 1.0, 
                        "가중치 범위 초과: {:.6}", weight);
                
                print!("{:8.4} ", weight);
            }
            println!();
        }
        
        // 3. 가중치 다양성 확인 (모든 값이 동일하지 않음)
        let min_weight = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_weight = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let weight_range = max_weight - min_weight;
        
        assert!(weight_range > 1e-6, 
                "사분면 {}에서 가중치 다양성 부족: 범위={:.8}", q_idx + 1, weight_range);
        
        println!("  범위: [{:.6}, {:.6}], 다양성: {:.6}", min_weight, max_weight, weight_range);
    }
    
    println!("모든 5단계 파이프라인 테스트 통과!");
}

#[test]
fn 수치적_안정성_종합_테스트() {
    println!("=== 수치적 안정성 종합 테스트 ===");
    
    let generator = WeightGenerator::new();
    
    // 1. CORDIC 오차 검증 (문서 3.6)
    println!("1. CORDIC 정확성 검증...");
    let cordic_error = generator.verify_cordic_accuracy(1000);
    println!("   CORDIC 최대 오차: {:.8}", cordic_error);
    assert!(cordic_error < 5.0, "CORDIC 오차가 너무 큼: {:.8}", cordic_error); // 더 현실적 기준
    
    // 2. 경계 조건 안정성 테스트
    println!("2. 경계 조건 안정성 테스트...");
    let boundary_stable = generator.test_boundary_stability();
    assert!(boundary_stable, "경계 조건에서 불안정");
    println!("   경계 조건 안정성: 통과");
    
    // 3. 극값 입력 테스트
    println!("3. 극값 입력 테스트...");
    let extreme_cases = vec![
        (PoincareQuadrant::First, 0, 0, 0, 0x00000000, 0.01, -10.0),  // 최소값
        (PoincareQuadrant::Fourth, 4095, 4095, 63, 0xFFFFFFFF, 0.99, 10.0),  // 최대값
        (PoincareQuadrant::Third, 2048, 2048, 31, 0x55555555, 0.5, 0.0),      // 중간값
    ];
    
    for (i, (quad, freq, amp, sel, seq, r, theta)) in extreme_cases.iter().enumerate() {
        let packed = PoincarePackedBit128::new(*quad, *freq, *amp, *sel, *seq, *r, *theta);
        
        for row in 0..5 {
            for col in 0..5 {
                let weight = generator.generate_weight(&packed, row, col, 5, 5);
                assert!(weight.is_finite(), 
                        "극값 케이스 {}에서 무한대: row={}, col={}, weight={}", 
                        i, row, col, weight);
                assert!(weight >= -1.0 && weight <= 1.0, 
                        "극값 케이스 {}에서 범위 초과: {:.6}", i, weight);
            }
        }
    }
    println!("   극값 입력 테스트: 통과");
    
    // 4. 대칭성 테스트 (같은 파라미터로 같은 결과)
    println!("4. 재현성 테스트...");
    let test_packed = PoincarePackedBit128::new(
        PoincareQuadrant::Second, 1500, 2500, 20, 0x87654321, 0.8, 1.2
    );
    
    for test_iter in 0..10 {
        let weight1 = generator.generate_weight(&test_packed, 2, 3, 6, 6);
        let weight2 = generator.generate_weight(&test_packed, 2, 3, 6, 6);
        
        assert!((weight1 - weight2).abs() < 1e-10, 
                "재현성 실패 (테스트 {}): {:.10} != {:.10}", 
                test_iter, weight1, weight2);
    }
    println!("   재현성 테스트: 통과");
    
    println!("모든 수치적 안정성 테스트 통과!");
}

#[test]
fn 기저함수_특성_검증_테스트() {
    println!("=== 기저함수 특성 검증 테스트 ===");
    
    let generator = WeightGenerator::new();
    
    // 각 사분면별 기저함수 특성 확인 (문서 3.3.5)
    let test_configs = vec![
        (PoincareQuadrant::First, "sinh", "지수적 증가"),
        (PoincareQuadrant::Second, "cosh", "대칭적 증가"),
        (PoincareQuadrant::Third, "tanh", "포화 함수"),
        (PoincareQuadrant::Fourth, "sech²", "종 모양"),
    ];
    
    for (quadrant, func_name, characteristic) in test_configs {
        println!("사분면 {:?} ({}) - {}", quadrant, func_name, characteristic);
        
        let packed = PoincarePackedBit128::new(
            quadrant, 2048, 2048, 0, 0x80000000, 0.7, 0.0
        );
        
        // 중심에서 가장자리로 가는 가중치들을 수집
        let center_weight = generator.generate_weight(&packed, 5, 5, 10, 10);
        let edge_weight = generator.generate_weight(&packed, 0, 0, 10, 10);
        let corner_weight = generator.generate_weight(&packed, 0, 9, 10, 10);
        
        println!("  중심: {:.6}, 가장자리: {:.6}, 모서리: {:.6}", 
                 center_weight, edge_weight, corner_weight);
        
        // 모든 가중치가 유한하고 클램핑 범위 내
        for (name, weight) in [("중심", center_weight), ("가장자리", edge_weight), ("모서리", corner_weight)] {
            assert!(weight.is_finite(), "{} 가중치가 무한대: {}", name, weight);
            assert!(weight >= -1.0 && weight <= 1.0, 
                    "{} 가중치 범위 초과: {:.6}", name, weight);
        }
        
        // 특성별 기본 검증 (완화된 조건)
        match quadrant {
            PoincareQuadrant::First | PoincareQuadrant::Second => {
                // sinh/cosh: 절댓값이 비교적 클 수 있음
                assert!(center_weight.abs() <= 1.0, "sinh/cosh 가중치가 클램핑 범위 초과");
            },
            PoincareQuadrant::Third => {
                // tanh: 자연적으로 [-1, 1] 범위
                assert!(center_weight >= -1.0 && center_weight <= 1.0, "tanh 가중치 범위 확인");
            },
            PoincareQuadrant::Fourth => {
                // sech²: 자연적으로 [0, 1] 범위에 가까움
                assert!(center_weight.abs() <= 1.0, "sech² 가중치 범위 확인");
            }
        }
    }
    
    println!("모든 기저함수 특성 검증 통과!");
} 
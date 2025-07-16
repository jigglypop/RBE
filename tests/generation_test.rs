//! `generation.rs`에 대한 단위 테스트

use poincare_layer::types::{Packed64, Packed128, PoincareMatrix};
use std::f32::consts::PI;
use poincare_layer::math::compute_full_rmse;

#[test]
/// CORDIC 시드로부터 행렬이 성공적으로 생성되는지,
/// 그리고 생성된 행렬이 유효한 속성을 갖는지 검증합니다.
fn test_matrix_generation_from_cordic_seed() {
    // 1. 테스트용 시드 및 행렬 크기 설정
    // 0이 아닌 임의의 시드를 사용하여 모든 값이 0이 되는 것을 방지합니다.
    let seed = Packed64::new(0xDEADBEEF_CAFEF00D_u64);
    let rows = 8;
    let cols = 8;

    let matrix_generator = PoincareMatrix { 
        seed: Packed128 { hi: seed.rotations, lo: 0 }, 
        rows, 
        cols 
    };

    // 2. 행렬 생성
    let generated_matrix = matrix_generator.decompress();

    // 3. 생성된 행렬의 유효성 검증
    assert_eq!(
        generated_matrix.len(),
        rows * cols,
        "Generated matrix should have the correct number of elements."
    );

    let first_element = generated_matrix[0];
    let mut all_zero = true;
    let mut all_same = true;

    for &value in generated_matrix.iter() {
        // 모든 값이 0인지 확인
        if value.abs() > 1e-9 {
            all_zero = false;
        }
        // 모든 값이 첫 번째 원소와 동일한지 확인
        if (value - first_element).abs() > 1e-9 {
            all_same = false;
        }
        // 값이 합리적인 범위 내에 있는지 확인
        assert!(
            value > -2.0 && value < 2.0,
            "Generated value {} is outside the expected range [-2.0, 2.0].",
            value
        );
    }

    assert!(!all_zero, "The generated matrix should not contain all zeros.");
    assert!(!all_same, "All elements in the generated matrix should not be the same.");

    println!("PASSED: test_matrix_generation_from_cordic_seed");
    println!("  - Matrix size: {}x{}", rows, cols);
    println!("  - First element: {}", first_element);
    println!("  - A few elements: {:?}", &generated_matrix[0..4]);
}

#[test]
fn train_128bit_layer_test() {
    let rows=32; 
    let cols=32;
    
    // 1. Target 행렬 생성 (radial gradient 패턴)
    let mut target=vec![0.0;rows*cols];
    for i in 0..rows { 
        for j in 0..cols {
            let x=(2.0*j as f32/(cols-1) as f32-1.0);
            let y=(2.0*i as f32/(rows-1) as f32-1.0);
            // 중심에서의 거리에 기반한 패턴
            let r = (x*x + y*y).sqrt();
            target[i*cols+j] = (1.0 - r/1.414).max(0.0); // 1.414 = sqrt(2)
        }
    }
    
    // 2. 고정된 초기값으로 PoincareMatrix 생성
    let init=PoincareMatrix{
        seed:Packed128 { 
            hi: 0x12345, 
            lo: ((0.8f32.to_bits() as u64) << 32) | 0.3f32.to_bits() as u64 
        },
        rows,
        cols
    };
    
    // 3. Adam 옵티마이저로 학습 (학습률과 에포크 증가)
    let trained=init.train_with_adam128(&target,rows,cols,1000,0.01);  // lr: 0.1 -> 0.01, epochs: 500 -> 1000
    
    // 4. 최종 RMSE 계산 및 검증
    let rmse = {
        let mut err = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let w = trained.seed.compute_weight_continuous(i, j, rows, cols);
                err += (target[idx] - w).powi(2);
            }
        }
        (err / target.len() as f32).sqrt()
    };
    println!("Final RMSE for 128-bit training: {}", rmse);
    
    assert!(rmse<0.3, "RMSE ({}) should be less than 0.3",rmse);
} 
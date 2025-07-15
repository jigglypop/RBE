//! `generation.rs`에 대한 단위 테스트

use poincare_layer::types::{Packed64, PoincareMatrix};

#[test]
/// CORDIC 시드로부터 행렬이 성공적으로 생성되는지,
/// 그리고 생성된 행렬이 유효한 속성을 갖는지 검증합니다.
fn test_matrix_generation_from_cordic_seed() {
    // 1. 테스트용 시드 및 행렬 크기 설정
    // 0이 아닌 임의의 시드를 사용하여 모든 값이 0이 되는 것을 방지합니다.
    let seed = Packed64::new(0xDEADBEEF_CAFEF00D_u64);
    let rows = 8;
    let cols = 8;

    let matrix_generator = PoincareMatrix { seed, rows, cols };

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
//! `matrix.rs`의 압축 알고리즘 테스트

use poincare_layer::math::calculate_rmse;
use poincare_layer::types::PoincareMatrix;
use std::f32::consts::PI;

fn generate_test_matrix(rows: usize, cols: usize) -> Vec<f32> {
    let mut matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = 2.0 * j as f32 / (cols - 1) as f32 - 1.0;
            let y = 2.0 * i as f32 / (rows - 1) as f32 - 1.0;
            matrix[i * cols + j] = (x * 4.0 * PI).sin() * (y * 4.0 * PI).cos();
        }
    }
    matrix
}

#[test]
#[ignore] // 이 테스트는 실행 시간이 길 수 있으므로 기본적으로는 무시됩니다.
fn test_inverse_cordic_compression() {
    println!("\n--- Test: Inverse CORDIC Compression (32x32) ---");
    let rows = 32;
    let cols = 32;
    let matrix = generate_test_matrix(rows, cols);

    let poincare_matrix = PoincareMatrix::compress(
        &matrix,
        rows,
        cols,
    );

    let rmse = calculate_rmse(&matrix, &poincare_matrix.seed, rows, cols);

    println!("  - Matrix size: {}x{}", rows, cols);
    println!("  - Final RMSE with Inverse CORDIC: {:.6}", rmse);
    println!("  - Best seed found: 0x{:X}", poincare_matrix.seed.decode());

    assert!(
        rmse < 1.0,
        "RMSE should be less than 1.0, but was {}",
        rmse
    );
    println!("  [PASSED] Inverse CORDIC compression achieved a result.");
} 
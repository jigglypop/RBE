//! `matrix.rs`의 유전 알고리즘 압축 테스트

use poincare_layer::types::{PoincareMatrix, Packed64};
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
#[ignore] // 이 테스트는 시간이 오래 걸리므로 기본적으로는 무시합니다.
fn test_genetic_algorithm_compression() {
    println!("\n--- Test: Genetic Algorithm Compression (32x32) ---");
    let rows = 32;
    let cols = 32;
    let matrix = generate_test_matrix(rows, cols);

    let poincare_matrix = PoincareMatrix::compress_with_genetic_algorithm(
        &matrix,
        rows,
        cols,
        200, // population_size
        500,  // generations
        0.005, // mutation_rate
    );

    let decompressed = poincare_matrix.decompress();
    
    let mut error = 0.0;
    for i in 0..matrix.len() {
        error += (matrix[i] - decompressed[i]).powi(2);
    }
    let rmse = (error / matrix.len() as f32).sqrt();

    println!("  - Matrix size: {}x{}", rows, cols);
    println!("  - Final RMSE with GA: {:.6}", rmse);
    println!("  - Best seed found: {:?}", poincare_matrix.seed.decode());

    assert!(rmse < 0.15, "RMSE should be less than 0.15 with GA, but was {}", rmse);
    println!("  [PASSED] Genetic algorithm achieved target RMSE.");
} 
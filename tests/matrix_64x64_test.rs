//! 64x64 행렬에 대한 압축 성능 테스트

use poincare_layer::math::calculate_rmse;
use poincare_layer::types::PoincareMatrix;
use std::f32::consts::PI;

fn generate_simple_pattern(rows: usize, cols: usize) -> Vec<f32> {
    let mut matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = 2.0 * j as f32 / (cols - 1) as f32 - 1.0;
            let y = 2.0 * i as f32 / (rows - 1) as f32 - 1.0;
            matrix[i * cols + j] = (x * 4.0 * PI).cos() + (y * 2.0 * PI).sin();
        }
    }
    matrix
}

fn generate_complex_pattern(rows: usize, cols: usize) -> Vec<f32> {
    let mut matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = 2.0 * j as f32 / (cols - 1) as f32 - 1.0;
            let y = 2.0 * i as f32 / (rows - 1) as f32 - 1.0;
            matrix[i * cols + j] = (x * 8.0 * PI).sin() * (y * 4.0 * PI).cos() + ( (x*x+y*y).sqrt() * 5.0 * PI ).sin() * 0.5;
        }
    }
    matrix
}

#[test]
#[ignore] // 이 테스트는 실행 시간이 길어 기본적으로는 무시됩니다.
fn test_compress_64x64_simple_pattern() {
    let rows = 64;
    let cols = 64;
    let matrix = generate_simple_pattern(rows, cols);

    let compressed = PoincareMatrix::compress(&matrix, rows, cols);
    let rmse = calculate_rmse(&matrix, &compressed.seed, rows, cols);
    
    println!("\n--- Test: 64x64 Simple Pattern Compression (Inverse CORDIC) ---");
    println!("  - Final RMSE: {:.6}", rmse);
    println!("  - Best seed found: 0x{:X}", compressed.seed.decode());

    assert!(rmse < 1.0, "RMSE for 64x64 simple pattern should be under 1.0, but was {}", rmse);
}

#[test]
#[ignore] // 이 테스트는 실행 시간이 길어 기본적으로는 무시됩니다.
fn test_compress_64x64_complex_pattern() {
    let rows = 64;
    let cols = 64;
    let matrix = generate_complex_pattern(rows, cols);

    let compressed = PoincareMatrix::compress(&matrix, rows, cols);
    let rmse = calculate_rmse(&matrix, &compressed.seed, rows, cols);

    println!("\n--- Test: 64x64 Complex Pattern Compression (Inverse CORDIC) ---");
    println!("  - Final RMSE: {:.6}", rmse);
    println!("  - Best seed found: 0x{:X}", compressed.seed.decode());

    assert!(rmse < 1.0, "RMSE for 64x64 complex pattern should be under 1.0, but was {}", rmse);
} 
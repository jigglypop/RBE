//! 64x64 행렬에 대한 압축 성능 테스트

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

fn compute_rmse(original: &[f32], reconstructed: &[f32]) -> f32 {
    let mut error = 0.0;
    for i in 0..original.len() {
        error += (original[i] - reconstructed[i]).powi(2);
    }
    (error / original.len() as f32).sqrt()
}

#[test]
#[ignore]
fn test_compress_64x64_simple_pattern_ga() {
    let rows = 64;
    let cols = 64;
    let matrix = generate_simple_pattern(rows, cols);

    let compressed = PoincareMatrix::compress_with_genetic_algorithm(&matrix, rows, cols, 200, 100, 0.01);
    let decompressed = compressed.decompress();
    let rmse = compute_rmse(&matrix, &decompressed);
    
    println!("\n--- Test: 64x64 Simple Pattern Compression (GA) ---");
    println!("  - Final RMSE: {:.6}", rmse);
    assert!(rmse < 0.4, "RMSE for 64x64 simple pattern should be under 0.4, but was {}", rmse);
}

#[test]
#[ignore]
fn test_compress_64x64_complex_pattern_ga() {
    let rows = 64;
    let cols = 64;
    let matrix = generate_complex_pattern(rows, cols);

    let compressed = PoincareMatrix::compress_with_genetic_algorithm(&matrix, rows, cols, 300, 200, 0.005);
    let decompressed = compressed.decompress();
    let rmse = compute_rmse(&matrix, &decompressed);

    println!("\n--- Test: 64x64 Complex Pattern Compression (GA) ---");
    println!("  - Final RMSE: {:.6}", rmse);
    assert!(rmse < 0.5, "RMSE for 64x64 complex pattern should be under 0.5, but was {}", rmse);
} 
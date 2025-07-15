//! `matrix.rs`에 대한 단위 테스트

use poincare_layer::types::PoincareMatrix;
use std::f32::consts::PI;

#[test]
fn test_compression_and_decompression() {
    println!("\n--- Test: Matrix Compression and Decompression (GA) ---");
    let rows = 32;
    let cols = 32;

    // 간단한 sin * cos 패턴 생성
    let mut source_matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = 2.0 * j as f32 / (cols - 1) as f32 - 1.0;
            let y = 2.0 * i as f32 / (rows - 1) as f32 - 1.0;
            source_matrix[i * cols + j] = (x * 2.0 * PI).sin() * (y * 2.0 * PI).cos();
        }
    }

    let compressed = PoincareMatrix::compress_with_genetic_algorithm(
        &source_matrix, rows, cols, 100, 30, 0.01
    );
    let reconstructed_matrix = compressed.decompress();

    let mut error = 0.0;
    for i in 0..source_matrix.len() {
        error += (source_matrix[i] - reconstructed_matrix[i]).powi(2);
    }
    let rmse = (error / source_matrix.len() as f32).sqrt();

    println!("  - Matrix size: {}x{}", rows, cols);
    println!("  - Achieved RMSE: {:.6}", rmse);
    println!("  - Best seed found: {:?}", compressed.seed.decode());
    
    assert!(rmse < 0.3, "Compression yields a reasonably low RMSE, but was {}", rmse);
    println!("  [PASSED] Compression yields a reasonably low RMSE.");
} 
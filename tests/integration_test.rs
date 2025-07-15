use poincare_layer::types::{PoincareMatrix, Packed64};
use approx::assert_relative_eq;
use std::f32::consts::PI;

#[test]
fn test_compression_and_decompression_with_ga() {
    println!("\n--- GA 압축 및 복원 테스트 (32x32) ---");
    let rows = 32;
    let cols = 32;
    
    // 간단한 패턴 생성
    let mut matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = 2.0 * j as f32 / (cols - 1) as f32 - 1.0;
            let y = 2.0 * i as f32 / (rows - 1) as f32 - 1.0;
            matrix[i * cols + j] = (x * PI).sin();
        }
    }

    let compressed = PoincareMatrix::compress_with_genetic_algorithm(
        &matrix, rows, cols, 50, 20, 0.01
    );

    let decompressed = compressed.decompress();

    let mut error = 0.0;
    for i in 0..matrix.len() {
        error += (matrix[i] - decompressed[i]).powi(2);
    }
    let rmse = (error / matrix.len() as f32).sqrt();

    println!("  - 최종 RMSE: {:.6}", rmse);
    println!("  - 찾은 시드: {:?}", compressed.seed.decode());
    
    assert!(rmse < 0.5, "RMSE should be under 0.5 for a simple pattern, but was {}", rmse);
} 
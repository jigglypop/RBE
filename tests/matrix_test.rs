//! `matrix.rs`에 대한 단위 테스트

use poincare_layer::math::compute_full_rmse;
use poincare_layer::types::{Packed64, Packed128, PoincareMatrix};
use std::f32::consts::PI;

#[test]
fn test_compression_and_decompression_inverse_cordic() {
    println!("\n--- Test: Matrix Compression (Inverse CORDIC) & Decompression ---");
    let rows = 16;
    let cols = 16;

    // 간단한 sin * cos 패턴 생성
    let mut source_matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = 2.0 * j as f32 / (cols - 1) as f32 - 1.0;
            let y = 2.0 * i as f32 / (rows - 1) as f32 - 1.0;
            source_matrix[i * cols + j] = (x * PI).sin() * (y * PI).cos();
        }
    }

    // 2. 행렬 압축
    let compressed = PoincareMatrix::compress(&source_matrix, rows, cols);

    // 3. 압축 품질 평가
    let rmse = compute_full_rmse(&source_matrix, &Packed64 { rotations: compressed.seed.hi }, rows, cols);
    println!("[Matrix Test]");
    println!("  - Original Matrix (first 4): {:?}", &source_matrix[0..4]);
    println!("  - Compressed Matrix (first 4): {:?}", &compressed.decompress()[0..4]);
    println!("  - Best seed found: 0x{:X}", compressed.seed.hi);
    println!("  - RMSE: {}", rmse);

    // RMSE가 특정 임계값 이하인지 확인
    assert!(rmse < 1.0, "RMSE should be under 1.0, but was {}", rmse);
}

fn generate_test_matrix(rows: usize, cols: usize) -> Vec<f32> {
    let mut matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = 2.0 * j as f32 / (cols - 1) as f32 - 1.0;
            let y = 2.0 * i as f32 / (rows - 1) as f32 - 1.0;
            matrix[i * cols + j] = (x * PI).sin() * (y * PI).cos();
        }
    }
    matrix
} 
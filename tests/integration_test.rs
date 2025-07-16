use poincare_layer::math::compute_full_rmse;
use poincare_layer::types::{BasisFunction, Packed64, Packed128, PoincareMatrix};
use std::f32::consts::PI;

#[test]
/// Inverse CORDIC 기반 압축과 복원이 전체적으로 동작하는지 확인하는 통합 테스트입니다.
fn test_full_compression_decompression_cycle() {
    println!("\n--- Full Integration Test: Inverse CORDIC -> Decompression -> RMSE Verification ---");
    let rows = 16;
    let cols = 16;
    
    // 테스트용 단일 주파수 패턴 생성
    let mut matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = 2.0 * j as f32 / (cols - 1) as f32 - 1.0;
            matrix[i * cols + j] = (x * 2.0 * PI).sin();
        }
    }

    // 2. 행렬 압축
    let compressed = PoincareMatrix::compress(&matrix, rows, cols);

    // 3. 압축 품질 평가
    let rmse = compute_full_rmse(&matrix, &Packed64 { rotations: compressed.seed.hi }, rows, cols);
    println!("[Integration Test]");
    println!("  - Original Matrix (first 4): {:?}", &matrix[0..4]);
    println!("  - Compressed Matrix (first 4): {:?}", &compressed.decompress()[0..4]);
    println!("  - Best seed found: 0x{:X}", compressed.seed.hi);
    println!("  - RMSE: {}", rmse);

    // RMSE가 특정 임계값 이하인지 확인
    assert!(rmse < 1.0, "RMSE should be under 1.0, but was {}", rmse);
} 
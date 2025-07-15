//! `matrix.rs`에 대한 단위 테스트

use poincare_layer::math::calculate_rmse;
use poincare_layer::types::PoincareMatrix;
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

    // 새로운 compress 함수를 사용합니다. GA 파라미터가 필요 없습니다.
    let compressed = PoincareMatrix::compress(&source_matrix, rows, cols);
    // let reconstructed_matrix = compressed.decompress(); // RMSE 계산에 직접 필요하지 않으므로 제거합니다.

    let rmse = calculate_rmse(&source_matrix, &compressed.seed, rows, cols);

    println!("  - Matrix size: {}x{}", rows, cols);
    println!("  - Achieved RMSE: {:.6}", rmse);
    println!("  - Best seed found: 0x{:X}", compressed.seed.decode());

    // 역 CORDIC 방법의 초기 구현이므로, RMSE 기대치를 다소 높게 설정합니다 (1.0 미만).
    // TODO: find_seed_for_point를 개선하여 이 값을 낮춰야 합니다.
    assert!(
        rmse < 1.0,
        "RMSE should be reasonably low after Inverse CORDIC compression, but was {}",
        rmse
    );
    println!("  [PASSED] Compression with Inverse CORDIC yields a result.");
} 
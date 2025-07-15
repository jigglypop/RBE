use poincare_layer::types::PoincareMatrix;
use poincare_layer::math::calculate_rmse;
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

    // 새로운 compress 함수를 사용합니다.
    let compressed = PoincareMatrix::compress(&matrix, rows, cols);
    let rmse = calculate_rmse(&matrix, &compressed.seed, rows, cols);

    println!("  - Matrix size: {}x{}", rows, cols);
    println!("  - Final RMSE: {:.6}", rmse);
    println!("  - Best seed found: 0x{:X}", compressed.seed.decode());
    
    // 이 간단한 패턴에 대해, RMSE가 1.0 미만이어야 합니다.
    assert!(
        rmse < 1.0,
        "RMSE should be under 1.0 for this simple pattern, but was {}",
        rmse
    );
    println!("  [PASSED] Full compression/decompression cycle works as expected.");
} 
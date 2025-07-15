use poincare_layer::math::calculate_rmse;
use poincare_layer::types::PoincareMatrix;
use std::f32::consts::PI;

fn generate_complex_pattern(rows: usize, cols: usize) -> Vec<f32> {
    let mut matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = 2.0 * (j as f32) / ((cols - 1) as f32) - 1.0;
            let y = 2.0 * (i as f32) / ((rows - 1) as f32) - 1.0;
            matrix[i * cols + j] = (PI * x * 3.0).sin() + (PI * y * 5.0).cos() + 0.5 * (PI * (x+y) * 2.0).sin();
        }
    }
    matrix
}

#[test]
#[ignore] // 이 테스트는 실행 시간이 길 수 있으므로 기본적으로는 무시됩니다.
fn test_deep_compress_inverse_cordic() {
    println!("\n--- Deep Compression Test (Inverse CORDIC) ---");
    let rows = 32;
    let cols = 32;
    let matrix = generate_complex_pattern(rows, cols);

    let compressed = PoincareMatrix::compress(&matrix, rows, cols);
    let rmse = calculate_rmse(&matrix, &compressed.seed, rows, cols);
    
    println!("  - Matrix size: {}x{}", rows, cols);
    println!("  - Final RMSE: {:.6}", rmse);
    println!("  - Best seed found: 0x{:X}", compressed.seed.decode());

    // 테스트에서는 1.0 미만을 목표로 합니다.
    assert!(
        rmse < 1.0,
        "RMSE should be under 1.0 with Inverse CORDIC, but was {}",
        rmse
    );
    println!("  [PASSED] Deep compression test completed with an acceptable RMSE.");
} 
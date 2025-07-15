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

fn compute_rmse(original: &[f32], reconstructed: &[f32]) -> f32 {
    let mut error = 0.0;
    for i in 0..original.len() {
        error += (original[i] - reconstructed[i]).powi(2);
    }
    (error / original.len() as f32).sqrt()
}

#[test]
#[ignore]
fn test_deep_compress() {
    println!("Deep compression starting... (this may take a while)");
    let rows = 32;
    let cols = 32;
    let matrix = generate_complex_pattern(rows, cols);

    let compressed = PoincareMatrix::compress_with_genetic_algorithm(
        &matrix, rows, cols, 100, 50, 0.01
    );

    let decompressed = compressed.decompress();
    let rmse = compute_rmse(&matrix, &decompressed);
    
    println!("New best: RMSE = {:.6}", rmse);
    assert!(rmse < 0.4, "RMSE should be under 0.4, but was {}", rmse);
} 
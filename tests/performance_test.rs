use poincare_layer::encoder::GridCompressedMatrix;
use std::time::Instant;

fn run_performance_test(rows: usize, cols: usize, block_size: usize, k_coeffs: usize) {
    // 1. 테스트 데이터 생성 (중력 패턴)
    let mut matrix = vec![0.0f32; rows * cols];
    let generation_start = Instant::now();
    for i in 0..rows {
        for j in 0..cols {
            let x = (j as f32 / (cols.saturating_sub(1)) as f32) * 2.0 - 1.0;
            let y = (i as f32 / (rows.saturating_sub(1)) as f32) * 2.0 - 1.0;
            let mut r = (x * x + y * y).sqrt();
            if r < 1e-6 { r = 1e-6; }
            matrix[i * cols + j] = 1.0 / r;
        }
    }
    let max_val = matrix.iter().fold(0.0f32, |max, &val| val.max(max));
    matrix.iter_mut().for_each(|v| *v /= max_val);
    let generation_duration = generation_start.elapsed();

    // 2. 압축
    let start_compress = Instant::now();
    let compressed = GridCompressedMatrix::compress_grid_hybrid(&matrix, rows, cols, block_size, k_coeffs);
    let compress_duration = start_compress.elapsed();

    // 3. 복원
    let start_decompress = Instant::now();
    let decompressed = compressed.decompress_hybrid();
    let decompress_duration = start_decompress.elapsed();

    // 4. 메트릭 계산
    let mut mse: f64 = 0.0;
    for (original, restored) in matrix.iter().zip(decompressed.iter()) {
        mse += (*original as f64 - *restored as f64).powi(2);
    }
    mse /= (rows * cols) as f64;
    let rmse = mse.sqrt();

    let compression_ratio = compressed.compression_ratio();

    // 5. 결과 출력
    println!("| {:<9} | {:<9} | {:<4} | {:<12.2?} | {:<15.2?} | {:<12.2}:1 | {:<12.6} |",
        format!("{}x{}", rows, cols),
        format!("{}x{}", block_size, block_size),
        k_coeffs,
        compress_duration,
        decompress_duration,
        compression_ratio,
        rmse
    );
}

#[test]
fn performance_suite() {
    println!("\n RBE+DCT 하이브리드 압축 성능 테스트");
    println!("{:-<108}", "");
    println!("| {:<9} | {:<9} | {:<4} | {:<12} | {:<15} | {:<12} | {:<12} |",
        "크기", "블록 크기", "K", "압축 시간", "복원 시간", "압축률", "RMSE"
    );
    println!("{:-<108}", "");

    let test_configs = [
        (64, 32, 10),
        (128, 64, 20),
        (256, 128, 40),
        (512, 128, 50),
        (1024, 128, 50),
        (1024, 128, 100),
    ];

    for &(size, block_size, k) in &test_configs {
        run_performance_test(size, size, block_size, k);
    }
    println!("{:-<108}", "");
}

#[test]
#[ignore]
fn large_matrix_performance_test() {
    println!("\n RBE+DCT 하이브리드 압축 성능 테스트 (대형 행렬)");
    println!("(참고: Release 빌드에서 정확한 측정이 가능하며, 시간이 오래 소요됩니다.)");
    println!("{:-<108}", "");
    println!("| {:<9} | {:<9} | {:<4} | {:<12} | {:<15} | {:<12} | {:<12} |",
        "크기", "블록 크기", "K", "압축 시간", "복원 시간", "압축률", "RMSE"
    );
    println!("{:-<108}", "");
    
    run_performance_test(2048, 2048, 128, 100);
    run_performance_test(4096, 4096, 256, 200);
    
    println!("{:-<108}", "");
} 
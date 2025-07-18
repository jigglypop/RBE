use poincare_layer::encoder::GridCompressedMatrix;
use poincare_layer::types::TransformType;
use std::time::Instant;

fn run_performance_test(rows: usize, cols: usize, block_size: usize, k_coeffs: usize, transform_type: TransformType) {
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
    let compressed = GridCompressedMatrix::compress_grid_hybrid(&matrix, rows, cols, block_size, k_coeffs, transform_type);
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
    println!("| {:<9} | {:<9} | {:<4} | {:<7} | {:<12.2?} | {:<15.2?} | {:<12.2}:1 | {:<12.6} |",
        format!("{}x{}", rows, cols),
        format!("{}x{}", block_size, block_size),
        k_coeffs,
        format!("{:?}", transform_type),
        compress_duration,
        decompress_duration,
        compression_ratio,
        rmse
    );
}

#[test]
fn performance_suite() {
    println!("\n RBE + DCT/DWT/Adaptive 하이브리드 압축 성능 비교 테스트");
    println!("{:-<120}", "");
    println!("| {:<9} | {:<9} | {:<4} | {:<10} | {:<12} | {:<15} | {:<12} | {:<12} |",
        "크기", "블록 크기", "K", "타입", "압축 시간", "복원 시간", "압축률", "RMSE"
    );
    println!("{:-<120}", "");

    let test_configs = [
        (256, 128, 40),
        (512, 128, 50),
        (1024, 128, 100),
    ];

    for &(size, block_size, k) in &test_configs {
        run_performance_test(size, size, block_size, k, TransformType::Dct);
        run_performance_test(size, size, block_size, k, TransformType::Dwt);
        run_performance_test(size, size, block_size, k, TransformType::Adaptive);
        println!("|{:-<118}|", "");
    }
    println!("{:-<120}", "");
}

#[test]
#[ignore]
fn large_matrix_performance_test() {
    println!("\n RBE + DCT/DWT/Adaptive 하이브리드 압축 성능 비교 테스트 (대형 행렬)");
    println!("{:-<120}", "");
    println!("| {:<9} | {:<9} | {:<4} | {:<10} | {:<12} | {:<15} | {:<12} | {:<12} |",
        "크기", "블록 크기", "K", "타입", "압축 시간", "복원 시간", "압축률", "RMSE"
    );
    println!("{:-<120}", "");
    
    run_performance_test(2048, 2048, 128, 100, TransformType::Dct);
    run_performance_test(2048, 2048, 128, 100, TransformType::Dwt);
    run_performance_test(2048, 2048, 128, 100, TransformType::Adaptive);
    
    println!("{:-<120}", "");
} 
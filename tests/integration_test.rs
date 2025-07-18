
/*
#[test]
/// Inverse CORDIC 기반 압축과 복원이 전체적으로 동작하는지 확인하는 통합 테스트입니다.
fn 전체_압축_복원_사이클_테스트() {
    println!("\n--- 전체 통합 테스트: Inverse CORDIC -> 압축 해제 -> RMSE 검증 ---");
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
    
    // 압축률 계산
    let matrix_size_bytes = rows * cols * 4; // f32 = 4 bytes
    let compressed_size_bytes = 16; // 128 bits = 16 bytes
    let compression_ratio = matrix_size_bytes / compressed_size_bytes;
    
    println!("[통합 테스트 결과]");
    println!("  - 원본 행렬 (처음 4개): {:?}", &matrix[0..4]);
    println!("  - 압축 해제 행렬 (처음 4개): {:?}", &compressed.decompress()[0..4]);
    println!("  - 찾은 최적 시드: 0x{:X}", compressed.seed.hi);
    println!("  - RMSE: {}", rmse);
    println!("  - 압축률: {}x{} 행렬({} bytes) -> 128 bits = {}:1", 
             rows, cols, matrix_size_bytes, compression_ratio);
    println!("  - 압축 효율: {:.2}% 크기로 압축", 100.0 / compression_ratio as f32);

    // RMSE가 특정 임계값 이하인지 확인
    assert!(rmse < 1.0, "RMSE should be under 1.0, but was {}", rmse);
} 
*/

#[test]
/// RBE+DCT 하이브리드 압축과 복원이 전체적으로 동작하는지 확인하는 통합 테스트입니다.
fn 하이브리드_압축_복원_사이클_테스트() {
    use poincare_layer::encoder::GridCompressedMatrix;

    println!("\n--- 하이브리드 통합 테스트: 압축 -> 복원 -> RMSE 검증 ---");
    let rows = 32;
    let cols = 32;
    let block_size = 16;
    let k_coeffs = 10;

    // 테스트용 중력 패턴 생성
    let mut matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
            let y = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
            let mut r = (x * x + y * y).sqrt();
            if r < 1e-6 { r = 1e-6; }
            matrix[i * cols + j] = 1.0 / r;
        }
    }
    // 정규화
    let max_val = matrix.iter().fold(0.0f32, |max, &val| val.max(max));
    matrix.iter_mut().for_each(|v| *v /= max_val);
    
    // 압축
    let compressed = GridCompressedMatrix::compress_grid_hybrid(&matrix, rows, cols, block_size, k_coeffs);
    
    // 복원
    let decompressed = compressed.decompress_hybrid();
    
    // RMSE 계산
    let mut mse = 0.0;
    for (original, restored) in matrix.iter().zip(decompressed.iter()) {
        mse += (*original - *restored).powi(2);
    }
    mse /= (rows * cols) as f32;
    let rmse = mse.sqrt();
    
    println!("[하이브리드 테스트 결과]");
    println!("  - 행렬 크기: {}x{}", rows, cols);
    println!("  - 블록 크기: {}x{}", block_size, block_size);
    println!("  - DCT 계수(K): {}", k_coeffs);
    println!("  - 압축률: {:.2}:1", compressed.compression_ratio());
    println!("  - RMSE: {}", rmse);

    assert!(rmse < 0.1, "RMSE should be very low, but was {}", rmse);
} 
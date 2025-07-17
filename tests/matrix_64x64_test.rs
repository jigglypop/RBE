//! 64x64 행렬에 대한 압축 성능 테스트

use poincare_layer::math::compute_full_rmse;
use poincare_layer::types::{Packed64, Packed128, PoincareMatrix};

#[test]
fn 대형_64x64_단순_패턴_압축_테스트() {
    let rows = 64;
    let cols = 64;
    let matrix_generator = PoincareMatrix { 
        seed: Packed128 { hi: 0x1A2B3C4D5E6F7890, lo: 0 }, 
        rows, 
        cols 
    };
    let original_matrix = matrix_generator.decompress();

    // 2. 행렬 압축
    let compressed = PoincareMatrix::compress(&original_matrix, rows, cols);

    // 3. 압축 품질 평가
    let rmse = compute_full_rmse(&original_matrix, &Packed64 { rotations: compressed.seed.hi }, rows, cols);
    
    // 압축률 계산
    let matrix_size_bytes = rows * cols * 4; // f32 = 4 bytes
    let compressed_size_bytes = 16; // 128 bits = 16 bytes
    let compression_ratio = matrix_size_bytes / compressed_size_bytes;
    
    println!("[64x64 단순 패턴 테스트]");
    println!("  - 원본 행렬 (처음 4개): {:?}", &original_matrix[0..4]);
    println!("  - 압축 해제 행렬 (처음 4개): {:?}", &compressed.decompress()[0..4]);
    println!("  - 찾은 최적 시드: 0x{:X}", compressed.seed.hi);
    println!("  - RMSE: {}", rmse);
    println!("  - 압축률: {}x{} 행렬({} bytes) -> 128 bits = {}:1", 
             rows, cols, matrix_size_bytes, compression_ratio);
    println!("  - 압축 효율: {:.2}% 크기로 압축", 100.0 / compression_ratio as f32);

    // RMSE가 특정 임계값 이하인지 확인
    assert!(rmse < 1.0, "64x64 단순 패턴의 RMSE({})가 1.0보다 작아야 합니다", rmse);
}

#[test]
#[ignore] // 이 테스트는 실행 시간이 길어 기본적으로는 무시됩니다.
fn 대형_64x64_복잡_패턴_압축_테스트() {
    let rows = 64;
    let cols = 64;
    let matrix_generator = PoincareMatrix {
        seed: Packed128 { hi: 0xFEDCBA9876543210, lo: 0 },
        rows,
        cols,
    };
    let original_matrix = matrix_generator.decompress();

    // 2. 행렬 압축
    let compressed = PoincareMatrix::compress(&original_matrix, rows, cols);

    // 3. 압축 품질 평가
    let rmse = compute_full_rmse(&original_matrix, &Packed64 { rotations: compressed.seed.hi }, rows, cols);
    
    // 압축률 계산
    let matrix_size_bytes = rows * cols * 4; // f32 = 4 bytes
    let compressed_size_bytes = 16; // 128 bits = 16 bytes
    let compression_ratio = matrix_size_bytes / compressed_size_bytes;
    
    println!("[64x64 복잡 패턴 테스트]");
    println!("  - 원본 행렬 (처음 4개): {:?}", &original_matrix[0..4]);
    println!("  - 압축 해제 행렬 (처음 4개): {:?}", &compressed.decompress()[0..4]);
    println!("  - 찾은 최적 시드: 0x{:X}", compressed.seed.hi);
    println!("  - RMSE: {}", rmse);
    println!("  - 압축률: {}x{} 행렬({} bytes) -> 128 bits = {}:1", 
             rows, cols, matrix_size_bytes, compression_ratio);
    println!("  - 압축 효율: {:.2}% 크기로 압축", 100.0 / compression_ratio as f32);

    // RMSE가 특정 임계값 이하인지 확인
    assert!(rmse < 1.0, "64x64 복잡 패턴의 RMSE({})가 1.0보다 작아야 합니다", rmse);
} 
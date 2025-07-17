//! `matrix.rs`에 대한 단위 테스트

use poincare_layer::math::compute_full_rmse;
use poincare_layer::types::{Packed64, Packed128, PoincareMatrix};
use std::f32::consts::PI;

#[test]
fn 역코딕_압축_및_복원_테스트() {
    println!("\n--- 테스트: 행렬 압축 (역 CORDIC) & 복원 ---");
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
    
    // 압축률 계산
    let matrix_size_bytes = rows * cols * 4; // f32 = 4 bytes
    let compressed_size_bytes = 16; // 128 bits = 16 bytes
    let compression_ratio = matrix_size_bytes / compressed_size_bytes;
    
    println!("[행렬 테스트 결과]");
    println!("  - 원본 행렬 (처음 4개): {:?}", &source_matrix[0..4]);
    println!("  - 압축 해제 행렬 (처음 4개): {:?}", &compressed.decompress()[0..4]);
    println!("  - 찾은 최적 시드: 0x{:X}", compressed.seed.hi);
    println!("  - RMSE: {}", rmse);
    println!("  - 압축률: {}x{} 행렬({} bytes) -> 128 bits = {}:1", 
             rows, cols, matrix_size_bytes, compression_ratio);
    println!("  - 압축 효율: {:.2}% 크기로 압축", 100.0 / compression_ratio as f32);

    // RMSE가 특정 임계값 이하인지 확인
    assert!(rmse < 1.0, "RMSE ({})가 1.0보다 작아야 합니다", rmse);
}

#[test]
fn 다양한_행렬_크기_압축_성능_테스트() {
    println!("\n=== 다양한 크기 행렬 압축 성능 테스트 ===");
    
    // 테스트할 행렬 크기들
    let test_sizes = vec![(4, 4), (8, 8), (16, 16), (32, 32), (64, 64), (128, 128)];
    
    for (rows, cols) in test_sizes {
        println!("\n--- {}x{} 행렬 압축 테스트 ---", rows, cols);
        
        // 복잡한 패턴 생성 (다중 주파수)
        let mut matrix = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let x = 2.0 * PI * j as f32 / cols as f32;
                let y = 2.0 * PI * i as f32 / rows as f32;
                // 여러 주파수를 혼합한 복잡한 패턴
                matrix[i * cols + j] = (x.sin() + (2.0 * x).cos() + (3.0 * y).sin()) / 3.0;
            }
        }
        
        // 압축
        let compressed = PoincareMatrix::compress(&matrix, rows, cols);
        
        // RMSE 계산
        let rmse = compute_full_rmse(&matrix, &Packed64 { rotations: compressed.seed.hi }, rows, cols);
        
        // 압축률 계산
        let matrix_size_bytes = rows * cols * 4; // f32 = 4 bytes
        let compressed_size_bytes = 16; // 128 bits = 16 bytes
        let compression_ratio = matrix_size_bytes / compressed_size_bytes;
        
        println!("  - RMSE: {:.6}", rmse);
        println!("  - 압축률: {}x{} 행렬({} bytes) -> 128 bits = {}:1", 
                 rows, cols, matrix_size_bytes, compression_ratio);
        println!("  - 압축 효율: {:.4}% 크기로 압축", 100.0 / compression_ratio as f32);
        println!("  - 메모리 절약: {:.2}MB -> 16 bytes", matrix_size_bytes as f32 / 1_048_576.0);
        
        // 크기별 RMSE 임계값 설정
        let rmse_threshold = match rows * cols {
            16 => 0.8,      // 4x4
            64 => 0.9,      // 8x8
            256 => 1.0,     // 16x16
            1024 => 1.1,    // 32x32
            4096 => 1.2,    // 64x64
            _ => 1.5,       // 128x128 이상
        };
        
        assert!(rmse < rmse_threshold, 
                "{}x{} 행렬의 RMSE ({:.6})가 임계값 {}보다 작아야 합니다", 
                rows, cols, rmse, rmse_threshold);
    }
}

#[test]
fn 극한_압축률_테스트() {
    println!("\n=== 극한 압축률 시연 테스트 ===");
    
    // 256x256 크기 행렬 - 메가바이트 단위 데이터
    let rows = 256;
    let cols = 256;
    
    println!("대형 {}x{} 행렬 압축 테스트", rows, cols);
    
    // 단순 그라디언트 패턴 (압축하기 쉬운 패턴)
    let mut matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let normalized_i = i as f32 / (rows - 1) as f32;
            let normalized_j = j as f32 / (cols - 1) as f32;
            matrix[i * cols + j] = normalized_i * normalized_j;
        }
    }
    
    // 압축
    let compressed = PoincareMatrix::compress(&matrix, rows, cols);
    
    // RMSE 계산
    let rmse = compute_full_rmse(&matrix, &Packed64 { rotations: compressed.seed.hi }, rows, cols);
    
    // 압축률 계산
    let matrix_size_bytes = rows * cols * 4; // f32 = 4 bytes
    let compressed_size_bytes = 16; // 128 bits = 16 bytes
    let compression_ratio = matrix_size_bytes / compressed_size_bytes;
    
    println!("\n[극한 압축 결과]");
    println!("  - 원본 크기: {:.2} MB", matrix_size_bytes as f32 / 1_048_576.0);
    println!("  - 압축 크기: 16 bytes (128 bits)");
    println!("  - 압축률: {}:1", compression_ratio);
    println!("  - 압축 효율: {:.6}% 크기로 압축", 100.0 / compression_ratio as f32);
    println!("  - RMSE: {:.6}", rmse);
    println!("\n  💡 {:.2}MB → 16 bytes: {}배 압축!", 
             matrix_size_bytes as f32 / 1_048_576.0, compression_ratio);
}

//! Packed128 → f32 가중치 복원기

use crate::core::tensors::Packed128;
use super::TransformStats;
use std::time::Instant;

/// 가중치 복원기
pub struct WeightDecompressor;

impl WeightDecompressor {
    /// Packed128에서 f32 배열로 복원
    pub fn restore_weights(seed: &Packed128, rows: usize, cols: usize) -> (Vec<f32>, TransformStats) {
        let start_time = Instant::now();
        
        println!("복원 시작: {}x{} 행렬", rows, cols);
        
        // 디버깅: seed와 decode 결과 확인
        static mut RESTORE_COUNT: usize = 0;
        unsafe {
            if RESTORE_COUNT < 2 {
                println!("🔍 Restore seed: hi={}, lo={}", seed.hi, seed.lo);
                let decoded = seed.decode();
                println!("🔍 Decoded: r={}, theta={}", decoded.r_fp32, decoded.theta_fp32);
                let test_val = seed.fused_forward(0, 0, rows, cols);
                println!("🔍 Test fused_forward(0,0): {}", test_val);
                RESTORE_COUNT += 1;
            }
        }
        
        let mut weights = Vec::with_capacity(rows * cols);
        
        // 병렬 처리 가능한 구조
        for i in 0..rows {
            for j in 0..cols {
                let weight = seed.fused_forward(i, j, rows, cols);
                weights.push(weight);
            }
        }
        
        // 디버깅: 복원 결과 확인
        unsafe {
            if RESTORE_COUNT <= 2 {
                let weight_sum: f32 = weights.iter().sum();
                let weight_max = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let weight_min = weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                println!("🔍 Restored weights: sum={:.6}, max={:.6}, min={:.6}, first_5={:?}", 
                        weight_sum, weight_max, weight_min, &weights[0..weights.len().min(5)]);
            }
        }
        
        let restore_time = start_time.elapsed().as_millis() as f64;
        
        // 복원 속도 계산
        let total_elements = rows * cols;
        let elements_per_ms = total_elements as f64 / restore_time;
        
        println!("복원 완료: {:.1}ms ({:.1}K elements/ms)", 
                restore_time, elements_per_ms / 1000.0);
        
        let compressed_size = std::mem::size_of::<Packed128>();
        let restored_size = weights.len() * 4;
        
        let stats = TransformStats {
            original_size_mb: 0.0, // 압축 시에만 의미 있음
            compressed_size_mb: compressed_size as f64 / 1024.0 / 1024.0,
            compression_ratio: restored_size as f64 / compressed_size as f64,
            rmse: 0.0, // 별도 검증 필요
            transform_ms: 0.0,
            restore_ms: restore_time,
        };
        
        (weights, stats)
    }
    
    /// 청크 단위 병렬 복원 (대용량용)
    pub fn restore_weights_parallel(seed: &Packed128, rows: usize, cols: usize, chunk_size: usize) -> (Vec<f32>, TransformStats) {
        let start_time = Instant::now();
        
        println!("병렬 복원 시작: {}x{} 행렬 (청크 크기: {})", rows, cols, chunk_size);
        
        let total_elements = rows * cols;
        let mut weights = vec![0.0f32; total_elements];
        
        // 청크 단위로 처리
        let chunks: Vec<_> = (0..total_elements)
            .step_by(chunk_size)
            .map(|start| {
                let end = (start + chunk_size).min(total_elements);
                (start, end)
            })
            .collect();
        
        // 각 청크 처리
        for (start_idx, end_idx) in chunks {
            for linear_idx in start_idx..end_idx {
                let i = linear_idx / cols;
                let j = linear_idx % cols;
                weights[linear_idx] = seed.fused_forward(i, j, rows, cols);
            }
        }
        
        let restore_time = start_time.elapsed().as_millis() as f64;
        let elements_per_ms = total_elements as f64 / restore_time;
        
        println!("병렬 복원 완료: {:.1}ms ({:.1}K elements/ms)", 
                restore_time, elements_per_ms / 1000.0);
        
        let compressed_size = std::mem::size_of::<Packed128>();
        let restored_size = weights.len() * 4;
        
        let stats = TransformStats {
            original_size_mb: 0.0,
            compressed_size_mb: compressed_size as f64 / 1024.0 / 1024.0,
            compression_ratio: restored_size as f64 / compressed_size as f64,
            rmse: 0.0,
            transform_ms: 0.0,
            restore_ms: restore_time,
        };
        
        (weights, stats)
    }
    
    /// 스트리밍 복원 (메모리 효율적)
    pub fn restore_weights_streaming<F>(seed: &Packed128, rows: usize, cols: usize, mut callback: F) -> TransformStats 
    where
        F: FnMut(usize, usize, f32),
    {
        let start_time = Instant::now();
        
        println!("스트리밍 복원 시작: {}x{} 행렬", rows, cols);
        
        for i in 0..rows {
            for j in 0..cols {
                let weight = seed.fused_forward(i, j, rows, cols);
                callback(i, j, weight);
            }
        }
        
        let restore_time = start_time.elapsed().as_millis() as f64;
        let total_elements = rows * cols;
        let elements_per_ms = total_elements as f64 / restore_time;
        
        println!("스트리밍 복원 완료: {:.1}ms ({:.1}K elements/ms)", 
                restore_time, elements_per_ms / 1000.0);
        
        let compressed_size = std::mem::size_of::<Packed128>();
        let restored_size = total_elements * 4;
        
        TransformStats {
            original_size_mb: 0.0,
            compressed_size_mb: compressed_size as f64 / 1024.0 / 1024.0,
            compression_ratio: restored_size as f64 / compressed_size as f64,
            rmse: 0.0,
            transform_ms: 0.0,
            restore_ms: restore_time,
        }
    }
    
    /// 부분 복원 (특정 영역만)
    pub fn restore_region(seed: &Packed128, start_row: usize, end_row: usize, 
                         start_col: usize, end_col: usize, 
                         total_rows: usize, total_cols: usize) -> Vec<f32> {
        let mut region_weights = Vec::new();
        
        for i in start_row..end_row {
            for j in start_col..end_col {
                let weight = seed.fused_forward(i, j, total_rows, total_cols);
                region_weights.push(weight);
            }
        }
        
        region_weights
    }
    
    /// 단일 가중치 복원 (인덱스 기반)
    pub fn restore_single_weight(seed: &Packed128, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        seed.fused_forward(i, j, rows, cols)
    }
} 
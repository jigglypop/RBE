//! 그리드로 압축된 행렬 처리

use crate::packed_params::{HybridEncodedBlock, TransformType};
use super::hybrid_encoder::HybridEncoder;
use std::sync::Mutex;
use rayon::prelude::*;

/// 그리드로 압축된 행렬
pub struct GridCompressedMatrix {
    pub blocks: Vec<HybridEncodedBlock>,
    pub grid_rows: usize,
    pub grid_cols: usize,
    pub block_size: usize,
    pub total_rows: usize,
    pub total_cols: usize,
}

impl GridCompressedMatrix {
    /// 그리드 기반 하이브리드 압축 (병렬, 캐싱)
    pub fn compress_grid_hybrid(
        matrix: &[f32],
        rows: usize,
        cols: usize,
        block_size: usize,
        k_coeffs: usize,
        transform_type: TransformType,
    ) -> Self {
        let grid_rows = (rows + block_size - 1) / block_size;
        let grid_cols = (cols + block_size - 1) / block_size;
        
        // 스레드별 인코더를 위한 Mutex
        let encoder = Mutex::new(HybridEncoder::new(k_coeffs, transform_type));

        let blocks: Vec<HybridEncodedBlock> = (0..grid_rows * grid_cols)
            .into_par_iter()
            .map(|block_idx| {
                let grid_i = block_idx / grid_cols;
                let grid_j = block_idx % grid_cols;
                
                let start_i = grid_i * block_size;
                let start_j = grid_j * block_size;
                let end_i = (start_i + block_size).min(rows);
                let end_j = (start_j + block_size).min(cols);
                
                let block_rows = end_i - start_i;
                let block_cols = end_j - start_j;
                
                let mut block_data = Vec::with_capacity(block_rows * block_cols);
                for i in start_i..end_i {
                    for j in start_j..end_j {
                        block_data.push(matrix[i * cols + j]);
                    }
                }
                
                // 각 스레드가 Mutex를 lock하고 인코더를 사용하여 블록 압축
                let mut encoder_guard = encoder.lock().unwrap();
                encoder_guard.encode_block(&block_data, block_rows, block_cols)
            })
            .collect();

        Self {
            blocks,
            grid_rows,
            grid_cols,
            block_size,
            total_rows: rows,
            total_cols: cols,
        }
    }

    /// 압축률 계산
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.total_rows * self.total_cols * 4; // f32 = 4 bytes
        let compressed_size: usize = self.blocks.iter().map(|b| {
            8 * 4 + // rbe_params: 8 * f32
            b.residuals.len() * (2 * 2 + 4) // residuals: N * (u16, u16, f32)
        }).sum();
        original_size as f32 / compressed_size as f32
    }
} 
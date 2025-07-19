//! 그리드 압축 매트릭스 디코딩 기능

use crate::encoder::GridCompressedMatrix;
use rayon::prelude::*;

impl GridCompressedMatrix {
    /// 그리드 압축된 행렬을 전체 복원 (병렬)
    pub fn decompress_hybrid(&self) -> Vec<f32> {
        let mut matrix = vec![0.0; self.total_rows * self.total_cols];

        // 디코딩된 블록들을 저장할 벡터
        let decoded_blocks: Vec<(usize, Vec<f32>)> = self.blocks.par_iter().enumerate().map(|(block_idx, block)| {
            (block_idx, block.decode())
        }).collect();

        // 순서를 보장하며 전체 행렬에 복사
        for (block_idx, block_data) in decoded_blocks {
            let grid_i = block_idx / self.grid_cols;
            let grid_j = block_idx % self.grid_cols;
            
            let start_i = grid_i * self.block_size;
            let start_j = grid_j * self.block_size;
            
            let block_rows = self.blocks[block_idx].rows;
            let block_cols = self.blocks[block_idx].cols;

            for bi in 0..block_rows {
                for bj in 0..block_cols {
                    let global_i = start_i + bi;
                    let global_j = start_j + bj;
                    if global_i < self.total_rows && global_j < self.total_cols {
                        matrix[global_i * self.total_cols + global_j] = block_data[bi * block_cols + bj];
                    }
                }
            }
        }
        matrix
    }
} 
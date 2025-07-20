//! 하이브리드 블록 디코딩 기능

use crate::packed_params::{HybridEncodedBlock, TransformType};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array, Array2};
use rustdct::DctPlanner;
use omni_wave::{wavelet as w, completely_reconstruct_2d};
use rayon::prelude::*;

impl HybridEncodedBlock {
    /// 하이브리드 압축 블록을 디코딩하여 원본 데이터 벡터를 반환합니다.
    pub fn decode(&self) -> Vec<f32> {
        let rows = self.rows;
        let cols = self.cols;

        // 1. RBE 기본 패턴 복원
        let mut a_matrix = DMatrix::from_element(rows * cols, 8, 0.0);
        for r in 0..rows {
            for c in 0..cols {
                let x = if cols > 1 { (c as f32 / (cols - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
                let y = if rows > 1 { (r as f32 / (rows - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
                let d = (x * x + y * y).sqrt();
                let pi = std::f32::consts::PI;

                let basis_row = [
                    1.0, d, d * d, (pi * x).cos(), (pi * y).cos(),
                    (2.0 * pi * x).cos(), (2.0 * pi * y).cos(), (pi * x).cos() * (pi * y).cos(),
                ];
                let matrix_row_index = r * cols + c;
                a_matrix.row_mut(matrix_row_index).copy_from_slice(&basis_row);
            }
        }
        let rbe_params_vec = DVector::from_row_slice(&self.rbe_params);
        let rbe_pattern_vec = a_matrix * rbe_params_vec;
        
        // 2. 잔차 행렬 복원 (IDCT 또는 IDWT)
        let mut coeffs_matrix = Array2::<f32>::zeros((rows, cols));
        for coeff in &self.residuals {
            let row_idx = coeff.index.0 as usize;
            let col_idx = coeff.index.1 as usize;
            if row_idx < rows && col_idx < cols {
                coeffs_matrix[(row_idx, col_idx)] = coeff.value;
            }
        }
        
        let residual_vec: Vec<f32> = match self.transform_type {
            TransformType::Dct => {
                let mut dct_planner = DctPlanner::<f32>::new();
                let idct = dct_planner.plan_dct2(rows * cols);
                let mut buffer = coeffs_matrix.into_raw_vec();
                idct.process_dct2(&mut buffer);
                // DMatrix::from_vec(rows, cols, buffer) / ((2 * rows * 2 * cols) as f32)
                buffer.iter_mut().for_each(|x| *x /= (4 * rows * cols) as f32);
                buffer
            },
            TransformType::Dwt => {
                let wavelet = w::BIOR_3_1; // 인코딩과 동일한 웨이블릿
                let mut buffer = ndarray::Array1::zeros(rows.max(cols) + wavelet.window_size() - 2);
                completely_reconstruct_2d(coeffs_matrix.view_mut(), buffer.view_mut(), wavelet);
                // DMatrix::from_iterator(rows, cols, coeffs_matrix.into_raw_vec().into_iter())
                coeffs_matrix.into_raw_vec()
            },
            TransformType::Adaptive => unreachable!("Decoder cannot receive Adaptive type"),
        };
        
        // 3. 최종 행렬 복원
        let final_vec: Vec<f32> = rbe_pattern_vec.iter()
            .zip(residual_vec.iter())
            .map(|(rbe_val, residual_val)| rbe_val + residual_val)
            .collect();
            
        final_vec
    }
}


/// 여러 개의 압축된 블록을 받아 전체 원본 데이터를 복원합니다.
pub fn decode_all_blocks(
    encoded_blocks: &[HybridEncodedBlock],
    height: usize,
    width: usize,
    block_size: usize,
) -> Vec<f32> {
    let mut reconstructed_data = vec![0.0f32; height * width];
    let blocks_per_width = (width + block_size - 1) / block_size;

    let results: Vec<(usize, Vec<f32>)> = encoded_blocks.par_iter().enumerate().map(|(block_idx, encoded_block)| {
        (block_idx, encoded_block.decode())
    }).collect();

    for (block_idx, decoded_block) in results {
        let block_i = block_idx / blocks_per_width;
        let block_j = block_idx % blocks_per_width;
        let start_i = block_i * block_size;
        let start_j = block_j * block_size;

        for i in 0..block_size {
            for j in 0..block_size {
                let global_i = start_i + i;
                let global_j = start_j + j;
                if global_i < height && global_j < width {
                    let decoded_idx = i * block_size + j;
                    if decoded_idx < decoded_block.len() {
                        reconstructed_data[global_i * width + global_j] = decoded_block[decoded_idx];
                    }
                }
            }
        }
    }

    reconstructed_data
} 
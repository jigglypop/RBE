//! 행렬 압축 해제(디코딩) 관련 기능
use crate::encoder::GridCompressedMatrix;
use crate::types::{HybridEncodedBlock, TransformType};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array, Array1, Array2};
use rayon::prelude::*;
use rustdct::DctPlanner;
use omni_wave::{wavelet as w, completely_reconstruct_2d};


impl HybridEncodedBlock {
    /// 하이브리드 압축 블록을 디코딩
    pub fn decode(&self) -> Vec<f32> {
        let rows = self.rows;
        let cols = self.cols;

        // --- 1. RBE 기본 패턴 복원 ---
        let mut a_matrix = DMatrix::from_element(rows * cols, 8, 0.0);
        for r in 0..rows {
            for c in 0..cols {
                let x = (c as f32 / (cols.saturating_sub(1)) as f32) * 2.0 - 1.0;
                let y = (r as f32 / (rows.saturating_sub(1)) as f32) * 2.0 - 1.0;
                let d = (x * x + y * y).sqrt();
                let pi = std::f32::consts::PI;

                let basis_row = [
                    1.0, d, d * d, (pi * x).cos(), (pi * y).cos(),
                    (2.0 * pi * x).cos(), (2.0 * pi * y).cos(), (pi * x).cos() * (pi * y).cos(),
                ];
                let matrix_row_index = r * cols + c;
                for i in 0..8 {
                    a_matrix[(matrix_row_index, i)] = basis_row[i];
                }
            }
        }
        let rbe_params_vec = DVector::from_row_slice(&self.rbe_params);
        let rbe_pattern_vec = a_matrix * rbe_params_vec;
        let rbe_pattern_matrix = DMatrix::from_vec(rows, cols, rbe_pattern_vec.data.into());


        // --- 2. 잔차 행렬 복원 (IDCT 또는 IDWT) ---
        let mut coeffs_matrix = Array2::<f32>::zeros((rows, cols));
        for coeff in &self.residuals {
            coeffs_matrix[(coeff.index.0 as usize, coeff.index.1 as usize)] = coeff.value;
        }
        
        match self.transform_type {
            TransformType::Dct => {
                let mut dct_planner = DctPlanner::<f32>::new();
                let idct_row = dct_planner.plan_dct3(cols);
                let idct_col = dct_planner.plan_dct3(rows);

                // 열에 대해 IDCT
                let mut transposed = coeffs_matrix.t().to_owned();
                for mut col in transposed.rows_mut() {
                    let mut col_vec = col.to_vec();
                    idct_row.process_dct3(&mut col_vec);
                    col.assign(&Array::from(col_vec));
                }
                
                // 행에 대해 IDCT
                let mut dct_matrix = transposed.t().to_owned();
                for mut row in dct_matrix.rows_mut() {
                    let mut row_vec = row.to_vec();
                    idct_col.process_dct3(&mut row_vec);
                    row.assign(&Array::from(row_vec));
                }

                // 정규화
                let normalization_factor = (2.0 * cols as f32) * (2.0 * rows as f32);
                coeffs_matrix = dct_matrix / normalization_factor;
            },
            TransformType::Dwt => {
                let wavelet = w::BIOR_3_1; // 인코딩과 동일한 웨이블릿 사용
                let mut buffer = Array1::zeros(rows.max(cols) + wavelet.window_size() - 2);
                completely_reconstruct_2d(coeffs_matrix.view_mut(), buffer.view_mut(), wavelet);
            },
            TransformType::Adaptive => {
                // 이 브랜치는 디코딩 시에 도달할 수 없습니다.
                // Adaptive는 인코딩 시에만 사용되는 로직입니다.
                unreachable!("Decoder should not receive an Adaptive transform type directly.");
            }
        }
        
        let residual_matrix_nalgebra = DMatrix::from_iterator(rows, cols, coeffs_matrix.into_raw_vec());
        
        // --- 3. 최종 행렬 복원 ---
        let final_matrix = rbe_pattern_matrix + residual_matrix_nalgebra;
        final_matrix.data.as_vec().clone()
    }
}

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
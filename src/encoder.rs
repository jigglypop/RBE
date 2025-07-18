//! 행렬 압축(인코딩) 관련 기능
use crate::types::{HybridEncodedBlock, ResidualCoefficient, TransformType};
use nalgebra::{DMatrix, DVector};
use rustdct::DctPlanner;
use std::sync::Mutex;
use rayon::prelude::*;
use ndarray::{Array, Array1, Array2};
use omni_wave::{wavelet as w, completely_decompose_2d};

/// RBE + DCT/Wavelet 하이브리드 인코더
pub struct HybridEncoder {
    pub k_coeffs: usize, // 유지할 잔차 계수의 개수
    pub transform_type: TransformType,
    // planner는 재사용 가능하므로 인코더가 소유하는 것이 효율적
    dct_planner_f32: DctPlanner<f32>,
}

impl HybridEncoder {
    pub fn new(k_coeffs: usize, transform_type: TransformType) -> Self {
        Self {
            k_coeffs,
            transform_type,
            dct_planner_f32: DctPlanner::new(),
        }
    }

    fn encode_single_transform(
        &mut self,
        rbe_params: [f32; 8],
        residual_vector: &DVector<f32>,
        rows: usize,
        cols: usize,
        transform_type: TransformType,
    ) -> HybridEncodedBlock {
        let mut residual_matrix = Array2::from_shape_vec((rows, cols), residual_vector.iter().cloned().collect()).unwrap();

        match transform_type {
            TransformType::Dct => {
                let dct_row = self.dct_planner_f32.plan_dct2(cols);
                let dct_col = self.dct_planner_f32.plan_dct2(rows);
                for mut row in residual_matrix.rows_mut() {
                    let mut row_vec = row.to_vec();
                    dct_row.process_dct2(&mut row_vec);
                    row.assign(&Array::from(row_vec));
                }
                let mut transposed = residual_matrix.t().to_owned();
                for mut col in transposed.rows_mut() {
                    let mut col_vec = col.to_vec();
                    dct_col.process_dct2(&mut col_vec);
                    col.assign(&Array::from(col_vec));
                }
                residual_matrix = transposed.t().to_owned();
            },
            TransformType::Dwt => {
                let wavelet = w::BIOR_3_1;
                let mut buffer = Array1::zeros(rows.max(cols) + wavelet.window_size() - 2);
                completely_decompose_2d(residual_matrix.view_mut(), buffer.view_mut(), wavelet);
            },
            TransformType::Adaptive => unreachable!(),
        }

        let mut coeffs: Vec<ResidualCoefficient> = residual_matrix
            .indexed_iter()
            .map(|((r, c), &val)| ResidualCoefficient {
                index: (r as u16, c as u16),
                value: val,
            })
            .collect();
        coeffs.sort_unstable_by(|a, b| b.value.abs().partial_cmp(&a.value.abs()).unwrap());
        let top_k_coeffs = coeffs.into_iter().take(self.k_coeffs).collect();

        HybridEncodedBlock {
            rbe_params,
            residuals: top_k_coeffs,
            rows,
            cols,
            transform_type,
        }
    }

    /// 단일 블록을 RBE+DCT/DWT로 압축
    pub fn encode_block(&mut self, block_data: &[f32], rows: usize, cols: usize) -> HybridEncodedBlock {
        // --- 1. RBE 파라미터 피팅 ---
        let mut a_matrix = DMatrix::from_element(rows * cols, 8, 0.0);
        let b_vector = DVector::from_row_slice(block_data);
        for r in 0..rows {
            for c in 0..cols {
                let x = (c as f32 / (cols.saturating_sub(1)) as f32) * 2.0 - 1.0;
                let y = (r as f32 / (rows.saturating_sub(1)) as f32) * 2.0 - 1.0;
                let d = (x * x + y * y).sqrt();
                let pi = std::f32::consts::PI;
                let basis_row = [
                    1.0, d, d * d, (pi * x).cos(), (pi * y).cos(),
                    (2.0 * pi * x).cos(), (2.0 * pi * y).cos(),
                    (pi * x).cos() * (pi * y).cos(),
                ];
                let matrix_row_index = r * cols + c;
                for i in 0..8 { a_matrix[(matrix_row_index, i)] = basis_row[i]; }
            }
        }
        
        let svd = a_matrix.clone().svd(true, true);
        let rbe_params_vec = svd.solve(&b_vector, 1e-6).expect("SVD solve failed");
        let rbe_params: [f32; 8] = core::array::from_fn(|i| rbe_params_vec[i]);
        let pred_vector = a_matrix * &rbe_params_vec;
        let residual_vector = &b_vector - pred_vector;

        // --- 2. 변환 적용 ---
        if self.transform_type == TransformType::Adaptive {
            let dct_block = self.encode_single_transform(rbe_params, &residual_vector, rows, cols, TransformType::Dct);
            let dwt_block = self.encode_single_transform(rbe_params, &residual_vector, rows, cols, TransformType::Dwt);

            let dct_decoded = dct_block.decode();
            let dwt_decoded = dwt_block.decode();

            let dct_mse = block_data.iter().zip(dct_decoded.iter()).map(|(o, d)| (*o - *d).powi(2)).sum::<f32>() / (rows * cols) as f32;
            let dwt_mse = block_data.iter().zip(dwt_decoded.iter()).map(|(o, d)| (*o - *d).powi(2)).sum::<f32>() / (rows * cols) as f32;

            if dct_mse <= dwt_mse { dct_block } else { dwt_block }
        } else {
            self.encode_single_transform(rbe_params, &residual_vector, rows, cols, self.transform_type)
        }
    }
}


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
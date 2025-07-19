//! RBE + DCT/Wavelet 하이브리드 인코더

use crate::packed_params::{HybridEncodedBlock, ResidualCoefficient, TransformType};
use nalgebra::{DMatrix, DVector};
use rustdct::DctPlanner;
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
        // 입력 데이터 크기 검증
        if block_data.len() != rows * cols {
            panic!("block_data 길이({})가 rows * cols({})와 일치하지 않음", block_data.len(), rows * cols);
        }
        
        // --- 1. RBE 파라미터 피팅 ---
        let mut a_matrix = DMatrix::from_element(rows * cols, 8, 0.0);
        let b_vector = DVector::from_row_slice(block_data);
        
        for r in 0..rows {
            for c in 0..cols {
                let x = if cols > 1 { (c as f32 / (cols - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
                let y = if rows > 1 { (r as f32 / (rows - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
                let d = (x * x + y * y).sqrt();
                let pi = std::f32::consts::PI;
                let basis_row = [
                    1.0, d, d * d, (pi * x).cos(), (pi * y).cos(),
                    (2.0 * pi * x).cos(), (2.0 * pi * y).cos(),
                    (pi * x).cos() * (pi * y).cos(),
                ];
                let matrix_row_index = r * cols + c;
                for i in 0..8 { 
                    a_matrix[(matrix_row_index, i)] = basis_row[i]; 
                }
            }
        }

        // Pseudo-inverse via SVD with proper dimension handling
        let a_matrix_clone = a_matrix.clone();
        let svd = a_matrix_clone.svd(true, true);
        let rbe_params_dv = if let (Some(u), Some(vt)) = (svd.u, svd.v_t) {
            let singular_values = svd.singular_values;
            let min_dim = singular_values.len().min(8).min(rows * cols);
            
            // Construct diagonal pseudo-inverse matrix
            let mut sigma_inv = DMatrix::zeros(min_dim, min_dim);
            for i in 0..min_dim {
                if singular_values[i] > 1e-6 { 
                    sigma_inv[(i, i)] = 1.0 / singular_values[i]; 
                }
            }
            
            // Proper pseudo-inverse calculation: V * Σ⁻¹ * Uᵀ * b
            let u_subset = u.columns(0, min_dim);
            let vt_subset = vt.rows(0, min_dim);
            
            vt_subset.transpose() * sigma_inv * u_subset.transpose() * &b_vector
        } else {
            DVector::zeros(8) // 백업
        };

        let rbe_params: [f32; 8] = [
            rbe_params_dv.get(0).copied().unwrap_or(0.0),
            rbe_params_dv.get(1).copied().unwrap_or(0.0),
            rbe_params_dv.get(2).copied().unwrap_or(0.0),
            rbe_params_dv.get(3).copied().unwrap_or(0.0),
            rbe_params_dv.get(4).copied().unwrap_or(0.0),
            rbe_params_dv.get(5).copied().unwrap_or(0.0),
            rbe_params_dv.get(6).copied().unwrap_or(0.0),
            rbe_params_dv.get(7).copied().unwrap_or(0.0),
        ];

        // --- 2. 잔차 계산 ---
        let rbe_reconstruction = a_matrix * rbe_params_dv;
        let residual_vector = &b_vector - &rbe_reconstruction;

        match self.transform_type {
            TransformType::Adaptive => {
                // DCT와 DWT 모두 시도하여 더 압축률이 좋은 것 선택
                let dct_block = self.encode_single_transform(rbe_params, &residual_vector, rows, cols, TransformType::Dct);
                let dwt_block = self.encode_single_transform(rbe_params, &residual_vector, rows, cols, TransformType::Dwt);
                
                if dct_block.residuals.len() <= dwt_block.residuals.len() {
                    dct_block
                } else {
                    dwt_block
                }
            },
            _ => self.encode_single_transform(rbe_params, &residual_vector, rows, cols, self.transform_type),
        }
    }
} 
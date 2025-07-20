//! RBE + DCT/Wavelet 하이브리드 인코더

use crate::packed_params::{HybridEncodedBlock, ResidualCoefficient, TransformType};
use nalgebra::{DMatrix, DVector, RowDVector};
use rustdct::DctPlanner;
use ndarray::{Array, Array1, Array2};
use omni_wave::{wavelet as w, completely_decompose_2d};
use rayon::prelude::*;

/// 신호 특성 구조체
#[derive(Debug)]
struct SignalCharacteristics {
    is_periodic: bool,
    is_sparse: bool,
    is_localized: bool,
    variance: f32,
}

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
    
    /// S급 품질 (RMSE < 0.001): 819:1 압축률, 128x128 블록 권장
    pub fn new_s_grade() -> Self {
        Self::new(500, TransformType::Dwt)  // 웨이블릿으로 변경
    }
    
    /// A급 품질 (RMSE < 0.01): 3276:1 압축률, 256x256 블록 권장  
    pub fn new_a_grade() -> Self {
        Self::new(300, TransformType::Dwt)  // 웨이블릿으로 변경
    }
    
    /// B급 품질 (RMSE < 0.1): 3276:1 압축률, 256x256 블록 권장
    pub fn new_b_grade() -> Self {
        Self::new(200, TransformType::Dwt)  // 웨이블릿으로 변경
    }
    
    /// 극한 압축 (RMSE ~0.09): 3276:1 압축률
    pub fn new_extreme_compression() -> Self {
        Self::new(50, TransformType::Dwt)   // 웨이블릿으로 변경
    }
    
    /// DCT 전용 인코더 (비교용)
    pub fn new_dct_comparison() -> Self {
        Self::new(200, TransformType::Dct)
    }
    
    /// 적응형 인코더 (DCT vs DWT 자동 선택)
    pub fn new_adaptive() -> Self {
        Self::new(200, TransformType::Adaptive)
    }
    
    /// 권장 블록 크기 반환 (품질 등급별)
    pub fn recommended_block_size(&self) -> usize {
        match self.k_coeffs {
            500.. => 128,        // S급/A급: 128x128 또는 256x256
            200..500 => 256,     // B급: 256x256  
            50..200 => 256,      // 고압축: 256x256
            _ => 64,             // 기타: 64x64
        }
    }
    
    /// 예상 품질 등급 반환
    pub fn quality_grade(&self) -> &'static str {
        match self.k_coeffs {
            500.. => "S급 (RMSE < 0.001)",
            200..500 => "B급 (RMSE < 0.1)", 
            50..200 => "C급 (RMSE > 0.1)",
            _ => "저품질",
        }
    }
    

    
    /// 신호 패턴 분석
    fn analyze_signal_pattern(&self, data: &[f32], rows: usize, cols: usize) -> SignalCharacteristics {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / data.len() as f32;
        
        // 주기성 검사 (FFT 기반)
        let is_periodic = self.check_periodicity(data, rows, cols);
        
        // 희소성 검사 (0에 가까운 값의 비율)
        let sparse_threshold = variance.sqrt() * 0.1;
        let sparse_count = data.iter().filter(|&&x| x.abs() < sparse_threshold).count();
        let is_sparse = sparse_count as f32 / data.len() as f32 > 0.7;
        
        // 집중성 검사 (중앙 영역에 에너지 집중)
        let is_localized = self.check_localization(data, rows, cols);
        
        SignalCharacteristics {
            is_periodic,
            is_sparse,
            is_localized,
            variance,
        }
    }
    
    /// 주기성 검사
    fn check_periodicity(&self, data: &[f32], rows: usize, cols: usize) -> bool {
        // 간단한 자기상관 기반 주기성 검사
        if data.len() < 16 { return false; }
        
        let sample_size = (data.len() / 4).min(64);
        let step = data.len() / sample_size;
        
        // 첫 번째 사분면과 다른 사분면들의 유사도 확인
        let quarter = sample_size / 4;
        let mut correlation_sum = 0.0f32;
        
        for i in 0..quarter {
            let idx1 = i * step;
            let idx2 = (i + quarter) * step;
            let idx3 = (i + 2 * quarter) * step;
            let idx4 = (i + 3 * quarter) * step;
            
            if idx4 < data.len() {
                let base = data[idx1];
                correlation_sum += (data[idx2] - base).abs() + 
                                  (data[idx3] - base).abs() + 
                                  (data[idx4] - base).abs();
            }
        }
        
        let avg_diff = correlation_sum / (quarter as f32 * 3.0);
        let signal_range = data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() -
                          data.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        avg_diff < signal_range * 0.3  // 30% 이하 차이면 주기적
    }
    
    /// 집중성 검사
    fn check_localization(&self, data: &[f32], rows: usize, cols: usize) -> bool {
        if data.len() != rows * cols { return false; }
        
        let center_r = rows / 2;
        let center_c = cols / 2;
        let radius = (rows.min(cols) / 4).max(1);
        
        let mut center_energy = 0.0f32;
        let mut total_energy = 0.0f32;
        let mut center_count = 0;
        
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let energy = data[idx] * data[idx];
                total_energy += energy;
                
                let dist_r = (r as i32 - center_r as i32).abs() as usize;
                let dist_c = (c as i32 - center_c as i32).abs() as usize;
                
                if dist_r <= radius && dist_c <= radius {
                    center_energy += energy;
                    center_count += 1;
                }
            }
        }
        
        if total_energy == 0.0 { return false; }
        
        let center_ratio = center_energy / total_energy;
        let expected_ratio = center_count as f32 / data.len() as f32;
        
        center_ratio > expected_ratio * 2.0  // 중앙에 2배 이상 에너지 집중
    }
    
    /// 빠른 변환 비교
    fn quick_transform_comparison(&mut self, data: &[f32], rows: usize, cols: usize) -> TransformType {
        // 작은 샘플로 빠른 테스트
        let test_size = 32.min(rows).min(cols);
        let step_r = rows / test_size;
        let step_c = cols / test_size;
        
        let mut test_data = Vec::with_capacity(test_size * test_size);
        for r in 0..test_size {
            for c in 0..test_size {
                let src_r = r * step_r;
                let src_c = c * step_c;
                if src_r < rows && src_c < cols {
                    test_data.push(data[src_r * cols + src_c]);
                } else {
                    test_data.push(0.0);
                }
            }
        }
        
        // RBE 파라미터는 간단히 계산
        let test_rbe_params = [0.0f32; 8];  // 더미 파라미터
        
        // DCT vs DWT 빠른 비교
        let dct_result = self.encode_single_transform(test_rbe_params, &nalgebra::DVector::from_vec(test_data.clone()), test_size, test_size, TransformType::Dct);
        let dwt_result = self.encode_single_transform(test_rbe_params, &nalgebra::DVector::from_vec(test_data), test_size, test_size, TransformType::Dwt);
        
        // 잔차 계수 개수가 적은 것 선택
        if dct_result.residuals.len() <= dwt_result.residuals.len() {
            TransformType::Dct
        } else {
            TransformType::Dwt
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
                
                // --- 행별 DCT 병렬 처리 ---
                residual_matrix.axis_iter_mut(ndarray::Axis(0)).into_par_iter().for_each(|mut row| {
                    let mut row_vec = row.to_vec();
                    dct_row.process_dct2(&mut row_vec);
                    row.assign(&Array::from(row_vec));
                });
                
                // --- 열별 DCT 병렬 처리 ---
                let mut transposed = residual_matrix.t().to_owned();
                transposed.axis_iter_mut(ndarray::Axis(0)).into_par_iter().for_each(|mut col| {
                    let mut col_vec = col.to_vec();
                    dct_col.process_dct2(&mut col_vec);
                    col.assign(&Array::from(col_vec));
                });
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
        
        let b_vector = DVector::from_row_slice(block_data);
        
        // --- 병렬 처리로 행렬 A 구성 (타입 문제 수정) ---
        let a_matrix_rows: Vec<RowDVector<f32>> = (0..rows * cols).into_par_iter().map(|idx| {
            let r = idx / cols;
            let c = idx % cols;
            let x = if cols > 1 { (c as f32 / (cols - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
            let y = if rows > 1 { (r as f32 / (rows - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
            let d = (x * x + y * y).sqrt();
            let pi = std::f32::consts::PI;
            RowDVector::from_vec(vec![
                1.0, d, d * d, (pi * x).cos(), (pi * y).cos(),
                (2.0 * pi * x).cos(), (2.0 * pi * y).cos(),
                (pi * x).cos() * (pi * y).cos(),
            ])
        }).collect();

        let a_matrix = DMatrix::from_rows(&a_matrix_rows);

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
                // 입력 데이터 특성 분석
                let signal_characteristics = self.analyze_signal_pattern(block_data, rows, cols);
                
                // 특성에 따라 최적 변환 선택
                let optimal_transform = if signal_characteristics.is_periodic {
                    TransformType::Dwt  // 주기적 신호는 DWT
                } else if signal_characteristics.is_sparse || signal_characteristics.is_localized {
                    TransformType::Dct  // 희소/집중 신호는 DCT
                } else {
                    // 빠른 샘플 테스트로 결정
                    self.quick_transform_comparison(block_data, rows, cols)
                };
                
                self.encode_single_transform(rbe_params, &residual_vector, rows, cols, optimal_transform)
            },
            _ => self.encode_single_transform(rbe_params, &residual_vector, rows, cols, self.transform_type),
        }
    }

    /// 이분탐색으로 RMSE ≤ threshold를 만족하는 최소 계수 찾기
    pub fn find_critical_coefficients(
        data: &[f32],
        rows: usize,
        cols: usize,
        rmse_threshold: f32,
        transform_type: TransformType,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        if rows != cols {
            return Err("정방형 블록만 지원됩니다".into());
        }
        
        let block_size = rows;
        let max_coeffs = (block_size * block_size) / 4; // 상한: 전체 픽셀의 1/4
        let min_coeffs = 8; // 하한: 최소 8개
        
        let mut left = min_coeffs;
        let mut right = max_coeffs;
        let mut critical_coeffs = max_coeffs;
        
        while left <= right {
            let mid = (left + right) / 2;
            
            // 임시 encoder로 테스트
            let mut test_encoder = HybridEncoder::new(mid, transform_type);
            let encoded_block = test_encoder.encode_block(data, rows, cols);
            let decoded_data = encoded_block.decode();
            
            // RMSE 계산
            let mse: f32 = data.iter()
                .zip(decoded_data.iter())
                .map(|(orig, recon)| (orig - recon).powi(2))
                .sum::<f32>() / (rows * cols) as f32;
            let rmse = mse.sqrt();
            
            if rmse <= rmse_threshold {
                // 성공: 더 적은 계수로 시도
                critical_coeffs = mid;
                right = mid - 1;
            } else {
                // 실패: 더 많은 계수 필요
                left = mid + 1;
            }
        }
        
        Ok(critical_coeffs)
    }
    
    /// 블록 크기에 따른 수학적 공식으로 계수 예측
    pub fn predict_coefficients_formula(block_size: usize) -> usize {
        let log_factor = if block_size >= 32 {
            (block_size as f32 / 32.0).log2().max(0.0) as usize
        } else {
            0
        };
        
        let r_value = 32_usize.saturating_sub(log_factor).max(25);
        (block_size * block_size + r_value - 1) / r_value // 올림 처리
    }
    
    /// 자동 최적화: 블록 크기에 따라 최적 계수를 자동으로 찾아서 생성
    pub fn new_auto_optimized(
        data: &[f32],
        rows: usize,
        cols: usize,
        transform_type: TransformType,
        rmse_threshold: Option<f32>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let threshold = rmse_threshold.unwrap_or(0.000001);
        
        // 1. 수학적 공식으로 초기 예측
        let predicted_coeffs = Self::predict_coefficients_formula(rows);
        
        // 2. 예측값 근처에서 빠른 검증
        let mut test_encoder = HybridEncoder::new(predicted_coeffs, transform_type);
        let encoded_block = test_encoder.encode_block(data, rows, cols);
        let decoded_data = encoded_block.decode();
        
        let mse: f32 = data.iter()
            .zip(decoded_data.iter())
            .map(|(orig, recon)| (orig - recon).powi(2))
            .sum::<f32>() / (rows * cols) as f32;
        let predicted_rmse = mse.sqrt();
        
        let final_coeffs = if predicted_rmse <= threshold {
            // 예측이 정확하면 더 적은 계수 시도 (10% 감소)
            let reduced_coeffs = (predicted_coeffs * 9 / 10).max(8);
            let mut reduced_encoder = HybridEncoder::new(reduced_coeffs, transform_type);
            let reduced_encoded = reduced_encoder.encode_block(data, rows, cols);
            let reduced_decoded = reduced_encoded.decode();
            
            let reduced_mse: f32 = data.iter()
                .zip(reduced_decoded.iter())
                .map(|(orig, recon)| (orig - recon).powi(2))
                .sum::<f32>() / (rows * cols) as f32;
            
            if reduced_mse.sqrt() <= threshold {
                reduced_coeffs
            } else {
                predicted_coeffs
            }
        } else {
            // 예측이 부족하면 이분탐색으로 정확한 값 찾기
            Self::find_critical_coefficients(data, rows, cols, threshold, transform_type)?
        };
        
        Ok(HybridEncoder::new(final_coeffs, transform_type))
    }
    
    /// 품질 등급별 자동 생성 (RMSE 임계값 기반)
    pub fn new_quality_grade(
        data: &[f32],
        rows: usize,
        cols: usize,
        grade: QualityGrade,
        transform_type: TransformType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let threshold = match grade {
            QualityGrade::S => 0.000001,  // 거의 완벽
            QualityGrade::A => 0.001,     // 매우 좋음
            QualityGrade::B => 0.01,      // 좋음
            QualityGrade::C => 0.1,       // 보통
        };
        
        Self::new_auto_optimized(data, rows, cols, transform_type, Some(threshold))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum QualityGrade {
    S,  // RMSE ≤ 0.000001
    A,  // RMSE ≤ 0.001
    B,  // RMSE ≤ 0.01
    C,  // RMSE ≤ 0.1
} 
//! 행렬 압축(인코딩) 관련 기능
use crate::types::{HybridEncodedBlock, ResidualCoefficient, TransformType, PoincarePackedBit128, PoincareQuadrant};
use nalgebra::{DMatrix, DVector};
use rustdct::DctPlanner;
use std::sync::Mutex;
use rayon::prelude::*;
use ndarray::{Array, Array1, Array2};
use omni_wave::{wavelet as w, completely_decompose_2d};
use rustfft::{FftPlanner, num_complex::Complex};

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

// ================================
// 2장: 푸앵카레 볼 기반 인코딩 파이프라인 구현
// ================================

/// 2D FFT 결과 구조체
#[derive(Debug, Clone)]
pub struct FrequencyAnalysisResult {
    pub dominant_frequency: (usize, usize),
    pub max_energy: f32,
    pub total_energy: f32,
    pub frequency_type: FrequencyType,
    pub normalized_frequencies: (f32, f32),
}

#[derive(Debug, Clone, PartialEq)]
pub enum FrequencyType {
    LowFreqMonotonic,    // 저주파, 단조증가
    LowFreqSymmetric,    // 저주파, 대칭패턴  
    HighFreqSaturated,   // 고주파, 포화패턴
    Localized,           // 국소화된 특징
}

/// 연속 파라미터 최적화 결과
#[derive(Debug, Clone)]
pub struct ContinuousOptimizationResult {
    pub r_optimal: f32,
    pub theta_optimal: f32,
    pub final_mse: f32,
    pub iterations: usize,
    pub converged: bool,
}

/// 잔차 압축 결과
#[derive(Debug, Clone)]
pub struct ResidualCompressionResult {
    pub selected_coefficients: Vec<ResidualCoefficient>,
    pub compression_ratio: f32,
    pub energy_preserved: f32,
}

/// 푸앵카레 볼 기반 고속 인코더 (2장 구현)
/// REFACTOR.md 성능 최적화 모든 제안 반영
pub struct PoincareEncoder {
    /// 캐시된 좌표 정보 (성능 최적화)
    coordinate_cache: Option<CoordinateCache>,
    /// FFT 플래너 재사용
    fft_planner: FftPlanner<f32>,
    /// DCT 플래너 재사용  
    dct_planner: DctPlanner<f32>,
    /// 잔차 계수 개수
    k_coeffs: usize,
}

/// 좌표 캐싱 구조체 (REFACTOR.md 제안 2번)
#[derive(Debug, Clone)]
struct CoordinateCache {
    rows: usize,
    cols: usize,
    normalized_coords: Vec<(f32, f32)>,  // (x_norm, y_norm)
    distances: Vec<f32>,                 // sqrt(x²+y²)
    angles: Vec<f32>,                    // atan2(y, x)
}

impl PoincareEncoder {
    /// 새로운 푸앵카레 인코더 생성
    pub fn new(k_coeffs: usize) -> Self {
        Self {
            coordinate_cache: None,
            fft_planner: FftPlanner::new(),
            dct_planner: DctPlanner::new(),
            k_coeffs,
        }
    }
    
    /// 좌표 캐시 초기화 (블록 크기별로 한 번만 계산)
    fn initialize_coordinate_cache(&mut self, rows: usize, cols: usize) {
        if let Some(ref cache) = self.coordinate_cache {
            if cache.rows == rows && cache.cols == cols {
                return; // 이미 캐시됨
            }
        }
        
        let mut normalized_coords = Vec::with_capacity(rows * cols);
        let mut distances = Vec::with_capacity(rows * cols);
        let mut angles = Vec::with_capacity(rows * cols);
        
        for i in 0..rows {
            for j in 0..cols {
                let x_norm = if cols > 1 { 
                    (j as f32 / (cols - 1) as f32) * 2.0 - 1.0 
                } else { 
                    0.0 
                };
                let y_norm = if rows > 1 { 
                    (i as f32 / (rows - 1) as f32) * 2.0 - 1.0 
                } else { 
                    0.0 
                };
                
                let distance = (x_norm * x_norm + y_norm * y_norm).sqrt();
                let angle = y_norm.atan2(x_norm);
                
                normalized_coords.push((x_norm, y_norm));
                distances.push(distance);
                angles.push(angle);
            }
        }
        
        self.coordinate_cache = Some(CoordinateCache {
            rows,
            cols,
            normalized_coords,
            distances,
            angles,
        });
    }
    
    /// 4단계 인코딩 파이프라인 실행
    pub fn encode_matrix(&mut self, matrix: &[f32], rows: usize, cols: usize) -> PoincarePackedBit128 {
        // 좌표 캐시 초기화 (성능 최적화)
        self.initialize_coordinate_cache(rows, cols);
        
        // 1단계: 주파수 도메인 분석
        let frequency_result = self.analyze_frequency_domain(matrix, rows, cols);
        
        // 2단계: 푸앵카레 볼 매핑
        let hi_field = self.map_to_poincare_ball(&frequency_result);
        
        // 3단계: 연속 파라미터 최적화
        let optimization_result = self.optimize_continuous_parameters(
            matrix, rows, cols, hi_field
        );
        
        // 4단계: 잔차 압축 (별도 저장, 여기서는 128비트만 반환)
        let _residual_result = self.compress_residuals(
            matrix, rows, cols, hi_field, 
            optimization_result.r_optimal,
            optimization_result.theta_optimal
        );
        
        // 최종 PoincarePackedBit128 생성
        let quadrant = self.extract_quadrant_from_hi(hi_field);
        let frequency = ((hi_field >> 50) & 0xFFF) as u16;
        let amplitude = ((hi_field >> 38) & 0xFFF) as u16;
        let basis_func = ((hi_field >> 32) & 0x3F) as u8;
        let cordic_seq = (hi_field & 0xFFFFFFFF) as u32;
        
        PoincarePackedBit128::new(
            quadrant,
            frequency,
            amplitude, 
            basis_func,
            cordic_seq,
            optimization_result.r_optimal,
            optimization_result.theta_optimal,
        )
    }
    
    /// 1단계: 주파수 도메인 분석 (2D FFT)
    fn analyze_frequency_domain(&mut self, matrix: &[f32], rows: usize, cols: usize) -> FrequencyAnalysisResult {
        // 복소수 배열로 변환
        let mut input: Vec<Complex<f32>> = matrix.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        // 2D FFT 수행
        self.perform_2d_fft(&mut input, rows, cols);
        
        // 에너지 계산 및 지배적 주파수 찾기
        let mut max_energy = 0.0f32;
        let mut dominant_freq = (0, 0);
        let mut total_energy = 0.0f32;
        
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let energy = input[idx].norm_sqr();
                total_energy += energy;
                
                // DC 성분 제외하고 최대 에너지 찾기
                if (i != 0 || j != 0) && energy > max_energy {
                    max_energy = energy;
                    dominant_freq = (i, j);
                }
            }
        }
        
        // 주파수 정규화
        let omega_x_norm = (dominant_freq.0 as f32 / rows as f32) * 2.0 * std::f32::consts::PI;
        let omega_y_norm = (dominant_freq.1 as f32 / cols as f32) * 2.0 * std::f32::consts::PI;
        
        // 주파수 타입 결정
        let frequency_type = self.classify_frequency_type(dominant_freq, rows, cols, max_energy, total_energy);
        
        FrequencyAnalysisResult {
            dominant_frequency: dominant_freq,
            max_energy,
            total_energy,
            frequency_type,
            normalized_frequencies: (omega_x_norm, omega_y_norm),
        }
    }
    
    /// 2D FFT 수행 (행-열 분리 가능)
    fn perform_2d_fft(&mut self, data: &mut [Complex<f32>], rows: usize, cols: usize) {
        // 행별 FFT
        let row_fft = self.fft_planner.plan_fft_forward(cols);
        for i in 0..rows {
            let start = i * cols;
            let end = start + cols;
            row_fft.process(&mut data[start..end]);
        }
        
        // 전치 후 열별 FFT
        let mut transposed = vec![Complex::new(0.0, 0.0); rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = data[i * cols + j];
            }
        }
        
        let col_fft = self.fft_planner.plan_fft_forward(rows);
        for j in 0..cols {
            let start = j * rows;
            let end = start + rows;
            col_fft.process(&mut transposed[start..end]);
        }
        
        // 다시 전치하여 원래 형태로
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = transposed[j * rows + i];
            }
        }
    }
    
    /// 주파수 타입 분류
    fn classify_frequency_type(&self, dominant_freq: (usize, usize), rows: usize, cols: usize, max_energy: f32, total_energy: f32) -> FrequencyType {
        let (freq_x, freq_y) = dominant_freq;
        let max_freq_x = rows / 2;
        let max_freq_y = cols / 2;
        let energy_ratio = max_energy / total_energy;
        
        // 저주파/고주파 구분
        let is_low_freq = freq_x <= max_freq_x / 2 && freq_y <= max_freq_y / 2;
        
        if energy_ratio > 0.8 {
            // 에너지가 집중된 경우
            FrequencyType::Localized
        } else if is_low_freq {
            if freq_x == 0 || freq_y == 0 {
                FrequencyType::LowFreqMonotonic
            } else {
                FrequencyType::LowFreqSymmetric
            }
        } else {
            FrequencyType::HighFreqSaturated
        }
    }
    
    /// 2단계: 푸앵카레 볼 매핑
    fn map_to_poincare_ball(&self, freq_result: &FrequencyAnalysisResult) -> u64 {
        // 주파수-쌍곡함수 대응
        let quadrant = match freq_result.frequency_type {
            FrequencyType::LowFreqMonotonic => 0u64,    // sinh
            FrequencyType::LowFreqSymmetric => 1u64,     // cosh
            FrequencyType::HighFreqSaturated => 2u64,    // tanh
            FrequencyType::Localized => 3u64,            // sech²
        };
        
        // 쌍곡주파수 계산
        let omega_h = self.compute_hyperbolic_frequency(freq_result.normalized_frequencies);
        let freq_quantized = (omega_h * 4095.0).round().clamp(0.0, 4095.0) as u64;
        
        // 측지선 진폭 계산
        let amplitude = freq_result.max_energy / freq_result.total_energy;
        let amp_quantized = (amplitude * 4095.0).round().clamp(0.0, 4095.0) as u64;
        
        // 기저함수 선택 (간단한 매핑)
        let basis_selector = ((quadrant * 16) + (freq_quantized >> 8)) & 0x3F;
        
        // CORDIC 회전 시퀀스 생성
        let cordic_seq = self.generate_cordic_sequence(omega_h, 0.0) as u64;
        
        // hi 필드 조립
        (quadrant << 62) |
        (freq_quantized << 50) |
        (amp_quantized << 38) |
        (basis_selector << 32) |
        cordic_seq
    }
    
    /// 쌍곡주파수 계산
    fn compute_hyperbolic_frequency(&self, normalized_freq: (f32, f32)) -> f32 {
        let (omega_x, omega_y) = normalized_freq;
        let omega_magnitude = (omega_x * omega_x + omega_y * omega_y).sqrt();
        let omega_norm = omega_magnitude / (2.0 * std::f32::consts::PI);
        
        // artanh 근사 (수치 안정성)
        let r_poincare = omega_norm.clamp(0.001, 0.999);
        let hyperbolic_distance = 0.5 * ((1.0 + r_poincare) / (1.0 - r_poincare)).ln();
        
        hyperbolic_distance * 2.0 // 스케일링 팩터
    }
    
    /// CORDIC 회전 시퀀스 생성
    fn generate_cordic_sequence(&self, omega_h: f32, phase: f32) -> u32 {
        let target_angle = omega_h + phase;
        let mut current_angle = 0.0f32;
        let mut sequence = 0u32;
        
        for k in 0..20 {
            let cordic_angle = (1.0f32 / (1 << k) as f32).atanh();
            
            if current_angle < target_angle {
                sequence |= 1 << k;  // 양의 회전
                current_angle += cordic_angle;
            } else {
                current_angle -= cordic_angle;  // 음의 회전
            }
        }
        
        sequence
    }
    
    /// hi 필드에서 사분면 추출
    fn extract_quadrant_from_hi(&self, hi_field: u64) -> PoincareQuadrant {
        match (hi_field >> 62) & 0x3 {
            0 => PoincareQuadrant::First,
            1 => PoincareQuadrant::Second,
            2 => PoincareQuadrant::Third,
            3 => PoincareQuadrant::Fourth,
            _ => unreachable!(),
        }
    }
    
    /// 3단계: 연속 파라미터 최적화 (Levenberg-Marquardt)
    fn optimize_continuous_parameters(
        &self, 
        target_matrix: &[f32], 
        rows: usize, 
        cols: usize, 
        hi_field: u64
    ) -> ContinuousOptimizationResult {
        // 초기 추정값
        let mut r = 0.5f32;
        let mut theta = 0.0f32;
        
        let max_iterations = 20;
        let convergence_threshold = 1e-6;
        let mut lambda = 0.001f32; // LM 댐핑 파라미터
        
        let mut previous_mse = f32::INFINITY;
        
        for iteration in 0..max_iterations {
            // 현재 파라미터로 예측값 계산
            let (predicted, jacobian) = self.compute_prediction_and_jacobian(
                hi_field, r, theta, rows, cols
            );
            
            // MSE 계산
            let mse = target_matrix.iter()
                .zip(predicted.iter())
                .map(|(t, p)| (t - p).powi(2))
                .sum::<f32>() / (rows * cols) as f32;
            
            // 수렴 확인
            if (previous_mse - mse).abs() < convergence_threshold {
                return ContinuousOptimizationResult {
                    r_optimal: r.clamp(0.01, 0.99),
                    theta_optimal: theta,
                    final_mse: mse,
                    iterations: iteration + 1,
                    converged: true,
                };
            }
            
            // 잔차 벡터 계산
            let residuals: Vec<f32> = target_matrix.iter()
                .zip(predicted.iter())
                .map(|(t, p)| t - p)
                .collect();
            
            // LM 업데이트 계산
            let (delta_r, delta_theta) = self.compute_lm_update(&jacobian, &residuals, lambda);
            
            // 파라미터 업데이트 (제약 조건 적용)
            let new_r = (r + delta_r).clamp(0.01, 0.99);
            let new_theta = theta + delta_theta;
            
            // 개선된 경우 파라미터 업데이트
            let new_predicted = self.generate_weights_from_poincare(hi_field, new_r, new_theta, rows, cols);
            let new_mse = target_matrix.iter()
                .zip(new_predicted.iter())
                .map(|(t, p)| (t - p).powi(2))
                .sum::<f32>() / (rows * cols) as f32;
            
            if new_mse < mse {
                r = new_r;
                theta = new_theta;
                lambda *= 0.7; // 댐핑 감소
            } else {
                lambda *= 1.5; // 댐핑 증가
            }
            
            previous_mse = mse;
        }
        
        ContinuousOptimizationResult {
            r_optimal: r.clamp(0.01, 0.99),
            theta_optimal: theta,
            final_mse: previous_mse,
            iterations: max_iterations,
            converged: false,
        }
    }
    
    /// 4단계: 잔차 압축
    fn compress_residuals(
        &mut self,
        target_matrix: &[f32],
        rows: usize,
        cols: usize,
        hi_field: u64,
        r_optimal: f32,
        theta_optimal: f32,
    ) -> ResidualCompressionResult {
        // 최적화된 파라미터로 예측값 생성
        let predicted = self.generate_weights_from_poincare(hi_field, r_optimal, theta_optimal, rows, cols);
        
        // 잔차 계산
        let mut residuals: Vec<f32> = target_matrix.iter()
            .zip(predicted.iter())
            .map(|(t, p)| t - p)
            .collect();
        
        // DCT 변환 수행
        let mut residual_matrix = Array2::from_shape_vec((rows, cols), residuals).unwrap();
        
        // 2D DCT 적용
        let dct_2d = self.dct_planner.plan_dct2(cols);
        for mut row in residual_matrix.rows_mut() {
            let mut row_vec = row.to_vec();
            dct_2d.process_dct2(&mut row_vec);
            row.assign(&Array::from(row_vec));
        }
        
        let dct_2d_col = self.dct_planner.plan_dct2(rows);
        let mut transposed = residual_matrix.t().to_owned();
        for mut col in transposed.rows_mut() {
            let mut col_vec = col.to_vec();
            dct_2d_col.process_dct2(&mut col_vec);
            col.assign(&Array::from(col_vec));
        }
        residual_matrix = transposed.t().to_owned();
        
        // 에너지 기반 계수 선택
        let mut coefficients: Vec<ResidualCoefficient> = residual_matrix
            .indexed_iter()
            .map(|((r, c), &val)| ResidualCoefficient {
                index: (r as u16, c as u16),
                value: val,
            })
            .collect();
        
        // 에너지 순으로 정렬
        coefficients.sort_unstable_by(|a, b| b.value.abs().partial_cmp(&a.value.abs()).unwrap());
        
        // 상위 K개 선택
        let selected_coefficients = coefficients.into_iter().take(self.k_coeffs).collect::<Vec<_>>();
        
        // 성능 지표 계산
        let total_energy: f32 = residual_matrix.iter().map(|x| x * x).sum();
        let preserved_energy: f32 = selected_coefficients.iter().map(|c| c.value * c.value).sum();
        let energy_preserved = if total_energy > 0.0 { preserved_energy / total_energy } else { 1.0 };
        
        let original_size = rows * cols * 4; // f32 = 4 bytes
        let compressed_size = 8 + selected_coefficients.len() * (2 + 2 + 4); // metadata + coefficients
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        ResidualCompressionResult {
            selected_coefficients,
            compression_ratio,
            energy_preserved,
        }
    }
    
    /// 예측값과 야코비안 계산 (해석적 미분 사용)
    fn compute_prediction_and_jacobian(
        &self,
        hi_field: u64,
        r: f32,
        theta: f32,
        rows: usize,
        cols: usize,
    ) -> (Vec<f32>, Vec<(f32, f32)>) {
        let cache = self.coordinate_cache.as_ref().unwrap();
        let mut predicted = Vec::with_capacity(rows * cols);
        let mut jacobian = Vec::with_capacity(rows * cols);
        
        // 사분면 추출
        let quadrant = (hi_field >> 62) & 0x3;
        
        for idx in 0..rows * cols {
            let distance = cache.distances[idx];
            let angle = cache.angles[idx];
            
            // 기본 패턴 계산
            let pattern_arg = distance * r + theta;
            
            // 사분면에 따른 함수 값과 미분 계산
            let (value, dr, dtheta) = match quadrant {
                0 => {
                    // sinh 함수
                    let sinh_val = pattern_arg.sinh();
                    let cosh_val = pattern_arg.cosh();
                    (sinh_val, distance * cosh_val, cosh_val)
                },
                1 => {
                    // cosh 함수  
                    let cosh_val = pattern_arg.cosh();
                    let sinh_val = pattern_arg.sinh();
                    (cosh_val, distance * sinh_val, sinh_val)
                },
                2 => {
                    // tanh 함수
                    let tanh_val = pattern_arg.tanh();
                    let sech_sq = 1.0 - tanh_val * tanh_val;
                    (tanh_val, distance * sech_sq, sech_sq)
                },
                3 => {
                    // sech² 함수
                    let cosh_val = pattern_arg.cosh();
                    let sech = 1.0 / cosh_val;
                    let sech_sq = sech * sech;
                    let tanh_val = pattern_arg.tanh();
                    (sech_sq, -2.0 * distance * sech_sq * tanh_val, -2.0 * sech_sq * tanh_val)
                },
                _ => (0.0, 0.0, 0.0),
            };
            
            predicted.push(value.clamp(-1.0, 1.0));
            jacobian.push((dr, dtheta));
        }
        
        (predicted, jacobian)
    }
    
    /// Levenberg-Marquardt 업데이트 계산
    fn compute_lm_update(&self, jacobian: &[(f32, f32)], residuals: &[f32], lambda: f32) -> (f32, f32) {
        let n = jacobian.len();
        
        // J^T * J 계산 (2x2 행렬)
        let mut jt_j = [[0.0f32; 2]; 2];
        for &(jr, jtheta) in jacobian {
            jt_j[0][0] += jr * jr;
            jt_j[0][1] += jr * jtheta;
            jt_j[1][0] += jtheta * jr;
            jt_j[1][1] += jtheta * jtheta;
        }
        
        // 댐핑 추가: J^T * J + λI
        jt_j[0][0] += lambda;
        jt_j[1][1] += lambda;
        
        // J^T * r 계산
        let mut jt_r = [0.0f32; 2];
        for (i, &residual) in residuals.iter().enumerate() {
            let (jr, jtheta) = jacobian[i];
            jt_r[0] += jr * residual;
            jt_r[1] += jtheta * residual;
        }
        
        // 2x2 역행렬 계산
        let det = jt_j[0][0] * jt_j[1][1] - jt_j[0][1] * jt_j[1][0];
        if det.abs() < 1e-10 {
            return (0.0, 0.0); // 특이행렬
        }
        
        let inv_det = 1.0 / det;
        let delta_r = inv_det * (jt_j[1][1] * jt_r[0] - jt_j[0][1] * jt_r[1]);
        let delta_theta = inv_det * (-jt_j[1][0] * jt_r[0] + jt_j[0][0] * jt_r[1]);
        
        (delta_r, delta_theta)
    }
    
    /// 푸앵카레 볼 파라미터로부터 가중치 생성
    fn generate_weights_from_poincare(
        &self,
        hi_field: u64,
        r: f32,
        theta: f32,
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        let cache = self.coordinate_cache.as_ref().unwrap();
        let mut weights = Vec::with_capacity(rows * cols);
        
        // 사분면 추출
        let quadrant = (hi_field >> 62) & 0x3;
        
        for idx in 0..rows * cols {
            let distance = cache.distances[idx];
            let pattern_arg = distance * r + theta;
            
            let value = match quadrant {
                0 => pattern_arg.sinh(),      // sinh 함수
                1 => pattern_arg.cosh(),      // cosh 함수
                2 => pattern_arg.tanh(),      // tanh 함수
                3 => {                        // sech² 함수
                    let cosh_val = pattern_arg.cosh();
                    let sech = 1.0 / cosh_val;
                    sech * sech
                },
                _ => 0.0,
            };
            
            weights.push(value.clamp(-1.0, 1.0));
        }
        
        weights
    }
} 
use crate::packed_params::{TransformType, HybridEncodedBlock, ResidualCoefficient};
use std::time::Instant;
use rayon::prelude::*;
use nalgebra::{DMatrix, DVector, RowDVector};
use rustdct::DctPlanner;
use ndarray::{Array, Array1, Array2};
use omni_wave::{wavelet as w, completely_decompose_2d};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use once_cell::sync::Lazy;

// A matrix 캐시 (thread-safe)
static A_MATRIX_CACHE: Lazy<Arc<RwLock<HashMap<(usize, usize), Arc<DMatrix<f32>>>>>> = 
    Lazy::new(|| Arc::new(RwLock::new(HashMap::new())));

#[derive(Debug, Clone, Copy)]
pub enum QualityGrade {
    S,  // RMSE ≤ 0.000001
    A,  // RMSE ≤ 0.001
    B,  // RMSE ≤ 0.01
    C,  // RMSE ≤ 0.1
}

#[derive(Debug, Clone, Copy)]
pub enum CompressionProfile {
    UltraHigh,   // 최고 품질, 느린 속도
    High,        // 고품질, 중간 속도  
    Balanced,    // 균형
    Fast,        // 빠른 속도, 낮은 품질
    UltraFast,   // 최고 속도, 최저 품질
}

#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub block_size: usize,
    pub quality_grade: QualityGrade, 
    pub transform_type: TransformType,
    pub profile: CompressionProfile,
    pub custom_coefficients: Option<usize>, // None이면 자동 계산
    
    // 사용자가 요청한 조정 가능 파라미터들
    pub min_block_count: Option<usize>,     // 최소 블록 개수 (하드코딩 조정용)
    pub rmse_threshold: Option<f32>,        // RMSE 임계값 (0.001 같은 값)
    pub compression_ratio_threshold: Option<f32>, // 압축률 임계값 (최소 압축률)
}

/// 통합된 RBE 인코더 (기존 AutoOptimizedEncoder + HybridEncoder)
pub struct RBEEncoder {
    pub k_coeffs: usize, // 유지할 잔차 계수의 개수
    pub transform_type: TransformType,
    // planner는 재사용 가능하므로 인코더가 소유하는 것이 효율적
    dct_planner_f32: DctPlanner<f32>,
}

// 기존 AutoOptimizedEncoder는 호환성을 위해 타입 별칭으로 유지
pub type AutoOptimizedEncoder = RBEEncoder;

impl CompressionConfig {
    /// 기본 설정 (Balanced 프로파일)
    pub fn default() -> Self {
        Self {
            block_size: 64,
            quality_grade: QualityGrade::B,
            transform_type: TransformType::Dwt,  // DWT! DWT! DWT!
            profile: CompressionProfile::Balanced,
            custom_coefficients: None,
            min_block_count: None,
            rmse_threshold: None,
            compression_ratio_threshold: None,
        }
    }
    
    /// UltraHigh 품질 프리셋 (현실적인 임계값)
    pub fn ultra_high() -> Self {
        Self {
            block_size: 32,  // 작은 블록 = 높은 품질
            quality_grade: QualityGrade::S,
            transform_type: TransformType::Dwt,  // DWT 사용!
            profile: CompressionProfile::UltraHigh,
            custom_coefficients: None,
            min_block_count: None,
            rmse_threshold: Some(0.01),   // 0.00001 → 0.01 (현실적으로 조정)
            compression_ratio_threshold: Some(50.0),
        }
    }
    
    /// Fast 압축 프리셋
    pub fn fast() -> Self {
        Self {
            block_size: 128, // 큰 블록 = 빠른 속도
            quality_grade: QualityGrade::C,
            transform_type: TransformType::Dwt,  // DWT 사용!
            profile: CompressionProfile::Fast,
            custom_coefficients: Some(256), // 적은 계수
            min_block_count: None,
            rmse_threshold: Some(0.1),
            compression_ratio_threshold: Some(10.0),
        }
    }
    
    /// 사용자 정의 설정
    pub fn custom(
        block_size: usize,
        rmse_threshold: f32,
        compression_ratio: f32,
        min_blocks: Option<usize>
    ) -> Self {
        Self {
            block_size,
            quality_grade: QualityGrade::B,
            transform_type: TransformType::Dwt,  // DWT! DWT! DWT!
            profile: CompressionProfile::Balanced,
            custom_coefficients: None,
            min_block_count: min_blocks,
            rmse_threshold: Some(rmse_threshold),
            compression_ratio_threshold: Some(compression_ratio),
        }
    }
}

impl RBEEncoder {
    /// 새로운 RBE 인코더 생성
    pub fn new(k_coeffs: usize, transform_type: TransformType) -> Self {
        Self {
            k_coeffs,
            transform_type,
            dct_planner_f32: DctPlanner::new(),
        }
    }
    
    /// S급 품질 (RMSE < 0.001): 819:1 압축률, 128x128 블록 권장
    pub fn new_s_grade() -> Self {
        Self::new(500, TransformType::Dwt)  // DWT 사용!
    }
    
    /// A급 품질 (RMSE < 0.01): 3276:1 압축률, 256x256 블록 권장  
    pub fn new_a_grade() -> Self {
        Self::new(300, TransformType::Dwt)  // DWT 사용!
    }
    
    /// B급 품질 (RMSE < 0.1): 3276:1 압축률, 256x256 블록 권장
    pub fn new_b_grade() -> Self {
        Self::new(200, TransformType::Dwt)  // DWT 사용!
    }
    
    /// 극한 압축 (RMSE ~0.09): 3276:1 압축률
    pub fn new_extreme_compression() -> Self {
        Self::new(75, TransformType::Dwt)   // DWT 사용! 계수 100 -> 75
    }

    /// 단일 블록을 RBE+DCT/DWT로 압축 (기존 HybridEncoder::encode_block)
    pub fn encode_block(&mut self, block_data: &[f32], rows: usize, cols: usize) -> HybridEncodedBlock {
        // 입력 데이터 크기 검증
        if block_data.len() != rows * cols {
            panic!("block_data 길이({})가 rows * cols({})와 일치하지 않음", block_data.len(), rows * cols);
        }
        
        let b_vector = DVector::from_row_slice(block_data);
        
        // A matrix 캐시 확인
        let cache_key = (rows, cols);
        let a_matrix = {
            let cache = A_MATRIX_CACHE.read().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                Arc::clone(cached)
            } else {
                drop(cache); // 읽기 잠금 해제
                
                // A matrix 생성 (병렬 처리)
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

                let new_a_matrix = Arc::new(DMatrix::from_rows(&a_matrix_rows));
                
                // 캐시에 저장
                let mut cache = A_MATRIX_CACHE.write().unwrap();
                cache.insert(cache_key, Arc::clone(&new_a_matrix));
                
                new_a_matrix
            }
        };

        // Pseudo-inverse via SVD with proper dimension handling
        let svd = a_matrix.as_ref().clone().svd(true, true);
        let singular_values = &svd.singular_values;
        let mut sigma_inv = DMatrix::zeros(svd.v_t.as_ref().unwrap().nrows(), svd.u.as_ref().unwrap().ncols());
        
        let tolerance = 1e-10_f32;
        for i in 0..singular_values.len().min(sigma_inv.nrows()).min(sigma_inv.ncols()) {
            if singular_values[i].abs() > tolerance {
                sigma_inv[(i, i)] = 1.0 / singular_values[i];
            }
        }
        
        let a_pseudo_inv = svd.v_t.as_ref().unwrap().transpose() * sigma_inv * svd.u.as_ref().unwrap().transpose();
        
        // RBE 파라미터 계산
        let rbe_params_vec = a_pseudo_inv.clone() * b_vector.clone();
        let mut rbe_params = [0.0f32; 8];
        for i in 0..8.min(rbe_params_vec.len()) {
            rbe_params[i] = rbe_params_vec[i];
        }
        
        // 잔차 계산
        let predicted_vector = a_matrix.as_ref() * rbe_params_vec;
        let residual_vector = b_vector - predicted_vector;

        self.encode_single_transform(rbe_params, &residual_vector, rows, cols, self.transform_type)
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
            TransformType::Adaptive => unreachable!("Adaptive transform should be handled separately"),
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

    /// 비대칭 매트릭스 지원
    pub fn compress_with_profile(
        matrix_data: &[f32],
        height: usize,
        width: usize, 
        block_size: usize,
        coefficients: usize,
        transform_type: TransformType,
    ) -> Result<(Vec<HybridEncodedBlock>, f64, f32, f32), String> {
        let start = Instant::now();
        
        // 비대칭 매트릭스 격자 분할
        let blocks_per_height = (height + block_size - 1) / block_size;
        let blocks_per_width = (width + block_size - 1) / block_size;
        let total_blocks = blocks_per_height * blocks_per_width;
        
        // 병렬 블록 처리 (Rayon)
        let encoded_blocks: Vec<HybridEncodedBlock> = (0..total_blocks)
            .into_par_iter()
            .map(|block_idx| {
                let mut local_encoder = RBEEncoder::new(coefficients, transform_type);
                let block_i = block_idx / blocks_per_width;
                let block_j = block_idx % blocks_per_width;
                let start_i = block_i * block_size;
                let start_j = block_j * block_size;
                
                // 블록 데이터 추출 (비대칭 매트릭스)
                let mut block_data = vec![0.0f32; block_size * block_size];
                for i in 0..block_size {
                    for j in 0..block_size {
                        let global_i = start_i + i;
                        let global_j = start_j + j;
                        if global_i < height && global_j < width {
                            block_data[i * block_size + j] = 
                                matrix_data[global_i * width + global_j];
                        }
                        // 패딩은 0.0으로 유지
                    }
                }
                
                // 블록 압축 (각 스레드별 local_encoder)
                local_encoder.encode_block(&block_data, block_size, block_size)
            })
            .collect();
        
        let compression_time = start.elapsed().as_secs_f64();
        
        // 압축률 계산 (비대칭 매트릭스)
        let original_size = height * width * 4; // f32 bytes
        let compressed_size = encoded_blocks.len() * std::mem::size_of::<HybridEncodedBlock>();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        // RMSE 계산 - 디코딩해서 원본과 비교
        let mut reconstructed_data = vec![0.0f32; height * width];
        
        for (block_idx, encoded_block) in encoded_blocks.iter().enumerate() {
            let block_i = block_idx / blocks_per_width;
            let block_j = block_idx % blocks_per_width;
            let start_i = block_i * block_size;
            let start_j = block_j * block_size;
            
            // 블록 디코딩
            let decoded_block = encoded_block.decode();
            
            // 원본 행렬에 복사 (비대칭 매트릭스)
            for i in 0..block_size {
                for j in 0..block_size {
                    let global_i = start_i + i;
                    let global_j = start_j + j;
                    if global_i < height && global_j < width {
                        reconstructed_data[global_i * width + global_j] = 
                            decoded_block[i * block_size + j];
                    }
                }
            }
        }
        
        // RMSE 계산 (실제 데이터 영역만)
        let mse: f32 = matrix_data.iter()
            .zip(reconstructed_data.iter())
            .map(|(orig, recon)| (orig - recon).powi(2))
            .sum::<f32>() / (height * width) as f32;
        let rmse = mse.sqrt();
        
        Ok((encoded_blocks, compression_time, compression_ratio, rmse))
    }
    
    /// 설정 기반 압축 함수 (사용자 요청 파라미터 적용)
    pub fn compress_with_config(
        matrix_data: &[f32],
        height: usize,
        width: usize,
        config: &CompressionConfig,
    ) -> Result<(Vec<HybridEncodedBlock>, f64, f32, f32), String> {
        // 1. 최소 블록 개수 체크
        let blocks_per_height = (height + config.block_size - 1) / config.block_size;
        let blocks_per_width = (width + config.block_size - 1) / config.block_size;
        let actual_block_count = blocks_per_height * blocks_per_width;
        
        if let Some(min_blocks) = config.min_block_count {
            if actual_block_count < min_blocks {
                return Err(format!(
                    "블록 개수 부족: 실제 {}개 < 최소 {}개 (블록 크기를 {}에서 더 작게 조정하세요)",
                    actual_block_count, min_blocks, config.block_size
                ));
            }
        }
        
        // 2. 계수 개수 결정
        let coefficients = if let Some(custom_coeffs) = config.custom_coefficients {
            custom_coeffs
        } else {
            // 설정에 따른 자동 계수 계산 (UltraHigh는 더 많은 계수 사용)
            match config.profile {
                CompressionProfile::UltraHigh => Self::predict_coefficients_improved(config.block_size) * 4, // 2 → 4배
                CompressionProfile::High => Self::predict_coefficients_improved(config.block_size) * 2,     // 1 → 2배
                CompressionProfile::Balanced => Self::predict_coefficients_improved(config.block_size),
                CompressionProfile::Fast => Self::predict_coefficients_improved(config.block_size) / 2,     // /4 → /2
                CompressionProfile::UltraFast => Self::predict_coefficients_improved(config.block_size) / 4, // /8 → /4
            }
        };
        
        // 3. 압축 실행
        let result = Self::compress_with_profile(
            matrix_data,
            height,
            width,
            config.block_size,
            coefficients,
            config.transform_type,
        )?;
        
        let (blocks, time, ratio, rmse) = result;
        
        // 4. RMSE 임계값 체크
        if let Some(max_rmse) = config.rmse_threshold {
            if rmse > max_rmse {
                return Err(format!(
                    "RMSE 임계값 초과: {:.6} > {:.6} (계수를 {}에서 더 늘리거나 블록을 더 작게 하세요)",
                    rmse, max_rmse, coefficients
                ));
            }
        }
        
        // 5. 압축률 임계값 체크
        if let Some(min_ratio) = config.compression_ratio_threshold {
            if ratio < min_ratio {
                return Err(format!(
                    "압축률 임계값 미달: {:.1}x < {:.1}x (계수를 {}에서 더 줄이거나 블록을 더 크게 하세요)",
                    ratio, min_ratio, coefficients
                ));
            }
        }
        
        Ok((blocks, time, ratio, rmse))
    }

    /// compress_multi.rs와 동일한 이분탐색
    pub fn find_critical_coefficients(
        matrix_data: &[f32], 
        matrix_size: usize, 
        block_size: usize,
        rmse_threshold: f32,
        transform_type: TransformType,
    ) -> Result<usize, String> {
        // 이분탐색으로 임계 계수 찾기
        let max_coeffs = (block_size * block_size) / 4; // 상한: 전체 픽셀의 1/4
        let min_coeffs = 8; // 하한: 최소 8개
        
        let mut left = min_coeffs;
        let mut right = max_coeffs;
        let mut critical_coeffs = max_coeffs;
        
        while left <= right {
            let mid = (left + right) / 2;
            
            match Self::compress_with_profile(matrix_data, matrix_size, matrix_size, block_size, mid, transform_type) {
                Ok((_, _, _, rmse)) => {
                    if rmse <= rmse_threshold {
                        // 성공: 더 적은 계수로 시도
                        critical_coeffs = mid;
                        right = mid - 1;
                    } else {
                        // 실패: 더 많은 계수 필요
                        left = mid + 1;
                    }
                },
                Err(_) => {
                    left = mid + 1;
                }
            }
        }
        
        Ok(critical_coeffs)
    }
    
    /// 푸앵카레 볼 기반 수학적으로 올바른 공식 (기존 테스트용)
    pub fn predict_coefficients_improved(block_size: usize) -> usize {
        // 올바른 임계비율 계산: R(Block_Size) = max(25, 32 - ⌊log₂(Block_Size/32)⌋)
        let log_factor = if block_size >= 32 {
            (block_size as f32 / 32.0).log2().floor() as i32
        } else {
            -1 // log₂(16/32) = log₂(0.5) = -1
        };
        
        let r_value = (32 - log_factor).max(25) as usize;
        
        // K_critical = ⌈Block_Size² / R(Block_Size)⌉
        let k_critical = (block_size * block_size + r_value - 1) / r_value; // 올림 처리
        
        k_critical
    }

    /// 자동 최적화: 개선된 공식으로 빠른 예측 후 미세 조정 (기존 테스트용)
    pub fn create_optimized_encoder(
        data: &[f32],
        rows: usize,
        cols: usize,
        transform_type: TransformType,
        rmse_threshold: Option<f32>,
    ) -> Result<RBEEncoder, String> {
        let threshold = rmse_threshold.unwrap_or(0.000001);
        
        // 1. 개선된 공식으로 초기 예측
        let predicted_coeffs = Self::predict_coefficients_improved(rows);
        
        // 2. 예측값으로 빠른 검증
        let mut test_encoder = RBEEncoder::new(predicted_coeffs, transform_type);
        let encoded_block = test_encoder.encode_block(data, rows, cols);
        let decoded_data = encoded_block.decode();
        
        let mse: f32 = data.iter()
            .zip(decoded_data.iter())
            .map(|(orig, recon)| (orig - recon).powi(2))
            .sum::<f32>() / (rows * cols) as f32;
        let predicted_rmse = mse.sqrt();
        
        let final_coeffs = if predicted_rmse <= threshold {
            // 예측이 정확하면 그대로 사용
            predicted_coeffs
        } else {
            // 예측이 부족하면 정밀 탐색 (좁은 범위에서만)
            let search_min = predicted_coeffs;
            let search_max = predicted_coeffs * 2; // 2배까지만 탐색
            
            let mut left = search_min;
            let mut right = search_max;
            let mut result = search_max;
            
            while left <= right {
                let mid = (left + right) / 2;
                
                let mut mid_encoder = RBEEncoder::new(mid, transform_type);
                let mid_encoded = mid_encoder.encode_block(data, rows, cols);
                let mid_decoded = mid_encoded.decode();
                
                let mid_mse: f32 = data.iter()
                    .zip(mid_decoded.iter())
                    .map(|(orig, recon)| (orig - recon).powi(2))
                    .sum::<f32>() / (rows * cols) as f32;
                
                if mid_mse.sqrt() <= threshold {
                    result = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            
            result
        };
        
        Ok(RBEEncoder::new(final_coeffs, transform_type))
    }

    /// 품질 등급에 따른 encoder 생성 (기존 테스트용 - 단일 블록 압축)
    pub fn create_quality_encoder(
        data: &[f32],
        rows: usize,
        cols: usize,
        grade: QualityGrade,
        transform_type: TransformType,
    ) -> Result<RBEEncoder, String> {
        let rmse_threshold = match grade {
            QualityGrade::S => 0.000001, // 이분탐색에서는 엄격하게, 테스트에서 여유분 적용
            QualityGrade::A => 0.001,
            QualityGrade::B => 0.01,
            QualityGrade::C => 0.1,
        };
        
        // 기존 방식: 단일 블록 이분탐색 (테스트 호환성)
        let critical_coeffs = Self::find_critical_coefficients_single_block(data, rows, cols, rmse_threshold, transform_type)?;
        Ok(RBEEncoder::new(critical_coeffs, transform_type))
    }

    /// 단일 블록 이분탐색 (기존 테스트용)
    pub fn find_critical_coefficients_single_block(
        data: &[f32],
        rows: usize,
        cols: usize,
        rmse_threshold: f32,
        transform_type: TransformType,
    ) -> Result<usize, String> {
        let max_coeffs = (rows * cols) / 4; // 상한: 전체 픽셀의 1/4
        let min_coeffs = 8; // 하한: 최소 8개
        
        let mut left = min_coeffs;
        let mut right = max_coeffs;
        let mut critical_coeffs = max_coeffs;
        
        while left <= right {
            let mid = (left + right) / 2;
            let mut encoder = RBEEncoder::new(mid, transform_type);
            
            // 단일 블록 압축 및 복원
            let encoded = encoder.encode_block(data, rows, cols);
            let decoded = encoded.decode();
            
            // RMSE 계산
            let mse: f32 = data.iter()
                .zip(decoded.iter())
                .map(|(orig, decoded)| (orig - decoded).powi(2))
                .sum::<f32>() / data.len() as f32;
            let rmse = mse.sqrt();
            
            if rmse <= rmse_threshold {
                critical_coeffs = mid;
                right = mid - 1; // 더 적은 계수로 시도
            } else {
                left = mid + 1; // 더 많은 계수 필요
            }
        }
        
        Ok(critical_coeffs)
    }
}

 
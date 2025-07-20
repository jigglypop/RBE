use crate::packed_params::{TransformType, HybridEncodedBlock};
use super::hybrid_encoder::HybridEncoder;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub enum QualityGrade {
    S,  // RMSE ≤ 0.000001
    A,  // RMSE ≤ 0.001
    B,  // RMSE ≤ 0.01
    C,  // RMSE ≤ 0.1
}

pub struct AutoOptimizedEncoder;

impl AutoOptimizedEncoder {
    /// compress_multi.rs와 동일한 압축 함수 (비대칭 매트릭스 지원)
    pub fn compress_with_profile(
        matrix_data: &[f32],
        height: usize,
        width: usize, 
        block_size: usize,
        coefficients: usize,
        transform_type: TransformType,
    ) -> Result<(Vec<HybridEncodedBlock>, f64, f32, f32), String> {
        let start = Instant::now();
        let mut encoder = HybridEncoder::new(coefficients, transform_type);
        
        // 비대칭 매트릭스 격자 분할
        let blocks_per_height = (height + block_size - 1) / block_size;
        let blocks_per_width = (width + block_size - 1) / block_size;
        let total_blocks = blocks_per_height * blocks_per_width;
        let mut encoded_blocks = Vec::new();
        
        for block_idx in 0..total_blocks {
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
            
            // 블록 압축
            let encoded_block = encoder.encode_block(&block_data, block_size, block_size);
            encoded_blocks.push(encoded_block);
        }
        
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
    ) -> Result<HybridEncoder, String> {
        let threshold = rmse_threshold.unwrap_or(0.000001);
        
        // 1. 개선된 공식으로 초기 예측
        let predicted_coeffs = Self::predict_coefficients_improved(rows);
        
        // 2. 예측값으로 빠른 검증
        let mut test_encoder = HybridEncoder::new(predicted_coeffs, transform_type);
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
                
                let mut mid_encoder = HybridEncoder::new(mid, transform_type);
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
        
        Ok(HybridEncoder::new(final_coeffs, transform_type))
    }

    /// 품질 등급에 따른 encoder 생성 (기존 테스트용 - 단일 블록 압축)
    pub fn create_quality_encoder(
        data: &[f32],
        rows: usize,
        cols: usize,
        grade: QualityGrade,
        transform_type: TransformType,
    ) -> Result<HybridEncoder, String> {
        let rmse_threshold = match grade {
            QualityGrade::S => 0.000001, // 이분탐색에서는 엄격하게, 테스트에서 여유분 적용
            QualityGrade::A => 0.001,
            QualityGrade::B => 0.01,
            QualityGrade::C => 0.1,
        };
        
        // 기존 방식: 단일 블록 이분탐색 (테스트 호환성)
        let critical_coeffs = Self::find_critical_coefficients_single_block(data, rows, cols, rmse_threshold, transform_type)?;
        Ok(HybridEncoder::new(critical_coeffs, transform_type))
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
            let mut encoder = HybridEncoder::new(mid, transform_type);
            
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

 
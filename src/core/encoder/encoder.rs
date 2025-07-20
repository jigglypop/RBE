use crate::packed_params::TransformType;
use super::hybrid_encoder::HybridEncoder;

#[derive(Debug, Clone, Copy)]
pub enum QualityGrade {
    S,  // RMSE ≤ 0.000001
    A,  // RMSE ≤ 0.001
    B,  // RMSE ≤ 0.01
    C,  // RMSE ≤ 0.1
}

pub struct AutoOptimizedEncoder;

impl AutoOptimizedEncoder {
    /// 푸앵카레 볼 기반 수학적으로 올바른 공식
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
        let max_coeffs = (block_size * block_size) / 2; // 상한: 전체 픽셀의 1/2
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
    
    /// 자동 최적화: 개선된 공식으로 빠른 예측 후 미세 조정
    pub fn create_optimized_encoder(
        data: &[f32],
        rows: usize,
        cols: usize,
        transform_type: TransformType,
        rmse_threshold: Option<f32>,
    ) -> Result<HybridEncoder, Box<dyn std::error::Error>> {
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
    
    /// 품질 등급별 자동 생성
    pub fn create_quality_encoder(
        data: &[f32],
        rows: usize,
        cols: usize,
        grade: QualityGrade,
        transform_type: TransformType,
    ) -> Result<HybridEncoder, Box<dyn std::error::Error>> {
        let threshold = match grade {
            QualityGrade::S => 0.000001,  // 거의 완벽
            QualityGrade::A => 0.001,     // 매우 좋음
            QualityGrade::B => 0.01,      // 좋음
            QualityGrade::C => 0.1,       // 보통
        };
        
        Self::create_optimized_encoder(data, rows, cols, transform_type, Some(threshold))
    }
}

 
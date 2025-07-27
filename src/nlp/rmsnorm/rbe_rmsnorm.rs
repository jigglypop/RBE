//! RBE 기반 RMSNorm 모듈
//! 
//! Root Mean Square Layer Normalization을 RBE로 압축하여 구현

use anyhow::Result;
use crate::core::{
    transform::{WeightCompressor, TransformStats},
    tensors::Packed128,
};

/// 압축 품질 등급
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityGrade {
    S, // 최고품질
    A, // 고품질  
    B, // 표준품질
    C, // 기본품질
}

/// RBE 기반 RMSNorm 레이어
#[derive(Debug)]
pub struct RBERMSNorm {
    /// 압축된 감마(스케일) 파라미터
    pub gamma_seed: Packed128,
    
    /// 정규화 차원
    pub normalized_shape: usize,
    /// epsilon 값 (수치적 안정성)
    pub eps: f32,
    /// 압축 품질 등급
    pub quality_grade: QualityGrade,
    
    /// 압축 통계
    pub compression_stats: Option<TransformStats>,
}

impl RBERMSNorm {
    /// 새로운 RBE RMSNorm 레이어 생성
    pub fn new(normalized_shape: usize, eps: f32, quality_grade: QualityGrade) -> Self {
        Self {
            gamma_seed: Packed128::default(),
            normalized_shape,
            eps,
            quality_grade,
            compression_stats: None,
        }
    }

    /// f32 감마 파라미터로부터 압축된 RMSNorm 레이어 생성
    pub fn from_weights(
        gamma_weights: &[f32],    // [normalized_shape]
        normalized_shape: usize,
        eps: f32,
        quality_grade: QualityGrade,
    ) -> Result<Self> {
        let mut rmsnorm = Self::new(normalized_shape, eps, quality_grade);
        
        println!("🔄 RBE RMSNorm 압축 시작...");
        
        // 감마 파라미터 압축
        if gamma_weights.len() != normalized_shape {
            return Err(anyhow::anyhow!(
                "Gamma weights size mismatch: {} vs {}", 
                gamma_weights.len(), normalized_shape
            ));
        }
        
        // 1D 벡터를 2D로 변환하여 압축 (1 x normalized_shape)
        let compressor = WeightCompressor::new(1, normalized_shape);
        let (gamma_seed, gamma_stats) = compressor.compress_weights(gamma_weights)?;
        rmsnorm.gamma_seed = gamma_seed;
        rmsnorm.compression_stats = Some(gamma_stats);
        
        println!("✅ RBE RMSNorm 압축 완료: {:.1}:1 압축률, RMSE {:.6}", 
                rmsnorm.compression_stats.as_ref().unwrap().compression_ratio,
                rmsnorm.compression_stats.as_ref().unwrap().rmse);
        
        Ok(rmsnorm)
    }

    /// RMSNorm 순전파
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() % self.normalized_shape != 0 {
            return Err(anyhow::anyhow!(
                "Input size {} not divisible by normalized_shape {}", 
                input.len(), self.normalized_shape
            ));
        }
        
        let batch_size = input.len() / self.normalized_shape;
        let mut output = vec![0.0f32; input.len()];
        
        for batch_idx in 0..batch_size {
            let start_idx = batch_idx * self.normalized_shape;
            let input_slice = &input[start_idx..start_idx + self.normalized_shape];
            let output_slice = &mut output[start_idx..start_idx + self.normalized_shape];
            
            // 1. RMS 계산
            let mean_square: f32 = input_slice.iter()
                .map(|&x| x * x)
                .sum::<f32>() / self.normalized_shape as f32;
            
            let rms = (mean_square + self.eps).sqrt();
            
            // 2. 정규화 및 스케일링
            for (i, (&input_val, output_val)) in input_slice.iter().zip(output_slice.iter_mut()).enumerate() {
                // 압축된 감마 파라미터 추출
                let gamma = self.gamma_seed.fused_forward(0, i, 1, self.normalized_shape);
                
                // RMSNorm: x * gamma / rms
                *output_val = input_val * gamma / rms;
            }
        }
        
        Ok(output)
    }

    /// 단일 벡터에 대한 RMSNorm (배치 처리 없음)
    pub fn forward_single(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.normalized_shape {
            return Err(anyhow::anyhow!(
                "Input size {} doesn't match normalized_shape {}", 
                input.len(), self.normalized_shape
            ));
        }
        
        // RMS 계산
        let mean_square: f32 = input.iter()
            .map(|&x| x * x)
            .sum::<f32>() / self.normalized_shape as f32;
        
        let rms = (mean_square + self.eps).sqrt();
        
        // 정규화 및 스케일링
        let mut output = vec![0.0f32; self.normalized_shape];
        for (i, (&input_val, output_val)) in input.iter().zip(output.iter_mut()).enumerate() {
            let gamma = self.gamma_seed.fused_forward(0, i, 1, self.normalized_shape);
            *output_val = input_val * gamma / rms;
        }
        
        Ok(output)
    }

    /// 감마 파라미터 추출
    pub fn get_gamma_parameter(&self, index: usize) -> Result<f32> {
        if index >= self.normalized_shape {
            return Err(anyhow::anyhow!("Index {} out of range", index));
        }
        
        Ok(self.gamma_seed.fused_forward(0, index, 1, self.normalized_shape))
    }

    /// 모든 감마 파라미터 추출
    pub fn get_all_gamma_parameters(&self) -> Vec<f32> {
        (0..self.normalized_shape)
            .map(|i| self.gamma_seed.fused_forward(0, i, 1, self.normalized_shape))
            .collect()
    }

    /// 통계 계산 (디버깅용)
    pub fn compute_stats(&self, input: &[f32]) -> Result<RMSNormStats> {
        if input.len() % self.normalized_shape != 0 {
            return Err(anyhow::anyhow!("Invalid input size"));
        }
        
        let batch_size = input.len() / self.normalized_shape;
        let mut mean_rms = 0.0f32;
        let mut min_rms = f32::INFINITY;
        let mut max_rms = f32::NEG_INFINITY;
        
        for batch_idx in 0..batch_size {
            let start_idx = batch_idx * self.normalized_shape;
            let input_slice = &input[start_idx..start_idx + self.normalized_shape];
            
            let mean_square: f32 = input_slice.iter()
                .map(|&x| x * x)
                .sum::<f32>() / self.normalized_shape as f32;
            
            let rms = (mean_square + self.eps).sqrt();
            
            mean_rms += rms;
            min_rms = min_rms.min(rms);
            max_rms = max_rms.max(rms);
        }
        
        mean_rms /= batch_size as f32;
        
        Ok(RMSNormStats {
            mean_rms,
            min_rms,
            max_rms,
            batch_size,
        })
    }

    /// 압축 통계 반환
    pub fn get_compression_stats(&self) -> Option<&TransformStats> {
        self.compression_stats.as_ref()
    }

    /// 메모리 사용량 계산
    pub fn get_memory_usage(&self) -> usize {
        std::mem::size_of::<Packed128>() + // 감마 시드
        std::mem::size_of::<Self>()
    }

    /// 압축률 계산
    pub fn get_compression_ratio(&self) -> f64 {
        if let Some(stats) = &self.compression_stats {
            stats.compression_ratio
        } else {
            0.0
        }
    }

    /// 파라미터 수 계산
    pub fn get_parameter_count(&self) -> usize {
        self.normalized_shape // 감마 파라미터만
    }
}

/// RMSNorm 통계
#[derive(Debug, Clone)]
pub struct RMSNormStats {
    pub mean_rms: f32,
    pub min_rms: f32,
    pub max_rms: f32,
    pub batch_size: usize,
} 
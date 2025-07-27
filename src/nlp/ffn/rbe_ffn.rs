//! RBE 기반 Feed-Forward Network 모듈
//! 
//! Transformer의 FFN을 RBE로 압축하여 구현

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

/// RBE 기반 Feed-Forward Network
#[derive(Debug)]
pub struct RBEFFN {
    /// 압축된 첫 번째 선형 변환 가중치 (up projection)
    pub up_weight_seed: Packed128,
    /// 압축된 두 번째 선형 변환 가중치 (down projection)  
    pub down_weight_seed: Packed128,
    
    /// 입력 차원
    pub input_dim: usize,
    /// 중간 차원 (일반적으로 input_dim * 4)
    pub intermediate_dim: usize,
    /// 압축 품질 등급
    pub quality_grade: QualityGrade,
    
    /// 압축 통계
    pub compression_stats: Option<Vec<TransformStats>>,
}

impl RBEFFN {
    /// 새로운 RBE FFN 레이어 생성
    pub fn new(input_dim: usize, intermediate_dim: usize, quality_grade: QualityGrade) -> Self {
        Self {
            up_weight_seed: Packed128::default(),
            down_weight_seed: Packed128::default(),
            input_dim,
            intermediate_dim,
            quality_grade,
            compression_stats: None,
        }
    }

    /// f32 가중치로부터 압축된 FFN 레이어 생성
    pub fn from_weights(
        up_weights: &[f32],      // [intermediate_dim, input_dim] 
        down_weights: &[f32],    // [input_dim, intermediate_dim]
        input_dim: usize,
        intermediate_dim: usize,
        quality_grade: QualityGrade,
    ) -> Result<Self> {
        let mut ffn = Self::new(input_dim, intermediate_dim, quality_grade);
        
        println!("🔄 RBE FFN 압축 시작...");
        
        let mut stats = Vec::new();
        
        // Up projection 가중치 압축
        if up_weights.len() != intermediate_dim * input_dim {
            return Err(anyhow::anyhow!(
                "Up weights size mismatch: {} vs {}x{}", 
                up_weights.len(), intermediate_dim, input_dim
            ));
        }
        
        let compressor = WeightCompressor::new(intermediate_dim, input_dim);
        let (up_seed, up_stats) = compressor.compress_weights(up_weights)?;
        ffn.up_weight_seed = up_seed;
        stats.push(up_stats);
        
        // Down projection 가중치 압축
        if down_weights.len() != input_dim * intermediate_dim {
            return Err(anyhow::anyhow!(
                "Down weights size mismatch: {} vs {}x{}", 
                down_weights.len(), input_dim, intermediate_dim
            ));
        }
        
        let compressor = WeightCompressor::new(input_dim, intermediate_dim);
        let (down_seed, down_stats) = compressor.compress_weights(down_weights)?;
        ffn.down_weight_seed = down_seed;
        stats.push(down_stats);
        
        ffn.compression_stats = Some(stats);
        
        let avg_ratio: f64 = ffn.compression_stats.as_ref().unwrap()
            .iter().map(|s| s.compression_ratio).sum::<f64>() / 2.0;
        let avg_rmse: f64 = ffn.compression_stats.as_ref().unwrap()
            .iter().map(|s| s.rmse).sum::<f64>() / 2.0;
            
        println!("✅ RBE FFN 압축 완료: {:.1}:1 압축률, RMSE {:.6}", avg_ratio, avg_rmse);
        
        Ok(ffn)
    }

    /// FFN 순전파
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() % self.input_dim != 0 {
            return Err(anyhow::anyhow!(
                "Input size {} not divisible by input_dim {}", 
                input.len(), self.input_dim
            ));
        }
        
        let batch_size = input.len() / self.input_dim;
        let mut output = vec![0.0f32; batch_size * self.input_dim];
        
        for batch_idx in 0..batch_size {
            let input_start = batch_idx * self.input_dim;
            let input_slice = &input[input_start..input_start + self.input_dim];
            
            // 1. Up projection: input_dim -> intermediate_dim
            let intermediate = self.up_projection(input_slice)?;
            
            // 2. 활성화 함수 (GELU)
            let activated = self.gelu_activation(&intermediate);
            
            // 3. Down projection: intermediate_dim -> input_dim  
            let final_output = self.down_projection(&activated)?;
            
            // 결과 저장
            let output_start = batch_idx * self.input_dim;
            output[output_start..output_start + self.input_dim].copy_from_slice(&final_output);
        }
        
        Ok(output)
    }

    /// Up projection (input_dim -> intermediate_dim)
    fn up_projection(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.input_dim {
            return Err(anyhow::anyhow!("Input size mismatch for up projection"));
        }
        
        let mut output = vec![0.0f32; self.intermediate_dim];
        
        // RBE 압축된 가중치로 행렬 곱셈
        for i in 0..self.intermediate_dim {
            let mut sum = 0.0f32;
            for j in 0..self.input_dim {
                let weight = self.up_weight_seed.fused_forward(
                    i, j, self.intermediate_dim, self.input_dim
                );
                sum += weight * input[j];
            }
            output[i] = sum;
        }
        
        Ok(output)
    }

    /// Down projection (intermediate_dim -> input_dim)
    fn down_projection(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.intermediate_dim {
            return Err(anyhow::anyhow!("Input size mismatch for down projection"));
        }
        
        let mut output = vec![0.0f32; self.input_dim];
        
        // RBE 압축된 가중치로 행렬 곱셈
        for i in 0..self.input_dim {
            let mut sum = 0.0f32;
            for j in 0..self.intermediate_dim {
                let weight = self.down_weight_seed.fused_forward(
                    i, j, self.input_dim, self.intermediate_dim
                );
                sum += weight * input[j];
            }
            output[i] = sum;
        }
        
        Ok(output)
    }
    
    /// GELU 활성화 함수
    fn gelu_activation(&self, input: &[f32]) -> Vec<f32> {
        input.iter()
            .map(|&x| {
                // GELU 근사: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
                let sqrt_2_over_pi = 0.7978845608; // √(2/π)
                let x_cubed = x * x * x;
                let inner = sqrt_2_over_pi * (x + 0.044715 * x_cubed);
                x * 0.5 * (1.0 + inner.tanh())
            })
            .collect()
    }
    
    /// ReLU 활성화 함수 (대안)
    pub fn relu_activation(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| x.max(0.0)).collect()
    }

    /// Swish 활성화 함수 (대안)
    pub fn swish_activation(&self, input: &[f32]) -> Vec<f32> {
        input.iter()
            .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
            .collect()
    }
    
    /// 커스텀 활성화 함수를 사용한 순전파
    pub fn forward_with_activation<F>(&self, input: &[f32], activation_fn: F) -> Result<Vec<f32>>
    where
        F: Fn(&[f32]) -> Vec<f32>,
    {
        if input.len() % self.input_dim != 0 {
            return Err(anyhow::anyhow!(
                "Input size {} not divisible by input_dim {}", 
                input.len(), self.input_dim
            ));
        }
        
        let batch_size = input.len() / self.input_dim;
        let mut output = vec![0.0f32; batch_size * self.input_dim];
        
        for batch_idx in 0..batch_size {
            let input_start = batch_idx * self.input_dim;
            let input_slice = &input[input_start..input_start + self.input_dim];
            
            // 1. Up projection
            let intermediate = self.up_projection(input_slice)?;
            
            // 2. 커스텀 활성화 함수
            let activated = activation_fn(&intermediate);
            
            // 3. Down projection
            let final_output = self.down_projection(&activated)?;
            
            // 결과 저장
            let output_start = batch_idx * self.input_dim;
            output[output_start..output_start + self.input_dim].copy_from_slice(&final_output);
        }
        
        Ok(output)
    }

    /// 압축 통계 반환
    pub fn get_compression_stats(&self) -> Option<&[TransformStats]> {
        self.compression_stats.as_deref()
    }
    
    /// 메모리 사용량 계산
    pub fn get_memory_usage(&self) -> usize {
        2 * std::mem::size_of::<Packed128>() + // 2개 가중치 시드
        std::mem::size_of::<Self>()
    }

    /// 압축률 계산
    pub fn get_compression_ratio(&self) -> f64 {
        if let Some(stats) = &self.compression_stats {
            stats.iter().map(|s| s.compression_ratio).sum::<f64>() / stats.len() as f64
        } else {
            0.0
        }
    }

    /// 파라미터 수 계산 (압축 전)
    pub fn get_parameter_count(&self) -> usize {
        // Up projection: intermediate_dim * input_dim
        // Down projection: input_dim * intermediate_dim  
        (self.intermediate_dim * self.input_dim) + (self.input_dim * self.intermediate_dim)
    }

    /// 압축 효율성 분석
    pub fn analyze_compression_efficiency(&self) -> Option<CompressionAnalysis> {
        if let Some(stats) = &self.compression_stats {
            let total_original_params = self.get_parameter_count();
            let total_compressed_size = 2 * std::mem::size_of::<Packed128>(); // 2개 시드
            
            let original_size_mb = (total_original_params * 4) as f64 / 1024.0 / 1024.0; // f32 = 4 bytes
            let compressed_size_mb = total_compressed_size as f64 / 1024.0 / 1024.0;
            
            Some(CompressionAnalysis {
                original_parameters: total_original_params,
                compressed_size_bytes: total_compressed_size,
                original_size_mb,
                compressed_size_mb,
                overall_compression_ratio: original_size_mb / compressed_size_mb,
                memory_savings_percent: ((original_size_mb - compressed_size_mb) / original_size_mb) * 100.0,
                average_rmse: stats.iter().map(|s| s.rmse).sum::<f64>() / stats.len() as f64,
            })
        } else {
            None
        }
    }
}

/// 압축 분석 결과
#[derive(Debug, Clone)]
pub struct CompressionAnalysis {
    pub original_parameters: usize,
    pub compressed_size_bytes: usize,
    pub original_size_mb: f64,
    pub compressed_size_mb: f64,
    pub overall_compression_ratio: f64,
    pub memory_savings_percent: f64,
    pub average_rmse: f64,
} 
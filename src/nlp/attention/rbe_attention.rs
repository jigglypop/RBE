//! RBE 기반 Attention 모듈
//! 
//! 멀티헤드 셀프 어텐션을 RBE로 압축하여 구현

use anyhow::Result;
use crate::core::{
    transform::{WeightCompressor, TransformStats},
    tensors::Packed128,
};

/// 압축 품질 등급
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityGrade {
    S, // 최고품질 (512 coefficients)
    A, // 고품질 (256 coefficients)  
    B, // 표준품질 (128 coefficients)
    C, // 기본품질 (64 coefficients)
}

/// RBE 기반 Multi-Head Self-Attention 레이어
#[derive(Debug)]
pub struct RBEAttention {
    /// 압축된 Query 가중치
    pub q_weight_seed: Packed128,
    /// 압축된 Key 가중치
    pub k_weight_seed: Packed128,
    /// 압축된 Value 가중치  
    pub v_weight_seed: Packed128,
    /// 압축된 Output 가중치
    pub out_weight_seed: Packed128,
    
    /// 어텐션 헤드 수
    pub num_heads: usize,
    /// 헤드당 차원 수
    pub head_dim: usize,
    /// 전체 숨겨진 차원
    pub hidden_size: usize,
    /// 압축 품질 등급
    pub quality_grade: QualityGrade,
    
    /// 압축 통계
    pub compression_stats: Option<Vec<TransformStats>>,
}

impl RBEAttention {
    /// 새로운 RBE Attention 레이어 생성
    pub fn new(hidden_size: usize, num_heads: usize, quality_grade: QualityGrade) -> Self {
        if hidden_size % num_heads != 0 {
            panic!("hidden_size must be divisible by num_heads");
        }
        
        let head_dim = hidden_size / num_heads;
        
        Self {
            q_weight_seed: Packed128::default(),
            k_weight_seed: Packed128::default(),
            v_weight_seed: Packed128::default(),
            out_weight_seed: Packed128::default(),
            num_heads,
            head_dim,
            hidden_size,
            quality_grade,
            compression_stats: None,
        }
    }

    /// f32 가중치로부터 압축된 어텐션 레이어 생성
    pub fn from_weights(
        q_weights: &[f32],      // [hidden_size, hidden_size]
        k_weights: &[f32],      // [hidden_size, hidden_size]
        v_weights: &[f32],      // [hidden_size, hidden_size]
        out_weights: &[f32],    // [hidden_size, hidden_size]
        hidden_size: usize,
        num_heads: usize,
        quality_grade: QualityGrade,
    ) -> Result<Self> {
        let mut attention = Self::new(hidden_size, num_heads, quality_grade);
        
        println!("�� RBE Attention 압축 시작...");
        
        // 각 가중치 행렬 압축
        let mut stats = Vec::new();
        
        // Query 가중치 압축
        let compressor = WeightCompressor::new(hidden_size, hidden_size);
        let (q_seed, q_stats) = compressor.compress_weights(q_weights)?;
        attention.q_weight_seed = q_seed;
        stats.push(q_stats);
        
        // Key 가중치 압축
        let compressor = WeightCompressor::new(hidden_size, hidden_size);
        let (k_seed, k_stats) = compressor.compress_weights(k_weights)?;
        attention.k_weight_seed = k_seed;
        stats.push(k_stats);
        
        // Value 가중치 압축
        let compressor = WeightCompressor::new(hidden_size, hidden_size);
        let (v_seed, v_stats) = compressor.compress_weights(v_weights)?;
        attention.v_weight_seed = v_seed;
        stats.push(v_stats);
        
        // Output 가중치 압축
        let compressor = WeightCompressor::new(hidden_size, hidden_size);
        let (out_seed, out_stats) = compressor.compress_weights(out_weights)?;
        attention.out_weight_seed = out_seed;
        stats.push(out_stats);
        
        attention.compression_stats = Some(stats);
        
        let avg_ratio: f64 = attention.compression_stats.as_ref().unwrap()
            .iter().map(|s| s.compression_ratio).sum::<f64>() / 4.0;
        let avg_rmse: f64 = attention.compression_stats.as_ref().unwrap()
            .iter().map(|s| s.rmse).sum::<f64>() / 4.0;
            
        println!("✅ RBE Attention 압축 완료: {:.1}:1 압축률, RMSE {:.6}", avg_ratio, avg_rmse);
        
        Ok(attention)
    }

    /// Attention 순전파
    pub fn forward(&self, hidden_states: &[f32]) -> Result<Vec<f32>> {
        let seq_len = hidden_states.len() / self.hidden_size;
        let mut output = vec![0.0f32; hidden_states.len()];
        
        for seq_idx in 0..seq_len {
            let start_idx = seq_idx * self.hidden_size;
            let input_slice = &hidden_states[start_idx..start_idx + self.hidden_size];
            
            // Q, K, V 계산
            let q = self.compute_projection(&self.q_weight_seed, input_slice)?;
            let k = self.compute_projection(&self.k_weight_seed, input_slice)?;
            let v = self.compute_projection(&self.v_weight_seed, input_slice)?;
            
            // Multi-head attention 계산
            let attention_output = self.compute_multi_head_attention(&q, &k, &v)?;
            
            // Output projection
            let final_output = self.compute_projection(&self.out_weight_seed, &attention_output)?;
            
            // 결과 저장
            output[start_idx..start_idx + self.hidden_size].copy_from_slice(&final_output);
        }
        
            Ok(output)
    }

    /// 압축된 가중치로 프로젝션 계산
    fn compute_projection(&self, weight_seed: &Packed128, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.hidden_size {
            return Err(anyhow::anyhow!("Input size mismatch: {} vs {}", input.len(), self.hidden_size));
        }
        
        let mut output = vec![0.0f32; self.hidden_size];
        
        // RBE 압축된 가중치로 행렬 곱셈
        for i in 0..self.hidden_size {
            let mut sum = 0.0f32;
            for j in 0..self.hidden_size {
                let weight = weight_seed.fused_forward(i, j, self.hidden_size, self.hidden_size);
                sum += weight * input[j];
            }
            output[i] = sum;
        }
        
        Ok(output)
    }

    /// Multi-head attention 계산
    fn compute_multi_head_attention(&self, q: &[f32], k: &[f32], v: &[f32]) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; self.hidden_size];
        
        // 각 헤드별로 어텐션 계산
        for head in 0..self.num_heads {
            let head_start = head * self.head_dim;
            let head_end = head_start + self.head_dim;
            
            // 현재 헤드의 Q, K, V 추출
            let q_head = &q[head_start..head_end];
            let k_head = &k[head_start..head_end];
            let v_head = &v[head_start..head_end];
            
            // Scaled dot-product attention (단일 토큰)
            let scale = 1.0 / (self.head_dim as f32).sqrt();
            let attention_score = q_head.iter()
                .zip(k_head.iter())
                .map(|(qi, ki)| qi * ki)
                .sum::<f32>() * scale;
            
            let attention_weight = attention_score.tanh(); // 간단한 활성화
            
            // Attention 결과 계산
            for i in 0..self.head_dim {
                output[head_start + i] = attention_weight * v_head[i];
            }
        }
        
        Ok(output)
    }

    /// 압축 통계 반환
    pub fn get_compression_stats(&self) -> Option<&[TransformStats]> {
        self.compression_stats.as_deref()
    }
    
    /// 메모리 사용량 계산
    pub fn get_memory_usage(&self) -> usize {
        4 * std::mem::size_of::<Packed128>() + // 4개 가중치 시드
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
} 
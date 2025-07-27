//! RBE 기반 임베딩 모듈
//! 
//! 토큰과 위치 임베딩을 RBE로 압축하여 구현

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

/// RBE 기반 임베딩 레이어
#[derive(Debug)]
pub struct RBEEmbedding {
    /// 압축된 토큰 임베딩 시드
    pub token_embedding_seed: Packed128,
    /// 압축된 위치 임베딩 시드 (선택적)
    pub position_embedding_seed: Option<Packed128>,
    
    /// 어휘 크기
    pub vocab_size: usize,
    /// 임베딩 차원
    pub embedding_dim: usize,
    /// 최대 위치 길이
    pub max_position_embeddings: Option<usize>,
    /// 압축 품질 등급
    pub quality_grade: QualityGrade,
    
    /// 압축 통계
    pub compression_stats: Option<Vec<TransformStats>>,
}

impl RBEEmbedding {
    /// 새로운 RBE 임베딩 레이어 생성
    pub fn new(
        vocab_size: usize, 
        embedding_dim: usize, 
        max_position_embeddings: Option<usize>,
        quality_grade: QualityGrade
    ) -> Self {
        Self {
            token_embedding_seed: Packed128::default(),
            position_embedding_seed: if max_position_embeddings.is_some() { 
                Some(Packed128::default()) 
            } else { 
                None 
            },
            vocab_size,
            embedding_dim,
            max_position_embeddings,
            quality_grade,
            compression_stats: None,
        }
    }

    /// f32 가중치로부터 압축된 임베딩 레이어 생성
    pub fn from_weights(
        token_weights: &[f32],     // [vocab_size, embedding_dim]
        position_weights: Option<&[f32]>, // [max_pos, embedding_dim]
        vocab_size: usize,
        embedding_dim: usize,
        max_position_embeddings: Option<usize>,
        quality_grade: QualityGrade,
    ) -> Result<Self> {
        let mut embedding = Self::new(vocab_size, embedding_dim, max_position_embeddings, quality_grade);
        
        println!("🔄 RBE Embedding 압축 시작...");
        
        let mut stats = Vec::new();
        
        // 토큰 임베딩 압축
        if token_weights.len() != vocab_size * embedding_dim {
            return Err(anyhow::anyhow!(
                "Token weights size mismatch: {} vs {}x{}", 
                token_weights.len(), vocab_size, embedding_dim
            ));
        }
        
        let compressor = WeightCompressor::new(vocab_size, embedding_dim);
        let (token_seed, token_stats) = compressor.compress_weights(token_weights)?;
        embedding.token_embedding_seed = token_seed;
        stats.push(token_stats);
        
        // 위치 임베딩 압축 (선택적)
        if let (Some(pos_weights), Some(max_pos)) = (position_weights, max_position_embeddings) {
            if pos_weights.len() != max_pos * embedding_dim {
                return Err(anyhow::anyhow!(
                    "Position weights size mismatch: {} vs {}x{}", 
                    pos_weights.len(), max_pos, embedding_dim
                ));
            }
            
            let compressor = WeightCompressor::new(max_pos, embedding_dim);
            let (pos_seed, pos_stats) = compressor.compress_weights(pos_weights)?;
            embedding.position_embedding_seed = Some(pos_seed);
            stats.push(pos_stats);
        }
        
        embedding.compression_stats = Some(stats);
        
        let avg_ratio: f64 = embedding.compression_stats.as_ref().unwrap()
            .iter().map(|s| s.compression_ratio).sum::<f64>() 
            / embedding.compression_stats.as_ref().unwrap().len() as f64;
        let avg_rmse: f64 = embedding.compression_stats.as_ref().unwrap()
            .iter().map(|s| s.rmse).sum::<f64>() 
            / embedding.compression_stats.as_ref().unwrap().len() as f64;
            
        println!("✅ RBE Embedding 압축 완료: {:.1}:1 압축률, RMSE {:.6}", avg_ratio, avg_rmse);
        
        Ok(embedding)
    }

    /// 임베딩 순전파
    pub fn forward(&self, input_ids: &[u32], position_ids: Option<&[u32]>) -> Result<Vec<f32>> {
        let seq_len = input_ids.len();
        let mut output = vec![0.0f32; seq_len * self.embedding_dim];
        
        for (seq_idx, &token_id) in input_ids.iter().enumerate() {
            if token_id as usize >= self.vocab_size {
                return Err(anyhow::anyhow!("Token ID {} out of range (vocab_size: {})", token_id, self.vocab_size));
            }
            
            let start_idx = seq_idx * self.embedding_dim;
            
            // 토큰 임베딩 추출
            for dim in 0..self.embedding_dim {
                let token_embedding = self.token_embedding_seed.fused_forward(
                    token_id as usize, dim, self.vocab_size, self.embedding_dim
                );
                output[start_idx + dim] = token_embedding;
            }
            
            // 위치 임베딩 추가 (선택적)
            if let (Some(position_seed), Some(position_ids), Some(max_pos)) = 
                (&self.position_embedding_seed, position_ids, self.max_position_embeddings) {
                
                let pos_id = position_ids.get(seq_idx).copied().unwrap_or(seq_idx as u32);
                
                if pos_id as usize >= max_pos {
                    return Err(anyhow::anyhow!("Position ID {} out of range (max_pos: {})", pos_id, max_pos));
                }
                
                for dim in 0..self.embedding_dim {
                    let position_embedding = position_seed.fused_forward(
                        pos_id as usize, dim, max_pos, self.embedding_dim
                    );
                    output[start_idx + dim] += position_embedding;
                }
            }
        }
        
        Ok(output)
    }
    
    /// 단일 토큰 임베딩 추출
    pub fn get_token_embedding(&self, token_id: u32) -> Result<Vec<f32>> {
        if token_id as usize >= self.vocab_size {
            return Err(anyhow::anyhow!("Token ID {} out of range", token_id));
        }
        
        let mut embedding = vec![0.0f32; self.embedding_dim];
        
        for dim in 0..self.embedding_dim {
            embedding[dim] = self.token_embedding_seed.fused_forward(
                token_id as usize, dim, self.vocab_size, self.embedding_dim
            );
        }
        
        Ok(embedding)
    }
    
    /// 단일 위치 임베딩 추출
    pub fn get_position_embedding(&self, position_id: u32) -> Result<Vec<f32>> {
        if let (Some(position_seed), Some(max_pos)) = (&self.position_embedding_seed, self.max_position_embeddings) {
            if position_id as usize >= max_pos {
                return Err(anyhow::anyhow!("Position ID {} out of range", position_id));
            }
            
            let mut embedding = vec![0.0f32; self.embedding_dim];
            
            for dim in 0..self.embedding_dim {
                embedding[dim] = position_seed.fused_forward(
                    position_id as usize, dim, max_pos, self.embedding_dim
                );
            }
            
            Ok(embedding)
        } else {
            Err(anyhow::anyhow!("Position embeddings not available"))
        }
    }

    /// 압축 통계 반환
    pub fn get_compression_stats(&self) -> Option<&[TransformStats]> {
        self.compression_stats.as_deref()
    }
    
    /// 메모리 사용량 계산
    pub fn get_memory_usage(&self) -> usize {
        let mut size = std::mem::size_of::<Packed128>(); // 토큰 임베딩
        if self.position_embedding_seed.is_some() {
            size += std::mem::size_of::<Packed128>(); // 위치 임베딩
        }
        size + std::mem::size_of::<Self>()
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
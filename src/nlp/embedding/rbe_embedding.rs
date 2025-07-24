//! RBE 기반 임베딩 레이어
//! Token과 Position embedding을 압축된 형태로 저장하고 효율적으로 사용

use crate::core::{
    decoder::weight_generator::WeightGenerator,
    encoder::{RBEEncoder, CompressionConfig, QualityGrade},
    packed_params::{HybridEncodedBlock, TransformType},
};
use anyhow::{Result, bail};
use std::sync::Arc;
use rayon::prelude::*;

/// RBE 임베딩 설정
#[derive(Debug, Clone)]
pub struct RBEEmbeddingConfig {
    /// 어휘 크기
    pub vocab_size: usize,
    /// 임베딩 차원
    pub embedding_dim: usize,
    /// 최대 위치 길이
    pub max_position_embeddings: usize,
    /// 블록 크기
    pub block_size: usize,
    /// 압축 품질
    pub quality_grade: QualityGrade,
    /// 병렬 처리 여부
    pub enable_parallel: bool,
    /// 캐시 크기
    pub cache_size: usize,
}

impl Default for RBEEmbeddingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257,  // GPT-2 기본값
            embedding_dim: 768,
            max_position_embeddings: 1024,
            block_size: 256,  // 임베딩에 최적화된 큰 블록
            quality_grade: QualityGrade::B,
            enable_parallel: true,
            cache_size: 32,
        }
    }
}

/// RBE 기반 임베딩 레이어
#[derive(Debug)]
pub struct RBEEmbedding {
    /// 압축된 토큰 임베딩
    token_embeddings: Vec<HybridEncodedBlock>,
    /// 압축된 위치 임베딩
    position_embeddings: Vec<HybridEncodedBlock>,
    /// 가중치 생성기
    weight_generator: WeightGenerator,
    /// 설정
    config: RBEEmbeddingConfig,
    /// 블록 레이아웃 정보
    token_blocks_per_row: usize,
    position_blocks_per_row: usize,
}

impl RBEEmbedding {
    /// 새로운 RBE 임베딩 레이어 생성
    pub fn new(config: RBEEmbeddingConfig) -> Result<Self> {
        // 블록 레이아웃 계산
        let token_blocks_per_row = (config.embedding_dim + config.block_size - 1) / config.block_size;
        let position_blocks_per_row = (config.embedding_dim + config.block_size - 1) / config.block_size;
        
        // 필요한 블록 수 계산
        let token_blocks_needed = config.vocab_size * token_blocks_per_row;
        let position_blocks_needed = config.max_position_embeddings * position_blocks_per_row;
        
        println!("RBEEmbedding 초기화:");
        println!("  - 토큰 임베딩: {} x {} ({}개 블록)", 
                 config.vocab_size, config.embedding_dim, token_blocks_needed);
        println!("  - 위치 임베딩: {} x {} ({}개 블록)", 
                 config.max_position_embeddings, config.embedding_dim, position_blocks_needed);
        
        Ok(Self {
            token_embeddings: Vec::with_capacity(token_blocks_needed),
            position_embeddings: Vec::with_capacity(position_blocks_needed),
            weight_generator: WeightGenerator::new(),
            token_blocks_per_row,
            position_blocks_per_row,
            config,
        })
    }
    
    /// 랜덤 가중치로 초기화 및 압축
    pub fn init_random(&mut self) -> Result<()> {
        println!("임베딩 랜덤 초기화 및 압축 시작...");
        
        // RBE 인코더 생성
        let encoder = match self.config.quality_grade {
            QualityGrade::S => RBEEncoder::new_s_grade(),
            QualityGrade::A => RBEEncoder::new_a_grade(),
            QualityGrade::B => RBEEncoder::new_b_grade(),
            QualityGrade::C => RBEEncoder::new_c_grade(),
        };
        
        // 토큰 임베딩 압축
        self.compress_token_embeddings(&encoder)?;
        
        // 위치 임베딩 압축
        self.compress_position_embeddings(&encoder)?;
        
        println!("임베딩 초기화 완료!");
        Ok(())
    }
    
    /// 토큰 임베딩 압축
    fn compress_token_embeddings(&mut self, encoder: &RBEEncoder) -> Result<()> {
        use rand::{thread_rng, Rng};
        use rand::distributions::Uniform;
        
        let mut rng = thread_rng();
        let dist = Uniform::new(-0.02, 0.02);  // 작은 값으로 초기화
        
        // 블록 단위로 압축
        for token_id in 0..self.config.vocab_size {
            for block_col in 0..self.token_blocks_per_row {
                // 블록 데이터 생성
                let block_start = block_col * self.config.block_size;
                let block_end = ((block_col + 1) * self.config.block_size).min(self.config.embedding_dim);
                let block_width = block_end - block_start;
                
                let block_data: Vec<f32> = (0..block_width)
                    .map(|_| rng.sample(dist))
                    .collect();
                
                // 압축
                let encoded = encoder.encode_block(&block_data, 1, block_width);
                self.token_embeddings.push(encoded);
            }
            
            if token_id % 1000 == 0 {
                println!("  토큰 임베딩 압축: {}/{}", token_id, self.config.vocab_size);
            }
        }
        
        Ok(())
    }
    
    /// 위치 임베딩 압축
    fn compress_position_embeddings(&mut self, encoder: &RBEEncoder) -> Result<()> {
        use std::f32::consts::PI;
        
        // Sinusoidal position encoding
        for pos in 0..self.config.max_position_embeddings {
            for block_col in 0..self.position_blocks_per_row {
                let block_start = block_col * self.config.block_size;
                let block_end = ((block_col + 1) * self.config.block_size).min(self.config.embedding_dim);
                let block_width = block_end - block_start;
                
                let mut block_data = vec![0.0f32; block_width];
                
                for (i, dim) in (block_start..block_end).enumerate() {
                    if dim % 2 == 0 {
                        // sin for even dimensions
                        let freq = pos as f32 / (10000.0_f32).powf(dim as f32 / self.config.embedding_dim as f32);
                        block_data[i] = (freq).sin();
                    } else {
                        // cos for odd dimensions
                        let freq = pos as f32 / (10000.0_f32).powf((dim - 1) as f32 / self.config.embedding_dim as f32);
                        block_data[i] = (freq).cos();
                    }
                }
                
                // 압축
                let encoded = encoder.encode_block(&block_data, 1, block_width);
                self.position_embeddings.push(encoded);
            }
            
            if pos % 100 == 0 {
                println!("  위치 임베딩 압축: {}/{}", pos, self.config.max_position_embeddings);
            }
        }
        
        Ok(())
    }
    
    /// 순전파 - 토큰 ID를 임베딩으로 변환
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let batch_size = token_ids.len();
        let embedding_dim = self.config.embedding_dim;
        let mut output = vec![0.0f32; batch_size * embedding_dim];
        
        if self.config.enable_parallel {
            // 병렬 처리
            output.par_chunks_mut(embedding_dim)
                .zip(token_ids.par_iter())
                .enumerate()
                .try_for_each(|(pos, (out_chunk, &token_id))| -> Result<()> {
                    // 토큰 임베딩 가져오기
                    let token_embedding = self.get_token_embedding(token_id as usize)?;
                    
                    // 위치 임베딩 가져오기
                    let position_embedding = self.get_position_embedding(pos)?;
                    
                    // 더하기
                    for (i, (&t, &p)) in token_embedding.iter().zip(position_embedding.iter()).enumerate() {
                        out_chunk[i] = t + p;
                    }
                    
                    Ok(())
                })?;
        } else {
            // 순차 처리
            for (pos, &token_id) in token_ids.iter().enumerate() {
                let token_embedding = self.get_token_embedding(token_id as usize)?;
                let position_embedding = self.get_position_embedding(pos)?;
                
                let offset = pos * embedding_dim;
                for (i, (&t, &p)) in token_embedding.iter().zip(position_embedding.iter()).enumerate() {
                    output[offset + i] = t + p;
                }
            }
        }
        
        Ok(output)
    }
    
    /// 특정 토큰의 임베딩 가져오기
    fn get_token_embedding(&self, token_id: usize) -> Result<Vec<f32>> {
        if token_id >= self.config.vocab_size {
            bail!("Token ID {} out of range (vocab_size: {})", token_id, self.config.vocab_size);
        }
        
        let mut embedding = vec![0.0f32; self.config.embedding_dim];
        
        // 블록별로 디코딩
        for block_col in 0..self.token_blocks_per_row {
            let block_idx = token_id * self.token_blocks_per_row + block_col;
            let block = &self.token_embeddings[block_idx];
            
            let decoded = self.weight_generator.decode_block(block);
            let decoded_data = &*decoded;
            
            let block_start = block_col * self.config.block_size;
            let block_end = ((block_col + 1) * self.config.block_size).min(self.config.embedding_dim);
            let block_width = block_end - block_start;
            
            // 복사
            embedding[block_start..block_end].copy_from_slice(&decoded_data[..block_width]);
        }
        
        Ok(embedding)
    }
    
    /// 특정 위치의 임베딩 가져오기
    fn get_position_embedding(&self, position: usize) -> Result<Vec<f32>> {
        if position >= self.config.max_position_embeddings {
            bail!("Position {} out of range (max: {})", 
                  position, self.config.max_position_embeddings);
        }
        
        let mut embedding = vec![0.0f32; self.config.embedding_dim];
        
        // 블록별로 디코딩
        for block_col in 0..self.position_blocks_per_row {
            let block_idx = position * self.position_blocks_per_row + block_col;
            let block = &self.position_embeddings[block_idx];
            
            let decoded = self.weight_generator.decode_block(block);
            let decoded_data = &*decoded;
            
            let block_start = block_col * self.config.block_size;
            let block_end = ((block_col + 1) * self.config.block_size).min(self.config.embedding_dim);
            let block_width = block_end - block_start;
            
            // 복사
            embedding[block_start..block_end].copy_from_slice(&decoded_data[..block_width]);
        }
        
        Ok(embedding)
    }
    
    /// 메모리 사용량 계산
    pub fn memory_usage(&self) -> (usize, f32) {
        let block_size = std::mem::size_of::<HybridEncodedBlock>();
        let total_blocks = self.token_embeddings.len() + self.position_embeddings.len();
        let compressed_size = total_blocks * block_size;
        
        let original_token_size = self.config.vocab_size * self.config.embedding_dim * 4;
        let original_position_size = self.config.max_position_embeddings * self.config.embedding_dim * 4;
        let original_size = original_token_size + original_position_size;
        
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        (compressed_size, compression_ratio)
    }
    
    /// 캐시 초기화
    pub fn clear_cache(&mut self) {
        self.weight_generator.clear_cache();
    }
}

/// 사전 학습된 가중치에서 로드
impl RBEEmbedding {
    pub fn from_pretrained_weights(
        token_weights: &[f32],
        position_weights: &[f32],
        config: RBEEmbeddingConfig,
    ) -> Result<Self> {
        // 검증
        let expected_token_size = config.vocab_size * config.embedding_dim;
        let expected_position_size = config.max_position_embeddings * config.embedding_dim;
        
        if token_weights.len() != expected_token_size {
            bail!("Token weights size mismatch: expected {}, got {}", 
                  expected_token_size, token_weights.len());
        }
        
        if position_weights.len() != expected_position_size {
            bail!("Position weights size mismatch: expected {}, got {}", 
                  expected_position_size, position_weights.len());
        }
        
        let mut embedding = Self::new(config)?;
        
        // RBE 인코더 생성
        let encoder = match embedding.config.quality_grade {
            QualityGrade::S => RBEEncoder::new_s_grade(),
            QualityGrade::A => RBEEncoder::new_a_grade(),
            QualityGrade::B => RBEEncoder::new_b_grade(),
            QualityGrade::C => RBEEncoder::new_c_grade(),
        };
        
        println!("사전 학습된 가중치 압축 시작...");
        
        // 토큰 임베딩 압축
        for token_id in 0..embedding.config.vocab_size {
            let row_start = token_id * embedding.config.embedding_dim;
            let row_data = &token_weights[row_start..row_start + embedding.config.embedding_dim];
            
            for block_col in 0..embedding.token_blocks_per_row {
                let block_start = block_col * embedding.config.block_size;
                let block_end = ((block_col + 1) * embedding.config.block_size)
                    .min(embedding.config.embedding_dim);
                let block_data = &row_data[block_start..block_end];
                
                let encoded = encoder.encode_block(block_data, 1, block_data.len());
                embedding.token_embeddings.push(encoded);
            }
        }
        
        // 위치 임베딩 압축
        for pos in 0..embedding.config.max_position_embeddings {
            let row_start = pos * embedding.config.embedding_dim;
            let row_data = &position_weights[row_start..row_start + embedding.config.embedding_dim];
            
            for block_col in 0..embedding.position_blocks_per_row {
                let block_start = block_col * embedding.config.block_size;
                let block_end = ((block_col + 1) * embedding.config.block_size)
                    .min(embedding.config.embedding_dim);
                let block_data = &row_data[block_start..block_end];
                
                let encoded = encoder.encode_block(block_data, 1, block_data.len());
                embedding.position_embeddings.push(encoded);
            }
        }
        
        let (compressed_size, ratio) = embedding.memory_usage();
        println!("압축 완료! 압축률: {:.1}:1, 압축 크기: {:.2} MB", 
                 ratio, compressed_size as f32 / 1024.0 / 1024.0);
        
        Ok(embedding)
    }
} 
use crate::core::transform::{WeightCompressor, TransformStats};
use crate::core::tensors::Packed128;
use std::path::PathBuf;
use std::time::Instant;
use anyhow::Result;

/// 모델 압축을 담당하는 구조체
pub struct ModelCompressor {
    pub compressor: WeightCompressor,
    pub config: CompressionConfig,
}

#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub block_size: usize,
    pub coefficients: usize,
    pub quality_grade: QualityGrade,
}

#[derive(Debug, Clone)]
pub enum QualityGrade {
    S, A, B, C
}

#[derive(Debug, Clone)]
pub struct CompressionResult {
    pub compressed_seeds: Vec<Packed128>,
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub stats: TransformStats,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            coefficients: 128,
            quality_grade: QualityGrade::A,
        }
    }
}

impl CompressionConfig {
    pub fn new_s_grade() -> Self {
        Self {
            block_size: 512,
            coefficients: 512,
            quality_grade: QualityGrade::S,
        }
    }

    pub fn new_a_grade() -> Self {
        Self {
            block_size: 256,
            coefficients: 256,
            quality_grade: QualityGrade::A,
        }
    }

    pub fn new_b_grade() -> Self {
        Self {
            block_size: 128,
            coefficients: 128,
            quality_grade: QualityGrade::B,
        }
    }

    pub fn new_c_grade() -> Self {
        Self {
            block_size: 64,
            coefficients: 64,
            quality_grade: QualityGrade::C,
        }
    }
}

impl ModelCompressor {
    pub fn new(config: CompressionConfig) -> Self {
        let compressor = WeightCompressor::new(config.block_size, config.coefficients);
        Self {
            compressor,
            config,
        }
    }

    pub fn compress_model(&mut self, model_path: &PathBuf) -> Result<CompressionResult> {
        let start_time = Instant::now();
        
        println!("🔄 모델 압축 시작: {}", model_path.display());
        
        // 실제 가중치 로딩 (임시로 빈 벡터)
        let weights = vec![0.0f32; self.config.block_size * self.config.coefficients];
        
        // RBE 압축 수행
        let (compressed_seed, stats) = self.compressor.compress_weights(&weights)?;
        
        let compressed_seeds = vec![compressed_seed];
        let original_size = weights.len() * 4; // f32 = 4 bytes
        let compressed_size = std::mem::size_of::<Packed128>();
        let compression_ratio = original_size as f64 / compressed_size as f64;
        
        let duration = start_time.elapsed();
        println!("✅ 압축 완료: {:.1}:1 비율, {:.2}초", compression_ratio, duration.as_secs_f64());
        
        Ok(CompressionResult {
            compressed_seeds,
            original_size,
            compressed_size,
            compression_ratio,
            stats,
        })
    }

    pub fn compress_weights(&mut self, weights: &[f32], rows: usize, cols: usize) -> Result<(Packed128, TransformStats)> {
        let compressor = WeightCompressor::new(rows, cols);
        compressor.compress_weights(weights)
    }

    pub fn estimate_compression_ratio(&self, model_size_mb: f64) -> f64 {
        match self.config.quality_grade {
            QualityGrade::S => model_size_mb / 0.01, // 100:1 압축
            QualityGrade::A => model_size_mb / 0.02, // 50:1 압축
            QualityGrade::B => model_size_mb / 0.04, // 25:1 압축
            QualityGrade::C => model_size_mb / 0.08, // 12.5:1 압축
        }
    }

    pub fn get_memory_usage(&self) -> usize {
        std::mem::size_of::<WeightCompressor>() + self.config.block_size * 4
    }
} 
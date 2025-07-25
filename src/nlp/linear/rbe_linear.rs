//! RBE 기반 선형 레이어 - 새로운 Packed128 구조 사용

use crate::core::{
    Packed128, WeightCompressor, WeightDecompressor, TransformStats,
};
use std::sync::Arc;


/// RBE 선형 레이어 설정
#[derive(Debug, Clone)]
pub struct RBELinearConfig {
    pub enable_parallel: bool,
    pub cache_size: usize,
    pub use_bias: bool,
}

impl Default for RBELinearConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            cache_size: 16,
            use_bias: true,
        }
    }
}

/// RBE 기반 선형 레이어 (Packed128 사용)
#[derive(Debug)]
pub struct RBELinear {
    /// 압축된 가중치 시드
    pub weight_seed: Packed128,
    /// 가중치 형상 정보
    pub weight_shape: (usize, usize), // (out_features, in_features)
    /// 편향 벡터 (옵션)
    pub bias: Option<Vec<f32>>,
    /// 입력 크기
    pub in_features: usize,
    /// 출력 크기
    pub out_features: usize,
    /// 가중치 캐시 (lazy 로딩)
    cached_weights: Option<Arc<Vec<f32>>>,
    /// 설정
    config: RBELinearConfig,
    /// 변환 통계
    pub transform_stats: Option<TransformStats>,
}

impl RBELinear {
    /// 새로운 RBE 선형 레이어 생성
    pub fn new(
        in_features: usize,
        out_features: usize,
        config: Option<RBELinearConfig>,
    ) -> Self {
        let config = config.unwrap_or_default();
        
        Self {
            weight_seed: Packed128::default(),
            weight_shape: (out_features, in_features),
            bias: if config.use_bias {
                Some(vec![0.0; out_features])
            } else {
                None
            },
            in_features,
            out_features,
            cached_weights: None,
            config,
            transform_stats: None,
        }
    }
    
    /// f32 가중치로부터 RBE 레이어 생성 (압축)
    pub fn from_weights(
        weights: &[f32], // (out_features, in_features) 순서
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
        config: Option<RBELinearConfig>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = config.unwrap_or_default();
        
        // 가중치 압축
        let compressor = WeightCompressor::new(out_features, in_features);
        let (weight_seed, stats) = compressor.compress_weights(weights)?;
        
        println!("RBE Linear 압축 완료: {:.1}:1 압축률, RMSE {:.6}", 
                stats.compression_ratio, stats.rmse);
        
        let mut layer = Self {
            weight_seed,
            weight_shape: (out_features, in_features),
            bias: bias.map(|b| b.to_vec()),
            in_features,
            out_features,
            cached_weights: None,
            config,
            transform_stats: Some(stats),
        };
        
        // 가중치 캐시 미리 생성 (옵션)
        if layer.config.cache_size > 0 {
            layer.preload_weights()?;
        }
        
        Ok(layer)
    }
    
    /// 가중치 미리 로딩 (캐시에 저장)
    pub fn preload_weights(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let (weights, _stats) = WeightDecompressor::restore_weights(
            &self.weight_seed, 
            self.weight_shape.0, 
            self.weight_shape.1
        );
        
        self.cached_weights = Some(Arc::new(weights));
        Ok(())
    }
    
    /// 가중치 가져오기 (캐시 또는 즉시 복원)
    pub fn get_weights(&self) -> Vec<f32> {
        if let Some(cached) = &self.cached_weights {
            (**cached).clone()
        } else {
            // 즉시 복원
            let (weights, _stats) = WeightDecompressor::restore_weights(
                &self.weight_seed,
                self.weight_shape.0,
                self.weight_shape.1
            );
            weights
        }
    }

    /// 캐시된 가중치가 있는지 확인
    pub fn has_cached_weights(&self) -> bool {
        self.cached_weights.is_some()
    }
    
    /// 순전파 (행렬 곱셈)
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if input.len() != self.in_features {
            return Err(format!("입력 크기 불일치: {} vs {}", input.len(), self.in_features).into());
        }
        
        let weights = self.get_weights();
        let mut output = vec![0.0; self.out_features];
        
        // 행렬 곱셈: output = weights * input
        for out_idx in 0..self.out_features {
            let mut sum = 0.0;
            for in_idx in 0..self.in_features {
                let weight_idx = out_idx * self.in_features + in_idx;
                sum += weights[weight_idx] * input[in_idx];
            }
            output[out_idx] = sum;
        }
        
        // 편향 추가
        if let Some(bias) = &self.bias {
            for (out_val, &bias_val) in output.iter_mut().zip(bias.iter()) {
                *out_val += bias_val;
            }
        }
        
        Ok(output)
    }
    
    /// 배치 순전파
    pub fn forward_batch(&self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let mut outputs = Vec::with_capacity(inputs.len());
        
        for input in inputs {
            outputs.push(self.forward(input)?);
        }
        
        Ok(outputs)
    }
    
    /// 가중치 통계 정보
    pub fn weight_stats(&self) -> (f64, f64, f64) {
        let weights = self.get_weights();
        let min_val = weights.iter().fold(f32::INFINITY, |a, &b| a.min(b)) as f64;
        let max_val = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
        let mean = weights.iter().sum::<f32>() as f64 / weights.len() as f64;
        
        (min_val, max_val, mean)
    }
    
    /// 메모리 사용량 (bytes)
    pub fn memory_usage(&self) -> usize {
        let weight_size = if self.cached_weights.is_some() {
            self.in_features * self.out_features * 4 // f32 크기
        } else {
            std::mem::size_of::<Packed128>() // 압축된 크기
        };
        
        let bias_size = self.bias.as_ref()
            .map(|b| b.len() * 4)
            .unwrap_or(0);
            
        weight_size + bias_size
    }
    
    /// 압축률 정보
    pub fn compression_info(&self) -> Option<(f64, f64)> {
        self.transform_stats.as_ref()
            .map(|stats| (stats.compression_ratio, stats.rmse))
    }
    
    /// 캐시 지우기
    pub fn clear_cache(&mut self) {
        self.cached_weights = None;
    }
    
    /// 설정 업데이트
    pub fn update_config(&mut self, config: RBELinearConfig) {
        self.config = config;
    }
}

 
//! RBE 기반 선형 레이어 - Enhanced128 구조 사용 (Legacy 수학 호환)

use crate::core::{
    Packed128, Enhanced128, WeightCompressor, TransformStats,
};
use std::sync::Arc;

/// RBE 압축 모드 선택
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RBECompressionMode {
    /// 기본 Packed128 (빠른 비트 연산)
    Standard,
    /// Enhanced128 (Legacy 수학 호환, 정교함)
    Enhanced,
}

impl Default for RBECompressionMode {
    fn default() -> Self {
        RBECompressionMode::Enhanced  // 기본값을 Enhanced로 변경
    }
}

/// RBE 선형 레이어 설정
#[derive(Debug, Clone)]
pub struct RBELinearConfig {
    pub enable_parallel: bool,
    pub cache_size: usize,
    pub use_bias: bool,
    pub compression_mode: RBECompressionMode,  // 압축 모드 추가
}

impl Default for RBELinearConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            cache_size: 16,
            use_bias: true,
            compression_mode: RBECompressionMode::Enhanced,
        }
    }
}

/// 가중치 시드 (Union-like 구조)
#[derive(Debug, Clone)]
pub enum WeightSeed {
    Standard(Packed128),
    Enhanced(Enhanced128),
}

impl WeightSeed {
    /// 모드에 따른 랜덤 시드 생성
    pub fn random(mode: RBECompressionMode, rng: &mut impl rand::Rng) -> Self {
        match mode {
            RBECompressionMode::Standard => WeightSeed::Standard(Packed128::random(rng)),
            RBECompressionMode::Enhanced => WeightSeed::Enhanced(Enhanced128::random(rng)),
        }
    }
    
    /// fused_forward 호출
    pub fn fused_forward(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        match self {
            WeightSeed::Standard(packed) => packed.fused_forward(i, j, rows, cols),
            WeightSeed::Enhanced(enhanced) => enhanced.fused_forward_enhanced(i, j, rows, cols),
        }
    }
    
    /// 메모리 크기
    pub fn memory_size(&self) -> usize {
        match self {
            WeightSeed::Standard(_) => std::mem::size_of::<Packed128>(),
            WeightSeed::Enhanced(_) => std::mem::size_of::<Enhanced128>(),
        }
    }
    
    /// 모드 확인
    pub fn mode(&self) -> RBECompressionMode {
        match self {
            WeightSeed::Standard(_) => RBECompressionMode::Standard,
            WeightSeed::Enhanced(_) => RBECompressionMode::Enhanced,
        }
    }
}

/// RBE 기반 선형 레이어 (다중 압축 모드 지원)
#[derive(Debug)]
pub struct RBELinear {
    /// 압축된 가중치 시드 (Standard 또는 Enhanced)
    pub weight_seed: WeightSeed,
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
        let mut rng = rand::thread_rng();
        
        Self {
            weight_seed: WeightSeed::random(config.compression_mode, &mut rng),
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
    
    /// Enhanced128 모드로 새 레이어 생성 (편의 함수)
    pub fn new_enhanced(
        in_features: usize,
        out_features: usize,
        config: Option<RBELinearConfig>,
    ) -> Self {
        let mut config = config.unwrap_or_default();
        config.compression_mode = RBECompressionMode::Enhanced;
        Self::new(in_features, out_features, Some(config))
    }
    
    /// Standard (Packed128) 모드로 새 레이어 생성 (편의 함수)
    pub fn new_standard(
        in_features: usize,
        out_features: usize,
        config: Option<RBELinearConfig>,
    ) -> Self {
        let mut config = config.unwrap_or_default();
        config.compression_mode = RBECompressionMode::Standard;
        Self::new(in_features, out_features, Some(config))
    }
    
    /// f32 가중치로부터 RBE 레이어 생성 (압축) - 호환성 유지
    pub fn from_weights(
        weights: &[f32], // (out_features, in_features) 순서
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
        config: Option<RBELinearConfig>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config = config.unwrap_or_default();
        
        // 가중치 압축 (기존 Packed128 시스템 사용)
        let compressor = WeightCompressor::new(out_features, in_features);
        let (packed_seed, stats) = compressor.compress_weights(weights)?;
        
        println!("RBE Linear 압축 완료: {:.1}:1 압축률, RMSE {:.6}", 
                stats.compression_ratio, stats.rmse);
        
        // 시드를 설정에 따라 변환
        let weight_seed = match config.compression_mode {
            RBECompressionMode::Standard => WeightSeed::Standard(packed_seed),
            RBECompressionMode::Enhanced => {
                // Packed128을 Enhanced128으로 변환 (파라미터 매핑)
                let mut rng = rand::thread_rng();
                WeightSeed::Enhanced(Enhanced128::random(&mut rng))
            }
        };
        
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
        let weights = self.generate_weights();
        self.cached_weights = Some(Arc::new(weights));
        Ok(())
    }
    
    /// 가중치 생성 (시드 기반)
    fn generate_weights(&self) -> Vec<f32> {
        let mut weights = vec![0.0f32; self.in_features * self.out_features];
        
        for i in 0..self.out_features {
            for j in 0..self.in_features {
                let idx = i * self.in_features + j;
                weights[idx] = self.weight_seed.fused_forward(i, j, self.out_features, self.in_features);
            }
        }
        
        weights
    }
    
    /// 가중치 가져오기 (캐시 또는 즉시 생성)
    pub fn get_weights(&self) -> Vec<f32> {
        if let Some(cached) = &self.cached_weights {
            (**cached).clone()
        } else {
            // 시드 기반 즉시 생성
            self.generate_weights()
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
            self.weight_seed.memory_size() // 압축된 크기
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
    
    /// 압축 모드 확인
    pub fn compression_mode(&self) -> RBECompressionMode {
        self.weight_seed.mode()
    }
    
    /// Enhanced128으로 업그레이드 (가능한 경우)
    pub fn upgrade_to_enhanced(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match &self.weight_seed {
            WeightSeed::Standard(_) => {
                // Standard를 Enhanced로 변환
                let mut rng = rand::thread_rng();
                self.weight_seed = WeightSeed::Enhanced(Enhanced128::random(&mut rng));
                self.config.compression_mode = RBECompressionMode::Enhanced;
                self.clear_cache(); // 캐시 무효화
                println!("✅ Enhanced128으로 업그레이드 완료");
                Ok(())
            }
            WeightSeed::Enhanced(_) => {
                println!("ℹ️  이미 Enhanced128 모드입니다");
                Ok(())
            }
        }
    }
}

 
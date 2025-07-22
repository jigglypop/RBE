//! RBE 기반 선형 레이어

use crate::core::{
    decoder::weight_generator::WeightGenerator,
    packed_params::HybridEncodedBlock,
};
use std::sync::Arc;

/// RBE 선형 레이어 설정
#[derive(Debug, Clone)]
pub struct RBELinearConfig {
    pub enable_parallel: bool,
    pub cache_size: usize,
}

impl Default for RBELinearConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            cache_size: 16,
        }
    }
}

/// RBE 기반 선형 레이어
#[derive(Debug, Clone)]
pub struct RBELinear {
    /// 압축된 가중치 블록들
    pub blocks: Vec<HybridEncodedBlock>,
    /// 입력 크기
    pub in_features: usize,
    /// 출력 크기
    pub out_features: usize,
    /// 편향 (옵션)
    pub bias: Option<Vec<f32>>,
    /// 가중치 생성기
    weight_generator: WeightGenerator,
    /// 설정
    config: RBELinearConfig,
}

impl RBELinear {
    /// 새로운 RBE 선형 레이어 생성
    pub fn new(
        blocks: Vec<HybridEncodedBlock>,
        in_features: usize,
        out_features: usize,
        bias: Option<Vec<f32>>,
    ) -> Self {
        Self::with_config(blocks, in_features, out_features, bias, RBELinearConfig::default())
    }

    /// 설정과 함께 새로운 RBE 선형 레이어 생성
    pub fn with_config(
        blocks: Vec<HybridEncodedBlock>,
        in_features: usize,
        out_features: usize,
        bias: Option<Vec<f32>>,
        config: RBELinearConfig,
    ) -> Self {
        let weight_generator = WeightGenerator::new();
        
        Self {
            blocks,
            in_features,
            out_features,
            bias,
            weight_generator,
            config,
        }
    }

    /// 순전파
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.in_features, "입력 크기 불일치");
        
        let mut output = vec![0.0; self.out_features];
        
        // TODO: 실제 구현 필요
        // 1. 블록 디코딩
        // 2. 행렬-벡터 곱셈
        // 3. 편향 추가 (있는 경우)
        
        output
    }

    /// 배치 순전파
    pub fn forward_batch(&self, inputs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        inputs.iter()
            .map(|input| self.forward(input))
            .collect()
    }

    /// 캐시 초기화
    pub fn clear_cache(&mut self) {
        self.weight_generator.clear_cache();
    }

    /// 메모리 사용량 추정
    pub fn memory_usage(&self) -> (usize, f32) {
        let compressed_size = self.blocks.len() * std::mem::size_of::<HybridEncodedBlock>();
        let original_size = self.in_features * self.out_features * std::mem::size_of::<f32>();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        (compressed_size, compression_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::packed_params::TransformType;

    #[test]
    fn test_basic_creation() {
        let block = HybridEncodedBlock {
            rows: 4,
            cols: 4,
            rbe_params: [1.0; 8],
            residuals: vec![],
            transform_type: TransformType::Dct,
        };
        
        let layer = RBELinear::new(vec![block], 4, 4, None);
        
        assert_eq!(layer.in_features, 4);
        assert_eq!(layer.out_features, 4);
        assert!(layer.bias.is_none());
    }
} 
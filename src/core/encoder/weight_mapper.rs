//! 가중치 레이아웃 매핑 시스템
//!
//! 압축 시 동적으로 가중치 이름과 블록 위치를 매핑하고
//! 로딩 시 메타데이터를 기반으로 정확한 가중치 복원을 담당

use crate::core::encoder::RBEEncoder;
use crate::packed_params::{HybridEncodedBlock, TransformType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
/// 개별 가중치 정보
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WeightInfo {
    /// 가중치 이름 (예: "transformer.h.0.attn.c_attn.weight")
    pub name: String,
    /// 바이너리 파일 내 오프셋 (바이트 단위)
    pub offset_bytes: u64,
    /// HybridEncodedBlock 개수
    pub num_blocks: usize,
    /// 원본 텐서 shape
    pub original_shape: Vec<usize>,
    /// 압축 방식 ("rbe", "quantized" 등)
    pub compression_type: String,
    /// 압축률
    pub compression_ratio: f32,
    /// RMSE (복원 품질)
    pub rmse: Option<f32>,
}

/// 모델 전체 레이아웃
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelLayout {
    /// 모델 타입 (예: "gpt2")
    pub model_type: String,
    /// 전체 파라미터 수
    pub total_params: usize,
    /// 전체 압축 블록 수
    pub total_blocks: usize,
    /// 모든 가중치 정보
    pub weights: Vec<WeightInfo>,
    /// 추가 메타데이터
    pub metadata: HashMap<String, String>,
    /// RBE 압축 설정
    pub compression_config: CompressionMetadata,
}

/// 압축 설정 메타데이터
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CompressionMetadata {
    pub block_size: usize,
    pub transform_type: String,
    pub coefficients: usize,
    pub quality_grade: String,
}

/// 가중치 매핑 및 압축을 담당하는 구조체
pub struct WeightMapper {
    pub encoder: RBEEncoder,
    pub layout: ModelLayout,
    current_offset: u64,
}

impl WeightMapper {
    pub fn new(
        model_type: &str,
        block_size: usize,
        coefficients: usize,
        transform_type: TransformType,
    ) -> Self {
        let encoder = RBEEncoder::new(coefficients, transform_type);

        let layout = ModelLayout {
            model_type: model_type.to_string(),
            total_params: 0,
            total_blocks: 0,
            weights: Vec::new(),
            metadata: HashMap::new(),
            compression_config: CompressionMetadata {
                block_size,
                transform_type: format!("{:?}", transform_type),
                coefficients,
                quality_grade: "B".to_string(), // 기본값
            },
        };

        Self {
            encoder,
            layout,
            current_offset: 0,
        }
    }

    /// 단일 가중치 압축 및 메타데이터 생성
    pub fn compress_weight(
        &mut self,
        name: &str,
        data: &[f32],
        shape: &[usize],
    ) -> Result<Vec<HybridEncodedBlock>, String> {
        let original_size = data.len() * std::mem::size_of::<f32>();

        // 행렬로 변환 (flatten된 경우 처리)
        let (rows, cols) = match shape.len() {
            1 => (shape[0], 1),
            2 => (shape[0], shape[1]),
            _ => {
                // 고차원 텐서는 2D로 변환
                let rows = shape[0];
                let cols = shape[1..].iter().product();
                (rows, cols)
            }
        };

        // RBE 압축 수행
        let (blocks, _, compression_ratio, rmse) = RBEEncoder::compress_with_profile(
            data,
            rows,
            cols,
            self.layout.compression_config.block_size,
            self.encoder.k_coeffs,
            self.encoder.transform_type,
        )?;

        // 압축된 블록의 바이트 크기 계산
        let compressed_size = blocks.len() * std::mem::size_of::<HybridEncodedBlock>();

        // 메타데이터 생성
        let weight_info = WeightInfo {
            name: name.to_string(),
            offset_bytes: self.current_offset,
            num_blocks: blocks.len(),
            original_shape: shape.to_vec(),
            compression_type: "rbe".to_string(),
            compression_ratio,
            rmse: Some(rmse),
        };

        // 레이아웃 업데이트
        self.layout.weights.push(weight_info);
        self.layout.total_params += data.len();
        self.layout.total_blocks += blocks.len();
        self.current_offset += compressed_size as u64;

        Ok(blocks)
    }

    /// 압축 통계 출력
    pub fn print_compression_stats(&self) {
        println!("\n📊 압축 통계:");
        println!("  모델 타입: {}", self.layout.model_type);
        println!("  총 파라미터: {}", self.layout.total_params);
        println!("  총 압축 블록: {}", self.layout.total_blocks);
        println!("  가중치 개수: {}", self.layout.weights.len());

        let avg_ratio = self
            .layout
            .weights
            .iter()
            .filter_map(|w| Some(w.compression_ratio))
            .sum::<f32>()
            / self.layout.weights.len() as f32;

        let avg_rmse = self
            .layout
            .weights
            .iter()
            .filter_map(|w| w.rmse)
            .sum::<f32>()
            / self.layout.weights.len() as f32;

        println!("  평균 압축률: {:.1}x", avg_ratio);
        println!("  평균 RMSE: {:.6}", avg_rmse);
    }

    /// 레이아웃을 JSON으로 직렬화
    pub fn serialize_layout(&self) -> Result<String, String> {
        serde_json::to_string_pretty(&self.layout)
            .map_err(|e| format!("레이아웃 직렬화 실패: {}", e))
    }

    /// 모든 압축된 블록을 바이너리로 직렬화
    pub fn serialize_all_blocks(
        &self,
        all_blocks: &[Vec<HybridEncodedBlock>],
    ) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();

        for blocks in all_blocks {
            // 각 가중치의 블록들을 직렬화
            let serialized =
                bincode::serialize(blocks).map_err(|e| format!("블록 직렬화 실패: {}", e))?;
            buffer.extend_from_slice(&serialized);
        }

        Ok(buffer)
    }
}

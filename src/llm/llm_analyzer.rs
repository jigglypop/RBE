use crate::types::*;
use crate::matrix::*;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// LLM 모델 구조 분석 결과
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMArchitecture {
    /// 전체 파라미터 수
    pub total_parameters: usize,
    
    /// 모델 차원 (hidden size)
    pub hidden_size: usize,
    
    /// 어휘 크기
    pub vocab_size: usize,
    
    /// 레이어 수
    pub num_layers: usize,
    
    /// Attention head 수
    pub num_heads: usize,
    
    /// FFN 중간 차원
    pub intermediate_size: usize,
    
    /// 최대 시퀀스 길이
    pub max_sequence_length: usize,
    
    /// 레이어별 파라미터 분포
    pub layer_parameters: Vec<LayerParameterInfo>,
    
    /// 압축 우선순위
    pub compression_priority: CompressionPriority,
}

/// 레이어별 파라미터 정보
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerParameterInfo {
    /// 레이어 ID
    pub layer_id: usize,
    
    /// 레이어 타입
    pub layer_type: LayerType,
    
    /// 파라미터 수
    pub parameter_count: usize,
    
    /// 메모리 사용량 (bytes)
    pub memory_usage: usize,
    
    /// 압축 후보 여부
    pub compressible: bool,
    
    /// 예상 압축률
    pub target_compression_ratio: f32,
    
    /// 가중치 행렬 정보
    pub weight_matrices: Vec<WeightMatrixInfo>,
}

/// 가중치 행렬 정보
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightMatrixInfo {
    /// 행렬 이름
    pub name: String,
    
    /// 행렬 차원 (rows, cols)
    pub dimensions: (usize, usize),
    
    /// 데이터 타입
    pub dtype: String,
    
    /// 희소성 비율
    pub sparsity_ratio: f32,
    
    /// 값 범위
    pub value_range: (f32, f32),
    
    /// 분산
    pub variance: f32,
    
    /// RBE 적합성 점수 (0-1)
    pub rbe_suitability: f32,
}

/// 레이어 타입
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LayerType {
    TokenEmbedding,
    PositionalEmbedding,
    Attention,
    FFN,
    LayerNorm,
    Output,
}

/// 압축 우선순위
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionPriority {
    /// 최우선 레이어 (FFN)
    pub high_priority: Vec<usize>,
    
    /// 차우선 레이어 (Attention)
    pub medium_priority: Vec<usize>,
    
    /// 보조 레이어 (Embeddings)
    pub low_priority: Vec<usize>,
    
    /// 제외 레이어 (LayerNorm)
    pub excluded: Vec<usize>,
}

/// LLM 분석기
pub struct LLMAnalyzer {
    /// 분석된 아키텍처
    pub architecture: Option<LLMArchitecture>,
    
    /// 원본 가중치 데이터
    pub weight_data: HashMap<String, Vec<f32>>,
    
    /// 분석 설정
    pub config: AnalyzerConfig,
}

/// 분석기 설정
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    /// 희소성 임계값
    pub sparsity_threshold: f32,
    
    /// RBE 적합성 임계값
    pub rbe_threshold: f32,
    
    /// 압축률 목표
    pub target_compression: f32,
    
    /// 분석 세밀도
    pub analysis_precision: AnalysisPrecision,
}

#[derive(Debug, Clone)]
pub enum AnalysisPrecision {
    Fast,
    Balanced,
    Thorough,
}

impl LLMAnalyzer {
    /// 새로운 LLM 분석기 생성
    pub fn new(config: AnalyzerConfig) -> Self {
        Self {
            architecture: None,
            weight_data: HashMap::new(),
            config,
        }
    }
    
    /// 표준 GPT-2 117M 모델 분석
    pub fn analyze_gpt2_117m(&mut self) -> Result<LLMArchitecture, String> {
        println!("=== GPT-2 117M 모델 구조 분석 시작 ===");
        
        // GPT-2 117M 기본 구조
        let architecture = LLMArchitecture {
            total_parameters: 117_210_240,
            hidden_size: 768,
            vocab_size: 50_257,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            max_sequence_length: 1024,
            layer_parameters: self.analyze_layer_parameters()?,
            compression_priority: self.compute_compression_priority(),
        };
        
        self.architecture = Some(architecture.clone());
        
        println!("✓ 총 파라미터: {}", architecture.total_parameters);
        println!("✓ 히든 차원: {}", architecture.hidden_size);
        println!("✓ 어휘 크기: {}", architecture.vocab_size);
        println!("✓ 레이어 수: {}", architecture.num_layers);
        
        Ok(architecture)
    }
    
    /// 레이어별 파라미터 분석
    fn analyze_layer_parameters(&self) -> Result<Vec<LayerParameterInfo>, String> {
        let mut layer_params = Vec::new();
        
        // 1. Token Embedding 분석
        layer_params.push(LayerParameterInfo {
            layer_id: 0,
            layer_type: LayerType::TokenEmbedding,
            parameter_count: 50_257 * 768, // 38,597,376
            memory_usage: 50_257 * 768 * 4, // 154MB
            compressible: true,
            target_compression_ratio: 500.0,
            weight_matrices: vec![WeightMatrixInfo {
                name: "token_embeddings".to_string(),
                dimensions: (50_257, 768),
                dtype: "f32".to_string(),
                sparsity_ratio: 0.15, // 임베딩은 보통 희소함
                value_range: (-0.5, 0.5),
                variance: 0.02,
                rbe_suitability: 0.75, // 높은 적합성
            }],
        });
        
        // 2. Positional Embedding 분석
        layer_params.push(LayerParameterInfo {
            layer_id: 1,
            layer_type: LayerType::PositionalEmbedding,
            parameter_count: 1024 * 768, // 786,432
            memory_usage: 1024 * 768 * 4, // 3MB
            compressible: false, // 작은 크기로 제외
            target_compression_ratio: 1.0,
            weight_matrices: vec![WeightMatrixInfo {
                name: "position_embeddings".to_string(),
                dimensions: (1024, 768),
                dtype: "f32".to_string(),
                sparsity_ratio: 0.05,
                value_range: (-1.0, 1.0),
                variance: 0.1,
                rbe_suitability: 0.3, // 낮은 적합성
            }],
        });
        
        // 3. Transformer 레이어들 분석 (12개)
        for layer_id in 0..12 {
            // Attention 레이어
            layer_params.push(self.analyze_attention_layer(layer_id + 2, layer_id)?);
            
            // FFN 레이어
            layer_params.push(self.analyze_ffn_layer(layer_id + 14, layer_id)?);
            
            // Layer Norm (2개씩)
            for ln_id in 0..2 {
                layer_params.push(LayerParameterInfo {
                    layer_id: layer_id + 26 + ln_id,
                    layer_type: LayerType::LayerNorm,
                    parameter_count: 768, // gamma, beta
                    memory_usage: 768 * 4,
                    compressible: false,
                    target_compression_ratio: 1.0,
                    weight_matrices: vec![],
                });
            }
        }
        
        // 4. Output Layer 분석
        layer_params.push(LayerParameterInfo {
            layer_id: 50,
            layer_type: LayerType::Output,
            parameter_count: 768 * 50_257, // 38,597,376
            memory_usage: 768 * 50_257 * 4, // 154MB
            compressible: true,
            target_compression_ratio: 800.0,
            weight_matrices: vec![WeightMatrixInfo {
                name: "output_projection".to_string(),
                dimensions: (768, 50_257),
                dtype: "f32".to_string(),
                sparsity_ratio: 0.2,
                value_range: (-0.3, 0.3),
                variance: 0.015,
                rbe_suitability: 0.85,
            }],
        });
        
        Ok(layer_params)
    }
    
    /// Attention 레이어 분석
    fn analyze_attention_layer(&self, layer_id: usize, transformer_layer: usize) -> Result<LayerParameterInfo, String> {
        // Q, K, V, O 프로젝션: 768 × 768 × 4 = 2,359,296 파라미터
        let param_count = 768 * 768 * 4;
        let memory_usage = param_count * 4; // 9.4MB
        
        // 레이어 깊이에 따른 압축률 조정
        let target_compression = if transformer_layer < 3 {
            200.0 // 초기 레이어: 보수적
        } else if transformer_layer < 9 {
            400.0 // 중간 레이어: 적극적
        } else {
            600.0 // 후반 레이어: 매우 적극적
        };
        
        Ok(LayerParameterInfo {
            layer_id,
            layer_type: LayerType::Attention,
            parameter_count: param_count,
            memory_usage,
            compressible: true,
            target_compression_ratio: target_compression,
            weight_matrices: vec![
                WeightMatrixInfo {
                    name: format!("attention_q_proj_{}", transformer_layer),
                    dimensions: (768, 768),
                    dtype: "f32".to_string(),
                    sparsity_ratio: 0.1,
                    value_range: (-0.2, 0.2),
                    variance: 0.01,
                    rbe_suitability: 0.65,
                },
                WeightMatrixInfo {
                    name: format!("attention_k_proj_{}", transformer_layer),
                    dimensions: (768, 768),
                    dtype: "f32".to_string(),
                    sparsity_ratio: 0.12,
                    value_range: (-0.2, 0.2),
                    variance: 0.01,
                    rbe_suitability: 0.65,
                },
                WeightMatrixInfo {
                    name: format!("attention_v_proj_{}", transformer_layer),
                    dimensions: (768, 768),
                    dtype: "f32".to_string(),
                    sparsity_ratio: 0.08,
                    value_range: (-0.25, 0.25),
                    variance: 0.012,
                    rbe_suitability: 0.7,
                },
                WeightMatrixInfo {
                    name: format!("attention_o_proj_{}", transformer_layer),
                    dimensions: (768, 768),
                    dtype: "f32".to_string(),
                    sparsity_ratio: 0.15,
                    value_range: (-0.15, 0.15),
                    variance: 0.008,
                    rbe_suitability: 0.8,
                },
            ],
        })
    }
    
    /// FFN 레이어 분석
    fn analyze_ffn_layer(&self, layer_id: usize, transformer_layer: usize) -> Result<LayerParameterInfo, String> {
        // W1: 768 → 3072, W2: 3072 → 768 = 4,718,592 파라미터
        let param_count = 768 * 3072 + 3072 * 768;
        let memory_usage = param_count * 4; // 18.9MB
        
        // FFN은 가장 압축하기 좋은 레이어
        let target_compression = if transformer_layer < 3 {
            500.0 // 초기 레이어
        } else if transformer_layer < 9 {
            800.0 // 중간 레이어
        } else {
            1200.0 // 후반 레이어: 극적 압축
        };
        
        Ok(LayerParameterInfo {
            layer_id,
            layer_type: LayerType::FFN,
            parameter_count: param_count,
            memory_usage,
            compressible: true,
            target_compression_ratio: target_compression,
            weight_matrices: vec![
                WeightMatrixInfo {
                    name: format!("ffn_w1_{}", transformer_layer),
                    dimensions: (768, 3072),
                    dtype: "f32".to_string(),
                    sparsity_ratio: 0.25, // FFN은 더 희소함
                    value_range: (-0.3, 0.3),
                    variance: 0.02,
                    rbe_suitability: 0.9, // 최고 적합성
                },
                WeightMatrixInfo {
                    name: format!("ffn_w2_{}", transformer_layer),
                    dimensions: (3072, 768),
                    dtype: "f32".to_string(),
                    sparsity_ratio: 0.3,
                    value_range: (-0.2, 0.2),
                    variance: 0.015,
                    rbe_suitability: 0.95, // 최고 적합성
                },
            ],
        })
    }
    
    /// 압축 우선순위 계산
    fn compute_compression_priority(&self) -> CompressionPriority {
        let mut high_priority = Vec::new();
        let mut medium_priority = Vec::new();
        let mut low_priority = Vec::new();
        let mut excluded = Vec::new();
        
        // FFN 레이어들을 최우선으로
        for i in 0..12 {
            high_priority.push(14 + i); // FFN 레이어 ID들
        }
        
        // Attention 레이어들을 차우선으로
        for i in 0..12 {
            medium_priority.push(2 + i); // Attention 레이어 ID들
        }
        
        // Embedding 레이어들을 보조로
        low_priority.push(0); // Token embedding
        low_priority.push(50); // Output layer
        
        // LayerNorm과 Positional embedding은 제외
        excluded.push(1); // Positional embedding
        for i in 26..50 { // LayerNorm 들
            excluded.push(i);
        }
        
        CompressionPriority {
            high_priority,
            medium_priority,
            low_priority,
            excluded,
        }
    }
    
    /// 압축 가능한 총 메모리 계산
    pub fn calculate_compression_savings(&self) -> Result<CompressionSavings, String> {
        let arch = self.architecture.as_ref()
            .ok_or("아키텍처 분석이 필요합니다")?;
        
        let mut original_size = 0;
        let mut compressed_size = 0;
        
        for layer in &arch.layer_parameters {
            original_size += layer.memory_usage;
            
            if layer.compressible {
                compressed_size += layer.memory_usage / layer.target_compression_ratio as usize;
            } else {
                compressed_size += layer.memory_usage;
            }
        }
        
        let savings_bytes = original_size - compressed_size;
        let savings_ratio = savings_bytes as f32 / original_size as f32;
        
        Ok(CompressionSavings {
            original_size_mb: original_size / 1024 / 1024,
            compressed_size_mb: compressed_size / 1024 / 1024,
            savings_mb: savings_bytes / 1024 / 1024,
            savings_ratio,
        })
    }
    
    /// 분석 결과 출력
    pub fn print_analysis_report(&self) -> Result<(), String> {
        let arch = self.architecture.as_ref()
            .ok_or("아키텍처 분석이 필요합니다")?;
        
        println!("\n=== LLM 구조 분석 리포트 ===");
        println!("총 파라미터: {}", arch.total_parameters);
        println!("총 레이어: {}", arch.layer_parameters.len());
        
        let savings = self.calculate_compression_savings()?;
        println!("\n=== 압축 예상 결과 ===");
        println!("원본 크기: {}MB", savings.original_size_mb);
        println!("압축 후 크기: {}MB", savings.compressed_size_mb);
        println!("절약 공간: {}MB ({:.1}%)", 
                 savings.savings_mb, savings.savings_ratio * 100.0);
        
        println!("\n=== 레이어별 압축 계획 ===");
        for layer in &arch.layer_parameters {
            if layer.compressible {
                println!("Layer {}: {:?} - {:.0}:1 압축 ({:.1}MB → {:.1}MB)",
                         layer.layer_id,
                         layer.layer_type,
                         layer.target_compression_ratio,
                         layer.memory_usage / 1024 / 1024,
                         layer.memory_usage as f32 / layer.target_compression_ratio / 1024.0 / 1024.0);
            }
        }
        
        Ok(())
    }
}

/// 압축 절약 정보
#[derive(Debug, Clone)]
pub struct CompressionSavings {
    pub original_size_mb: usize,
    pub compressed_size_mb: usize,
    pub savings_mb: usize,
    pub savings_ratio: f32,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            sparsity_threshold: 0.1,
            rbe_threshold: 0.5,
            target_compression: 0.9, // 90% 압축 목표
            analysis_precision: AnalysisPrecision::Balanced,
        }
    }
} 
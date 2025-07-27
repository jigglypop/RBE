//! KoGPT-2 RBE 통합 구현체
//! 실제 KoGPT-2 모델을 RBE로 변환하고 추론을 수행합니다

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use anyhow::{Result, anyhow};

use crate::core::encoder::encoder::RBEEncoder;
use crate::core::decoder::optimized_decoder::OptimizedRBEDecoder;
use crate::core::tensors::{Packed128, DecodedParams};

// NLP 레이어들
use crate::nlp::embedding::rbe_embedding::{RBEEmbedding, RBEEmbeddingConfig};
use crate::nlp::linear::rbe_linear::{RBELinear, RBELinearConfig};
use crate::nlp::attention::rbe_attention::{RBEAttention, RBEAttentionConfig};
use crate::nlp::ffn::rbe_ffn::{RBEFFN, RBEFFNConfig, ActivationType};
use crate::nlp::layernorm::rbe_layernorm::{RBELayerNorm, RBELayerNormConfig};
use crate::nlp::softmax::rbe_softmax::RBESoftmax;
use crate::nlp::model_tools::analyzer::QualityGrade;

use serde::{Serialize, Deserialize};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// KoGPT-2 모델 설정
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KoGPT2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_epsilon: f32,
    pub dropout_prob: f32,
}

impl Default for KoGPT2Config {
    fn default() -> Self {
        // skt/kogpt2-base-v2 기본 설정
        Self {
            vocab_size: 51200,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 1024,
            layer_norm_epsilon: 1e-5,
            dropout_prob: 0.1,
        }
    }
}

/// Transformer 블록
#[derive(Serialize, Deserialize)]
pub struct TransformerBlock {
    pub attention: RBEAttention,
    pub ln1: RBELayerNorm,
    pub ffn: RBEFFN,
    pub ln2: RBELayerNorm,
}

/// KoGPT-2 RBE 모델
#[derive(Serialize, Deserialize)]
pub struct KoGPT2RBE {
    pub config: KoGPT2Config,
    pub wte: RBEEmbedding,      // 토큰 임베딩
    pub wpe: RBEEmbedding,      // 위치 임베딩
    pub blocks: Vec<TransformerBlock>,
    pub ln_f: RBELayerNorm,     // 최종 LayerNorm
    pub lm_head: RBELinear,     // 언어 모델 헤드
}

impl KoGPT2RBE {
    pub fn init_after_load(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.wte.init_after_load()?;
        self.wpe.init_after_load()?;
        for block in &mut self.blocks {
            block.attention.init_after_load()?;
            block.ffn.init_after_load()?;
            // LayerNorm does not need it
        }
        self.lm_head.init_after_load()?;
        Ok(())
    }

    /// 새 모델 생성
    pub fn new(config: KoGPT2Config) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut blocks = Vec::new();
        
        for _ in 0..config.num_layers {
            // Attention 설정
            let attn_config = RBEAttentionConfig {
                hidden_dim: config.hidden_size,
                num_heads: config.num_heads,
                head_dim: config.hidden_size / config.num_heads,
                attention_dropout: config.dropout_prob,
                output_dropout: config.dropout_prob,
                ..Default::default()
            };
            
            // LayerNorm 설정
            let ln_config = RBELayerNormConfig {
                normalized_shape: vec![config.hidden_size],
                eps: config.layer_norm_epsilon as f64,
                elementwise_affine: true,
                use_fused_ops: true,
            };
            
            // FFN 설정
            let ffn_config = RBEFFNConfig {
                hidden_dim: config.hidden_size,
                intermediate_dim: config.intermediate_size,
                activation: ActivationType::GeluNew,  // GPT-2는 GeluNew 사용
                dropout: config.dropout_prob,
                ..Default::default()
            };
            
            blocks.push(TransformerBlock {
                attention: RBEAttention::new(attn_config)?,
                ln1: RBELayerNorm::new(ln_config.clone())?,
                ffn: RBEFFN::new(ffn_config)?,
                ln2: RBELayerNorm::new(ln_config)?,
            });
        }
        
        // 토큰 임베딩 설정
        let wte_config = RBEEmbeddingConfig {
            vocab_size: config.vocab_size,
            embedding_dim: config.hidden_size,
            max_position_embeddings: config.max_position_embeddings,
            ..Default::default()
        };
        
        // 위치 임베딩 설정
        let wpe_config = RBEEmbeddingConfig {
            vocab_size: config.max_position_embeddings,  // 위치 임베딩은 vocab_size가 max_position
            embedding_dim: config.hidden_size,
            max_position_embeddings: config.max_position_embeddings,
            quality_grade: QualityGrade::B, // 예시 품질 등급
            ..Default::default()
        };
        
        // 최종 LayerNorm 설정
        let ln_f_config = RBELayerNormConfig {
            normalized_shape: vec![config.hidden_size],
            eps: config.layer_norm_epsilon as f64,
            elementwise_affine: true,
            use_fused_ops: true,
        };
        
        // LM Head 설정
        let lm_head_config = RBELinearConfig {
            use_bias: false,  // GPT-2 LM head는 bias 없음
            ..Default::default()
        };
        
        Ok(Self {
            config: config.clone(),
            wte: RBEEmbedding::new(wte_config)?,
            wpe: RBEEmbedding::new(wpe_config)?,
            blocks,
            ln_f: RBELayerNorm::new(ln_f_config)?,
            lm_head: RBELinear::new(config.hidden_size, config.vocab_size, Some(lm_head_config)),
        })
    }
    
    /// PyTorch 가중치에서 RBE 모델 로드
    pub fn from_pytorch_weights(
        config: KoGPT2Config,
        weights_path: &Path,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let weights_data = fs::read(weights_path)?;
        let weights = Arc::new(safetensors::SafeTensors::deserialize(&weights_data)?);
        
        let mut model = Self::new(config.clone())?;

        let total_tasks = 2 + config.num_layers * 4 + 2; // wte, wpe, (attn, ffn) * layers, ln_f, lm_head
        let pb = ProgressBar::new(total_tasks as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) - {msg}")
                .unwrap()
                .progress_chars("=> "),
        );
        println!("📦 가중치를 RBE 형식으로 변환 중...");

        // Embedding layers
        pb.set_message("토큰 임베딩(wte) 변환...");
        // 토큰 임베딩
        if let Ok(wte_weight) = weights.tensor("transformer.wte.weight") {
            let wte_data: Vec<f32> = wte_weight.data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            
            let wte_config = RBEEmbeddingConfig {
                vocab_size: config.vocab_size,
                embedding_dim: config.hidden_size,
                quality_grade: QualityGrade::B, // 예시 품질 등급
                ..Default::default()
            };

            model.wte = RBEEmbedding::from_pretrained_weights(&wte_data, None, wte_config)
                .map_err(|e| anyhow!(e.to_string()))?;
        } else {
            return Err("transformer.wte.weight not found".into());
        };
        pb.inc(1);
        
        pb.set_message("위치 임베딩(wpe) 변환...");
        // 위치 임베딩
        if let Ok(wpe_weight) = weights.tensor("transformer.wpe.weight") {
            let wpe_data: Vec<f32> = wpe_weight.data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            let wpe_config = RBEEmbeddingConfig {
                vocab_size: config.max_position_embeddings,
                embedding_dim: config.hidden_size,
                max_position_embeddings: config.max_position_embeddings,
                quality_grade: QualityGrade::B, // 예시 품질 등급
                ..Default::default()
            };
            model.wpe = RBEEmbedding::from_pretrained_weights(&wpe_data, None, wpe_config)
                .map_err(|e| anyhow!(e.to_string()))?;
        } else {
            return Err("transformer.wpe.weight not found".into());
        };
        pb.inc(1);
        
        // Transformer blocks
        for i in 0..config.num_layers {
            let prefix = format!("transformer.h.{}", i);
            
            pb.set_message(format!("블록 {} - Attention 변환...", i));
            // Attention 레이어
            if let Ok(c_attn_tensor) = weights.tensor(&format!("{}.attn.c_attn.weight", prefix)) {
                let c_attn_data: Vec<f32> = c_attn_tensor.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();

                let hidden_size = model.config.hidden_size;
                let q_weights = &c_attn_data[..hidden_size * hidden_size];
                let k_weights = &c_attn_data[hidden_size * hidden_size..2 * hidden_size * hidden_size];
                let v_weights = &c_attn_data[2 * hidden_size * hidden_size..];

                let linear_config = RBELinearConfig { use_bias: false, ..Default::default() };
                
                model.blocks[i].attention.q_proj = RBELinear::from_weights(q_weights, None, hidden_size, hidden_size, Some(linear_config.clone()))?;
                model.blocks[i].attention.k_proj = RBELinear::from_weights(k_weights, None, hidden_size, hidden_size, Some(linear_config.clone()))?;
                model.blocks[i].attention.v_proj = RBELinear::from_weights(v_weights, None, hidden_size, hidden_size, Some(linear_config.clone()))?;
            }

            if let Ok(c_proj_tensor) = weights.tensor(&format!("{}.attn.c_proj.weight", prefix)) {
                 let c_proj_data: Vec<f32> = c_proj_tensor.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                
                let hidden_size = model.config.hidden_size;
                let linear_config = RBELinearConfig { use_bias: false, ..Default::default() };
                model.blocks[i].attention.out_proj = RBELinear::from_weights(&c_proj_data, None, hidden_size, hidden_size, Some(linear_config))?;
            }
            
            // LayerNorm 1
            if let Ok(ln1_weight) = weights.tensor(&format!("{}.ln_1.weight", prefix)) {
                let ln1_data: Vec<f32> = ln1_weight.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                
                let ln1_bias = if let Ok(bias) = weights.tensor(&format!("{}.ln_1.bias", prefix)) {
                    let bias_data: Vec<f32> = bias.data()
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    Some(bias_data)
                } else {
                    None
                };
                
                let ln_config = RBELayerNormConfig {
                    normalized_shape: vec![model.config.hidden_size],
                    eps: model.config.layer_norm_epsilon as f64,
                    ..Default::default()
                };
                model.blocks[i].ln1 = RBELayerNorm::from_pretrained(Some(ln1_data), ln1_bias, ln_config)?;
            }
            
            // FFN
            let ffn_prefix = format!("{}.mlp", prefix);
            let up_weights = if let Ok(tensor) = weights.tensor(&format!("{}.c_fc.weight", ffn_prefix)) {
                tensor.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<f32>>()
            } else { continue; };

            let up_bias: Option<Vec<f32>> = if let Ok(tensor) = weights.tensor(&format!("{}.c_fc.bias", ffn_prefix)) {
                Some(tensor.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<f32>>())
            } else { None };

            let down_weights = if let Ok(tensor) = weights.tensor(&format!("{}.c_proj.weight", ffn_prefix)) {
                tensor.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<f32>>()
            } else { continue; };

            let down_bias: Option<Vec<f32>> = if let Ok(tensor) = weights.tensor(&format!("{}.c_proj.bias", ffn_prefix)) {
                Some(tensor.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<f32>>())
            } else { None };

            let linear_config = RBELinearConfig { use_bias: up_bias.is_some(), ..Default::default() };
            model.blocks[i].ffn.up_proj = RBELinear::from_weights(
                &up_weights, up_bias.as_deref(), model.config.hidden_size, model.config.intermediate_size, Some(linear_config)
            )?;

            let linear_config = RBELinearConfig { use_bias: down_bias.is_some(), ..Default::default() };
            model.blocks[i].ffn.down_proj = RBELinear::from_weights(
                &down_weights, down_bias.as_deref(), model.config.intermediate_size, model.config.hidden_size, Some(linear_config)
            )?;

            // LayerNorm 2
            if let Ok(ln2_weight) = weights.tensor(&format!("{}.ln_2.weight", prefix)) {
                let ln2_data: Vec<f32> = ln2_weight.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                
                let ln2_bias = if let Ok(bias) = weights.tensor(&format!("{}.ln_2.bias", prefix)) {
                    let bias_data: Vec<f32> = bias.data()
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    Some(bias_data)
                } else {
                    None
                };
                
                 let ln_config = RBELayerNormConfig {
                    normalized_shape: vec![model.config.hidden_size],
                    eps: model.config.layer_norm_epsilon as f64,
                    ..Default::default()
                };
                model.blocks[i].ln2 = RBELayerNorm::from_pretrained(Some(ln2_data), ln2_bias, ln_config)?;
            }
            pb.inc(4); // q, k, v, out_proj
            
            pb.set_message(format!("블록 {} - FFN 변환...", i));
            // FFN
            let ffn_prefix = format!("{}.mlp", prefix);
            let up_weights = if let Ok(tensor) = weights.tensor(&format!("{}.c_fc.weight", ffn_prefix)) {
                tensor.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<f32>>()
            } else { continue; };

            let up_bias: Option<Vec<f32>> = if let Ok(tensor) = weights.tensor(&format!("{}.c_fc.bias", ffn_prefix)) {
                Some(tensor.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<f32>>())
            } else { None };

            let down_weights = if let Ok(tensor) = weights.tensor(&format!("{}.c_proj.weight", ffn_prefix)) {
                tensor.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<f32>>()
            } else { continue; };

            let down_bias: Option<Vec<f32>> = if let Ok(tensor) = weights.tensor(&format!("{}.c_proj.bias", ffn_prefix)) {
                Some(tensor.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<f32>>())
            } else { None };

            let linear_config = RBELinearConfig { use_bias: up_bias.is_some(), ..Default::default() };
            model.blocks[i].ffn.up_proj = RBELinear::from_weights(
                &up_weights, up_bias.as_deref(), model.config.hidden_size, model.config.intermediate_size, Some(linear_config)
            )?;

            let linear_config = RBELinearConfig { use_bias: down_bias.is_some(), ..Default::default() };
            model.blocks[i].ffn.down_proj = RBELinear::from_weights(
                &down_weights, down_bias.as_deref(), model.config.intermediate_size, model.config.hidden_size, Some(linear_config)
            )?;

            // LayerNorm 2
            if let Ok(ln2_weight) = weights.tensor(&format!("{}.ln_2.weight", prefix)) {
                let ln2_data: Vec<f32> = ln2_weight.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                
                let ln2_bias = if let Ok(bias) = weights.tensor(&format!("{}.ln_2.bias", prefix)) {
                    let bias_data: Vec<f32> = bias.data()
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    Some(bias_data)
                } else {
                    None
                };
                
                 let ln_config = RBELayerNormConfig {
                    normalized_shape: vec![model.config.hidden_size],
                    eps: model.config.layer_norm_epsilon as f64,
                    ..Default::default()
                };
                model.blocks[i].ln2 = RBELayerNorm::from_pretrained(Some(ln2_data), ln2_bias, ln_config)?;
            }
            pb.inc(2); // ln1, ln2
        }

        // Final layers
        pb.set_message("최종 LayerNorm 변환...");
        // 최종 LayerNorm
        if let Ok(ln_f_weight) = weights.tensor("transformer.ln_f.weight") {
            let ln_f_data: Vec<f32> = ln_f_weight.data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            
            let ln_f_bias = if let Ok(bias) = weights.tensor("transformer.ln_f.bias") {
                let bias_data: Vec<f32> = bias.data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                Some(bias_data)
            } else {
                None
            };
            
            let ln_config = RBELayerNormConfig {
                normalized_shape: vec![model.config.hidden_size],
                eps: model.config.layer_norm_epsilon as f64,
                ..Default::default()
            };
            model.ln_f = RBELayerNorm::from_pretrained(Some(ln_f_data), ln_f_bias, ln_config)?;
            println!("✅ Loaded final LayerNorm");
        }
        
        // 언어 모델 헤드 (wte와 가중치 공유)
        if let Ok(lm_head_weight) = weights.tensor("lm_head.weight") {
            let lm_head_data: Vec<f32> = lm_head_weight.data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            model.lm_head = RBELinear::from_weights(
                &lm_head_data,
                None,
                model.config.hidden_size,
                model.config.vocab_size,
                Some(RBELinearConfig { use_bias: false, ..Default::default() }),
            )?;
            println!("✅ Loaded LM head");
        }
        
        pb.finish_with_message("✅ 모든 가중치 변환 완료!");
        Ok(model)
    }
    
    /// 순전파 (추론)
    pub fn forward(&mut self, input_ids: &[usize]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let _batch_size = 1;
        let seq_len = input_ids.len();
        let _hidden_size = self.config.hidden_size;
        
        // usize를 u32로 변환
        let input_ids_u32: Vec<u32> = input_ids.iter()
            .map(|&id| id as u32)
            .collect();
        
        // 1. 토큰 임베딩
        let mut hidden_states = self.wte.forward(&input_ids_u32)?;
        
        // 2. 위치 임베딩 추가
        let position_ids: Vec<u32> = (0..seq_len as u32).collect();
        let position_embeds = self.wpe.forward(&position_ids)?;
        
        // hidden_states += position_embeds
        for i in 0..hidden_states.len() {
            hidden_states[i] += position_embeds[i];
        }
        
        // 3. Transformer 블록들 통과 - 각 토큰별로 처리
        for (block_idx, block) in self.blocks.iter_mut().enumerate() {
            println!("🔄 Processing block {}...", block_idx);
            
            // 각 토큰을 개별적으로 처리
            let mut processed_tokens = Vec::new();
            
            for token_idx in 0..seq_len {
                let start_idx = token_idx * self.config.hidden_size;
                let end_idx = start_idx + self.config.hidden_size;
                
                if end_idx > hidden_states.len() {
                    return Err(format!("토큰 {}의 인덱스 범위 초과: {}..{} (전체 크기: {})", 
                                     token_idx, start_idx, end_idx, hidden_states.len()).into());
                }
                
                let mut token_hidden = hidden_states[start_idx..end_idx].to_vec();
                
                // LayerNorm 1
                let ln1_out = block.ln1.forward(&token_hidden)?;
                
                // Self-Attention (단일 토큰이므로 mask 없음)
                let attn_out = block.attention.forward(&ln1_out, None)?;
                
                // Residual connection
                for i in 0..token_hidden.len() {
                    token_hidden[i] += attn_out[i];
                }
                
                // LayerNorm 2
                let ln2_out = block.ln2.forward(&token_hidden)?;
                
                // FFN
                let ffn_out = block.ffn.forward(&ln2_out)?;
                
                // Residual connection
                for i in 0..token_hidden.len() {
                    token_hidden[i] += ffn_out[i];
                }
                
                processed_tokens.extend(token_hidden);
            }
            
            hidden_states = processed_tokens;
            println!("✅ Block {} processed", block_idx);
        }
        
        // 4. 최종 LayerNorm - 각 토큰별로 처리
        let mut final_hidden = Vec::new();
        for token_idx in 0..seq_len {
            let start_idx = token_idx * self.config.hidden_size;
            let end_idx = start_idx + self.config.hidden_size;
            let token_hidden = &hidden_states[start_idx..end_idx];
            
            let ln_out = self.ln_f.forward(token_hidden)?;
            final_hidden.extend(ln_out);
        }
        
        // 5. LM Head (언어 모델 출력) - 마지막 토큰만 처리
        let last_token_start = (seq_len - 1) * self.config.hidden_size;
        let last_token_end = last_token_start + self.config.hidden_size;
        let last_token_hidden = &final_hidden[last_token_start..last_token_end];
        
        let logits = self.lm_head.forward(last_token_hidden)?;
        
        println!("✅ Forward pass completed!");
        println!("   Final logits size: {}", logits.len());
        
        Ok(logits)
    }
    
    /// 텍스트 생성
    pub fn generate(
        &mut self,
        input_ids: &[usize],
        max_length: usize,
        temperature: f32,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let mut generated = input_ids.to_vec();
        let softmax = RBESoftmax::new(-1);  // 마지막 차원에 적용
        
        while generated.len() < max_length {
            // 현재까지의 시퀀스로 다음 토큰 예측
            let logits = self.forward(&generated)?;
            
            // forward 함수가 마지막 위치의 logits만 반환하므로 전체를 사용
            let last_logits = &logits;
            
            // Temperature 적용
            let scaled_logits: Vec<f32> = last_logits.iter()
                .map(|&x| x / temperature)
                .collect();
            
            // Softmax로 확률 변환
            let probs = softmax.forward(&scaled_logits);
            
            // 가장 높은 확률의 토큰 선택 (greedy decoding)
            let next_token = probs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            
            generated.push(next_token);
            
            // EOS 토큰이면 종료
            if next_token == 1 {  // assuming 1 is EOS token
                break;
            }
        }
        
        Ok(generated)
    }
    
    /// 모델 압축률 통계
    pub fn compression_stats(&self) -> CompressionStats {
        let mut total_original_params = 0;
        let mut total_compressed_bytes = 0;
        
        // 각 레이어별 통계 수집
        let (comp, _ratio) = self.wte.memory_usage();
        total_original_params += self.config.vocab_size * self.config.hidden_size;
        total_compressed_bytes += comp;
        
        let (comp, _ratio) = self.wpe.memory_usage();
        total_original_params += self.config.max_position_embeddings * self.config.hidden_size;
        total_compressed_bytes += comp;
        
        for block in &self.blocks {
            let (comp, _ratio) = block.attention.memory_usage();
            total_original_params += 4 * self.config.hidden_size * self.config.hidden_size;
            total_compressed_bytes += comp;
            
            let ln1_params = block.ln1.gamma.as_ref().map_or(0, |g| g.len()) + block.ln1.beta.as_ref().map_or(0, |b| b.len());
            total_original_params += ln1_params;
            total_compressed_bytes += ln1_params * 4; // f32
            
            let (comp, _ratio) = block.ffn.memory_usage();
            total_original_params += self.config.hidden_size * self.config.intermediate_size + self.config.intermediate_size * self.config.hidden_size;
            total_compressed_bytes += comp;
            
            let ln2_params = block.ln2.gamma.as_ref().map_or(0, |g| g.len()) + block.ln2.beta.as_ref().map_or(0, |b| b.len());
            total_original_params += ln2_params;
            total_compressed_bytes += ln2_params * 4; // f32
        }
        
        let ln_f_params = self.ln_f.gamma.as_ref().map_or(0, |g| g.len()) + self.ln_f.beta.as_ref().map_or(0, |b| b.len());
        total_original_params += ln_f_params;
        total_compressed_bytes += ln_f_params * 4;

        total_original_params += self.config.hidden_size * self.config.vocab_size;
        total_compressed_bytes += self.lm_head.memory_usage();
        
        let total_original_bytes = total_original_params * 4;
        
        CompressionStats {
            original_params: total_original_params,
            compressed_params: total_compressed_bytes / 16, // 128bit = 16 bytes
            compression_ratio: total_original_bytes as f32 / total_compressed_bytes as f32,
            original_size_mb: total_original_bytes as f32 / (1024.0 * 1024.0),
            compressed_size_mb: total_compressed_bytes as f32 / (1024.0 * 1024.0),
        }
    }
}

/// 압축 통계
#[derive(Debug)]
pub struct CompressionStats {
    pub original_params: usize,
    pub compressed_params: usize,
    pub compression_ratio: f32,
    pub original_size_mb: f32,
    pub compressed_size_mb: f32,
}

impl std::fmt::Display for CompressionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, 
            "📊 Compression Statistics:\n\
             - Original params: {}\n\
             - Compressed params: {}\n\
             - Compression ratio: {:.1}:1\n\
             - Original size: {:.2} MB\n\
             - Compressed size: {:.2} MB\n\
             - Size reduction: {:.1}%",
            self.original_params,
            self.compressed_params,
            self.compression_ratio,
            self.original_size_mb,
            self.compressed_size_mb,
            (1.0 - self.compressed_size_mb / self.original_size_mb) * 100.0
        )
    }
} 
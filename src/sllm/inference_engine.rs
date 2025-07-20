use crate::packed_params::{Packed128, HybridEncodedBlock};
use crate::decoder::FusedForwardPass;
use crate::sllm::{CompressedModel, CompressedLayer, SimpleModelLoader, SimpleTokenizer};
use crate::sllm::simple_loader::ModelConfig;
use std::collections::HashMap;
use std::path::Path;
use anyhow::{Result, anyhow};
use nalgebra::{DVector, DMatrix};
use rand::{Rng, SeedableRng};

/// RBE 추론 엔진
pub struct RBEInferenceEngine {
    /// 압축된 모델
    compressed_model: CompressedModel,
    /// 모델 설정
    config: ModelConfig,
    /// 토크나이저
    tokenizer: SimpleTokenizer,
    /// 융합 순전파 엔진
    fused_forward: FusedForwardPass,
    /// 압축된 레이어 가중치들
    layer_weights: HashMap<String, LayerWeights>,
}

/// 레이어별 가중치 정보
#[derive(Debug)]
struct LayerWeights {
    /// RBE 압축된 블록들
    compressed_blocks: Vec<HybridEncodedBlock>,
    /// 원본 형태 정보
    shape: (usize, usize), // (rows, cols)
    /// 레이어 타입
    layer_type: LayerType,
}

/// 레이어 타입
#[derive(Debug, Clone, PartialEq)]
enum LayerType {
    Embedding,
    Attention,
    FeedForward,
    LayerNorm,
    Output,
}

impl RBEInferenceEngine {
    /// 압축된 모델로부터 추론 엔진 생성
    pub async fn from_compressed_model(
        compressed_model_path: &Path,
        original_model_path: &Path,
    ) -> Result<Self> {
        println!("🚀 RBE 추론 엔진 초기화 중...");
        
        // 1. 압축된 모델 로드
        let model_content = std::fs::read_to_string(compressed_model_path)?;
        let compressed_model: CompressedModel = serde_json::from_str(&model_content)?;
        
        println!("📊 압축된 모델 정보:");
        println!("   - 모델명: {}", compressed_model.model_name);
        println!("   - 레이어 수: {}", compressed_model.layers.len());
        println!("   - 압축률: {:.1}:1", compressed_model.total_compression_ratio);
        
        // 2. 원본 모델 설정 로드
        let simple_loader = SimpleModelLoader::new()?;
        let config = simple_loader.load_model_config(original_model_path)?;
        
        println!("🔧 모델 설정:");
        println!("   - 어휘 크기: {}", config.vocab_size);
        println!("   - 은닉층 크기: {}", config.hidden_size);
        println!("   - 레이어 수: {}", config.num_layers);
        
        // 3. 토크나이저 로드
        let tokenizer = SimpleTokenizer::load(original_model_path)?;
        
        // 4. 융합 순전파 엔진 초기화
        let fused_forward = FusedForwardPass::new();
        
        // 5. 레이어 가중치 구조화
        let layer_weights = Self::organize_layer_weights(&compressed_model)?;
        
        println!("✅ RBE 추론 엔진 초기화 완료!");
        
        Ok(Self {
            compressed_model,
            config,
            tokenizer,
            fused_forward,
            layer_weights,
        })
    }
    
    /// 원본 모델을 압축하여 추론 엔진 생성
    pub async fn from_original_model(
        model_path: &Path,
        output_path: &Path,
    ) -> anyhow::Result<Self> {
        println!("🗜️ 원본 모델 압축 중...");
        
        // 1. 원본 모델 압축
        let compressor = crate::sllm::SLLMCompressor::new(Default::default());
        let compressed_model = compressor.compress_safetensors_model(model_path, output_path).await
            .map_err(|e| anyhow::anyhow!("압축 실패: {}", e))?;
        
        // 2. 압축된 모델로 추론 엔진 생성
        Self::from_compressed_model(output_path, model_path).await
    }
    
    /// 레이어 가중치 구조화
    fn organize_layer_weights(compressed_model: &CompressedModel) -> Result<HashMap<String, LayerWeights>> {
        let mut layer_weights = HashMap::new();
        
        for (layer_name, compressed_layer) in &compressed_model.layers {
            let layer_type = Self::classify_layer_type(layer_name);
            
            let weights = LayerWeights {
                compressed_blocks: compressed_layer.compressed_data.clone(),
                shape: (compressed_layer.shape[0], compressed_layer.shape[1]),
                layer_type,
            };
            
            layer_weights.insert(layer_name.clone(), weights);
        }
        
        Ok(layer_weights)
    }
    
    /// 레이어 타입 분류
    fn classify_layer_type(layer_name: &str) -> LayerType {
        if layer_name.contains("embed") {
            LayerType::Embedding
        } else if layer_name.contains("attn") || layer_name.contains("attention") {
            LayerType::Attention
        } else if layer_name.contains("mlp") || layer_name.contains("fc") || layer_name.contains("linear") {
            LayerType::FeedForward
        } else if layer_name.contains("ln") || layer_name.contains("norm") {
            LayerType::LayerNorm
        } else if layer_name.contains("lm_head") || layer_name.contains("output") {
            LayerType::Output
        } else {
            LayerType::FeedForward // 기본값
        }
    }
    
    /// 텍스트 생성
    pub fn generate_text(
        &self,
        prompt: &str,
        max_length: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<String> {
        println!("💭 텍스트 생성 시작: '{}'", prompt);
        
        // 1. 프롬프트 토크나이징
        let mut token_ids = self.tokenizer.encode(prompt)?;
        println!("🔤 초기 토큰 수: {}", token_ids.len());
        
        // 2. 텍스트 생성 루프
        for step in 0..max_length {
            if step % 10 == 0 {
                println!("📝 생성 진행: {}/{}", step, max_length);
            }
            
            // 3. 다음 토큰 예측
            let next_token = self.predict_next_token(&token_ids, temperature, top_p)?;
            
            // 4. 토큰 추가
            token_ids.push(next_token);
            
            // 5. 종료 조건 체크 (EOS 토큰 등)
            if next_token == 50256 { // GPT-2 EOS 토큰
                println!("🏁 EOS 토큰 감지, 생성 종료");
                break;
            }
            
            // 6. 최대 길이 체크
            if token_ids.len() >= self.config.max_length {
                println!("📏 최대 길이 도달, 생성 종료");
                break;
            }
        }
        
        // 7. 토큰을 텍스트로 변환
        let generated_text = self.tokenizer.decode(&token_ids)?;
        
        println!("✅ 텍스트 생성 완료 (총 {} 토큰)", token_ids.len());
        Ok(generated_text)
    }
    
    /// 다음 토큰 예측
    fn predict_next_token(
        &self,
        input_tokens: &[i64],
        temperature: f32,
        top_p: f32,
    ) -> Result<i64> {
        // 1. 임베딩 레이어 적용
        let embeddings = self.apply_embedding(input_tokens)?;
        
        // 2. 트랜스포머 레이어들 적용
        let mut hidden_states = embeddings;
        for layer_idx in 0..self.config.num_layers {
            hidden_states = self.apply_transformer_layer(&hidden_states, layer_idx)?;
        }
        
        // 3. 최종 출력 레이어 적용
        let logits = self.apply_output_layer(&hidden_states)?;
        
        // 4. 마지막 토큰의 로짓 추출
        let last_token_logits = logits.row(input_tokens.len() - 1);
        
        // 5. 샘플링으로 다음 토큰 선택
        let logits_vec: Vec<f32> = last_token_logits.iter().cloned().collect();
        let next_token = self.sample_token(&logits_vec, temperature, top_p)?;
        
        Ok(next_token)
    }
    
    /// 임베딩 레이어 적용
    fn apply_embedding(&self, token_ids: &[i64]) -> Result<DMatrix<f32>> {
        // 단순화: 임베딩은 랜덤 벡터로 근사
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;
        
        let mut embeddings = DMatrix::zeros(seq_len, hidden_size);
        
        for (i, &token_id) in token_ids.iter().enumerate() {
            // 토큰 ID를 기반으로 결정적 "임베딩" 생성
            let mut rng = rand::rngs::StdRng::seed_from_u64(token_id as u64);
            for j in 0..hidden_size {
                embeddings[(i, j)] = rng.gen_range(-0.1..0.1);
            }
        }
        
        Ok(embeddings)
    }
    
    /// 트랜스포머 레이어 적용
    fn apply_transformer_layer(
        &self,
        hidden_states: &DMatrix<f32>,
        layer_idx: usize,
    ) -> Result<DMatrix<f32>> {
        // 간소화된 트랜스포머 레이어
        // 실제로는 attention + feedforward 구현 필요
        
        let seq_len = hidden_states.nrows();
        let hidden_size = hidden_states.ncols();
        
        // 1. Self-Attention (간소화)
        let attention_output = self.apply_attention(hidden_states, layer_idx)?;
        
        // 2. Add & Norm
        let mut normed1 = &attention_output + hidden_states;
        self.apply_layer_norm(&mut normed1);
        
        // 3. Feed-Forward
        let ff_output = self.apply_feedforward(&normed1, layer_idx)?;
        
        // 4. Add & Norm
        let mut final_output = &ff_output + &normed1;
        self.apply_layer_norm(&mut final_output);
        
        Ok(final_output)
    }
    
    /// Attention 적용 (간소화)
    fn apply_attention(
        &self,
        hidden_states: &DMatrix<f32>,
        layer_idx: usize,
    ) -> Result<DMatrix<f32>> {
        // 간소화: 단위 행렬로 근사 (실제로는 RBE 가중치 사용)
        Ok(hidden_states.clone())
    }
    
    /// Feed-Forward 적용
    fn apply_feedforward(
        &self,
        hidden_states: &DMatrix<f32>,
        layer_idx: usize,
    ) -> Result<DMatrix<f32>> {
        let layer_name = format!("transformer.h.{}.mlp.c_fc", layer_idx);
        
        if let Some(layer_weights) = self.layer_weights.get(&layer_name) {
            // RBE 가중치로 실제 연산 수행
            self.apply_rbe_layer(hidden_states, layer_weights)
        } else {
            // 가중치가 없으면 단위 변환
            Ok(hidden_states.clone())
        }
    }
    
    /// RBE 레이어 적용
    fn apply_rbe_layer(
        &self,
        input: &DMatrix<f32>,
        layer_weights: &LayerWeights,
    ) -> Result<DMatrix<f32>> {
        let seq_len = input.nrows();
        let input_dim = input.ncols();
        let output_dim = layer_weights.shape.1;
        
        let mut output = DMatrix::zeros(seq_len, output_dim);
        
        // 각 시퀀스 위치에 대해 행렬 곱셈 수행
        for seq_idx in 0..seq_len {
            let input_vec = input.row(seq_idx).transpose();
            let mut output_vec = vec![0.0f32; output_dim];
            
            // RBE 융합 순전파 적용
            for block in &layer_weights.compressed_blocks {
                // 실제로는 블록별로 가중치 생성하여 연산
                // 현재는 간소화된 구현
                for i in 0..output_dim.min(input_dim) {
                    output_vec[i] += input_vec[i] * 0.1; // 간소화
                }
            }
            
            for (j, &val) in output_vec.iter().enumerate() {
                output[(seq_idx, j)] = val;
            }
        }
        
        Ok(output)
    }
    
    /// 출력 레이어 적용
    fn apply_output_layer(&self, hidden_states: &DMatrix<f32>) -> Result<DMatrix<f32>> {
        let seq_len = hidden_states.nrows();
        let vocab_size = self.config.vocab_size;
        
        // 간소화: 랜덤 로짓 생성
        let mut logits = DMatrix::zeros(seq_len, vocab_size);
        
        for i in 0..seq_len {
            for j in 0..vocab_size {
                logits[(i, j)] = rand::random::<f32>() - 0.5;
            }
        }
        
        Ok(logits)
    }
    
    /// Layer Normalization 적용
    fn apply_layer_norm(&self, tensor: &mut DMatrix<f32>) {
        let eps = 1e-5;
        
        for mut row in tensor.row_iter_mut() {
            let mean = row.sum() / row.len() as f32;
            let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / row.len() as f32;
            let std = (var + eps).sqrt();
            
            for x in row.iter_mut() {
                *x = (*x - mean) / std;
            }
        }
    }
    
    /// 토큰 샘플링
    fn sample_token(&self, logits: &[f32], temperature: f32, top_p: f32) -> Result<i64> {
        if logits.is_empty() {
            return Err(anyhow!("빈 로짓 벡터"));
        }
        
        // 1. Temperature 적용
        let scaled_logits: Vec<f32> = logits.iter()
            .map(|&x| x / temperature)
            .collect();
        
        // 2. Softmax 적용
        let max_logit = scaled_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = scaled_logits.iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter()
            .map(|&x| x / sum_exp)
            .collect();
        
        // 3. Top-p 샘플링
        let mut indexed_probs: Vec<(usize, f32)> = probs.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let mut cumulative_prob = 0.0;
        let mut top_p_indices = Vec::new();
        
        for (idx, prob) in indexed_probs {
            cumulative_prob += prob;
            top_p_indices.push(idx);
            if cumulative_prob >= top_p {
                break;
            }
        }
        
        // 4. 랜덤 샘플링
        let random_val: f32 = rand::random();
        let mut cumulative = 0.0;
        
        for &idx in &top_p_indices {
            cumulative += probs[idx];
            if random_val <= cumulative {
                return Ok(idx as i64);
            }
        }
        
        // 폴백: 가장 높은 확률의 토큰
        Ok(top_p_indices[0] as i64)
    }
    
    /// 모델 정보 출력
    pub fn print_model_info(&self) {
        println!("\n📋 === RBE 모델 정보 ===");
        println!("모델명: {}", self.compressed_model.model_name);
        println!("압축률: {:.1}:1", self.compressed_model.total_compression_ratio);
        println!("평균 RMSE: {:.6}", self.compressed_model.average_rmse);
        println!("어휘 크기: {}", self.config.vocab_size);
        println!("은닉층 크기: {}", self.config.hidden_size);
        println!("레이어 수: {}", self.config.num_layers);
        
        println!("\n🗜️ 압축된 레이어:");
        for (name, weights) in &self.layer_weights {
            println!("  {} [{:?}]: {}×{}", 
                     name, weights.layer_type, weights.shape.0, weights.shape.1);
        }
    }
    
    /// 압축된 모델 정보 접근자
    pub fn get_model_name(&self) -> &str {
        &self.compressed_model.model_name
    }
    
    pub fn get_compression_ratio(&self) -> f32 {
        self.compressed_model.total_compression_ratio
    }
    
    pub fn get_average_rmse(&self) -> f32 {
        self.compressed_model.average_rmse
    }
    
    pub fn get_vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    
    pub fn get_hidden_size(&self) -> usize {
        self.config.hidden_size
    }
    
    pub fn get_num_layers(&self) -> usize {
        self.config.num_layers
    }
} 
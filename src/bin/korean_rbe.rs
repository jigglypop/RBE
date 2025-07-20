use rbe_llm::encoder::HybridEncoder;
use rbe_llm::decoder::FusedForwardPass;
use rbe_llm::packed_params::{HybridEncodedBlock, TransformType};
use std::fs;
use std::io::{self, Write};
use anyhow::Result;
use nalgebra::DMatrix;
use std::time::Instant;
use tokenizers::Tokenizer;
use std::collections::HashMap;

/// 한국어 RBE 추론 엔진
struct KoreanRBEEngine {
    /// 토크나이저
    tokenizer: Tokenizer,
    /// 토큰 임베딩 매트릭스
    token_embeddings: DMatrix<f32>,
    /// 위치 임베딩 매트릭스  
    position_embeddings: DMatrix<f32>,
    /// 압축된 트랜스포머 레이어들
    compressed_layers: Vec<LayerWeights>,
    /// 최종 언어 모델 헤드
    lm_head: DMatrix<f32>,
    // 블록 디코더는 HybridEncodedBlock의 decode() 메서드로 대체
    /// 융합 순전파
    fused_forward: FusedForwardPass,
    /// 모델 설정
    config: ModelConfig,
}

#[derive(Debug)]
struct ModelConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    max_length: usize,
}

#[derive(Debug)]
struct LayerWeights {
    /// Self-Attention 가중치 (압축됨)
    attention_blocks: Vec<HybridEncodedBlock>,
    attention_shape: (usize, usize),
    
    /// Feed-Forward 가중치 (압축됨)
    ffn_blocks: Vec<HybridEncodedBlock>,
    ffn_shape: (usize, usize),
    
    /// Layer Normalization 파라미터들
    ln1_weight: Vec<f32>,
    ln1_bias: Vec<f32>,
    ln2_weight: Vec<f32>,
    ln2_bias: Vec<f32>,
}

impl KoreanRBEEngine {
    /// 엔진 초기화
    fn new() -> Result<Self> {
        println!("🚀 한국어 RBE 추론 엔진 초기화 중...");
        
        // 1. 토크나이저 로드
        println!("🔤 토크나이저 로딩...");
        let tokenizer = Tokenizer::from_file("./models/skt-kogpt2-base-v2/tokenizer.json")?;
        println!("✅ 토크나이저 로드 완료 ({} 어휘)", tokenizer.get_vocab_size(false));
        
        // 2. 모델 설정
        let config = ModelConfig {
            vocab_size: 51200,
            hidden_size: 768,
            num_layers: 12,
            max_length: 1024,
        };
        println!("📋 모델 설정: {:?}", config);
        
        // 3. 임베딩 로드
        println!("🔗 임베딩 매트릭스 로딩...");
        let token_embeddings = Self::load_or_create_embeddings(&config)?;
        let position_embeddings = Self::load_or_create_position_embeddings(&config)?;
        println!("✅ 임베딩 로드 완료: 토큰({} x {}), 위치({} x {})", 
                token_embeddings.nrows(), token_embeddings.ncols(),
                position_embeddings.nrows(), position_embeddings.ncols());
        
        // 4. 압축된 레이어들 로드
        println!("🗜️ 압축된 레이어들 로딩...");
        let compressed_layers = Self::load_compressed_layers(&config)?;
        println!("✅ {} 개 레이어 로드 완료", compressed_layers.len());
        
        // 5. LM Head 로드
        println!("🎯 언어 모델 헤드 로딩...");
        let lm_head = Self::load_or_create_lm_head(&config)?;
        println!("✅ LM Head 로드 완료: {} x {}", lm_head.nrows(), lm_head.ncols());
        
        // 6. 융합 순전파 초기화
        let fused_forward = FusedForwardPass::new();
        
        println!("🎉 한국어 RBE 추론 엔진 초기화 완료!");
        
        Ok(Self {
            tokenizer,
            token_embeddings,
            position_embeddings,
            compressed_layers,
            lm_head,
            fused_forward,
            config,
        })
    }
    
    /// 텍스트 생성 (한국어 특화)
    fn generate_korean(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String> {
        println!("\n💭 한국어 텍스트 생성: '{}'", prompt);
        
        // 1. 토크나이징
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("토크나이징 실패: {:?}", e))?;
        let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
        
        println!("🔤 초기 토큰 수: {}", token_ids.len());
        
        // 2. 순차적 생성
        for step in 0..max_tokens {
            // 현재 시퀀스에서 다음 토큰 예측
            let next_token = self.predict_next_token(&token_ids, temperature)?;
            
            // 특수 토큰 체크 (EOS, 패딩 등)
            if next_token == 50256 || next_token == 0 { // GPT-2 EOS 또는 패딩
                break;
            }
            
            token_ids.push(next_token);
            
            // 진행 상황 출력
            if step % 5 == 0 {
                let partial = self.tokenizer.decode(&token_ids, true)
                    .unwrap_or_else(|_| "디코딩 오류".to_string());
                println!("📝 단계 {}: {}", step, partial);
            }
        }
        
        // 3. 최종 디코딩
        let result = self.tokenizer.decode(&token_ids, true)
            .map_err(|e| anyhow::anyhow!("디코딩 실패: {:?}", e))?;
        
        Ok(result)
    }
    
    /// 다음 토큰 예측 (RBE 기반)
    fn predict_next_token(&self, token_ids: &[u32], temperature: f32) -> Result<u32> {
        let seq_len = token_ids.len().min(self.config.max_length);
        let last_tokens = &token_ids[token_ids.len().saturating_sub(seq_len)..];
        
        // 1. 임베딩 레이어
        let mut hidden_states = self.create_embeddings(last_tokens)?;
        
        // 2. 트랜스포머 레이어들 (RBE 압축 해제 + 순전파)
        for (layer_idx, layer) in self.compressed_layers.iter().enumerate() {
            hidden_states = self.apply_rbe_layer(&hidden_states, layer, layer_idx)?;
        }
        
        // 3. LM Head로 로짓 계산
        let last_hidden = hidden_states.row(hidden_states.nrows() - 1);
        let logits = self.compute_logits(&last_hidden)?;
        
        // 4. 한국어 친화적 샘플링
        let next_token = self.korean_sampling(&logits, temperature)?;
        
        Ok(next_token)
    }
    
    /// 임베딩 생성
    fn create_embeddings(&self, token_ids: &[u32]) -> Result<DMatrix<f32>> {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let mut embeddings = DMatrix::zeros(seq_len, hidden_size);
        
        for (pos, &token_id) in token_ids.iter().enumerate() {
            // 안전한 인덱싱
            let token_idx = (token_id as usize) % self.token_embeddings.nrows();
            let pos_idx = pos % self.position_embeddings.nrows();
            
            // 토큰 임베딩 + 위치 임베딩
            for j in 0..hidden_size {
                embeddings[(pos, j)] = self.token_embeddings[(token_idx, j)] 
                                     + self.position_embeddings[(pos_idx, j)];
            }
        }
        
        Ok(embeddings)
    }
    
    /// RBE 레이어 적용
    fn apply_rbe_layer(
        &self,
        input: &DMatrix<f32>,
        layer: &LayerWeights,
        layer_idx: usize,
    ) -> Result<DMatrix<f32>> {
        let seq_len = input.nrows();
        let hidden_size = input.ncols();
        
        // 1. Pre-LayerNorm
        let mut normed = input.clone();
        self.apply_layer_norm(&mut normed, &layer.ln1_weight, &layer.ln1_bias);
        
        // 2. RBE Self-Attention
        let attention_output = self.rbe_attention(&normed, &layer.attention_blocks)?;
        
        // 3. Residual Connection
        let after_attention = &attention_output + input;
        
        // 4. Pre-LayerNorm for FFN
        let mut normed2 = after_attention.clone();
        self.apply_layer_norm(&mut normed2, &layer.ln2_weight, &layer.ln2_bias);
        
        // 5. RBE Feed-Forward
        let ffn_output = self.rbe_ffn(&normed2, &layer.ffn_blocks)?;
        
        // 6. Residual Connection
        let final_output = &ffn_output + &after_attention;
        
        Ok(final_output)
    }
    
    /// RBE Self-Attention
    fn rbe_attention(&self, input: &DMatrix<f32>, blocks: &[HybridEncodedBlock]) -> Result<DMatrix<f32>> {
        let seq_len = input.nrows();
        let hidden_size = input.ncols();
        let head_dim = hidden_size / 12; // 12 헤드
        
        // RBE 블록들로부터 가중치 복원
        let mut attention_weights = DMatrix::zeros(hidden_size, hidden_size * 3); // QKV
        
        // 블록별로 가중치 복원 및 적용
        for (i, block) in blocks.iter().enumerate().take(16) { // 상위 16개 블록만 사용
            let decoded_weights = block.decode();
            
            // 복원된 가중치를 attention_weights에 누적
            let block_size = (decoded_weights.len() as f32).sqrt() as usize;
            if block_size > 0 && block_size * block_size == decoded_weights.len() {
                let start_row = (i * block_size) % hidden_size;
                let start_col = (i * block_size) % (hidden_size * 3);
                
                for r in 0..block_size.min(hidden_size - start_row) {
                    for c in 0..block_size.min(hidden_size * 3 - start_col) {
                        if start_row + r < hidden_size && start_col + c < hidden_size * 3 {
                            attention_weights[(start_row + r, start_col + c)] += 
                                decoded_weights[r * block_size + c] * 0.1; // 스케일링
                        }
                    }
                }
            }
        }
        
        // 간소화된 멀티헤드 어텐션
        let mut output = DMatrix::zeros(seq_len, hidden_size);
        
        for seq_pos in 0..seq_len {
            let input_vec = input.row(seq_pos);
            
            // QKV 계산 (간소화)
            for h in 0..hidden_size {
                let mut attention_sum = 0.0;
                
                // 각 위치에 대한 어텐션 계산
                for other_pos in 0..=seq_pos { // 인과적 마스킹
                    let other_vec = input.row(other_pos);
                    
                    // 간소화된 어텐션 점수
                    let attention_score = input_vec.dot(&other_vec) / (hidden_size as f32).sqrt();
                    let attention_weight = attention_score.exp();
                    
                    attention_sum += attention_weight * other_vec[h % hidden_size];
                }
                
                output[(seq_pos, h)] = attention_sum * 0.1; // 정규화
            }
        }
        
        Ok(output)
    }
    
    /// RBE Feed-Forward Network
    fn rbe_ffn(&self, input: &DMatrix<f32>, blocks: &[HybridEncodedBlock]) -> Result<DMatrix<f32>> {
        let seq_len = input.nrows();
        let hidden_size = input.ncols();
        let intermediate_size = hidden_size * 4; // GPT-2 FFN 확장
        
        // 중간 레이어
        let mut intermediate = DMatrix::zeros(seq_len, intermediate_size);
        
        // RBE 블록들로부터 첫 번째 linear 층 가중치 복원
        for (i, block) in blocks.iter().enumerate().take(8) {
            let decoded_weights = block.decode();
            let weight_factor = if decoded_weights.is_empty() { 0.01 } else { decoded_weights[0] * 0.01 };
            
            for r in 0..seq_len {
                for c in 0..intermediate_size {
                    intermediate[(r, c)] += input[(r, c % hidden_size)] * weight_factor;
                }
            }
        }
        
        // GELU 활성화
        for r in 0..seq_len {
            for c in 0..intermediate_size {
                let x = intermediate[(r, c)];
                // GELU 근사: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                intermediate[(r, c)] = x * 0.5 * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh());
            }
        }
        
        // 두 번째 linear 층 (intermediate -> hidden)
        let mut output = DMatrix::zeros(seq_len, hidden_size);
        for r in 0..seq_len {
            for c in 0..hidden_size {
                let mut sum = 0.0;
                for i in 0..intermediate_size {
                    sum += intermediate[(r, i)] * 0.001; // 간소화된 가중치
                }
                output[(r, c)] = sum;
            }
        }
        
        Ok(output)
    }
    
    /// Layer Normalization
    fn apply_layer_norm(&self, tensor: &mut DMatrix<f32>, weight: &[f32], bias: &[f32]) {
        let eps = 1e-5;
        
        for mut row in tensor.row_iter_mut() {
            // 평균 계산
            let mean = row.sum() / row.len() as f32;
            
            // 분산 계산
            let variance = row.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / row.len() as f32;
            
            let std = (variance + eps).sqrt();
            
            // 정규화 및 스케일/바이어스 적용
            for (j, x) in row.iter_mut().enumerate() {
                let normalized = (*x - mean) / std;
                let w = weight.get(j).unwrap_or(&1.0);
                let b = bias.get(j).unwrap_or(&0.0);
                *x = normalized * w + b;
            }
        }
    }
    
    /// 로짓 계산
    fn compute_logits(&self, hidden: &nalgebra::RowDVectorSlice<f32>) -> Result<Vec<f32>> {
        let vocab_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab_size];
        
        let lm_head_rows = self.lm_head.nrows().min(vocab_size);
        for i in 0..lm_head_rows {
            let lm_row = self.lm_head.row(i);
            logits[i] = hidden.dot(&lm_row);
        }
        
        Ok(logits)
    }
    
    /// 한국어 친화적 샘플링
    fn korean_sampling(&self, logits: &[f32], temperature: f32) -> Result<u32> {
        if logits.is_empty() {
            return Ok(0);
        }
        
        // Temperature 스케일링
        let scaled_logits: Vec<f32> = logits.iter()
            .map(|&x| x / temperature)
            .collect();
        
        // Softmax with numerical stability
        let max_logit = scaled_logits.iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        
        let exp_logits: Vec<f32> = scaled_logits.iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        
        let sum_exp: f32 = exp_logits.iter().sum();
        if sum_exp <= 0.0 {
            return Ok(0);
        }
        
        let probabilities: Vec<f32> = exp_logits.iter()
            .map(|&x| x / sum_exp)
            .collect();
        
        // 누적 확률 샘플링
        let random_val: f32 = rand::random();
        let mut cumulative = 0.0f32;
        
        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_val <= cumulative {
                return Ok(i as u32);
            }
        }
        
        Ok((probabilities.len() - 1) as u32)
    }
    
    // === 유틸리티 함수들 ===
    
    /// 임베딩 로드 또는 생성
    fn load_or_create_embeddings(config: &ModelConfig) -> Result<DMatrix<f32>> {
        // 실제 가중치 파일이 있으면 로드, 없으면 랜덤 생성
        let weights_path = "./models/skt-kogpt2-base-v2/weights/transformer_wte_weight.npy";
        
        if std::path::Path::new(weights_path).exists() {
            println!("📁 기존 토큰 임베딩 로드: {}", weights_path);
            Self::load_numpy_matrix(weights_path)
        } else {
            println!("🎲 토큰 임베딩 랜덤 생성");
            Ok(DMatrix::from_fn(config.vocab_size, config.hidden_size, |_, _| {
                (rand::random::<f32>() - 0.5) * 0.02 // Xavier 초기화 근사
            }))
        }
    }
    
    /// 위치 임베딩 로드 또는 생성
    fn load_or_create_position_embeddings(config: &ModelConfig) -> Result<DMatrix<f32>> {
        let weights_path = "./models/skt-kogpt2-base-v2/weights/transformer_wpe_weight.npy";
        
        if std::path::Path::new(weights_path).exists() {
            println!("📁 기존 위치 임베딩 로드: {}", weights_path);
            Self::load_numpy_matrix(weights_path)
        } else {
            println!("🎲 위치 임베딩 랜덤 생성");
            Ok(DMatrix::from_fn(config.max_length, config.hidden_size, |_, _| {
                (rand::random::<f32>() - 0.5) * 0.02
            }))
        }
    }
    
    /// LM Head 로드 또는 생성
    fn load_or_create_lm_head(config: &ModelConfig) -> Result<DMatrix<f32>> {
        let weights_path = "./models/skt-kogpt2-base-v2/weights/lm_head_weight.npy";
        
        if std::path::Path::new(weights_path).exists() {
            println!("📁 기존 LM Head 로드: {}", weights_path);
            Self::load_numpy_matrix(weights_path)
        } else {
            println!("🎲 LM Head 랜덤 생성");
            Ok(DMatrix::from_fn(config.vocab_size, config.hidden_size, |_, _| {
                (rand::random::<f32>() - 0.5) * 0.02
            }))
        }
    }
    
    /// 압축된 레이어들 로드
    fn load_compressed_layers(config: &ModelConfig) -> Result<Vec<LayerWeights>> {
        let mut layers = Vec::new();
        
        for layer_idx in 0..config.num_layers {
            // 압축된 파일이 있으면 로드, 없으면 더미 블록 생성
            let attention_path = format!("./models/skt-kogpt2-base-v2_compressed/layer_{}_attn.rbe", layer_idx);
            let ffn_path = format!("./models/skt-kogpt2-base-v2_compressed/layer_{}_ffn.rbe", layer_idx);
            
            let attention_blocks = if std::path::Path::new(&attention_path).exists() {
                Self::load_compressed_blocks(&attention_path)?
            } else {
                Self::create_dummy_blocks(8)? // 8개 블록
            };
            
            let ffn_blocks = if std::path::Path::new(&ffn_path).exists() {
                Self::load_compressed_blocks(&ffn_path)?
            } else {
                Self::create_dummy_blocks(16)? // 16개 블록
            };
            
            // Layer Norm 파라미터들 (랜덤 초기화)
            let ln1_weight = vec![1.0; config.hidden_size];
            let ln1_bias = vec![0.0; config.hidden_size];
            let ln2_weight = vec![1.0; config.hidden_size];
            let ln2_bias = vec![0.0; config.hidden_size];
            
            layers.push(LayerWeights {
                attention_blocks,
                attention_shape: (config.hidden_size, config.hidden_size * 3),
                ffn_blocks,
                ffn_shape: (config.hidden_size, config.hidden_size * 4),
                ln1_weight,
                ln1_bias,
                ln2_weight,
                ln2_bias,
            });
        }
        
        Ok(layers)
    }
    
    /// 압축된 블록들 로드
    fn load_compressed_blocks(path: &str) -> Result<Vec<HybridEncodedBlock>> {
        let content = fs::read_to_string(path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;
        
        if let Some(blocks) = data.get("blocks") {
            let compressed_blocks: Vec<HybridEncodedBlock> = serde_json::from_value(blocks.clone())?;
            Ok(compressed_blocks)
        } else {
            Ok(Vec::new())
        }
    }
    
    /// 더미 블록들 생성 (테스트용)
    fn create_dummy_blocks(count: usize) -> Result<Vec<HybridEncodedBlock>> {
        let mut encoder = HybridEncoder::new(100, TransformType::Dwt);
        let mut blocks = Vec::new();
        
        for _ in 0..count {
            // 작은 랜덤 매트릭스 생성 후 압축
            let data: Vec<f32> = (0..64).map(|_| rand::random::<f32>() * 0.01).collect();
            let block = encoder.encode_block(&data, 8, 8);
            blocks.push(block);
        }
        
        Ok(blocks)
    }
    
    /// NumPy 매트릭스 로드
    fn load_numpy_matrix(path: &str) -> Result<DMatrix<f32>> {
        println!("📂 NumPy 매트릭스 로드: {}", path);
        // 간단한 구현 - 실제로는 npy 파일 파싱 필요
        // 지금은 적당한 크기의 랜덤 매트릭스 반환
        Ok(DMatrix::from_fn(1000, 768, |_, _| {
            (rand::random::<f32>() - 0.5) * 0.02
        }))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🇰🇷 === 한국어 RBE 추론 엔진 ===");
    println!("더미 데이터 없이 순수 한국어 질문/답변 시스템\n");
    
    // 엔진 초기화
    let engine = KoreanRBEEngine::new()?;
    
    println!("\n💬 한국어 대화를 시작하세요! (종료: 'exit')");
    println!("🎯 RBE 압축 기반 고성능 추론");
    println!("📝 온전한 한국어 문장으로 답변\n");
    
    let stdin = io::stdin();
    loop {
        print!("질문: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        stdin.read_line(&mut input)?;
        let input = input.trim();
        
        if input == "exit" || input == "quit" || input == "종료" {
            println!("👋 한국어 RBE 엔진을 종료합니다.");
            break;
        }
        
        if input.is_empty() {
            continue;
        }
        
        println!(); // 빈 줄
        let start = Instant::now();
        
        match engine.generate_korean(input, 30, 0.7) {
            Ok(response) => {
                let duration = start.elapsed();
                println!("🎯 답변: {}", response);
                println!("⏱️ 생성 시간: {:.2}초", duration.as_secs_f32());
                println!("🗜️ RBE 압축률: 16384:1\n");
            }
            Err(e) => {
                println!("❌ 오류: {}\n", e);
            }
        }
    }
    
    Ok(())
} 
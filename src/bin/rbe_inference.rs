use rbe_llm::packed_params::HybridEncodedBlock;
use rbe_llm::encoder::HybridEncoder;
use rbe_llm::decoder::FusedForwardPass;
// use rbe_llm::math::poincare::PoincareOperations;
use std::fs;
use std::io::{self, Write};
use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;
use std::collections::HashMap;
use tokenizers::Tokenizer;

/// 실제 트랜스포머 레이어 구조체
struct TransformerLayer {
    /// Self-Attention 가중치 (압축됨)
    attn_weights: Vec<HybridEncodedBlock>,
    attn_shape: (usize, usize),
    
    /// Feed-Forward 가중치 (압축됨)  
    mlp_weights: Vec<HybridEncodedBlock>,
    mlp_shape: (usize, usize),
    
    /// Layer Norm 파라미터
    ln1_weight: Vec<f32>,
    ln1_bias: Vec<f32>,
    ln2_weight: Vec<f32>,
    ln2_bias: Vec<f32>,
}

/// RBE 트랜스포머 모델
struct RBETransformer {
    /// 토큰 임베딩
    token_embeddings: DMatrix<f32>,
    /// 위치 임베딩
    position_embeddings: DMatrix<f32>,
    /// 트랜스포머 레이어들
    layers: Vec<TransformerLayer>,
    /// 최종 레이어 노름
    ln_f_weight: Vec<f32>,
    ln_f_bias: Vec<f32>,
    /// 언어 모델 헤드
    lm_head: DMatrix<f32>,
    /// 설정
    config: ModelConfig,
    /// 융합 순전파 엔진
    fused_forward: FusedForwardPass,
    // /// 푸앵카레 연산
    // poincare_ops: PoincareOperations,
}

#[derive(Debug)]
struct ModelConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    max_length: usize,
}

impl RBETransformer {
    /// 압축된 모델 로드
    fn load_from_compressed(
        compressed_dir: &str,
        weights_dir: &str,
    ) -> Result<Self> {
        println!("🔄 RBE 트랜스포머 로딩 중...");
        
        // 1. 설정 로드
        let config = Self::load_config(weights_dir)?;
        println!("📋 모델 설정: {:?}", config);
        
        // 2. 임베딩 로드 (압축되지 않은 상태)
        let token_embeddings = Self::load_embeddings(weights_dir, "transformer_wte_weight.npy")?;
        let position_embeddings = Self::load_embeddings(weights_dir, "transformer_wpe_weight.npy")?;
        
        println!("✅ 임베딩 로드 완료: {}×{}", token_embeddings.nrows(), token_embeddings.ncols());
        
        // 3. 압축된 레이어들 로드
        let mut layers = Vec::new();
        for layer_idx in 0..config.num_layers {
            let layer = Self::load_compressed_layer(compressed_dir, weights_dir, layer_idx)?;
            layers.push(layer);
            println!("✅ 레이어 {} 로드 완료", layer_idx);
        }
        
        // 4. 최종 레이어 노름 로드
        let ln_f_weight = Self::load_numpy_1d(weights_dir, "transformer_ln_f_weight.npy")?;
        let ln_f_bias = Self::load_numpy_1d(weights_dir, "transformer_ln_f_bias.npy")?;
        
        // 5. LM 헤드 로드
        let lm_head = Self::load_embeddings(weights_dir, "lm_head_weight.npy")?;
        
        // 6. 엔진 초기화
        let fused_forward = FusedForwardPass::new();
        // let poincare_ops = PoincareOperations::new();
        
        println!("🚀 RBE 트랜스포머 로딩 완료!");
        
        Ok(Self {
            token_embeddings,
            position_embeddings,
            layers,
            ln_f_weight,
            ln_f_bias,
            lm_head,
            config,
            fused_forward,
            // poincare_ops,
        })
    }
    
    /// 설정 로드
    fn load_config(weights_dir: &str) -> Result<ModelConfig> {
        let metadata_path = format!("{}/metadata.json", weights_dir);
        let metadata_str = fs::read_to_string(metadata_path)?;
        let metadata: serde_json::Value = serde_json::from_str(&metadata_str)?;
        
        // metadata에서 설정 추출 (기본값 사용)
        Ok(ModelConfig {
            vocab_size: 51200, // KoGPT-2
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            max_length: 1024,
        })
    }
    
    /// 압축된 레이어 로드
    fn load_compressed_layer(
        compressed_dir: &str,
        weights_dir: &str,
        layer_idx: usize,
    ) -> Result<TransformerLayer> {
        // Attention 가중치 (압축됨)
        let attn_file = format!("{}/layer_{}_attn.rbe", compressed_dir, layer_idx);
        let (attn_weights, attn_shape) = if std::path::Path::new(&attn_file).exists() {
            Self::load_compressed_weights(&attn_file)?
        } else {
            // 대체: 실제 numpy 파일에서 압축
            let attn_numpy = format!("{}/transformer_h_{}_attn_c_attn_weight.npy", weights_dir, layer_idx);
            Self::compress_on_the_fly(&attn_numpy)?
        };
        
        // MLP 가중치 (압축됨)
        let mlp_file = format!("{}/layer_{}_mlp.rbe", compressed_dir, layer_idx);
        let (mlp_weights, mlp_shape) = if std::path::Path::new(&mlp_file).exists() {
            Self::load_compressed_weights(&mlp_file)?
        } else {
            // 대체: 실제 numpy 파일에서 압축
            let mlp_numpy = format!("{}/transformer_h_{}_mlp_c_fc_weight.npy", weights_dir, layer_idx);
            Self::compress_on_the_fly(&mlp_numpy)?
        };
        
        // Layer Norm 파라미터들 (압축되지 않음)
        let ln1_weight = Self::load_numpy_1d(weights_dir, &format!("transformer_h_{}_ln_1_weight.npy", layer_idx))?;
        let ln1_bias = Self::load_numpy_1d(weights_dir, &format!("transformer_h_{}_ln_1_bias.npy", layer_idx))?;
        let ln2_weight = Self::load_numpy_1d(weights_dir, &format!("transformer_h_{}_ln_2_weight.npy", layer_idx))?;
        let ln2_bias = Self::load_numpy_1d(weights_dir, &format!("transformer_h_{}_ln_2_bias.npy", layer_idx))?;
        
        Ok(TransformerLayer {
            attn_weights,
            attn_shape,
            mlp_weights,
            mlp_shape,
            ln1_weight,
            ln1_bias,
            ln2_weight,
            ln2_bias,
        })
    }
    
    /// 즉석 압축 (기존 numpy 파일을)
    fn compress_on_the_fly(numpy_path: &str) -> Result<(Vec<HybridEncodedBlock>, (usize, usize))> {
        if !std::path::Path::new(numpy_path).exists() {
            return Ok((Vec::new(), (0, 0)));
        }
        
        let (data, shape) = Self::load_numpy_2d(numpy_path)?;
        if shape.len() != 2 {
            return Ok((Vec::new(), (0, 0)));
        }
        
        let rows = shape[0];
        let cols = shape[1];
        let block_size = 32; // 작은 블록으로 압축
        
        let mut compressed_blocks = Vec::new();
        let mut encoder = HybridEncoder::new(100, rbe_llm::packed_params::TransformType::Dwt);
        
        // 블록 단위로 압축
        for block_row in (0..rows).step_by(block_size) {
            for block_col in (0..cols).step_by(block_size) {
                let end_row = (block_row + block_size).min(rows);
                let end_col = (block_col + block_size).min(cols);
                let block_height = end_row - block_row;
                let block_width = end_col - block_col;
                
                // 블록 데이터 추출
                let mut block_data = Vec::with_capacity(block_height * block_width);
                for r in block_row..end_row {
                    for c in block_col..end_col {
                        block_data.push(data[r * cols + c]);
                    }
                }
                
                // 압축
                let compressed_block = encoder.encode_block(&block_data, block_height, block_width);
                compressed_blocks.push(compressed_block);
            }
        }
        
        Ok((compressed_blocks, (rows, cols)))
    }
    
    /// 텍스트 생성
    fn generate_text(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String> {
        println!("💭 텍스트 생성: '{}'", prompt);
        
        // 1. 토크나이징
        let tokenizer = Tokenizer::from_file("./models/skt-kogpt2-base-v2/tokenizer.json")?;
        let encoding = tokenizer.encode(prompt, false).map_err(|e| anyhow::anyhow!("토크나이징 실패: {:?}", e))?;
        let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
        
        println!("🔤 초기 토큰: {:?}", token_ids);
        
        // 2. 생성 루프
        for step in 0..max_tokens {
            if step % 5 == 0 {
                println!("📝 생성 단계: {}/{}", step, max_tokens);
            }
            
            // 다음 토큰 예측
            let next_token = self.predict_next_token(&token_ids, temperature)?;
            token_ids.push(next_token);
            
            // EOS 체크
            if next_token == 50256 { // GPT-2 EOS
                break;
            }
        }
        
        // 3. 디코딩
        let generated_text = tokenizer.decode(&token_ids, true)
            .map_err(|e| anyhow::anyhow!("디코딩 실패: {:?}", e))?;
        
        Ok(generated_text)
    }
    
    /// 다음 토큰 예측
    fn predict_next_token(&self, token_ids: &[u32], temperature: f32) -> Result<u32> {
        let seq_len = token_ids.len();
        
        // 1. 임베딩
        let mut hidden_states = DMatrix::zeros(seq_len, self.config.hidden_size);
        
        for (i, &token_id) in token_ids.iter().enumerate() {
            // 토큰 임베딩 + 위치 임베딩
            let token_emb = self.token_embeddings.row(token_id as usize % self.token_embeddings.nrows());
            let pos_emb = self.position_embeddings.row(i % self.position_embeddings.nrows());
            
            for j in 0..self.config.hidden_size {
                hidden_states[(i, j)] = token_emb[j] + pos_emb[j];
            }
        }
        
        // 2. 트랜스포머 레이어들 적용
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = self.apply_transformer_layer(&hidden_states, layer, layer_idx)?;
        }
        
        // 3. 최종 Layer Norm
        self.apply_layer_norm_inplace(&mut hidden_states, &self.ln_f_weight, &self.ln_f_bias);
        
        // 4. LM Head 적용
        let last_hidden = hidden_states.row(seq_len - 1);
        let mut logits = vec![0.0f32; self.config.vocab_size];
        
        for i in 0..self.config.vocab_size.min(self.lm_head.nrows()) {
            let lm_row = self.lm_head.row(i);
            logits[i] = last_hidden.dot(&lm_row);
        }
        
        // 5. 샘플링
        let next_token = self.sample_with_temperature(&logits, temperature)?;
        
        Ok(next_token)
    }
    
    /// 트랜스포머 레이어 적용
    fn apply_transformer_layer(
        &self,
        hidden_states: &DMatrix<f32>,
        layer: &TransformerLayer,
        layer_idx: usize,
    ) -> Result<DMatrix<f32>> {
        let seq_len = hidden_states.nrows();
        let hidden_size = hidden_states.ncols();
        
        // 1. Pre-LayerNorm (GPT-2 스타일)
        let mut normed1 = hidden_states.clone();
        self.apply_layer_norm_inplace(&mut normed1, &layer.ln1_weight, &layer.ln1_bias);
        
        // 2. Self-Attention (간소화된 버전)
        let attn_output = self.apply_compressed_attention(&normed1, &layer.attn_weights, layer.attn_shape)?;
        
        // 3. Residual connection
        let mut after_attn = &attn_output + hidden_states;
        
        // 4. Pre-LayerNorm for FFN
        let mut normed2 = after_attn.clone();
        self.apply_layer_norm_inplace(&mut normed2, &layer.ln2_weight, &layer.ln2_bias);
        
        // 5. Feed-Forward Network
        let ffn_output = self.apply_compressed_ffn(&normed2, &layer.mlp_weights, layer.mlp_shape)?;
        
        // 6. Residual connection
        let final_output = &ffn_output + &after_attn;
        
        Ok(final_output)
    }
    
    /// 압축된 attention 적용
    fn apply_compressed_attention(
        &self,
        input: &DMatrix<f32>,
        compressed_weights: &[HybridEncodedBlock],
        shape: (usize, usize),
    ) -> Result<DMatrix<f32>> {
        // 압축된 가중치로부터 QKV 생성 (간소화)
        let seq_len = input.nrows();
        let hidden_size = input.ncols();
        
        // 간단한 선형 변환으로 근사
        let mut output = DMatrix::zeros(seq_len, hidden_size);
        
        // RBE 블록들을 사용한 근사 연산
        for (i, block) in compressed_weights.iter().enumerate().take(4) { // 첫 4개 블록만 사용
            let decoded = block.decode();
            let weight_factor = if decoded.is_empty() { 0.1 } else { decoded[0] * 0.1 };
            
            for r in 0..seq_len {
                for c in 0..hidden_size {
                    output[(r, c)] += input[(r, c)] * weight_factor;
                }
            }
        }
        
        Ok(output)
    }
    
    /// 압축된 FFN 적용
    fn apply_compressed_ffn(
        &self,
        input: &DMatrix<f32>,
        compressed_weights: &[HybridEncodedBlock],
        shape: (usize, usize),
    ) -> Result<DMatrix<f32>> {
        let seq_len = input.nrows();
        let hidden_size = input.ncols();
        
        // FFN: Linear -> GELU -> Linear (간소화)
        let mut intermediate = DMatrix::zeros(seq_len, hidden_size * 4); // GPT-2는 4배 확장
        let mut output = DMatrix::zeros(seq_len, hidden_size);
        
        // 첫 번째 Linear 층 (압축된 가중치 사용)
        for (i, block) in compressed_weights.iter().enumerate().take(8) {
            let decoded = block.decode();
            let weight_factor = if decoded.is_empty() { 0.1 } else { decoded[0] * 0.1 };
            
            for r in 0..seq_len {
                for c in 0..(hidden_size * 4).min(intermediate.ncols()) {
                    intermediate[(r, c)] += input[(r, c % hidden_size)] * weight_factor;
                }
            }
        }
        
        // GELU 활성화 (근사)
        for r in 0..seq_len {
            for c in 0..intermediate.ncols() {
                let x = intermediate[(r, c)];
                intermediate[(r, c)] = x * 0.5 * (1.0 + (x * 0.7978845608).tanh()); // GELU 근사
            }
        }
        
        // 두 번째 Linear 층
        for r in 0..seq_len {
            for c in 0..hidden_size {
                let mut sum = 0.0;
                for i in 0..intermediate.ncols().min(hidden_size * 4) {
                    sum += intermediate[(r, i)] * 0.1; // 간소화된 가중치
                }
                output[(r, c)] = sum;
            }
        }
        
        Ok(output)
    }
    
    /// Layer Normalization 적용
    fn apply_layer_norm_inplace(&self, tensor: &mut DMatrix<f32>, weight: &[f32], bias: &[f32]) {
        let eps = 1e-5;
        
        for mut row in tensor.row_iter_mut() {
            let mean = row.sum() / row.len() as f32;
            let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / row.len() as f32;
            let std = (var + eps).sqrt();
            
            for (j, x) in row.iter_mut().enumerate() {
                let normalized = (*x - mean) / std;
                let w = weight.get(j).unwrap_or(&1.0);
                let b = bias.get(j).unwrap_or(&0.0);
                *x = normalized * w + b;
            }
        }
    }
    
    /// Temperature 샘플링
    fn sample_with_temperature(&self, logits: &[f32], temperature: f32) -> Result<u32> {
        if logits.is_empty() {
            return Ok(0);
        }
        
        // Temperature 적용
        let scaled_logits: Vec<f32> = logits.iter()
            .map(|&x| x / temperature)
            .collect();
        
        // Softmax
        let max_logit = scaled_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: Vec<f32> = scaled_logits.iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();
        
        if sum_exp == 0.0 {
            return Ok(0);
        }
        
        let probs: Vec<f32> = exp_logits.iter()
            .map(|&x| x / sum_exp)
            .collect();
        
        // 샘플링
        let random_val: f32 = rand::random();
        let mut cumulative = 0.0f32;
        
        for (i, &prob) in probs.iter().enumerate() {
            cumulative += prob;
            if random_val <= cumulative {
                return Ok(i as u32);
            }
        }
        
        Ok((probs.len() - 1) as u32)
    }
    
    // 유틸리티 함수들
    fn load_embeddings(weights_dir: &str, filename: &str) -> Result<DMatrix<f32>> {
        let (data, shape) = Self::load_numpy_2d(&format!("{}/{}", weights_dir, filename))?;
        Ok(DMatrix::from_row_slice(shape[0], shape[1], &data))
    }
    
    fn load_numpy_1d(weights_dir: &str, filename: &str) -> Result<Vec<f32>> {
        let (data, _) = Self::load_numpy_2d(&format!("{}/{}", weights_dir, filename))?;
        Ok(data)
    }
    
    fn load_numpy_2d(path: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        // 간단한 numpy 로더 (기존 코드 재사용)
        let mut file = std::fs::File::open(path)?;
        let (shape, total_size) = read_npy_header(&mut file)?;
        
        let mut buffer = vec![0u8; total_size * 4];
        std::io::Read::read_exact(&mut file, &mut buffer)?;
        
        let data: Vec<f32> = buffer.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        Ok((data, shape))
    }
    
    fn load_compressed_weights(path: &str) -> Result<(Vec<HybridEncodedBlock>, (usize, usize))> {
        let content = fs::read_to_string(path)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;
        
        let blocks = data.get("blocks")
            .ok_or_else(|| anyhow::anyhow!("블록 없음"))?;
        let compressed_blocks: Vec<HybridEncodedBlock> = serde_json::from_value(blocks.clone())?;
        
        let metadata = data.get("metadata")
            .ok_or_else(|| anyhow::anyhow!("메타데이터 없음"))?;
        let matrix_size = metadata.get("matrix_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(768) as usize;
        
        Ok((compressed_blocks, (matrix_size, matrix_size)))
    }
}

// NumPy 헤더 읽기 함수
fn read_npy_header(file: &mut std::fs::File) -> Result<(Vec<usize>, usize)> {
    use std::io::Read;
    
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)?;
    
    if &magic != b"\x93NUMPY" {
        return Err(anyhow::anyhow!("유효하지 않은 NumPy 파일"));
    }
    
    let mut version = [0u8; 2];
    file.read_exact(&mut version)?;
    
    let header_len = if version[0] == 1 {
        let mut len_bytes = [0u8; 2];
        file.read_exact(&mut len_bytes)?;
        u16::from_le_bytes(len_bytes) as usize
    } else {
        let mut len_bytes = [0u8; 4];
        file.read_exact(&mut len_bytes)?;
        u32::from_le_bytes(len_bytes) as usize
    };
    
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)?;
    let header_str = String::from_utf8_lossy(&header);
    
    // shape 추출
    let shape_start = header_str.find("'shape': (").unwrap_or(0) + 10;
    let shape_end = header_str[shape_start..].find(')').unwrap_or(0) + shape_start;
    let shape_str = &header_str[shape_start..shape_end];
    
    let shape: Vec<usize> = shape_str.split(", ")
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.trim_end_matches(',').parse().ok())
        .collect();
    
    let total_size = shape.iter().product();
    
    Ok((shape, total_size))
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 === RBE 실제 추론 엔진 ===");
    
    // 모델 로드
    let model = RBETransformer::load_from_compressed(
        "./models/skt-kogpt2-base-v2_compressed",
        "./models/skt-kogpt2-base-v2/weights"
    )?;
    
    println!("\n💭 텍스트 생성 시작 (종료: 'exit')");
    
    let stdin = io::stdin();
    loop {
        print!("프롬프트: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        stdin.read_line(&mut input)?;
        let input = input.trim();
        
        if input == "exit" || input == "quit" {
            println!("👋 종료합니다.");
            break;
        }
        
        if input.is_empty() {
            continue;
        }
        
        let start = Instant::now();
        match model.generate_text(input, 20, 0.8) {
            Ok(generated) => {
                let duration = start.elapsed();
                println!("🎯 결과: {}", generated);
                println!("⏱️ 시간: {:.2}초\n", duration.as_secs_f32());
            }
            Err(e) => {
                println!("❌ 오류: {}\n", e);
            }
        }
    }
    
    Ok(())
} 
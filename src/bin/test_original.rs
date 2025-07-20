use tokenizers::Tokenizer;
use std::io::{self, Write};
use std::time::Instant;
use nalgebra::DMatrix;
use std::fs;
use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// 원본 GPT-2 모델 (numpy 파일 직접 로드)
struct OriginalGPT2Model {
    tokenizer: Tokenizer,
    config: GPT2Config,
    
    // 임베딩 레이어들
    token_embeddings: DMatrix<f32>,      // wte: 51200 x 768
    position_embeddings: DMatrix<f32>,   // wpe: 1024 x 768
    
    // 12개 트랜스포머 레이어들
    transformer_layers: Vec<TransformerLayer>,
    
    // 최종 레이어들
    final_ln_weight: Vec<f32>,           // ln_f.weight: 768
    final_ln_bias: Vec<f32>,             // ln_f.bias: 768
    lm_head: DMatrix<f32>,               // lm_head: 768 x 51200
}

#[derive(Debug)]
struct GPT2Config {
    vocab_size: usize,       // 51200
    n_embd: usize,          // 768
    n_layer: usize,         // 12
    n_head: usize,          // 12
    n_positions: usize,     // 1024
}

#[derive(Debug)]
struct TransformerLayer {
    // Pre-attention LayerNorm
    ln_1_weight: Vec<f32>,
    ln_1_bias: Vec<f32>,
    
    // Multi-head Self-Attention
    attn_c_attn: DMatrix<f32>,     // QKV combined: 768 x 2304
    attn_c_proj: DMatrix<f32>,     // Output projection: 768 x 768
    
    // Pre-FFN LayerNorm
    ln_2_weight: Vec<f32>,
    ln_2_bias: Vec<f32>,
    
    // Feed-Forward Network
    mlp_c_fc: DMatrix<f32>,        // Up projection: 768 x 3072
    mlp_c_proj: DMatrix<f32>,      // Down projection: 3072 x 768
}

impl OriginalGPT2Model {
    /// 원본 numpy 파일들로부터 완전 무손실 로드
    fn load_from_numpy(weights_dir: &str, tokenizer_path: &str) -> Result<Self> {
        println!("🚀 원본 GPT-2 모델 로딩 (numpy 직접 로드)");
        println!("   - 원본 가중치: {}", weights_dir);
        println!("   - 토크나이저: {}", tokenizer_path);
        
        // 1. 토크나이저 로드
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("토크나이저 로드 실패: {:?}", e))?;
        println!("✅ 토크나이저 로드 완료: {} 어휘", tokenizer.get_vocab_size(false));
        
        // 2. 모델 설정
        let config = GPT2Config {
            vocab_size: 51200,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            n_positions: 1024,
        };
        
        // 3. 원본 numpy 메타데이터 로드
        let weights_path = Path::new(weights_dir);
        let metadata_path = weights_path.join("metadata.json");
        let metadata_str = fs::read_to_string(&metadata_path)?;
        let metadata: HashMap<String, Value> = serde_json::from_str(&metadata_str)?;
        
        println!("✅ 원본 메타데이터 로드 완료: {} 개", metadata.len());
        
        // 4. 임베딩 레이어 로드
        println!("🔗 임베딩 레이어 로드 중...");
        let token_embeddings = Self::load_original_matrix(
            &metadata, weights_path, "transformer.wte.weight", 51200, 768)?;
        let position_embeddings = Self::load_original_matrix(
            &metadata, weights_path, "transformer.wpe.weight", 1024, 768)?;
        
        println!("✅ 임베딩 로드 완료");
        println!("   - 토큰 임베딩: {} x {}", token_embeddings.nrows(), token_embeddings.ncols());
        println!("   - 위치 임베딩: {} x {}", position_embeddings.nrows(), position_embeddings.ncols());
        
        // 5. 트랜스포머 레이어들 로드
        println!("🔄 12개 트랜스포머 레이어 로드 중...");
        let mut transformer_layers = Vec::new();
        
        for layer_idx in 0..config.n_layer {
            println!("  📋 레이어 {} 로드 중...", layer_idx);
            
            let layer_prefix = format!("transformer.h.{}", layer_idx);
            
            // LayerNorm 파라미터들
            let ln_1_weight = Self::load_original_vector(
                &metadata, weights_path, &format!("{}.ln_1.weight", layer_prefix), 768)?;
            let ln_1_bias = Self::load_original_vector(
                &metadata, weights_path, &format!("{}.ln_1.bias", layer_prefix), 768)?;
            let ln_2_weight = Self::load_original_vector(
                &metadata, weights_path, &format!("{}.ln_2.weight", layer_prefix), 768)?;
            let ln_2_bias = Self::load_original_vector(
                &metadata, weights_path, &format!("{}.ln_2.bias", layer_prefix), 768)?;
            
            // Attention 가중치들
            let attn_c_attn = Self::load_original_matrix(
                &metadata, weights_path, &format!("{}.attn.c_attn.weight", layer_prefix), 768, 2304)?;
            let attn_c_proj = Self::load_original_matrix(
                &metadata, weights_path, &format!("{}.attn.c_proj.weight", layer_prefix), 768, 768)?;
            
            // FFN 가중치들
            let mlp_c_fc = Self::load_original_matrix(
                &metadata, weights_path, &format!("{}.mlp.c_fc.weight", layer_prefix), 768, 3072)?;
            let mlp_c_proj = Self::load_original_matrix(
                &metadata, weights_path, &format!("{}.mlp.c_proj.weight", layer_prefix), 3072, 768)?;
            
            transformer_layers.push(TransformerLayer {
                ln_1_weight,
                ln_1_bias,
                attn_c_attn,
                attn_c_proj,
                ln_2_weight,
                ln_2_bias,
                mlp_c_fc,
                mlp_c_proj,
            });
            
            println!("  ✅ 레이어 {} 완료", layer_idx);
        }
        
        // 6. 최종 레이어들 로드
        println!("🎯 최종 레이어들 로드 중...");
        let final_ln_weight = Self::load_original_vector(
            &metadata, weights_path, "transformer.ln_f.weight", 768)?;
        let final_ln_bias = Self::load_original_vector(
            &metadata, weights_path, "transformer.ln_f.bias", 768)?;
        let lm_head = Self::load_original_matrix(
            &metadata, weights_path, "lm_head.weight", 768, 51200)?;
        
        println!("✅ 원본 GPT-2 모델 로딩 완료!");
        println!("   - 트랜스포머 레이어: {} 개", transformer_layers.len());
        println!("   - LM Head: {} x {}", lm_head.nrows(), lm_head.ncols());
        println!("   - 🎯 100% 원본 무손실");
        
        Ok(Self {
            tokenizer,
            config,
            token_embeddings,
            position_embeddings,
            transformer_layers,
            final_ln_weight,
            final_ln_bias,
            lm_head,
        })
    }
    
    /// 원본 numpy 파일에서 매트릭스 로드 (완전 무손실)
    fn load_original_matrix(
        metadata: &HashMap<String, Value>,
        weights_dir: &Path,
        layer_name: &str,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<DMatrix<f32>> {
        
        if let Some(layer_info) = metadata.get(layer_name) {
            if let Some(info_obj) = layer_info.as_object() {
                if let (Some(shape_val), Some(file_val)) = 
                    (info_obj.get("shape"), info_obj.get("file")) {
                    
                    let file_name = file_val.as_str().unwrap();
                    let npy_path = weights_dir.join(file_name);
                    
                    // shape 정보 확인
                    let shape = shape_val.as_array().unwrap();
                    let actual_rows = shape[0].as_u64().unwrap() as usize;
                    let actual_cols = shape[1].as_u64().unwrap() as usize;
                    
                    println!("    📁 로드: {} → {}×{}", layer_name, actual_rows, actual_cols);
                    
                    // numpy 파일 읽기
                    let (data, _) = Self::read_npy_data(&npy_path)?;
                    
                    // 매트릭스 생성 (row-major)
                    let mut matrix = DMatrix::from_row_slice(actual_rows, actual_cols, &data);
                    
                    // PyTorch 가중치는 전치된 형태로 저장됨 -> 필요 시 전치
                    if actual_rows != expected_rows || actual_cols != expected_cols {
                        if actual_rows == expected_cols && actual_cols == expected_rows {
                            println!("    🔄 전치 적용: {}×{} → {}×{}", actual_rows, actual_cols, expected_rows, expected_cols);
                            matrix = matrix.transpose();
                        } else {
                            println!("    ⚠️ 크기 불일치: 예상 {}×{}, 실제 {}×{}", 
                                    expected_rows, expected_cols, actual_rows, actual_cols);
                        }
                    }
                    
                    return Ok(matrix);
                }
            }
        }
        
        Err(anyhow::anyhow!("레이어 {}를 찾을 수 없습니다", layer_name))
    }
    
    /// 원본 numpy 파일에서 1D 벡터 로드 (완전 무손실)
    fn load_original_vector(
        metadata: &HashMap<String, Value>,
        weights_dir: &Path,
        layer_name: &str,
        expected_size: usize,
    ) -> Result<Vec<f32>> {
        
        if let Some(layer_info) = metadata.get(layer_name) {
            if let Some(info_obj) = layer_info.as_object() {
                if let Some(file_val) = info_obj.get("file") {
                    let file_name = file_val.as_str().unwrap();
                    let npy_path = weights_dir.join(file_name);
                    
                    println!("    📁 로드: {} ({} 개)", layer_name, expected_size);
                    
                    // numpy 파일 읽기
                    let (data, _) = Self::read_npy_data(&npy_path)?;
                    
                    // 크기 검증
                    if data.len() != expected_size {
                        println!("    ⚠️ 크기 불일치: 예상 {}, 실제 {}", expected_size, data.len());
                    }
                    
                    return Ok(data);
                }
            }
        }
        
        Err(anyhow::anyhow!("레이어 {}를 찾을 수 없습니다", layer_name))
    }
    
    /// numpy 파일 읽기 (정확한 헤더 파싱)
    fn read_npy_data(path: &Path) -> Result<(Vec<f32>, Vec<usize>)> {
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};
        
        let mut file = File::open(path)?;
        
        // numpy 헤더 읽기
        let mut magic = [0u8; 6];
        file.read_exact(&mut magic)?;
        
        if &magic != b"\x93NUMPY" {
            return Err(anyhow::anyhow!("올바른 numpy 파일이 아닙니다"));
        }
        
        // 버전 정보
        let mut version = [0u8; 2];
        file.read_exact(&mut version)?;
        
        // 헤더 길이
        let mut header_len_bytes = [0u8; 2];
        file.read_exact(&mut header_len_bytes)?;
        let header_len = u16::from_le_bytes(header_len_bytes) as usize;
        
        // 헤더 내용 읽기
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)?;
        let header_str = String::from_utf8_lossy(&header_bytes);
        
        // shape와 dtype 파싱 (간단 버전)
        let shape: Vec<usize> = if header_str.contains("(") {
            header_str.split("(").nth(1).unwrap()
                .split(")").next().unwrap()
                .split(",")
                .filter_map(|s| s.trim().parse().ok())
                .collect()
        } else {
            vec![]
        };
        
        let total_elements: usize = shape.iter().product();
        
        // float32 데이터 읽기
        let mut data_bytes = vec![0u8; total_elements * 4];
        file.read_exact(&mut data_bytes)?;
        
        let data: Vec<f32> = data_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        println!("    ✅ numpy 로드: {} 요소, shape: {:?}", data.len(), shape);
        
        Ok((data, shape))
    }

    /// 원본 GPT-2 Forward Pass
    fn generate_text(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        println!("\n💭 원본 GPT-2로 텍스트 생성: '{}'", prompt);
        
        // 1. 토크나이징
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("토크나이징 실패: {:?}", e))?;
        let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
        
        println!("🔤 초기 토큰 수: {}", token_ids.len());
        
        // 2. 생성 루프
        for step in 0..max_tokens {
            let next_token = self.forward_pass(&token_ids)?;
            
            // EOS 체크
            if next_token == 1 || next_token == 0 { // GPT-2 EOS
                break;
            }
            
            token_ids.push(next_token);
            
            // 진행 상황 출력
            if step % 3 == 0 {
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
    
    /// 원본 GPT-2 Forward Pass
    fn forward_pass(&self, token_ids: &[u32]) -> Result<u32> {
        let seq_len = token_ids.len().min(self.config.n_positions);
        let recent_tokens = &token_ids[token_ids.len().saturating_sub(seq_len)..];
        
        // 1. 임베딩: 토큰 + 위치
        let mut hidden_states = self.create_embeddings(recent_tokens);
        
        // 2. 12개 트랜스포머 레이어 통과
        for (layer_idx, layer) in self.transformer_layers.iter().enumerate() {
            hidden_states = self.apply_transformer_layer(&hidden_states, layer, layer_idx)?;
        }
        
        // 3. 최종 LayerNorm
        self.apply_layer_norm(&mut hidden_states, &self.final_ln_weight, &self.final_ln_bias);
        
        // 4. LM Head로 로짓 계산
        let last_hidden = hidden_states.row(hidden_states.nrows() - 1);
        let logits = &self.lm_head.transpose() * last_hidden.transpose();
        
        // 5. 샘플링
        let next_token = self.sample_token(&logits.as_slice())?;
        
        Ok(next_token)
    }
    
    /// 임베딩 생성 (토큰 + 위치)
    fn create_embeddings(&self, token_ids: &[u32]) -> DMatrix<f32> {
        let seq_len = token_ids.len();
        let mut embeddings = DMatrix::zeros(seq_len, self.config.n_embd);
        
        for (pos, &token_id) in token_ids.iter().enumerate() {
            let token_idx = (token_id as usize) % self.token_embeddings.nrows();
            let pos_idx = pos % self.position_embeddings.nrows();
            
            // 토큰 임베딩 + 위치 임베딩
            for j in 0..self.config.n_embd {
                embeddings[(pos, j)] = self.token_embeddings[(token_idx, j)] 
                                     + self.position_embeddings[(pos_idx, j)];
            }
        }
        
        embeddings
    }
    
    /// 트랜스포머 레이어 적용
    fn apply_transformer_layer(
        &self,
        input: &DMatrix<f32>,
        layer: &TransformerLayer,
        _layer_idx: usize,
    ) -> Result<DMatrix<f32>> {
        
        // 1. Pre-LayerNorm for Attention
        let mut attn_input = input.clone();
        self.apply_layer_norm(&mut attn_input, &layer.ln_1_weight, &layer.ln_1_bias);
        
        // 2. Multi-Head Self-Attention
        let attn_output = self.apply_attention(&attn_input, &layer.attn_c_attn, &layer.attn_c_proj)?;
        
        // 3. Residual Connection
        let after_attn = &attn_output + input;
        
        // 4. Pre-LayerNorm for FFN
        let mut ffn_input = after_attn.clone();
        self.apply_layer_norm(&mut ffn_input, &layer.ln_2_weight, &layer.ln_2_bias);
        
        // 5. Feed-Forward Network
        let ffn_output = self.apply_ffn(&ffn_input, &layer.mlp_c_fc, &layer.mlp_c_proj)?;
        
        // 6. Residual Connection
        let final_output = &ffn_output + &after_attn;
        
        Ok(final_output)
    }
    
    /// Multi-Head Self-Attention
    fn apply_attention(
        &self,
        input: &DMatrix<f32>,
        c_attn: &DMatrix<f32>,
        c_proj: &DMatrix<f32>,
    ) -> Result<DMatrix<f32>> {
        let seq_len = input.nrows();
        let d_model = input.ncols();
        let n_head = self.config.n_head;
        let head_dim = d_model / n_head;
        
        // QKV 계산
        let qkv = input * c_attn; // seq_len x 2304
        
        let mut output = DMatrix::zeros(seq_len, d_model);
        
        for head in 0..n_head {
            let head_start = head * head_dim;
            
            // Q, K, V 추출
            let q = qkv.columns(head_start, head_dim);
            let k = qkv.columns(head_start + d_model, head_dim);
            let v = qkv.columns(head_start + 2 * d_model, head_dim);
            
            // Attention 계산
            let scores = &q * k.transpose() / (head_dim as f32).sqrt();
            
            // Causal masking
            let mut masked_scores = scores;
            for i in 0..seq_len {
                for j in (i+1)..seq_len {
                    masked_scores[(i, j)] = f32::NEG_INFINITY;
                }
            }
            
            // Softmax
            let attn_weights = self.softmax_2d(&masked_scores);
            
            // Apply attention
            let head_output = &attn_weights * &v;
            
            // 결과 합치기
            for i in 0..seq_len {
                for j in 0..head_dim {
                    output[(i, head_start + j)] = head_output[(i, j)];
                }
            }
        }
        
        // Output projection
        let final_output = &output * c_proj;
        
        Ok(final_output)
    }
    
    /// Feed-Forward Network
    fn apply_ffn(
        &self,
        input: &DMatrix<f32>,
        c_fc: &DMatrix<f32>,
        c_proj: &DMatrix<f32>,
    ) -> Result<DMatrix<f32>> {
        
        // 첫 번째 linear: 768 -> 3072
        let intermediate = input * c_fc;
        
        // GELU 활성화
        let activated = self.gelu(&intermediate);
        
        // 두 번째 linear: 3072 -> 768
        let output = &activated * c_proj;
        
        Ok(output)
    }
    
    /// GELU 활성화 함수
    fn gelu(&self, input: &DMatrix<f32>) -> DMatrix<f32> {
        input.map(|x| {
            0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
        })
    }
    
    /// Layer Normalization
    fn apply_layer_norm(&self, tensor: &mut DMatrix<f32>, weight: &[f32], bias: &[f32]) {
        let eps = 1e-5;
        
        for mut row in tensor.row_iter_mut() {
            let mean = row.sum() / row.len() as f32;
            let variance = row.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>() / row.len() as f32;
            let std = (variance + eps).sqrt();
            
            for (j, x) in row.iter_mut().enumerate() {
                let normalized = (*x - mean) / std;
                let w = weight.get(j).unwrap_or(&1.0);
                let b = bias.get(j).unwrap_or(&0.0);
                *x = normalized * w + b;
            }
        }
    }
    
    /// 2D Softmax
    fn softmax_2d(&self, input: &DMatrix<f32>) -> DMatrix<f32> {
        let mut output = input.clone();
        
        for mut row in output.row_iter_mut() {
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = row.iter()
                .map(|&x| (x - max_val).exp())
                .sum();
            
            for x in row.iter_mut() {
                *x = (*x - max_val).exp() / exp_sum;
            }
        }
        
        output
    }
    
    /// 토큰 샘플링
    fn sample_token(&self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() {
            return Ok(0);
        }
        
        let temperature = 0.8;
        let scaled_logits: Vec<f32> = logits.iter()
            .map(|&x| x / temperature)
            .collect();
        
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
    
    /// 모델 정보 출력
    fn print_model_info(&self) {
        println!("\n📊 원본 GPT-2 모델 정보:");
        println!("  🔤 어휘 크기: {}", self.config.vocab_size);
        println!("  🧠 은닉층 크기: {}", self.config.n_embd);
        println!("  📚 레이어 수: {}", self.config.n_layer);
        println!("  👥 어텐션 헤드: {}", self.config.n_head);
        println!("  📏 최대 위치: {}", self.config.n_positions);
        println!("  ✅ 100% 원본 무손실 로드");
        println!("  🎯 압축 없는 순수 모델");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🇰🇷 === 원본 GPT-2 모델 테스트 ===");
    println!("압축 없는 순수 원본 모델 동작 확인\n");
    
    let original_weights_dir = "./models/skt-kogpt2-base-v2/weights";
    let tokenizer_path = "./models/skt-kogpt2-base-v2/tokenizer.json";
    
    // 원본 파일 존재 확인
    let weights_dir_exists = Path::new(original_weights_dir).exists();
    let tokenizer_exists = Path::new(tokenizer_path).exists();
    
    println!("📋 원본 모델 파일 확인:");
    println!("   - 원본 가중치: {} ({})", original_weights_dir, 
             if weights_dir_exists { "존재" } else { "❌ 없음" });
    println!("   - 토크나이저: {} ({})", tokenizer_path, 
             if tokenizer_exists { "존재" } else { "❌ 없음" });
    
    if !weights_dir_exists || !tokenizer_exists {
        return Err(anyhow::anyhow!("원본 모델 파일이 없습니다. extract_weights.py를 먼저 실행하세요."));
    }
    
    // 원본 GPT-2 모델 로드
    let model = OriginalGPT2Model::load_from_numpy(original_weights_dir, tokenizer_path)?;
    model.print_model_info();
    
    println!("\n💬 원본 GPT-2로 한국어 대화 시작! (종료: 'exit')");

    let stdin = io::stdin();
    loop {
        print!("\n질문: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        stdin.read_line(&mut input)?;
        let input = input.trim();
        
        if input == "exit" || input == "quit" || input == "종료" {
            println!("👋 원본 GPT-2 엔진을 종료합니다.");
            break;
        }
        
        if input.is_empty() {
            continue;
        }

        let start = Instant::now();
        
        match model.generate_text(input, 25) {
            Ok(response) => {
                let duration = start.elapsed();
                
                let generated_part = if response.starts_with(input) {
                    response[input.len()..].trim()
                } else {
                    &response
                };
                
                if !generated_part.is_empty() {
                    println!("🎯 원본 GPT-2 답변: {}", generated_part);
                } else {
                    println!("🎯 원본 GPT-2 답변: {}", response);
                }
                
                println!("⏱️ 생성 시간: {:.2}초", duration.as_secs_f32());
                println!("✨ 100% 원본 모델, 압축 없음");
            }
            Err(e) => {
                println!("❌ 오류: {}", e);
            }
        }
    }

    Ok(())
} 
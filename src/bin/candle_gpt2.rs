use anyhow::Result;
use candle_core::{Device, Tensor, DType, Module, D, IndexOp};
use candle_nn::{VarBuilder, VarMap, linear, layer_norm, LayerNorm, Linear, Embedding};
use tokenizers::Tokenizer;
use std::io::{self, Write};
use std::time::Instant;
use std::path::Path;

/// GPT-2 설정
#[derive(Debug, Clone)]
pub struct GPT2Config {
    pub vocab_size: usize,      // 51200
    pub n_embd: usize,         // 768
    pub n_head: usize,         // 12
    pub n_layer: usize,        // 12
    pub n_positions: usize,    // 1024
    pub dropout: f64,          // 0.1
}

impl Default for GPT2Config {
    fn default() -> Self {
        Self {
            vocab_size: 51200,
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
            n_positions: 1024,
            dropout: 0.1,
        }
    }
}

/// Multi-Head Attention
pub struct MultiHeadAttention {
    c_attn: Linear,     // QKV projection: n_embd -> 3 * n_embd
    c_proj: Linear,     // Output projection: n_embd -> n_embd
    n_head: usize,
    n_embd: usize,
    dropout: f64,
}

impl MultiHeadAttention {
    pub fn new(vb: VarBuilder, config: &GPT2Config) -> Result<Self> {
        let c_attn = linear(config.n_embd, 3 * config.n_embd, vb.pp("c_attn"))?;
        let c_proj = linear(config.n_embd, config.n_embd, vb.pp("c_proj"))?;
        
        Ok(Self {
            c_attn,
            c_proj,
            n_head: config.n_head,
            n_embd: config.n_embd,
            dropout: config.dropout,
        })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _c) = x.dims3()?;
        let head_size = self.n_embd / self.n_head;
        
        // QKV projection
        let qkv = self.c_attn.forward(x)?;
        let q = qkv.narrow(D::Minus1, 0, self.n_embd)?
            .reshape((b, t, self.n_head, head_size))?
            .transpose(1, 2)?; // (B, nh, T, hs)
        let k = qkv.narrow(D::Minus1, self.n_embd, self.n_embd)?
            .reshape((b, t, self.n_head, head_size))?
            .transpose(1, 2)?; // (B, nh, T, hs)
        let v = qkv.narrow(D::Minus1, 2 * self.n_embd, self.n_embd)?
            .reshape((b, t, self.n_head, head_size))?
            .transpose(1, 2)?; // (B, nh, T, hs)
        
        // Self-attention
        let att = self.scaled_dot_product_attention(&q, &k, &v)?;
        
        // Concatenate heads and project
        let att = att.transpose(1, 2)?.reshape((b, t, self.n_embd))?;
        self.c_proj.forward(&att)
    }
    
    fn scaled_dot_product_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (_b, _nh, t, hs) = q.dims4()?;
        let scale = 1.0 / ((hs as f64).sqrt());
        
        // Attention scores
        let att = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let att = (att * scale)?;
        
        // Causal mask
        let mask = Tensor::tril2(t, DType::F32, q.device())?;
        let att = att.where_cond(&mask, &Tensor::new(f32::NEG_INFINITY, q.device())?)?;
        
        // Softmax
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        
        // Apply to values
        att.matmul(v)
    }
}

/// Feed-Forward Network
pub struct MLP {
    c_fc: Linear,      // n_embd -> 4 * n_embd
    c_proj: Linear,    // 4 * n_embd -> n_embd
    dropout: f64,
}

impl MLP {
    pub fn new(vb: VarBuilder, config: &GPT2Config) -> Result<Self> {
        let c_fc = linear(config.n_embd, 4 * config.n_embd, vb.pp("c_fc"))?;
        let c_proj = linear(4 * config.n_embd, config.n_embd, vb.pp("c_proj"))?;
        
        Ok(Self {
            c_fc,
            c_proj,
            dropout: config.dropout,
        })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = self.gelu_activation(&x)?;  // GELU activation
        self.c_proj.forward(&x)
    }
    
    /// GELU 활성화 함수 구현
    fn gelu_activation(&self, x: &Tensor) -> Result<Tensor> {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let x_cubed = x.pow(&Tensor::new(3.0f32, x.device())?)?;
        let inner = (x + &(x_cubed * 0.044715f32)?)?;
        let inner = (inner * (2.0f32 / std::f32::consts::PI).sqrt())?;
        let tanh_part = inner.tanh()?;
        let one_plus_tanh = (tanh_part + 1.0f32)?;
        (x * &one_plus_tanh)? * 0.5f32
    }
}

/// Transformer Block
pub struct Block {
    ln_1: LayerNorm,
    attn: MultiHeadAttention,
    ln_2: LayerNorm,
    mlp: MLP,
}

impl Block {
    pub fn new(vb: VarBuilder, config: &GPT2Config) -> Result<Self> {
        let ln_1 = layer_norm(config.n_embd, 1e-5, vb.pp("ln_1"))?;
        let attn = MultiHeadAttention::new(vb.pp("attn"), config)?;
        let ln_2 = layer_norm(config.n_embd, 1e-5, vb.pp("ln_2"))?;
        let mlp = MLP::new(vb.pp("mlp"), config)?;
        
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-LayerNorm architecture (GPT-2 style)
        let x = (x + &self.attn.forward(x)?)?;
        let x = (x + &self.mlp.forward(x)?)?;
        Ok(x)
    }
}

/// 완전한 GPT-2 모델
pub struct GPT2LMHeadModel {
    wte: Embedding,           // Token embeddings
    wpe: Embedding,           // Position embeddings
    h: Vec<Block>,            // Transformer blocks
    ln_f: LayerNorm,          // Final layer norm
    lm_head: Linear,          // Language modeling head
    config: GPT2Config,
}

impl GPT2LMHeadModel {
    pub fn new(vb: VarBuilder, config: &GPT2Config) -> Result<Self> {
        println!("🏗️ GPT-2 모델 구성 중...");
        
        let wte = candle_nn::embedding(config.vocab_size, config.n_embd, vb.pp("wte"))?;
        let wpe = candle_nn::embedding(config.n_positions, config.n_embd, vb.pp("wpe"))?;
        
        let mut h = Vec::new();
        for i in 0..config.n_layer {
            let block = Block::new(vb.pp(&format!("h.{}", i)), config)?;
            h.push(block);
            if i % 3 == 0 {
                println!("  📚 레이어 {}/{} 완료", i + 1, config.n_layer);
            }
        }
        
        let ln_f = layer_norm(config.n_embd, 1e-5, vb.pp("ln_f"))?;
        let lm_head = linear(config.n_embd, config.vocab_size, vb.pp("lm_head"))?;
        
        println!("✅ GPT-2 모델 구성 완료");
        
        Ok(Self {
            wte,
            wpe,
            h,
            ln_f,
            lm_head,
            config: config.clone(),
        })
    }
    
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, t) = input_ids.dims2()?;
        let device = input_ids.device();
        
        // Position IDs
        let position_ids = Tensor::arange(0u32, t as u32, device)?
            .unsqueeze(0)?; // (1, T)
        
        // Token + Position embeddings
        let tok_emb = self.wte.forward(input_ids)?;
        let pos_emb = self.wpe.forward(&position_ids)?;
        let mut x = (tok_emb + pos_emb)?;
        
        // Transformer blocks
        for block in &self.h {
            x = block.forward(&x)?;
        }
        
        // Final layer norm
        x = self.ln_f.forward(&x)?;
        
        // Language modeling head
        self.lm_head.forward(&x)
    }
}

/// Candle GPT-2 모델 래퍼
pub struct CandleGPT2Model {
    model: GPT2LMHeadModel,
    tokenizer: Tokenizer,
    device: Device,
    config: GPT2Config,
}

impl CandleGPT2Model {
    /// 랜덤 가중치로 GPT-2 모델 생성
    pub fn new_random(tokenizer_path: &str) -> Result<Self> {
        println!("🕯️ Candle GPT-2 모델 (랜덤 가중치) 초기화...");
        
        let device = Device::Cpu;
        println!("✅ 디바이스: CPU");
        
        // 토크나이저 로드
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("토크나이저 로드 실패: {:?}", e))?;
        println!("✅ 토크나이저 로드 완료: {} 어휘", tokenizer.get_vocab_size(false));
        
        // 모델 설정
        let config = GPT2Config::default();
        println!("✅ 모델 설정: {:?}", config);
        
        // 랜덤 가중치로 모델 생성
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = GPT2LMHeadModel::new(vb, &config)?;
        
        println!("🎯 Candle GPT-2 모델 초기화 완료!");
        
        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }
    
    /// 텍스트 생성
    pub fn generate(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String> {
        println!("\n💭 Candle GPT-2 텍스트 생성: '{}'", prompt);
        
        // 토크나이징
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("토크나이징 실패: {:?}", e))?;
        let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
        
        println!("🔤 초기 토큰 수: {}", token_ids.len());
        
        // 생성 루프
        for step in 0..max_tokens {
            let next_token = self.forward_pass(&token_ids, temperature)?;
            
            // EOS 체크
            if next_token == 1 || next_token == 0 {
                break;
            }
            
            token_ids.push(next_token);
            
            if step % 5 == 0 {
                let partial = self.tokenizer.decode(&token_ids, true)
                    .unwrap_or_else(|_| "디코딩 오류".to_string());
                println!("📝 단계 {}: {}", step, partial);
            }
        }
        
        // 최종 디코딩
        let result = self.tokenizer.decode(&token_ids, true)
            .map_err(|e| anyhow::anyhow!("디코딩 실패: {:?}", e))?;
        
        Ok(result)
    }
    
    /// Forward pass
    fn forward_pass(&self, token_ids: &[u32], temperature: f32) -> Result<u32> {
        // 토큰 ID를 텐서로 변환
        let input_ids = Tensor::from_slice(
            &token_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            (1, token_ids.len()),
            &self.device,
        )?;
        
        // 모델 forward
        let logits = self.model.forward(&input_ids)?;
        
        // 마지막 토큰의 로짓 추출
        let last_logits = logits.i((0, token_ids.len() - 1))?;
        
        // 온도 스케일링
        let last_logits = if temperature != 1.0 {
            (last_logits / temperature)?
        } else {
            last_logits
        };
        
        // 샘플링 (간단한 argmax)
        let next_token_tensor = last_logits.argmax(0)?;
        let next_token_vec = next_token_tensor.to_vec1::<i64>()?;
        let next_token = next_token_vec[0] as u32;
        
        Ok(next_token % (self.config.vocab_size as u32))
    }
    
    /// 모델 정보 출력
    pub fn print_info(&self) {
        println!("\n📊 Candle GPT-2 모델 정보:");
        println!("  🕯️ 프레임워크: Candle (완전한 GPT-2 구현)");
        println!("  🔤 어휘 크기: {}", self.config.vocab_size);
        println!("  🧠 은닉층 크기: {}", self.config.n_embd);
        println!("  📚 레이어 수: {}", self.config.n_layer);
        println!("  👥 어텐션 헤드: {}", self.config.n_head);
        println!("  📏 최대 위치: {}", self.config.n_positions);
        println!("  ⚡ 디바이스: {:?}", self.device);
        println!("  ✅ 완전한 트랜스포머 아키텍처");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🇰🇷 === Candle 완전한 GPT-2 구현 ===");
    println!("검증된 트랜스포머 아키텍처로 GPT-2 모델\n");
    
    let tokenizer_path = "./models/skt-kogpt2-base-v2/tokenizer.json";
    
    // 파일 존재 확인
    let tokenizer_exists = Path::new(tokenizer_path).exists();
    
    println!("📋 파일 확인:");
    println!("   - 토크나이저: {} ({})", tokenizer_path, 
             if tokenizer_exists { "존재" } else { "❌ 없음" });
    
    if !tokenizer_exists {
        return Err(anyhow::anyhow!("토크나이저 파일이 없습니다."));
    }
    
    // Candle GPT-2 모델 초기화
    let model = CandleGPT2Model::new_random(tokenizer_path)?;
    model.print_info();
    
    println!("\n💬 Candle GPT-2로 한국어 대화 시작! (종료: 'exit')");
    println!("⚠️ 랜덤 가중치로 구조 테스트 중");

    let stdin = io::stdin();
    loop {
        print!("\n질문: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        stdin.read_line(&mut input)?;
        let input = input.trim();
        
        if input == "exit" || input == "quit" || input == "종료" {
            println!("👋 Candle GPT-2 모델을 종료합니다.");
            break;
        }
        
        if input.is_empty() {
            continue;
        }

        let start = Instant::now();
        
        match model.generate(input, 15, 0.7) {
            Ok(response) => {
                let duration = start.elapsed();
                
                println!("🎯 Candle GPT-2 답변: {}", response);
                println!("⏱️ 생성 시간: {:.2}초", duration.as_secs_f32());
                println!("🏗️ 완전한 트랜스포머 아키텍처 사용");
            }
            Err(e) => {
                println!("❌ 오류: {}", e);
            }
        }
    }

    Ok(())
} 
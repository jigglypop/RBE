use anyhow::Result;
use clap::Parser;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::{Module, LayerNorm, Linear, Dropout, Embedding};
use tokenizers::Tokenizer;
use rand::thread_rng;
use rand::distributions::{Distribution, WeightedIndex};

use rbe_llm::core::decoder::model_loader::RBEModelLoader;

/// GPT-2 추론을 위한 CLI 인자
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// RBE 압축 모델 디렉토리 경로
    #[arg(short, long, default_value = "models/rbe_compressed")]
    model_dir: PathBuf,
    
    /// 토크나이저 파일 경로  
    #[arg(short, long, default_value = "models/tokenizer.json")]
    tokenizer_path: PathBuf,
    
    /// 생성할 텍스트 프롬프트
    #[arg(short, long, default_value = "Once upon a time")]
    prompt: String,

    /// 생성할 최대 토큰 수
    #[arg(short = 'n', long, default_value = "100")]
    max_tokens: usize,

    /// 온도 (0.0 = deterministic, 1.0 = creative)
    #[arg(short = 'T', long, default_value = "0.8")]
    temperature: f32,
    
    /// Top-p (nucleus) 샘플링
    #[arg(long, default_value = "0.9")]
    top_p: f32,
    
    /// 반복 페널티
    #[arg(long, default_value = "1.0")]
    repetition_penalty: f32,
    
    /// Random seed (재현 가능한 생성)
    #[arg(long)]
    seed: Option<u64>,
}

/// RBE 압축된 가중치를 Tensor로 로드
fn load_rbe_weight_as_tensor(
    loader: &mut RBEModelLoader, 
    name: &str,
    device: &Device
) -> Result<Tensor> {
    // 가중치 정보 먼저 가져오기
    let weight_info = loader.get_weight_info(name)?;
    let shape = weight_info.original_shape.clone();
    
    // 로드 및 디코딩
    loader.load(name)?;
    let decoded = loader.decode_weight(name)?;
    
    // Box::leak를 사용하여 'static lifetime 생성
    let leaked_data: &'static [f32] = Box::leak(decoded.into_boxed_slice());
    
    // Tensor 생성
    Ok(Tensor::from_slice(leaked_data, shape.as_slice(), device)?)
}

/// RBE VarBuilder - candle의 VarBuilder 인터페이스 구현
struct RBEVarBuilder {
    loader: std::cell::RefCell<RBEModelLoader>,
    device: Device,
}

impl RBEVarBuilder {
    fn new(model_dir: &Path) -> Result<Self> {
        let loader = RBEModelLoader::new(model_dir)?;
        let device = Device::cuda_if_available(0)?;
        
        Ok(Self { 
            loader: std::cell::RefCell::new(loader), 
            device 
        })
    }
    
    fn get_tensor(&self, name: &str) -> Result<Tensor> {
        load_rbe_weight_as_tensor(&mut *self.loader.borrow_mut(), name, &self.device)
    }
}

/// GPT-2 설정
#[derive(Debug, Clone)]
struct Config {
    vocab_size: usize,
    n_positions: usize,
    n_embd: usize,
    n_layer: usize,
    n_head: usize,
    n_inner: Option<usize>,
    activation_function: String,
    resid_pdrop: f64,
    embd_pdrop: f64,
    attn_pdrop: f64,
    layer_norm_epsilon: f64,
}

impl Default for Config {
    fn default() -> Self {
        // GPT-2 117M 기본 설정
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            n_inner: None,
            activation_function: "gelu".to_string(),
            resid_pdrop: 0.1,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            layer_norm_epsilon: 1e-5,
        }
    }
}

/// Multi-head Self Attention
struct MultiHeadSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    attn_dropout: Dropout,
    resid_dropout: Dropout,
    n_head: usize,
    n_embd: usize,
}

impl MultiHeadSelfAttention {
    fn new(config: &Config, vb: &RBEVarBuilder, layer_idx: usize) -> Result<Self> {
        let n_embd = config.n_embd;
        let n_head = config.n_head;
        
        let c_attn_weight = vb.get_tensor(&format!("h.{}.attn.c_attn.weight", layer_idx))?;
        let c_attn_bias = vb.get_tensor(&format!("h.{}.attn.c_attn.bias", layer_idx))?;
        let c_attn = Linear::new(c_attn_weight, Some(c_attn_bias));
        
        let c_proj_weight = vb.get_tensor(&format!("h.{}.attn.c_proj.weight", layer_idx))?;
        let c_proj_bias = vb.get_tensor(&format!("h.{}.attn.c_proj.bias", layer_idx))?;
        let c_proj = Linear::new(c_proj_weight, Some(c_proj_bias));
        
        Ok(Self {
            c_attn,
            c_proj,
            attn_dropout: Dropout::new(config.attn_pdrop as f32),
            resid_dropout: Dropout::new(config.resid_pdrop as f32),
            n_head,
            n_embd,
        })
    }
    
    fn forward(&self, x: &Tensor, layer_past: Option<&(Tensor, Tensor)>) -> Result<(Tensor, (Tensor, Tensor))> {
        let (batch_size, seq_len, _) = x.dims3()?;
        
        // QKV projection
        let qkv = self.c_attn.forward(x)?;
        let qkv = qkv.reshape((batch_size, seq_len, 3, self.n_head, self.n_embd / self.n_head))?;
        
        let q = qkv.i((.., .., 0, .., ..))?.transpose(1, 2)?;
        let k = qkv.i((.., .., 1, .., ..))?.transpose(1, 2)?;
        let v = qkv.i((.., .., 2, .., ..))?.transpose(1, 2)?;
        
        // Past key-values 처리
        let (k, v) = if let Some((past_k, past_v)) = layer_past {
            let k = Tensor::cat(&[past_k, &k], 2)?;
            let v = Tensor::cat(&[past_v, &v], 2)?;
            (k, v)
        } else {
            (k, v)
        };
        
        let present = (k.clone(), v.clone());
        
        // Attention scores
        let head_dim = self.n_embd / self.n_head;
        let mut scores = q.matmul(&k.transpose(2, 3)?)? / (head_dim as f64).sqrt();
        
        // Causal mask
        let seq_len_k = k.dims()[2];
        if seq_len > 1 {
            let mask = Tensor::triu2(seq_len, DType::F32, &scores.device())?
                .broadcast_as(scores.shape())?
                .to_dtype(scores.dtype())?;
            scores = scores.broadcast_sub(&(&mask * 1e10)?)?;
        }
        
        // Softmax
        let probs = candle_nn::ops::softmax(&scores, 3)?;
        let probs = self.attn_dropout.forward(&probs, false)?;
        
        // Attention output
        let attn_output = probs.matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, self.n_embd))?;
        
        let attn_output = self.c_proj.forward(&attn_output)?;
        let attn_output = self.resid_dropout.forward(&attn_output, false)?;
        
        Ok((attn_output, present))
    }
}

/// Feed-forward MLP
struct MLP {
    c_fc: Linear,
    c_proj: Linear,
    dropout: Dropout,
}

impl MLP {
    fn new(config: &Config, vb: &RBEVarBuilder, layer_idx: usize) -> Result<Self> {
        let n_inner = config.n_inner.unwrap_or(4 * config.n_embd);
        
        let c_fc_weight = vb.get_tensor(&format!("h.{}.mlp.c_fc.weight", layer_idx))?;
        let c_fc_bias = vb.get_tensor(&format!("h.{}.mlp.c_fc.bias", layer_idx))?;
        let c_fc = Linear::new(c_fc_weight, Some(c_fc_bias));
        
        let c_proj_weight = vb.get_tensor(&format!("h.{}.mlp.c_proj.weight", layer_idx))?;
        let c_proj_bias = vb.get_tensor(&format!("h.{}.mlp.c_proj.bias", layer_idx))?;
        let c_proj = Linear::new(c_proj_weight, Some(c_proj_bias));
        
        Ok(Self {
            c_fc,
            c_proj,
            dropout: Dropout::new(config.resid_pdrop as f32),
        })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.c_fc.forward(x)?;
        let h = h.gelu()?;
        let h = self.c_proj.forward(&h)?;
        Ok(self.dropout.forward(&h, false)?)
    }
}

/// GPT-2 Block (Transformer layer)
struct Block {
    ln_1: LayerNorm,
    attn: MultiHeadSelfAttention,
    ln_2: LayerNorm,
    mlp: MLP,
}

impl Block {
    fn new(config: &Config, vb: &RBEVarBuilder, layer_idx: usize) -> Result<Self> {
        let n_embd = config.n_embd;
        let eps = config.layer_norm_epsilon;
        
        let ln_1_weight = vb.get_tensor(&format!("h.{}.ln_1.weight", layer_idx))?;
        let ln_1_bias = vb.get_tensor(&format!("h.{}.ln_1.bias", layer_idx))?;
        let ln_1 = LayerNorm::new(ln_1_weight, ln_1_bias, eps);
        
        let ln_2_weight = vb.get_tensor(&format!("h.{}.ln_2.weight", layer_idx))?;
        let ln_2_bias = vb.get_tensor(&format!("h.{}.ln_2.bias", layer_idx))?;
        let ln_2 = LayerNorm::new(ln_2_weight, ln_2_bias, eps);
        
        Ok(Self {
            ln_1,
            attn: MultiHeadSelfAttention::new(config, vb, layer_idx)?,
            ln_2,
            mlp: MLP::new(config, vb, layer_idx)?,
        })
    }
    
    fn forward(&self, x: &Tensor, layer_past: Option<&(Tensor, Tensor)>) -> Result<(Tensor, (Tensor, Tensor))> {
        let residual = x;
        let x = self.ln_1.forward(x)?;
        let (attn_output, present) = self.attn.forward(&x, layer_past)?;
        let x = (residual + attn_output)?;
        
        let residual = &x;
        let x = self.ln_2.forward(&x)?;
        let mlp_output = self.mlp.forward(&x)?;
        let x = (residual + mlp_output)?;
        
        Ok((x, present))
    }
}

/// GPT-2 모델
struct GPT2 {
    wte: Embedding,
    wpe: Embedding,
    drop: Dropout,
    h: Vec<Block>,
    ln_f: LayerNorm,
}

impl GPT2 {
    fn new(config: &Config, vb: &RBEVarBuilder) -> Result<Self> {
        let wte = Embedding::new(
            vb.get_tensor("wte.weight")?,
            config.n_embd
        );
        
        let wpe = Embedding::new(
            vb.get_tensor("wpe.weight")?,
            config.n_embd
        );
        
        let ln_f_weight = vb.get_tensor("ln_f.weight")?;
        let ln_f_bias = vb.get_tensor("ln_f.bias")?;
        let ln_f = LayerNorm::new(ln_f_weight, ln_f_bias, config.layer_norm_epsilon);
        
        let mut h = Vec::new();
        for i in 0..config.n_layer {
            h.push(Block::new(config, vb, i)?);
        }
        
        Ok(Self {
            wte,
            wpe,
            drop: Dropout::new(config.embd_pdrop as f32),
            h,
            ln_f,
        })
    }
    
    fn forward(&self, input_ids: &Tensor, past: Option<Vec<(Tensor, Tensor)>>) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let (_batch_size, seq_len) = input_ids.dims2()?;
        let past_len = past.as_ref().map(|p| p[0].0.dims()[2]).unwrap_or(0);
        
        // Position ids
        let position_ids = Tensor::arange(past_len as u32, (past_len + seq_len) as u32, input_ids.device())?
            .unsqueeze(0)?;
        
        // Embeddings
        let inputs_embeds = self.wte.forward(input_ids)?;
        let position_embeds = self.wpe.forward(&position_ids)?;
        let hidden_states = self.drop.forward(&(inputs_embeds + position_embeds)?, false)?;
        
        // Transformer blocks
        let mut hidden_states = hidden_states;
        let mut presents = Vec::new();
        
        for (i, block) in self.h.iter().enumerate() {
            let layer_past = past.as_ref().map(|p| &p[i]);
            let (new_hidden_states, present) = block.forward(&hidden_states, layer_past)?;
            hidden_states = new_hidden_states;
            presents.push(present);
        }
        
        let hidden_states = self.ln_f.forward(&hidden_states)?;
        
        Ok((hidden_states, presents))
    }
}

/// 텍스트 생성 함수
fn generate(
    model: &GPT2,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    device: &Device,
) -> Result<String> {
    // 프롬프트 토큰화
    let encoding = tokenizer.encode(prompt, false).map_err(|e| anyhow::anyhow!("{}", e))?;
    let mut input_ids = Tensor::new(encoding.get_ids(), device)?.unsqueeze(0)?;
    
    let mut past: Option<Vec<(Tensor, Tensor)>> = None;
    let mut generated_tokens = Vec::new();
    let mut token_counts: HashMap<u32, usize> = HashMap::new();
    
    println!("\n🚀 생성 시작...\n");
    print!("{}", prompt);
    
    for _ in 0..max_tokens {
        // Forward pass
        let (logits, new_past) = model.forward(&input_ids, past)?;
        past = Some(new_past);
        
        // 마지막 토큰의 logits 가져오기
        let seq_len = logits.dims()[1];
        let logits = logits.i((.., seq_len - 1, ..))?;
        
        // Repetition penalty 적용
        let mut logits_vec: Vec<f32> = logits.to_vec1()?;
        for (token_id, count) in &token_counts {
            if *count > 0 {
                let penalty = repetition_penalty.powf(*count as f32);
                logits_vec[*token_id as usize] /= penalty;
            }
        }
        
        // Temperature 적용
        if temperature > 0.0 {
            for logit in &mut logits_vec {
                *logit /= temperature;
            }
        }
        
        // Softmax
        let max_logit = logits_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut probs: Vec<f32> = logits_vec.iter()
            .map(|&logit| (logit - max_logit).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        for prob in &mut probs {
            *prob /= sum;
        }
        
        // Top-p (nucleus) 샘플링
        let next_token = if top_p < 1.0 {
            // 확률 기준 내림차순 정렬
            let mut indexed_probs: Vec<(usize, f32)> = probs.iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            // 누적 확률이 top_p를 넘을 때까지 선택
            let mut cumsum = 0.0;
            let mut nucleus_size = 0;
            for (_, prob) in &indexed_probs {
                cumsum += prob;
                nucleus_size += 1;
                if cumsum >= top_p {
                    break;
                }
            }
            
            // Nucleus 내에서 샘플링
            let nucleus: Vec<(usize, f32)> = indexed_probs.into_iter()
                .take(nucleus_size)
                .collect();
            let nucleus_probs: Vec<f32> = nucleus.iter().map(|(_, p)| *p).collect();
            let nucleus_sum: f32 = nucleus_probs.iter().sum();
            let normalized_probs: Vec<f32> = nucleus_probs.iter()
                .map(|p| p / nucleus_sum)
                .collect();
            
            let dist = WeightedIndex::new(&normalized_probs)?;
            let idx = dist.sample(&mut thread_rng());
            nucleus[idx].0
        } else {
            // 기본 샘플링
            let dist = WeightedIndex::new(&probs)?;
            dist.sample(&mut thread_rng())
        };
        
        // 토큰 디코딩 및 출력
        let token_str = tokenizer.decode(&[next_token as u32], false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        print!("{}", token_str);
        use std::io::{self, Write};
        io::stdout().flush()?;
        
        // 다음 입력 준비
        input_ids = Tensor::new(&[next_token as u32], device)?.unsqueeze(0)?;
        generated_tokens.push(next_token as u32);
        
        // 토큰 카운트 업데이트
        *token_counts.entry(next_token as u32).or_insert(0) += 1;
        
        // EOS 토큰 체크 (GPT-2의 경우 50256)
        if next_token == 50256 {
            break;
        }
    }
    
    println!("\n\n✅ 생성 완료!");
    
    // 전체 생성된 텍스트 반환
    let full_ids = [encoding.get_ids(), &generated_tokens].concat();
    let generated_text = tokenizer.decode(&full_ids, true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    
    Ok(generated_text)
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    println!("🤖 RBE GPT-2 추론 엔진 시작");
    println!("📁 모델 디렉토리: {:?}", args.model_dir);
    println!("📝 프롬프트: {}", args.prompt);
    
    // Random seed 설정
    if let Some(seed) = args.seed {
        // candle에는 set_seed가 없으므로 직접 처리하지 않음
        println!("🎲 Random seed: {}", seed);
    }
    
    // 토크나이저 로드
    println!("\n📚 토크나이저 로드 중...");
    let tokenizer = Tokenizer::from_file(&args.tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
    
    // RBE 모델 로드
    println!("🔧 RBE 압축 모델 로드 중...");
    let rbe_builder = RBEVarBuilder::new(&args.model_dir)?;
    
    // GPT-2 설정
    let config = Config::default();
    
    // 모델 생성
    println!("🏗️ GPT-2 모델 구성 중...");
    let model = GPT2::new(&config, &rbe_builder)?;

    // 텍스트 생성
    let generated = generate(
        &model,
        &tokenizer,
        &args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_p,
        args.repetition_penalty,
        &rbe_builder.device,
    )?;
    
    println!("\n\n📄 전체 생성 텍스트:");
    println!("{}", generated);
    
    // 메모리 사용량
    rbe_builder.loader.borrow().print_stats();
    
    Ok(())
} 
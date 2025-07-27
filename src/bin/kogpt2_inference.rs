//! KoGPT-2 모델을 RBE 형식으로 변환하고 추론하는 CLI

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tokio;

use rbe_llm::core::tensors::Packed128;
use rbe_llm::nlp::kogpt2_rbe::{KoGPT2Config, KoGPT2RBE};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Convert {
        #[arg(short, long, default_value = "models/skt-kogpt2-base-v2/model.safetensors")]
        input: PathBuf,
        #[arg(short, long, default_value = "models/rbe_model")]
        output_dir: PathBuf,
        #[arg(long)]
        force: bool,
    },
    Generate {
        #[arg(short, long, default_value = "models/rbe_model")]
        model_dir: PathBuf,
        #[arg(short, long)]
        prompt: String,
        #[arg(long, default_value_t = 50)]
        max_length: usize,
        #[arg(long, default_value_t = 1.0)]
        temperature: f32,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Convert { input, output_dir, force } => {
            if !*force && output_dir.exists() {
                println!("'{}'가 이미 존재합니다. --force로 강제 변환.", output_dir.display());
                return Ok(());
            }
            println!("PyTorch 가중치를 RBE 모델로 변환합니다...");
            let model = KoGPT2RBE::from_pytorch_weights(KoGPT2Config::default(), input)
                .map_err(|e| anyhow!(e.to_string()))?;
            println!("모델 변환 완료. 저장 중...");
            save_rbe_model(&model, output_dir)?;
            println!("모델 저장 완료.");
        }
        Commands::Generate { model_dir, prompt, max_length, temperature } => {
            println!("RBE 모델을 로드합니다...");
            let mut model = load_rbe_model(model_dir)?;
            println!("모델 로드 완료.");
            
            // 임시로 하드코딩된 토큰 사용 (인공지능이란 무엇인가?)
            let input_ids: Vec<usize> = vec![13753, 8263, 7166, 10479, 24488, 406];
            println!("입력 토큰: {:?}", input_ids);
            
            /*
            let tokenizer_path = Path::new(".").join("tokenizer.json");
            println!("토크나이저를 로드합니다: {}", tokenizer_path.display());
            let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("토크나이저 로드 실패: {}, 현재 경로: {}", e, std::env::current_dir().unwrap().display()))?;
            let encoding = tokenizer.encode(prompt.as_str(), true).map_err(|e| anyhow!(e.to_string()))?;
            let input_ids: Vec<usize> = encoding.get_ids().iter().map(|&x| x as usize).collect();
            */
        
            println!("텍스트 생성 중...");
            let generated_ids = model.generate(&input_ids, *max_length, *temperature)
                .map_err(|e| anyhow!(e.to_string()))?;
            
            println!("\n생성된 토큰 ID: {:?}", generated_ids);
            println!("Python으로 디코딩하려면: python3 tokenizer.py decode {}", 
                     generated_ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
            
            /*
            let generated_text = tokenizer.decode(&generated_ids.iter().map(|&t| t as u32).collect::<Vec<_>>(), true)
                .map_err(|e| anyhow!(e.to_string()))?;
            println!("\n생성된 텍스트:\n---\n{}\n---", generated_text);
            */
        }
    }
    Ok(())
}

fn save_rbe_model(model: &KoGPT2RBE, dir: &Path) -> Result<()> {
    fs::create_dir_all(dir)?;
    fs::write(dir.join("config.json"), serde_json::to_string_pretty(&model.config)?)?;
    let mut file = fs::File::create(dir.join("model.bin"))?;
    
    // 개별 변수 저장 헬퍼 함수
    let empty_vec = vec![];
    let write_seeds = |f: &mut fs::File, s: &Vec<Packed128>| -> std::io::Result<()> { f.write_all(&(s.len() as u64).to_le_bytes())?; for seed in s { f.write_all(&seed.hi.to_le_bytes())?; f.write_all(&seed.lo.to_le_bytes())?; } Ok(()) };
    let write_seed = |f: &mut fs::File, s: &Packed128| -> std::io::Result<()> { f.write_all(&s.hi.to_le_bytes())?; f.write_all(&s.lo.to_le_bytes())?; Ok(()) };
    let write_f32s = |f: &mut fs::File, d: &Option<Vec<f32>>| -> std::io::Result<()> { let v = d.as_ref().unwrap_or(&empty_vec); f.write_all(&(v.len() as u64).to_le_bytes())?; for val in v { f.write_all(&val.to_le_bytes())?; } Ok(()) };

    write_seeds(&mut file, &model.wte.rbe_weights.seeds)?;
    write_seeds(&mut file, &model.wpe.rbe_weights.seeds)?;
    for b in &model.blocks {
        write_f32s(&mut file, &b.ln1.gamma)?; write_f32s(&mut file, &b.ln1.beta)?;
        write_seed(&mut file, &b.attention.q_proj.weight_seed)?; write_f32s(&mut file, &b.attention.q_proj.bias)?;
        write_seed(&mut file, &b.attention.k_proj.weight_seed)?; write_f32s(&mut file, &b.attention.k_proj.bias)?;
        write_seed(&mut file, &b.attention.v_proj.weight_seed)?; write_f32s(&mut file, &b.attention.v_proj.bias)?;
        write_seed(&mut file, &b.attention.out_proj.weight_seed)?; write_f32s(&mut file, &b.attention.out_proj.bias)?;
        write_f32s(&mut file, &b.ln2.gamma)?; write_f32s(&mut file, &b.ln2.beta)?;
        write_seed(&mut file, &b.ffn.up_proj.weight_seed)?; write_f32s(&mut file, &b.ffn.up_proj.bias)?;
        write_seed(&mut file, &b.ffn.down_proj.weight_seed)?; write_f32s(&mut file, &b.ffn.down_proj.bias)?;
    }
    write_f32s(&mut file, &model.ln_f.gamma)?; write_f32s(&mut file, &model.ln_f.beta)?;
    write_seed(&mut file, &model.lm_head.weight_seed)?; write_f32s(&mut file, &model.lm_head.bias)?;
    Ok(())
}

fn load_rbe_model(dir: &Path) -> Result<KoGPT2RBE> {
    let config: KoGPT2Config = serde_json::from_str(&fs::read_to_string(dir.join("config.json"))?)?;
    let mut model = KoGPT2RBE::new(config)
        .map_err(|e| anyhow!(e.to_string()))?;
    let mut file = fs::File::open(dir.join("model.bin"))?;

    let read_seeds = |f: &mut fs::File| -> Result<Vec<Packed128>,std::io::Error> { let mut b=[0;8]; f.read_exact(&mut b)?; let l=u64::from_le_bytes(b) as usize; let mut s=Vec::with_capacity(l); for _ in 0..l { let mut h=[0;8]; let mut o=[0;8]; f.read_exact(&mut h)?; f.read_exact(&mut o)?; s.push(Packed128{hi:u64::from_le_bytes(h),lo:u64::from_le_bytes(o)}); } Ok(s) };
    let read_seed = |f: &mut fs::File| -> Result<Packed128,std::io::Error> { let mut h=[0;8]; let mut o=[0;8]; f.read_exact(&mut h)?; f.read_exact(&mut o)?; Ok(Packed128{hi:u64::from_le_bytes(h),lo:u64::from_le_bytes(o)}) };
    let read_f32s = |f: &mut fs::File| -> Result<Option<Vec<f32>>,std::io::Error> { let mut b=[0;8]; f.read_exact(&mut b)?; let l=u64::from_le_bytes(b) as usize; if l==0 { return Ok(None); } let mut v=Vec::with_capacity(l); for _ in 0..l { let mut d=[0;4]; f.read_exact(&mut d)?; v.push(f32::from_le_bytes(d)); } Ok(Some(v)) };

    model.wte.rbe_weights.seeds = read_seeds(&mut file)?;
    model.wpe.rbe_weights.seeds = read_seeds(&mut file)?;
    
    // 임베딩만 초기화
    model.wte.init_after_load().map_err(|e| anyhow!(e.to_string()))?;
    model.wpe.init_after_load().map_err(|e| anyhow!(e.to_string()))?;
    
    for b in &mut model.blocks {
        b.ln1.gamma = read_f32s(&mut file)?; b.ln1.beta = read_f32s(&mut file)?;
        b.attention.q_proj.weight_seed = read_seed(&mut file)?; b.attention.q_proj.bias = read_f32s(&mut file)?;
        b.attention.k_proj.weight_seed = read_seed(&mut file)?; b.attention.k_proj.bias = read_f32s(&mut file)?;
        b.attention.v_proj.weight_seed = read_seed(&mut file)?; b.attention.v_proj.bias = read_f32s(&mut file)?;
        b.attention.out_proj.weight_seed = read_seed(&mut file)?; b.attention.out_proj.bias = read_f32s(&mut file)?;
        b.ln2.gamma = read_f32s(&mut file)?; b.ln2.beta = read_f32s(&mut file)?;
        b.ffn.up_proj.weight_seed = read_seed(&mut file)?; b.ffn.up_proj.bias = read_f32s(&mut file)?;
        b.ffn.down_proj.weight_seed = read_seed(&mut file)?; b.ffn.down_proj.bias = read_f32s(&mut file)?;
    }
    model.ln_f.gamma = read_f32s(&mut file)?; model.ln_f.beta = read_f32s(&mut file)?;
    model.lm_head.weight_seed = read_seed(&mut file)?; model.lm_head.bias = read_f32s(&mut file)?;
    
    // init_after_load를 호출하지 않음 - 이미 압축된 가중치를 사용
    // model.init_after_load()
    //     .map_err(|e| anyhow!(e.to_string()))?;
    Ok(model)
}

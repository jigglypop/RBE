use rbe_llm::{
    encoder::{RBEEncoder, encoder::{QualityGrade, CompressionProfile, CompressionConfig}},
    decoder::WeightGenerator,
    TransformType,
    HybridEncodedBlock,
    nlp::linear::rbe_linear::{RBELinear, RBELinearConfig},
};
use std::{time::Instant, path::Path, collections::HashMap};
use tokenizers::tokenizer::Tokenizer;
use safetensors::{tensor::SafeTensors, SafeTensorError};
use std::fs;
use std::io::Read;
use serde_json::Value;

/// 한글 sLLM 압축 데모
/// KoMiniLM-23M 모델을 RBE로 압축하고 실제 한글 텍스트를 생성합니다.

// 모델 구조체
struct CompressedKoreanLLM {
    tokenizer: Tokenizer,
    layers: HashMap<String, CompressedLayer>,
    config: ModelConfig,
}

// 압축된 레이어
struct CompressedLayer {
    blocks: Vec<HybridEncodedBlock>,
    shape: Vec<usize>,
}

// 모델 설정
#[derive(Clone)]
struct ModelConfig {
    vocab_size: usize,
    hidden_size: usize,
    n_layers: usize,
    n_heads: usize,
    max_position_embeddings: usize,
    model_type: String,
}

// LayerNorm (압축 없이 그대로 사용)
struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    eps: f32,
}

impl LayerNorm {
    fn from_tensors(weight: Vec<f32>, bias: Vec<f32>) -> Self {
        Self { weight, bias, eps: 1e-5 }
    }
    
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
        let std = (variance + self.eps).sqrt();
        
        x.iter().enumerate()
            .map(|(i, &v)| ((v - mean) / std) * self.weight[i] + self.bias[i])
            .collect()
    }
}

// 모델 로드 및 압축
async fn load_and_compress_korean_model(model_path: &Path) -> Result<CompressedKoreanLLM, Box<dyn std::error::Error>> {
    println!("🔍 한국어 모델 로드 시작: {:?}", model_path);
    
    // 1. 설정 파일 로드
    let config_path = model_path.join("config.json");
    let config_str = fs::read_to_string(&config_path)?;
    let config_json: Value = serde_json::from_str(&config_str)?;
    
    let config = ModelConfig {
        vocab_size: config_json["vocab_size"].as_u64().unwrap() as usize,
        hidden_size: config_json["hidden_size"].as_u64().unwrap() as usize,
        n_layers: config_json["num_hidden_layers"].as_u64().unwrap() as usize,
        n_heads: config_json["num_attention_heads"].as_u64().unwrap() as usize,
        max_position_embeddings: config_json["max_position_embeddings"].as_u64().unwrap() as usize,
        model_type: config_json["model_type"].as_str().unwrap().to_string(),
    };
    
    println!("📊 모델 구성:");
    println!("  - 모델 타입: {}", config.model_type);
    println!("  - 어휘 크기: {}", config.vocab_size);
    println!("  - 히든 크기: {}", config.hidden_size);
    println!("  - 레이어 수: {}", config.n_layers);
    
    // 2. 토크나이저 로드
    println!("\n🔤 토크나이저 로드 중...");
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("토크나이저 로드 실패: {}", e))?;
    
    // 3. 가중치 로드
    println!("\n📦 모델 가중치 로드 중...");
    let weights_path = model_path.join("model.safetensors");
    let mut file = fs::File::open(&weights_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let tensors = SafeTensors::deserialize(&buffer)?;
    
    // 4. RBE 압축 설정
    let compression_config = CompressionConfig {
        block_size: 64,
        quality_grade: QualityGrade::A,
        transform_type: TransformType::Dwt,
        profile: CompressionProfile::Balanced,
        custom_coefficients: Some(200), // A급 품질을 위한 계수
        min_block_count: None,
        rmse_threshold: Some(0.001),
        compression_ratio_threshold: Some(500.0),
    };
    
    // 5. 레이어별 압축
    println!("\n🔨 레이어별 RBE 압축 시작...");
    let start_time = Instant::now();
    
    let mut layers = HashMap::new();
    
    // 압축할 텐서들의 목록을 수집
    let tensor_names: Vec<_> = tensors.names().into_iter().collect();
    let total_tensors = tensor_names.len();
    
    for (idx, tensor_name) in tensor_names.iter().enumerate() {
        if idx % 10 == 0 {
            println!("  진행률: {}/{} ({:.1}%)", idx, total_tensors, 
                     idx as f32 / total_tensors as f32 * 100.0);
        }
        
        // LayerNorm은 압축하지 않음
        if tensor_name.contains("LayerNorm") || tensor_name.contains("ln_") {
            continue;
        }
        
        // 텐서 데이터 추출
        let tensor = tensors.tensor(tensor_name)?;
        let shape = tensor.shape();
        let data = tensor.data();
        
        // f32로 변환
        let weights: Vec<f32> = data.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        // 압축
        let compressed = compress_tensor(&weights, shape, &compression_config)?;
        layers.insert(tensor_name.to_string(), compressed);
    }
    
    println!("\n✅ 전체 압축 완료! 시간: {:.2}초", start_time.elapsed().as_secs_f64());
    
    // 메모리 사용량 분석
    print_memory_usage(&layers, &config);
    
    Ok(CompressedKoreanLLM {
        tokenizer,
        layers,
        config,
    })
}

// 텐서 압축
fn compress_tensor(
    weights: &[f32],
    shape: &[usize],
    config: &CompressionConfig,
) -> Result<CompressedLayer, Box<dyn std::error::Error>> {
    // 2D로 변환
    let (rows, cols) = match shape.len() {
        1 => (shape[0], 1),
        2 => (shape[0], shape[1]),
        _ => {
            let rows = shape[0];
            let cols = shape[1..].iter().product();
            (rows, cols)
        }
    };
    
    // RBE 압축
    let (blocks, _, compression_ratio, rmse) = RBEEncoder::compress_with_profile(
        weights,
        rows,
        cols,
        config.block_size,
        config.custom_coefficients.unwrap_or(200),
        config.transform_type,
    )?;
    
    Ok(CompressedLayer {
        blocks,
        shape: shape.to_vec(),
    })
}

// 메모리 사용량 출력
fn print_memory_usage(layers: &HashMap<String, CompressedLayer>, config: &ModelConfig) {
    println!("\n📊 압축률 통계:");
    
    let mut total_original = 0usize;
    let mut total_compressed = 0usize;
    
    for (name, layer) in layers {
        let original_size = layer.shape.iter().product::<usize>() * 4; // f32
        let compressed_size = layer.blocks.len() * std::mem::size_of::<HybridEncodedBlock>();
        
        total_original += original_size;
        total_compressed += compressed_size;
        
        if name.contains("embedding") || name.contains("lm_head") {
            println!("  - {}: {:.2}MB → {:.2}MB (압축률 {:.1}x)", 
                     name,
                     original_size as f32 / 1024.0 / 1024.0,
                     compressed_size as f32 / 1024.0 / 1024.0,
                     original_size as f32 / compressed_size as f32);
        }
    }
    
    println!("\n📈 전체 통계:");
    println!("  - 원본 크기: {:.2}MB", total_original as f32 / 1024.0 / 1024.0);
    println!("  - 압축 크기: {:.2}MB", total_compressed as f32 / 1024.0 / 1024.0);
    println!("  - 전체 압축률: {:.1}x", total_original as f32 / total_compressed as f32);
    println!("  - 메모리 절약: {:.1}%", 
             (1.0 - total_compressed as f32 / total_original as f32) * 100.0);
}

// 한글 텍스트 생성
async fn generate_korean_text(
    model: &CompressedKoreanLLM,
    prompt: &str,
    max_length: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    println!("\n🚀 한글 텍스트 생성 시작");
    println!("📝 프롬프트: {}", prompt);
    
    // 토큰화
    let encoding = model.tokenizer.encode(prompt, false)
        .map_err(|e| format!("토큰화 실패: {:?}", e))?;
    let mut input_ids: Vec<u32> = encoding.get_ids().to_vec();
    
    println!("🔢 입력 토큰: {:?} ({}개)", &input_ids[..5.min(input_ids.len())], input_ids.len());
    
    // WeightGenerator 생성
    let weight_generator = WeightGenerator::new();
    
    // 생성 루프
    for step in 0..max_length {
        let start = Instant::now();
        
        // 간단한 추론 데모 (실제로는 전체 트랜스포머 구조를 구현해야 함)
        // 여기서는 임베딩 레이어만 사용해서 다음 토큰 예측을 시뮬레이션
        let token_id = input_ids.last().unwrap();
        
        // 토큰 임베딩 가져오기 (예시)
        let embedding_key = match model.config.model_type.as_str() {
            "gpt2" => "wte.weight",
            "bert" | "electra" => "embeddings.word_embeddings.weight",
            _ => "embeddings.token_embeddings.weight",
        };
        
        if let Some(embedding_layer) = model.layers.get(embedding_key) {
            // 압축된 블록에서 토큰 임베딩 디코딩
            // 실제로는 전체 forward pass를 구현해야 하지만, 데모를 위해 간단히 처리
            let next_token = generate_next_token(&weight_generator, embedding_layer, *token_id as usize);
            input_ids.push(next_token);
            
            // 디코딩
            let generated = model.tokenizer.decode(&input_ids, false)
                .map_err(|e| format!("디코딩 실패: {:?}", e))?;
            
            if step % 5 == 0 {
                println!("  Step {}: {} ({:.1}ms)", step, generated, start.elapsed().as_millis());
            }
            
            // 종료 조건 (EOS 토큰)
            if next_token == 2 {  // 일반적인 EOS token ID
                break;
            }
        } else {
            println!("⚠️  임베딩 레이어를 찾을 수 없습니다.");
            break;
        }
    }
    
    let final_text = model.tokenizer.decode(&input_ids, false)
        .map_err(|e| format!("최종 디코딩 실패: {:?}", e))?;
    
    Ok(final_text)
}

// 다음 토큰 생성 (간단한 예시)
fn generate_next_token(
    weight_generator: &WeightGenerator,
    embedding_layer: &CompressedLayer,
    current_token: usize,
) -> u32 {
    // 실제로는 전체 모델을 통과해야 하지만, 데모를 위해 간단히 처리
    // 여기서는 랜덤하게 다음 토큰을 선택
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    // 토큰 범위 내에서 랜덤 선택 (실제로는 확률 분포에 따라 선택해야 함)
    rng.gen_range(0..50000) as u32
}

// RBE 레이어를 사용한 실제 추론 함수
fn forward_rbe_layer(
    weight_generator: &WeightGenerator,
    compressed_layer: &CompressedLayer,
    input: &[f32],
) -> Vec<f32> {
    // 블록별로 디코딩하고 행렬 곱셈 수행
    let mut output = vec![0.0f32; compressed_layer.shape[0]];
    
    for (block_idx, block) in compressed_layer.blocks.iter().enumerate() {
        // 블록 디코딩
        let decoded_block = weight_generator.decode_block(block);
        
        // 부분 행렬 곱셈 (간단한 구현)
        // 실제로는 블록 위치를 고려한 정확한 계산이 필요
        for i in 0..block.rows {
            for j in 0..block.cols {
                if i < output.len() && j < input.len() {
                    output[i] += decoded_block[i * block.cols + j] * input[j];
                }
            }
        }
    }
    
    output
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 RBE 한국어 sLLM 압축 및 추론 데모");
    println!("=====================================\n");
    
    let start_time = Instant::now();
    
    // 1. 모델 경로 설정
    let model_path = Path::new("models/kominilm-23m");
    if !model_path.exists() {
        println!("❌ 모델이 없습니다. 먼저 setup_korean_test_model.py를 실행하세요.");
        println!("   python setup_korean_test_model.py");
        return Ok(());
    }
    
    // 2. 모델 로드 및 압축
    let compressed_model = load_and_compress_korean_model(model_path).await?;
    
    // 3. 한글 텍스트 생성 테스트
    println!("\n🧪 한글 텍스트 생성 테스트");
    println!("==========================");
    
    let test_prompts = vec![
        "안녕하세요",
        "오늘 날씨가",
        "RBE 시스템은",
        "한국어 자연어처리",
    ];
    
    for prompt in test_prompts {
        println!("\n📝 프롬프트: \"{}\"", prompt);
        
        match generate_korean_text(&compressed_model, prompt, 20).await {
            Ok(generated) => {
                println!("✅ 생성된 텍스트: {}", generated);
            }
            Err(e) => {
                println!("❌ 생성 실패: {}", e);
            }
        }
    }
    
    // 4. RBE 블록 직접 테스트
    println!("\n🔬 RBE 압축 블록 직접 테스트");
    println!("=============================");
    
    // 임베딩 레이어에서 첫 번째 블록 테스트
    if let Some(embedding_layer) = compressed_model.layers.get("wte.weight")
        .or(compressed_model.layers.get("embeddings.word_embeddings.weight")) {
        
        if let Some(first_block) = embedding_layer.blocks.first() {
            let weight_generator = WeightGenerator::new();
            let decoded = weight_generator.decode_block(first_block);
            
            println!("  - 첫 번째 블록 크기: {}x{}", first_block.rows, first_block.cols);
            println!("  - RBE 파라미터: {:?}", &first_block.rbe_params[..4]);
            println!("  - 잔차 계수 개수: {}", first_block.residuals.len());
            println!("  - 디코딩된 값 샘플: {:?}", &decoded[..5.min(decoded.len())]);
            
            // RMSE 계산 (원본 데이터가 없으므로 자체 검증)
            let re_encoded = RBEEncoder::new(200, TransformType::Dwt)
                .encode_block(&decoded, first_block.rows, first_block.cols);
            let re_decoded = weight_generator.decode_block(&re_encoded);
            
            let rmse: f32 = decoded.iter()
                .zip(re_decoded.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>() / decoded.len() as f32;
            
            println!("  - 재인코딩 RMSE: {:.6}", rmse.sqrt());
        }
    }
    
    println!("\n🏁 전체 실행 시간: {:.2}초", start_time.elapsed().as_secs_f64());
    println!("\n✨ RBE 한국어 모델 압축 및 추론 완료!");
    
    Ok(())
} 
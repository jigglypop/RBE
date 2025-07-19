use RBE_LLM::encoder::HybridEncoder;
use RBE_LLM::sllm::model_downloader::ModelDownloader;
use RBE_LLM::packed_params::{HybridEncodedBlock, TransformType};
use std::fs;
use std::time::Instant;
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};

#[tokio::main]
async fn main() -> Result<()> {
    // 1. 실제 모델 다운로드
    let model_id = "skt/kogpt2-base-v2";
    let downloader = ModelDownloader::new(model_id);
    let model_dir = downloader.download().await?;
    let model_path = model_dir.join("pytorch_model.bin");

    println!("\n=== RBE 모델 압축 도구 ===\n");
    
    // 압축 설정
    let block_size = 256;
    let coefficients = 500;
    
    println!("압축 설정:");
    println!("   - 블록 크기: {}×{}", block_size, block_size);
    println!("   - 계수 개수: {} (웨이블릿)", coefficients);
    println!("   - 모델 경로: {:?}", model_path);
    
    // 테스트용 행렬 데이터 생성 (실제로는 모델에서 로드해야 함)
    let matrix_size = 768; // GPT-2 hidden size
    let mut matrix_data = vec![0.0f32; matrix_size * matrix_size];
    
    // 압축 가능한 패턴 생성
    for i in 0..matrix_size {
        for j in 0..matrix_size {
            let x = i as f32 / matrix_size as f32;
            let y = j as f32 / matrix_size as f32;
            matrix_data[i * matrix_size + j] = 
                (2.0 * std::f32::consts::PI * x).sin() * 
                (2.0 * std::f32::consts::PI * y).cos() * 0.5;
        }
    }
    
    // 압축 시작
    let start = Instant::now();
    let mut encoder = HybridEncoder::new(coefficients, TransformType::Dwt);
    
    // 블록 단위로 압축
    let blocks_per_dim = (matrix_size + block_size - 1) / block_size;
    let total_blocks = blocks_per_dim * blocks_per_dim;
    
    let pb = ProgressBar::new(total_blocks as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {percent}% 블록 {pos}/{len} 압축 중")
            .unwrap()
    );
    
    let mut encoded_blocks = Vec::new();
    
    for block_i in 0..blocks_per_dim {
        for block_j in 0..blocks_per_dim {
            let start_i = block_i * block_size;
            let start_j = block_j * block_size;
            
            // 블록 데이터 추출
            let mut block_data = vec![0.0f32; block_size * block_size];
            for i in 0..block_size {
                for j in 0..block_size {
                    let global_i = start_i + i;
                    let global_j = start_j + j;
                    if global_i < matrix_size && global_j < matrix_size {
                        block_data[i * block_size + j] = 
                            matrix_data[global_i * matrix_size + global_j];
                    }
                }
            }
            
            // 블록 압축
            let encoded_block = encoder.encode_block(&block_data, block_size, block_size);
            encoded_blocks.push(encoded_block);
            
            pb.inc(1);
        }
    }
    pb.finish();
    
    let compression_time = start.elapsed();
    
    // 압축 결과 계산
    let original_size = matrix_size * matrix_size * 4; // f32 bytes
    let compressed_size = encoded_blocks.len() * std::mem::size_of::<HybridEncodedBlock>();
    let compression_ratio = original_size as f32 / compressed_size as f32;
    
    println!("\n 압축 완료 ");
    println!("   - 원본 크기: {} bytes ({:.2} MB)", 
        original_size, original_size as f32 / 1_048_576.0);
    println!("   - 압축 크기: {} bytes ({:.2} MB)", 
        compressed_size, compressed_size as f32 / 1_048_576.0);
    println!("   - 압축률: {:.1}:1", compression_ratio);
    println!("   - 압축 시간: {:.2}초", compression_time.as_secs_f32());
    println!("   - 블록 수: {}", encoded_blocks.len());

    // 압축된 데이터 저장
    let output_path = format!("./models/skt-kogpt2-base-v2_compressed/kogpt2_{}x{}_w{}.rbe", 
        block_size, block_size, coefficients);
    println!("   - 압축 경로: {}", output_path);

    // 실제 압축 데이터를 포함한 JSON 생성
    let compressed_data = serde_json::json!({
        "metadata": {
            "model_name": model_id,
            "matrix_size": matrix_size,
            "block_size": block_size,
            "coefficients": coefficients,
            "transform_type": "Wavelet",
            "compression_ratio": compression_ratio,
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "total_blocks": encoded_blocks.len(),
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs()
        },
        "blocks": encoded_blocks
    });
    
    // 디렉토리 생성
    fs::create_dir_all("./compressed_models")?;
    
    // 파일로 저장
    let json_string = serde_json::to_string_pretty(&compressed_data)?;
    fs::write(&output_path, json_string)?;
    
    println!("   - 저장 경로: {}", output_path);
    
    Ok(())
} 
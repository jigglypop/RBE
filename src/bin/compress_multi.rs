use RBE_LLM::encoder::HybridEncoder;
use RBE_LLM::sllm::model_downloader::ModelDownloader;
use RBE_LLM::types::{HybridEncodedBlock, TransformType};
use std::fs;
use std::time::Instant;
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use serde_json;
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct CompressionProfile {
    name: &'static str,
    block_size: usize,
    coefficients: usize,
    quality_level: &'static str,
}

fn compress_with_profile(
    matrix_data: &[f32],
    matrix_size: usize,
    profile: &CompressionProfile,
    multi_progress: &MultiProgress,
) -> Result<(Vec<HybridEncodedBlock>, f64, f32)> {
    let pb = multi_progress.add(ProgressBar::new(100));
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!("[{{bar:40}}] {{percent}}% {} ({}x{}, {} 계수)", 
                profile.name, profile.block_size, profile.block_size, profile.coefficients))
            .unwrap()
    );
    
    let start = Instant::now();
    let mut encoder = HybridEncoder::new(profile.coefficients, TransformType::Dwt);
    
    // 블록 단위로 압축
    let blocks_per_dim = (matrix_size + profile.block_size - 1) / profile.block_size;
    let total_blocks = blocks_per_dim * blocks_per_dim;
    let mut encoded_blocks = Vec::new();
    
    for block_idx in 0..total_blocks {
        let block_i = block_idx / blocks_per_dim;
        let block_j = block_idx % blocks_per_dim;
        let start_i = block_i * profile.block_size;
        let start_j = block_j * profile.block_size;
        
        // 블록 데이터 추출
        let mut block_data = vec![0.0f32; profile.block_size * profile.block_size];
        for i in 0..profile.block_size {
            for j in 0..profile.block_size {
                let global_i = start_i + i;
                let global_j = start_j + j;
                if global_i < matrix_size && global_j < matrix_size {
                    block_data[i * profile.block_size + j] = 
                        matrix_data[global_i * matrix_size + global_j];
                }
            }
        }
        
        // 블록 압축
        let encoded_block = encoder.encode_block(&block_data, profile.block_size, profile.block_size);
        encoded_blocks.push(encoded_block);
        
        pb.set_position((block_idx * 100 / total_blocks) as u64);
    }
    
    pb.finish();
    
    let compression_time = start.elapsed().as_secs_f64();
    
    // 압축률 계산
    let original_size = matrix_size * matrix_size * 4; // f32 bytes
    let compressed_size = encoded_blocks.len() * std::mem::size_of::<HybridEncodedBlock>();
    let compression_ratio = original_size as f32 / compressed_size as f32;
    
    Ok((encoded_blocks, compression_time, compression_ratio))
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("\n=== RBE 다중 압축 프로파일 테스트 ===\n");
    
    // 압축 프로파일 정의
    let profiles = vec![
        CompressionProfile {
            name: "극한 압축",
            block_size: 256,
            coefficients: 50,
            quality_level: "극저",
        },
        CompressionProfile {
            name: "초고압축",
            block_size: 256,
            coefficients: 100,
            quality_level: "매우 낮음",
        },
        CompressionProfile {
            name: "고압축",
            block_size: 256,
            coefficients: 200,
            quality_level: "낮음",
        },
        CompressionProfile {
            name: "표준 압축",
            block_size: 256,
            coefficients: 500,
            quality_level: "중간",
        },
        CompressionProfile {
            name: "균형 압축",
            block_size: 128,
            coefficients: 500,
            quality_level: "중상",
        },
        CompressionProfile {
            name: "고품질",
            block_size: 64,
            coefficients: 1000,
            quality_level: "높음",
        },
        CompressionProfile {
            name: "초고품질",
            block_size: 32,
            coefficients: 2000,
            quality_level: "매우 높음",
        },
    ];
    
    // 테스트용 행렬 데이터 생성
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
    
    // 멀티 프로그레스 바
    let multi_progress = MultiProgress::new();
    
    // 결과 저장용
    let mut results = Vec::new();
    
    println!("압축 프로파일 테스트 시작...\n");
    
    // 각 프로파일로 압축
    for profile in &profiles {
        let (encoded_blocks, compression_time, compression_ratio) = 
            compress_with_profile(&matrix_data, matrix_size, profile, &multi_progress)?;
        
        // 압축된 데이터 저장
        let output_path = format!("./models/skt-kogpt2-base-v2_compressed/kogpt2_{}x{}_w{}.rbe", 
            profile.block_size, profile.block_size, profile.coefficients);
        
        let compressed_data = serde_json::json!({
            "metadata": {
                "profile_name": profile.name,
                "quality_level": profile.quality_level,
                "matrix_size": matrix_size,
                "block_size": profile.block_size,
                "coefficients": profile.coefficients,
                "transform_type": "Wavelet",
                "compression_ratio": compression_ratio,
                "original_size_bytes": matrix_size * matrix_size * 4,
                "compressed_size_bytes": encoded_blocks.len() * std::mem::size_of::<HybridEncodedBlock>(),
                "total_blocks": encoded_blocks.len(),
                "compression_time_sec": compression_time,
                "timestamp": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs()
            },
            "blocks": encoded_blocks
        });
        
        // 디렉토리 생성
        fs::create_dir_all("./models/skt-kogpt2-base-v2_compressed")?;
        
        // 파일로 저장
        let json_string = serde_json::to_string(&compressed_data)?;
        fs::write(&output_path, json_string)?;
        
        results.push((profile.clone(), compression_ratio, compression_time, output_path));
    }
    
    // 결과 요약 출력
    println!("\n=== 압축 결과 요약 ===\n");
    println!("{:<15} | {:<10} | {:<10} | {:<15} | {:<12} | {:<10}",
        "프로파일", "블록크기", "계수", "압축률", "압축시간(초)", "품질");
    println!("{:-<85}", "");
    
    for (profile, ratio, time, path) in &results {
        println!("{:<15} | {:<10} | {:<10} | {:<15.1} | {:<12.2} | {:<10}",
            profile.name, 
            format!("{}x{}", profile.block_size, profile.block_size),
            profile.coefficients,
            ratio,
            time,
            profile.quality_level
        );
    }
    
    // 상세 정보 저장
    let summary = serde_json::json!({
        "test_date": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
        "matrix_size": matrix_size,
        "profiles": results.iter().map(|(profile, ratio, time, path)| {
            serde_json::json!({
                "name": profile.name,
                "block_size": profile.block_size,
                "coefficients": profile.coefficients,
                "quality_level": profile.quality_level,
                "compression_ratio": ratio,
                "compression_time_sec": time,
                "output_path": path,
                "estimated_rmse": 1.0 / (profile.coefficients as f32).sqrt() // 추정치
            })
        }).collect::<Vec<_>>()
    });
    
    fs::write("./models/skt-kogpt2-base-v2_compressed/compression_comparison.json", 
        serde_json::to_string_pretty(&summary)?)?;
    
    println!("\n✅ 모든 압축 프로파일 테스트 완료!");
    println!("📊 상세 결과: ./models/skt-kogpt2-base-v2_compressed/compression_comparison.json");
    
    Ok(())
} 
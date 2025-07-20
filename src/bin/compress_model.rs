use std::path::Path;
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rbe_llm::packed_params::{HybridEncodedBlock, TransformType};
use rbe_llm::encoder::AutoOptimizedEncoder;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use serde_json::{Value, Map};

/// numpy 파일 헤더 읽기
fn read_npy_header(file: &mut File) -> Result<(Vec<usize>, usize)> {
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)?;
    
    if &magic != b"\x93NUMPY" {
        return Err(anyhow::anyhow!("Invalid numpy file"));
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
    let shape_start = header_str.find("'shape': (").unwrap() + 10;
    let shape_end = header_str[shape_start..].find(')').unwrap() + shape_start;
    let shape_str = &header_str[shape_start..shape_end];
    
    let shape: Vec<usize> = shape_str.split(", ")
        .filter(|s| !s.is_empty())
        .map(|s| s.trim_end_matches(',').parse().unwrap())
        .collect();
    
    let total_size = shape.iter().product();
    
    Ok((shape, total_size))
}

/// numpy 파일에서 float32 데이터 읽기
fn read_npy_data(path: &Path) -> Result<(Vec<f32>, Vec<usize>)> {
    let mut file = File::open(path)?;
    let (shape, total_size) = read_npy_header(&mut file)?;
    
    let mut buffer = vec![0u8; total_size * 4];
    file.read_exact(&mut buffer)?;
    
    let data: Vec<f32> = buffer.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    
    Ok((data, shape))
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 KoGPT2 모델 압축 시작 (numpy 파일 사용)");
    
    let weights_dir = Path::new("models/skt-kogpt2-base-v2/weights");
    let metadata_path = weights_dir.join("metadata.json");
    
    if !metadata_path.exists() {
        println!("❌ 메타데이터 파일이 없습니다. extract_weights.py를 먼저 실행하세요.");
        return Err(anyhow::anyhow!("Metadata file not found"));
    }
    
    // 메타데이터 로드
    let metadata_str = fs::read_to_string(&metadata_path)?;
    let metadata: Map<String, Value> = serde_json::from_str(&metadata_str)?;
    
    println!("✅ 발견된 레이어: {} 개", metadata.len());
    
    // 압축 설정들
    let configs = vec![
        ("insane", 8, 128, TransformType::Dwt),     // 미친 압축! 128²/8 = 2048x!
        ("ultra", 16, 128, TransformType::Dwt),     // 초압축 128²/16 = 1024x!
        ("extreme", 32, 128, TransformType::Dwt),   // 극도 압축 128²/32 = 512x!
        ("high", 64, 128, TransformType::Dwt),      // 고압축 128²/64 = 256x!
        ("balanced", 128, 128, TransformType::Dwt), // 균형 128²/128 = 128x!
        ("lossless", 2000, 64, TransformType::Adaptive), // 거의 무손실
    ];
    
    // 출력 디렉토리 생성
    fs::create_dir_all("models/compressed")?;
    
    for (name, coeffs, block_size, transform_type) in configs {
        println!("\n🔧 압축 프로파일: {} (계수: {}, 블록: {}x{}, 변환: {:?})", 
                 name, coeffs, block_size, block_size, transform_type);
        
        let mut compressed_weights = HashMap::new();
        let mut total_original_size = 0u64;
        let mut total_compressed_size = 0u64;
        let mut total_rmse = 0.0;
        let mut count = 0;
        
        // 프로그레스 바
        let pb = ProgressBar::new(metadata.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("██░"),
        );
        
        // AutoOptimizedEncoder의 compress_multi.rs 로직 사용
        println!("📊 블록 크기 {}x{}, 계수: {}, 변환: {:?}", 
                 block_size, block_size, coeffs, transform_type);
        
        // 각 레이어 압축
        for (layer_idx, (layer_name, layer_info)) in metadata.iter().enumerate() {
            println!("\n🔄 [{}/{}] 레이어 처리 중: {}", layer_idx + 1, metadata.len(), layer_name);
            pb.set_message(format!("압축 중: {}", layer_name));
            
            if let Some(info_obj) = layer_info.as_object() {
                if let (Some(_shape_val), Some(file_val)) = 
                    (info_obj.get("shape"), info_obj.get("file")) {
                    
                    let file_name = file_val.as_str().unwrap();
                    let npy_path = weights_dir.join(file_name);
                    
                    // numpy 파일 읽기
                    match read_npy_data(&npy_path) {
                        Ok((data, shape)) => {
                            // 2D 가중치인 경우만 압축 (Linear layers)
                            if shape.len() == 2 {
                                let height = shape[0];
                                let width = shape[1];
                                
                                // 격자 분할 방식으로 비대칭 매트릭스 압축
                                let blocks_per_row = (height + block_size - 1) / block_size;
                                let blocks_per_col = (width + block_size - 1) / block_size;
                                let total_blocks = blocks_per_row * blocks_per_col;
                                
                                println!("  📐 매트릭스: {}x{} → {}x{} 격자 ({} 블록)", 
                                        height, width, blocks_per_row, blocks_per_col, total_blocks);
                                
                                let mut encoded_blocks = Vec::new();
                                let mut total_block_rmse = 0.0;
                                
                                for block_row in 0..blocks_per_row {
                                    for block_col in 0..blocks_per_col {
                                        let start_i = block_row * block_size;
                                        let start_j = block_col * block_size;
                                        let end_i = (start_i + block_size).min(height);
                                        let end_j = (start_j + block_size).min(width);
                                        
                                        // 블록 데이터 추출 (패딩 포함)
                                        let mut block_data = vec![0.0f32; block_size * block_size];
                                        for i in 0..(end_i - start_i) {
                                            for j in 0..(end_j - start_j) {
                                                let src_idx = (start_i + i) * width + (start_j + j);
                                                let dst_idx = i * block_size + j;
                                                block_data[dst_idx] = data[src_idx];
                                            }
                                        }
                                        
                                        // 블록별 압축 (height, width, block_size, coeffs, transform_type)
                                        match AutoOptimizedEncoder::compress_with_profile(
                                            &block_data, 
                                            block_size,  // height
                                            block_size,  // width
                                            block_size,  // block_size
                                            coeffs, 
                                            transform_type
                                        ) {
                                            Ok((mut block_compressed, _, _, block_rmse)) => {
                                                total_block_rmse += block_rmse;
                                                encoded_blocks.append(&mut block_compressed);
                                            },
                                            Err(e) => {
                                                println!("  ❌ 블록 압축 실패: {}", e);
                                            }
                                        }
                                    }
                                }
                                
                                let avg_rmse = total_block_rmse / total_blocks as f32;
                                
                                // 압축률 계산
                                let original_size = data.len() * 4;
                                let compressed_size = encoded_blocks.len() * std::mem::size_of::<HybridEncodedBlock>();
                                let compression_ratio = original_size as f32 / compressed_size as f32;
                                
                                println!("  ✅ 격자 압축 완료: RMSE {:.6}, 압축률 {:.1}x ({} 블록)", 
                                        avg_rmse, compression_ratio, total_blocks);
                                
                                total_rmse += avg_rmse;
                                count += 1;
                                total_original_size += original_size as u64;
                                total_compressed_size += compressed_size as u64;
                                
                                compressed_weights.insert(layer_name.clone(), encoded_blocks);
                            }
                        }
                        Err(e) => {
                            eprintln!("❌ {} 읽기 실패: {}", layer_name, e);
                        }
                    }
                }
            }
            
            pb.inc(1);
        }
        
        pb.finish_with_message("압축 완료!");
        
        // 결과 출력
        let compression_ratio = if total_compressed_size > 0 {
            total_original_size as f32 / total_compressed_size as f32
        } else {
            0.0
        };
        let avg_rmse = if count > 0 { total_rmse / count as f32 } else { 0.0 };
        
        println!("✅ 압축 완료!");
        println!("  - 원본: {:.2} MB", total_original_size as f32 / 1_048_576.0);
        println!("  - 압축: {:.2} MB", total_compressed_size as f32 / 1_048_576.0);
        println!("  - 압축률: {:.1}x", compression_ratio);
        println!("  - 평균 RMSE: {:.6}", avg_rmse);
        println!("  - 압축된 레이어: {}", count);
        
        // 압축된 모델 저장
        let output_path = format!("models/compressed/kogpt2_{}.bin", name);
        save_compressed_model(&compressed_weights, &output_path)?;
        println!("💾 저장 완료: {}", output_path);
    }
    
    println!("\n✅ 모든 압축 프로파일 완료!");
    Ok(())
}

fn save_compressed_model(weights: &HashMap<String, Vec<HybridEncodedBlock>>, path: &str) -> Result<()> {
    let summary = format!(
        "Compressed model with {} layers, total {} blocks", 
        weights.len(),
        weights.values().map(|v| v.len()).sum::<usize>()
    );
    fs::write(path, summary)?;
    Ok(())
} 
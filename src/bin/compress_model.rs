use clap::{Arg, Command};
use anyhow::{Context, Result};
use std::path::Path;
use std::fs::{self, File};
use std::io::Write;
use std::time::Instant;

use rbe_llm::core::encoder::{RBEEncoder, WeightMapper, ModelLayout, WeightInfo};
use rbe_llm::core::packed_params::{TransformType, HybridEncodedBlock};

/// GPT-2 모델 압축용 설정
struct CompressionSettings {
    block_size: usize,
    coefficients: usize,
    transform_type: TransformType,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            block_size: 64,
            coefficients: 133,
            transform_type: TransformType::Dwt,
        }
    }
}

fn main() -> Result<()> {
    let matches = Command::new("RBE Model Compressor")
        .version("1.0")
        .author("RBE Team")
        .about("GPT-2 모델을 RBE로 압축합니다")
        .arg(Arg::new("input")
            .short('i')
            .long("input")
            .value_name("PATH")
            .help("입력 모델 디렉토리 (extract_weights.py로 추출한 텐서들)")
            .required(true))
        .arg(Arg::new("output")
            .short('o')
            .long("output")
            .value_name("PATH")
            .help("출력 디렉토리")
            .required(true))
        .arg(Arg::new("block-size")
            .short('b')
            .long("block-size")
            .value_name("SIZE")
            .help("블록 크기 (기본값: 64)")
            .default_value("64"))
        .arg(Arg::new("coefficients")
            .short('c')
            .long("coefficients")
            .value_name("NUM")
            .help("유지할 계수 개수 (기본값: 133)")
            .default_value("133"))
        .arg(Arg::new("transform")
            .short('t')
            .long("transform")
            .value_name("TYPE")
            .help("변환 타입: dct, dwt (기본값: dwt)")
            .default_value("dwt"))
        .get_matches();
    
    let input_dir = Path::new(matches.get_one::<String>("input").unwrap());
    let output_dir = Path::new(matches.get_one::<String>("output").unwrap());
    let block_size = matches.get_one::<String>("block-size").unwrap().parse()?;
    let coefficients = matches.get_one::<String>("coefficients").unwrap().parse()?;
    let transform_type = match matches.get_one::<String>("transform").unwrap().as_str() {
        "dct" => TransformType::Dct,
        "dwt" => TransformType::Dwt,
        _ => TransformType::Dwt,
    };
    
    // 출력 디렉토리 생성
    fs::create_dir_all(output_dir)?;

    // 압축 시작
    compress_gpt2_with_layout(
        input_dir,
        output_dir,
        CompressionSettings {
            block_size,
            coefficients,
            transform_type,
        }
    )?;
    
    Ok(())
}

/// GPT-2 모델을 메타데이터와 함께 압축
fn compress_gpt2_with_layout(
    input_dir: &Path,
    output_dir: &Path,
    settings: CompressionSettings,
) -> Result<()> {
    println!("🚀 GPT-2 모델 압축 시작");
    println!("  입력: {}", input_dir.display());
    println!("  출력: {}", output_dir.display());
    println!("  설정: 블록={}, 계수={}, 변환={:?}", 
        settings.block_size, settings.coefficients, settings.transform_type);
    
    let start_time = Instant::now();
    
    // 1. 가중치 매퍼 생성
    let mut mapper = WeightMapper::new(
        "gpt2",
        settings.block_size,
        settings.coefficients,
        settings.transform_type,
    );
    
    // 2. 압축된 블록들을 저장할 벡터
    let mut all_compressed_blocks = Vec::new();
    
    // 3. 모든 텐서 파일 찾기 및 압축
    let tensor_files = find_tensor_files(input_dir)?;
    println!("\n📁 {} 개의 텐서 파일 발견", tensor_files.len());
    
    for (idx, tensor_path) in tensor_files.iter().enumerate() {
        let weight_name = tensor_path.file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("잘못된 파일 이름"))?;
        
        println!("\n[{}/{}] 압축 중: {}", idx + 1, tensor_files.len(), weight_name);
        
        // 텐서 로드
        let (data, shape) = load_tensor(&tensor_path)?;
        println!("  Shape: {:?}, 크기: {:.2} MB", 
            shape, data.len() as f64 * 4.0 / 1_048_576.0);
        
        // 압축
        let blocks = mapper.compress_weight(weight_name, &data, &shape)?;
        println!("  압축 완료: {} 블록", blocks.len());
        
        all_compressed_blocks.push(blocks);
    }
    
    // 4. 모든 블록을 바이너리로 직렬화
    let bin_path = output_dir.join("rbe_model.bin");
    let bin_data = mapper.serialize_all_blocks(&all_compressed_blocks)?;
    let mut bin_file = File::create(&bin_path)?;
    bin_file.write_all(&bin_data)?;
    println!("\n✅ 바이너리 파일 생성: {} ({:.2} MB)", 
        bin_path.display(), bin_data.len() as f64 / 1_048_576.0);
    
    // 5. 레이아웃 파일 저장
    let layout_path = output_dir.join("rbe_layout.json");
    let layout_json = mapper.serialize_layout()?;
    fs::write(&layout_path, layout_json)?;
    println!("✅ 레이아웃 파일 생성: {}", layout_path.display());
    
    // 6. 압축 통계 출력
    mapper.print_compression_stats();
    
    let elapsed = start_time.elapsed();
    println!("\n⏱️  총 소요 시간: {:.2}초", elapsed.as_secs_f64());
    
    Ok(())
}

/// 텐서 파일들 찾기
fn find_tensor_files(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("pt") {
            files.push(path);
        }
    }
    
    // GPT-2 레이어 순서대로 정렬
    files.sort_by(|a, b| {
        let a_name = a.file_stem().unwrap().to_str().unwrap();
        let b_name = b.file_stem().unwrap().to_str().unwrap();
        
        // 레이어 번호 추출하여 정렬
        let a_layer = extract_layer_number(a_name);
        let b_layer = extract_layer_number(b_name);
        
        match (a_layer, b_layer) {
            (Some(a_n), Some(b_n)) => a_n.cmp(&b_n),
            _ => a_name.cmp(b_name),
        }
    });
    
    Ok(files)
}

/// 레이어 번호 추출 (예: "transformer.h.0.attn" -> 0)
fn extract_layer_number(name: &str) -> Option<usize> {
    if name.contains(".h.") {
        let parts: Vec<&str> = name.split('.').collect();
        for (i, part) in parts.iter().enumerate() {
            if *part == "h" && i + 1 < parts.len() {
                return parts[i + 1].parse().ok();
            }
        }
    }
    None
}

/// PyTorch 텐서 파일 로드 (extract_weights.py 형식)
fn load_tensor(path: &Path) -> Result<(Vec<f32>, Vec<usize>)> {
    // 간단한 바이너리 형식 가정:
    // [shape_len:u32][shape:u32*shape_len][data:f32*product(shape)]
    
    let data = fs::read(path)?;
    let mut cursor = 0;
    
    // Shape 길이 읽기
    let shape_len = u32::from_le_bytes([
        data[cursor], data[cursor+1], data[cursor+2], data[cursor+3]
    ]) as usize;
    cursor += 4;
    
    // Shape 읽기
    let mut shape = Vec::with_capacity(shape_len);
    for _ in 0..shape_len {
        let dim = u32::from_le_bytes([
            data[cursor], data[cursor+1], data[cursor+2], data[cursor+3]
        ]) as usize;
        shape.push(dim);
        cursor += 4;
    }
    
    // 데이터 읽기
    let total_elements: usize = shape.iter().product();
    let mut tensor_data = Vec::with_capacity(total_elements);
    
    for _ in 0..total_elements {
        let value = f32::from_le_bytes([
            data[cursor], data[cursor+1], data[cursor+2], data[cursor+3]
        ]);
        tensor_data.push(value);
        cursor += 4;
    }
    
    Ok((tensor_data, shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_layer_number_extraction() {
        assert_eq!(extract_layer_number("transformer.h.0.attn.c_attn.weight"), Some(0));
        assert_eq!(extract_layer_number("transformer.h.11.mlp.c_fc.weight"), Some(11));
        assert_eq!(extract_layer_number("transformer.wte.weight"), None);
    }
} 
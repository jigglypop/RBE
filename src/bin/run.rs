use RBE_LLM::packed_params::HybridEncodedBlock;
use std::fs;
use std::io::{self, Write};
use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;
use std::collections::HashMap;
use serde_json::Value;
use tokenizers::Tokenizer;
// hf-hub는 더 이상 여기서 필요하지 않습니다.

fn load_compressed_model(filepath: &str) -> Result<Vec<HybridEncodedBlock>> {
    let json_content = fs::read_to_string(filepath)?;
    let data: serde_json::Value = serde_json::from_str(&json_content)?;
    let blocks_data = data.get("blocks").ok_or_else(|| anyhow::anyhow!("JSON에서 'blocks' 키를 찾을 수 없습니다."))?;
    serde_json::from_value(blocks_data.clone())
        .map_err(|e| anyhow::anyhow!("블록 데이터 파싱에 실패했습니다: {}", e))
}

// 실제 어휘집 파일을 로드하는 원래의 간단한 함수
fn load_vocabulary(filepath: &str) -> Result<HashMap<u32, String>> {
    // tokenizers 라이브러리를 사용하여 tokenizer 로드
    let tokenizer = Tokenizer::from_file(filepath)
        .map_err(|e| anyhow::anyhow!("tokenizer.json 로드 실패: {}. 파일 경로: {}", e, filepath))?;
    
    // vocab 추출
    let vocab = tokenizer.get_vocab(false);
    
    // String -> u32 매핑을 u32 -> String으로 변환
    let id_to_token: HashMap<u32, String> = vocab.into_iter()
        .map(|(token, id)| (id, token))
        .collect();
    
    if id_to_token.is_empty() {
        return Err(anyhow::anyhow!("어휘집이 비어있습니다."));
    }
    
    Ok(id_to_token)
}

// 압축된 블록들로부터 전체 가중치 행렬을 복원하는 함수
fn reconstruct_matrix(blocks: &[HybridEncodedBlock], matrix_size: usize, block_size: usize) -> DMatrix<f32> {
    let mut full_matrix = DMatrix::from_element(matrix_size, matrix_size, 0.0);
    let blocks_per_dim = (matrix_size + block_size - 1) / block_size;

    for (idx, block) in blocks.iter().enumerate() {
        let block_i = idx / blocks_per_dim;
        let block_j = idx % blocks_per_dim;
        let start_i = block_i * block_size;
        let start_j = block_j * block_size;

        let decoded_block = block.decode();

        for r in 0..block_size {
            for c in 0..block_size {
                if start_i + r < matrix_size && start_j + c < matrix_size {
                    full_matrix[(start_i + r, start_j + c)] = decoded_block[r * block_size + c];
                }
            }
        }
    }
    full_matrix
}

// 사용자 프롬프트를 간단한 입력 벡터로 변환하는 함수
fn prompt_to_vector(prompt: &str, dim: usize) -> DVector<f32> {
    let mut vec = DVector::from_element(dim, 0.0);
    for (i, ch) in prompt.chars().enumerate() {
        let hash = (ch as usize + i).wrapping_mul(31);
        vec[hash % dim] += (hash % 100) as f32 / 100.0;
    }
    if vec.norm() > 0.0 {
        vec.normalize_mut();
    }
    vec
}

// 모델의 출력 벡터를 실제 텍스트로 변환하는 함수
fn vector_to_text(vector: &DVector<f32>, id_to_vocab: &HashMap<u32, String>) -> String {
    let mut indices: Vec<_> = (0..vector.len()).collect();
    indices.sort_unstable_by(|&a, &b| vector[b].partial_cmp(&vector[a]).unwrap_or(std::cmp::Ordering::Equal));

    // 상위 10개 예측 토큰 ID를 가져옴
    let top_tokens = indices.iter()
        .take(10)
        .map(|&i| {
            // 모델의 출력 차원(768)을 어휘집 크기(50000+)에 맞게 매핑
            let vocab_id = (i as u32) % (id_to_vocab.len() as u32);
            id_to_vocab.get(&vocab_id).cloned().unwrap_or_else(|| "[UNK]".to_string())
        })
        .collect::<Vec<_>>()
        .join(" ");
    
    top_tokens
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("실제 어휘집을 사용하는 RBE 모델 추론기");
    println!("(종료하려면 'exit' 또는 'quit'를 입력하세요)\n");

    let model_id = "skt/kogpt2-base-v2";
    let model_file = "./models/skt-kogpt2-base-v2_compressed/kogpt2_256x256_w500.rbe";
    // hf-hub가 관리하는 실제 캐시 경로를 사용하도록 수정해야 할 수 있지만,
    // 우선은 ModelDownloader가 다운로드할 기본 경로를 가정합니다.
    // downloader의 로직을 보면 `~/.cache/huggingface/hub/models--skt--kogpt2-base-v2/snapshots/...` 와 유사한 경로에 저장됩니다.
    // 여기서는 간단하게 로컬 경로를 사용합니다.
    let vocab_file = format!("./models/{}/tokenizer.json", model_id.replace('/', "-"));


    // 실제 어휘집 로드
    let vocab = match load_vocabulary(&vocab_file) {
        Ok(v) => {
            if v.is_empty() {
                println!("경고: 어휘집을 로드했지만 단어가 없습니다.");
                None
            } else {
                println!("'{}'에서 실제 어휘집 로드 완료 ({}개 단어)", vocab_file, v.len());
                Some(v)
            }
        },
        Err(e) => {
            println!("오류: {}", e);
            None
        }
    };
    
    // 압축된 가중치 행렬 로드 및 복원
    let matrix = match std::path::Path::new(model_file).exists() {
        true => {
            println!("압축 모델 '{}' 로드 중...", model_file);
            let blocks = load_compressed_model(model_file)?;
            println!("{}개 블록 로드 완료. 가중치 행렬 복원 중...", blocks.len());
            let start = Instant::now();
            let restored_matrix = reconstruct_matrix(&blocks, 768, 256);
            println!("가중치 행렬 복원 완료! ({:.2}초 소요)\n", start.elapsed().as_secs_f32());
            Some(restored_matrix)
        }
        false => {
            println!("경고: 압축 모델 파일 '{}'을 찾을 수 없습니다.", model_file);
            println!("'compress_model'을 먼저 실행해주세요.\n");
            None
        }
    };

    if matrix.is_none() || vocab.is_none() {
        println!("필수 파일이 없어 추론을 시작할 수 없습니다.");
        return Ok(());
    }
    let full_matrix = matrix.unwrap();
    let id_to_vocab = vocab.unwrap();

    let stdin = io::stdin();
    loop {
        print!("질문: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        stdin.read_line(&mut input)?;
        let input = input.trim();
        
        if input == "exit" || input == "quit" {
            println!("프로그램을 종료합니다.");
            break;
        }
        if input.is_empty() {
            continue;
        }

        let start = Instant::now();
        let input_vector = prompt_to_vector(input, 768);
        let output_vector = &full_matrix * input_vector;
        let response = vector_to_text(&output_vector, &id_to_vocab);
        let inference_time = start.elapsed();
        
        println!("응답: {}", response);
        println!("(추론 시간: {:.3}ms)\n", inference_time.as_secs_f64() * 1000.0);
    }

    Ok(())
} 
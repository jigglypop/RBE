use RBE_LLM::types::HybridEncodedBlock;
use std::fs;
use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use std::time::Instant;
use std::collections::HashMap;
use tokenizers::Tokenizer;
use indicatif::{ProgressBar, ProgressStyle};

// 모델 프로파일 구조체
#[derive(Debug, Clone)]
struct ModelProfile {
    name: String,
    file_path: String,
}

// 테스트 결과 구조체
#[derive(Debug)]
struct TestResult {
    model_name: String,
    question: String,
    response: String,
    inference_time_ms: f64,
    tokens_per_second: f32,
}

fn load_compressed_model(filepath: &str) -> Result<Vec<HybridEncodedBlock>> {
    let json_content = fs::read_to_string(filepath)?;
    let data: serde_json::Value = serde_json::from_str(&json_content)?;
    let blocks_data = data.get("blocks")
        .ok_or_else(|| anyhow::anyhow!("JSON에서 'blocks' 키를 찾을 수 없습니다."))?;
    serde_json::from_value(blocks_data.clone())
        .map_err(|e| anyhow::anyhow!("블록 데이터 파싱에 실패했습니다: {}", e))
}

fn load_vocabulary(filepath: &str) -> Result<HashMap<u32, String>> {
    let tokenizer = Tokenizer::from_file(filepath)
        .map_err(|e| anyhow::anyhow!("tokenizer.json 로드 실패: {}. 파일 경로: {}", e, filepath))?;
    let vocab = tokenizer.get_vocab(false);
    let id_to_token: HashMap<u32, String> = vocab.into_iter()
        .map(|(token, id)| (id, token))
        .collect();
    Ok(id_to_token)
}

fn reconstruct_matrix(blocks: &[HybridEncodedBlock], rows: usize, cols: usize) -> DMatrix<f32> {
    let mut matrix = DMatrix::zeros(rows, cols);
    
    if blocks.is_empty() {
        return matrix;
    }
    
    // 블록 크기 확인
    let block_rows = blocks[0].rows;
    let block_cols = blocks[0].cols;
    let blocks_per_row = (cols + block_cols - 1) / block_cols;
    
    for (block_idx, block) in blocks.iter().enumerate() {
        let block_row = block_idx / blocks_per_row;
        let block_col = block_idx % blocks_per_row;
        let start_row = block_row * block_rows;
        let start_col = block_col * block_cols;
        
        let block_matrix = block.decode();
        
        for i in 0..block.rows {
            for j in 0..block.cols {
                let global_i = start_row + i;
                let global_j = start_col + j;
                if global_i < rows && global_j < cols {
                    matrix[(global_i, global_j)] = block_matrix[i * block.cols + j];
                }
            }
        }
    }
    
    matrix
}

fn prompt_to_vector(prompt: &str, dim: usize) -> DVector<f32> {
    let mut vec = DVector::zeros(dim);
    let chars: Vec<char> = prompt.chars().collect();
    
    for (i, ch) in chars.iter().enumerate() {
        let hash = (*ch as u32).wrapping_mul(31_u32.wrapping_pow(i as u32));
        let idx = (hash as usize) % dim;
        vec[idx] += 1.0;
    }
    
    if vec.norm() > 0.0 {
        vec.normalize_mut();
    }
    vec
}

fn vector_to_text(vector: &DVector<f32>, id_to_vocab: &HashMap<u32, String>) -> String {
    let mut indices: Vec<_> = (0..vector.len()).collect();
    indices.sort_unstable_by(|&a, &b| 
        vector[b].partial_cmp(&vector[a]).unwrap_or(std::cmp::Ordering::Equal));

    let top_tokens = indices.iter()
        .take(10)
        .map(|&i| {
            let vocab_id = (i as u32) % (id_to_vocab.len() as u32);
            id_to_vocab.get(&vocab_id).cloned().unwrap_or_else(|| "[UNK]".to_string())
        })
        .collect::<Vec<_>>()
        .join(" ");
    
    top_tokens
}

fn test_model(
    model_profile: &ModelProfile,
    questions: &[&str],
    vocab: &HashMap<u32, String>,
    progress: &ProgressBar,
) -> Result<Vec<TestResult>> {
    progress.set_message(format!("테스트 중: {}", model_profile.name));
    
    // 모델 로드
    let blocks = load_compressed_model(&model_profile.file_path)?;
    
    // 실제 블록 구조를 기반으로 행렬 크기 결정
    if blocks.is_empty() {
        return Err(anyhow::anyhow!("압축된 블록이 없습니다."));
    }
    
    // 첫 번째 블록에서 정보 가져오기
    let first_block = &blocks[0];
    let block_size = first_block.rows; // 블록 크기
    let num_blocks_sqrt = (blocks.len() as f64).sqrt() as usize;
    let matrix_size = num_blocks_sqrt * block_size;
    
    let matrix = reconstruct_matrix(&blocks, matrix_size, matrix_size);
    
    let mut results = Vec::new();
    
    // 각 질문에 대해 테스트
    for question in questions {
        let start = Instant::now();
        
        // 입력 벡터 생성 및 추론
        let input_vector = prompt_to_vector(question, matrix_size);
        let output_vector = &matrix * input_vector;
        let response = vector_to_text(&output_vector, vocab);
        
        let inference_time = start.elapsed();
        let inference_ms = inference_time.as_secs_f64() * 1000.0;
        let tokens_per_second = 10.0 / inference_time.as_secs_f32(); // 10 토큰 생성 가정
        
        results.push(TestResult {
            model_name: model_profile.name.clone(),
            question: question.to_string(),
            response,
            inference_time_ms: inference_ms,
            tokens_per_second,
        });
    }
    
    Ok(results)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("\n=== RBE 압축 모델 성능 비교 테스트 ===\n");
    
    // 한글 질문 세트 (5개)
    let questions = vec![
        "오늘 날씨가 어때요?",
        "한국의 수도는 어디인가요?",
        "인공지능이 무엇인지 설명해주세요.",
        "가장 좋아하는 음식은 무엇인가요?",
        "미래의 기술은 어떻게 발전할까요?",
    ];
    
    // 테스트할 모델 프로파일들
    let model_profiles = vec![
        ModelProfile {
            name: "극한 압축 (256x256, 50계수)".to_string(),
            file_path: "./models/skt-kogpt2-base-v2_compressed/kogpt2_256x256_w50.rbe".to_string(),
        },
        ModelProfile {
            name: "초고압축 (256x256, 100계수)".to_string(),
            file_path: "./models/skt-kogpt2-base-v2_compressed/kogpt2_256x256_w100.rbe".to_string(),
        },
        ModelProfile {
            name: "고압축 (256x256, 200계수)".to_string(),
            file_path: "./models/skt-kogpt2-base-v2_compressed/kogpt2_256x256_w200.rbe".to_string(),
        },
        ModelProfile {
            name: "표준 압축 (256x256, 500계수)".to_string(),
            file_path: "./models/skt-kogpt2-base-v2_compressed/kogpt2_256x256_w500.rbe".to_string(),
        },
        ModelProfile {
            name: "균형 압축 (128x128, 500계수)".to_string(),
            file_path: "./models/skt-kogpt2-base-v2_compressed/kogpt2_128x128_w500.rbe".to_string(),
        },
        ModelProfile {
            name: "고품질 (64x64, 1000계수)".to_string(),
            file_path: "./models/skt-kogpt2-base-v2_compressed/kogpt2_64x64_w1000.rbe".to_string(),
        },
        ModelProfile {
            name: "초고품질 (32x32, 2000계수)".to_string(),
            file_path: "./models/skt-kogpt2-base-v2_compressed/kogpt2_32x32_w2000.rbe".to_string(),
        },
    ];
    
    // 어휘집 로드
    println!("어휘집 로드 중...");
    let vocab_file = "./models/skt-kogpt2-base-v2/tokenizer.json";
    let vocab = load_vocabulary(vocab_file)?;
    println!("어휘집 로드 완료: {}개 토큰\n", vocab.len());
    
    // 진행률 표시
    let progress = ProgressBar::new(model_profiles.len() as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {pos}/{len} {msg}")
            .unwrap()
    );
    
    // 모든 모델 테스트
    let mut all_results = Vec::new();
    
    for model_profile in &model_profiles {
        match test_model(model_profile, &questions, &vocab, &progress) {
            Ok(results) => {
                all_results.extend(results);
                progress.inc(1);
            }
            Err(e) => {
                eprintln!("모델 {} 테스트 실패: {}", model_profile.name, e);
                progress.inc(1);
                continue;
            }
        }
    }
    
    progress.finish_with_message("테스트 완료!");
    
    // 결과 출력
    println!("\n=== 테스트 결과 ===\n");
    
    // 질문별로 결과 그룹화 및 출력
    for question in &questions {
        println!("\n질문: {}", question);
        println!("{:-<80}", "");
        
        let mut question_results: Vec<&TestResult> = all_results.iter()
            .filter(|r| r.question == *question)
            .collect();
        
        // 추론 속도 순으로 정렬
        question_results.sort_by(|a, b| 
            a.inference_time_ms.partial_cmp(&b.inference_time_ms).unwrap());
        
        for result in question_results {
            println!("모델: {}", result.model_name);
            println!("응답: {}", result.response);
            println!("추론 시간: {:.2}ms ({:.1} tokens/s)", 
                result.inference_time_ms, result.tokens_per_second);
            println!();
        }
    }
    
    // 전체 통계
    println!("\n=== 모델별 평균 성능 ===");
    println!("{:-<80}", "");
    println!("{:<40} | {:<15} | {:<15}", "모델", "평균 추론시간(ms)", "평균 토큰/초");
    println!("{:-<80}", "");
    
    for profile in &model_profiles {
        let model_results: Vec<&TestResult> = all_results.iter()
            .filter(|r| r.model_name == profile.name)
            .collect();
        
        if !model_results.is_empty() {
            let avg_time = model_results.iter()
                .map(|r| r.inference_time_ms)
                .sum::<f64>() / model_results.len() as f64;
            
            let avg_tokens = model_results.iter()
                .map(|r| r.tokens_per_second)
                .sum::<f32>() / model_results.len() as f32;
            
            println!("{:<40} | {:<15.2} | {:<15.1}", 
                profile.name, avg_time, avg_tokens);
        }
    }
    
    // 결과 저장
    let results_json = serde_json::json!({
        "test_date": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
        "questions": questions,
        "results": all_results.iter().map(|r| {
            serde_json::json!({
                "model": r.model_name,
                "question": r.question,
                "response": r.response,
                "inference_time_ms": r.inference_time_ms,
                "tokens_per_second": r.tokens_per_second,
            })
        }).collect::<Vec<_>>(),
    });
    
    fs::write("./models/skt-kogpt2-base-v2_compressed/test_results.json", 
        serde_json::to_string_pretty(&results_json)?)?;
    
    println!("\n✅ 테스트 완료!");
    println!("📊 상세 결과: ./models/skt-kogpt2-base-v2_compressed/test_results.json");
    
    Ok(())
} 
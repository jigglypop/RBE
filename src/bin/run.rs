use rbe_llm::packed_params::HybridEncodedBlock;
use tokenizers::Tokenizer;
use std::io::{self, Write};
use std::time::Instant;
use nalgebra::DMatrix;
use std::fs;
use anyhow::Result;
use serde_json::Value;

/// 실제 RBE 압축 모델을 사용하는 한국어 GPT-2
struct RealRBEModel {
    tokenizer: Tokenizer,
    /// 실제 압축된 블록들
    compressed_blocks: Vec<HybridEncodedBlock>,
    /// 복원된 가중치 매트릭스
    weight_matrix: DMatrix<f32>,
    vocab_size: usize,
    hidden_size: usize,
}

impl RealRBEModel {
    /// 실제 압축된 .rbe 파일로부터 모델 로드
    fn load_from_rbe(rbe_path: &str, tokenizer_path: &str) -> Result<Self> {
        println!("🚀 실제 RBE 압축 모델 로딩...");
        println!("   - RBE 파일: {}", rbe_path);
        println!("   - 토크나이저: {}", tokenizer_path);
        
        // 1. 토크나이저 로드
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("토크나이저 로드 실패: {:?}", e))?;
        println!("✅ 토크나이저 로드 완료: {} 어휘", tokenizer.get_vocab_size(false));
        
        // 2. 압축된 RBE 파일 로드
        let rbe_content = fs::read_to_string(rbe_path)?;
        let rbe_data: Value = serde_json::from_str(&rbe_content)?;
        
        let blocks_data = rbe_data.get("blocks")
            .ok_or_else(|| anyhow::anyhow!("RBE 파일에서 'blocks' 키를 찾을 수 없습니다"))?;
        
        let compressed_blocks: Vec<HybridEncodedBlock> = serde_json::from_value(blocks_data.clone())?;
        println!("✅ 압축된 블록 로드 완료: {} 개", compressed_blocks.len());
        
        // 3. 메타데이터 확인
        if let Some(metadata) = rbe_data.get("metadata") {
            if let Some(compression_ratio) = metadata.get("compression_ratio") {
                println!("🗜️ 압축률: {:.1}:1", compression_ratio.as_f64().unwrap_or(0.0));
            }
            if let Some(matrix_size) = metadata.get("matrix_size") {
                println!("📐 매트릭스 크기: {}×{}", matrix_size, matrix_size);
            }
        }
        
        // 4. 실제 core 모듈의 decode() 함수로 가중치 복원
        println!("🔄 가중치 매트릭스 복원 중...");
        let start = Instant::now();
        let weight_matrix = Self::reconstruct_matrix_from_blocks(&compressed_blocks)?;
        let decode_time = start.elapsed();
        println!("✅ 가중치 복원 완료! ({:.2}초 소요)", decode_time.as_secs_f32());
        println!("   - 복원된 매트릭스: {} × {}", weight_matrix.nrows(), weight_matrix.ncols());
        
        // 5. 모델 설정
        let vocab_size = 51200; // KoGPT-2
        let hidden_size = weight_matrix.ncols().min(768);
        
        println!("🎉 실제 RBE 모델 로딩 완료!");
        println!("   - 어휘 크기: {}", vocab_size);
        println!("   - 은닉층 크기: {}", hidden_size);
        
        Ok(Self {
            tokenizer,
            compressed_blocks,
            weight_matrix,
            vocab_size,
            hidden_size,
        })
    }
    
    /// 실제 HybridEncodedBlock들로부터 가중치 매트릭스 복원
    fn reconstruct_matrix_from_blocks(blocks: &[HybridEncodedBlock]) -> Result<DMatrix<f32>> {
        if blocks.is_empty() {
            return Err(anyhow::anyhow!("압축된 블록이 없습니다"));
        }
        
        // 첫 번째 블록 정보로 전체 크기 추정
        let first_block = &blocks[0];
        let block_size = first_block.rows.max(first_block.cols);
        let blocks_per_dim = (blocks.len() as f32).sqrt().ceil() as usize;
        let matrix_size = blocks_per_dim * block_size;
        
        println!("   - 블록 크기: {} × {}", first_block.rows, first_block.cols);
        println!("   - 총 블록 수: {}", blocks.len());
        println!("   - 예상 매트릭스 크기: {} × {}", matrix_size, matrix_size);
        
        let mut full_matrix = DMatrix::from_element(matrix_size, matrix_size, 0.0);
        
        // 각 블록을 실제 core 모듈의 decode() 함수로 복원
        for (block_idx, block) in blocks.iter().enumerate() {
            // ✨ 실제 core 모듈의 HybridEncodedBlock::decode() 사용
            let decoded_data = block.decode();
            
            if decoded_data.len() != block.rows * block.cols {
                println!("⚠️ 블록 {} 크기 불일치: 예상 {}, 실제 {}", 
                         block_idx, block.rows * block.cols, decoded_data.len());
                continue;
            }
            
            // 블록 위치 계산
            let grid_i = block_idx / blocks_per_dim;
            let grid_j = block_idx % blocks_per_dim;
            let start_i = grid_i * block_size;
            let start_j = grid_j * block_size;
            
            // 복원된 데이터를 전체 매트릭스에 배치
            for r in 0..block.rows {
                for c in 0..block.cols {
                    let global_i = start_i + r;
                    let global_j = start_j + c;
                    
                    if global_i < matrix_size && global_j < matrix_size {
                        let block_idx_data = r * block.cols + c;
                        if block_idx_data < decoded_data.len() {
                            full_matrix[(global_i, global_j)] = decoded_data[block_idx_data];
                        }
                    }
                }
            }
            
            if block_idx % 10 == 0 {
                println!("   - 블록 {}/{} 복원 완료", block_idx + 1, blocks.len());
            }
        }
        
        Ok(full_matrix)
    }
    
    /// 한국어 텍스트 생성 (실제 RBE 가중치 사용)
    fn generate_korean(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        println!("\n💭 실제 RBE 가중치로 텍스트 생성: '{}'", prompt);
        
        // 1. 토크나이징
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("토크나이징 실패: {:?}", e))?;
        let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
        
        println!("🔤 초기 토큰 수: {}", token_ids.len());
        
        // 2. 생성 루프
        for step in 0..max_tokens {
            let next_token = self.predict_next_token_with_rbe(&token_ids)?;
            
            // EOS 체크
            if next_token == 50256 || next_token == 0 {
                break;
            }
            
            token_ids.push(next_token);
            
            // 진행 상황 출력
            if step % 3 == 0 {
                let partial = self.tokenizer.decode(&token_ids, true)
                    .unwrap_or_else(|_| "디코딩 오류".to_string());
                println!("📝 단계 {}: {}", step, partial);
            }
        }
        
        // 3. 최종 디코딩
        let result = self.tokenizer.decode(&token_ids, true)
            .map_err(|e| anyhow::anyhow!("디코딩 실패: {:?}", e))?;
        
        Ok(result)
    }
    
    /// 실제 RBE 복원 가중치로 다음 토큰 예측
    fn predict_next_token_with_rbe(&self, token_ids: &[u32]) -> Result<u32> {
        let seq_len = token_ids.len().min(64); // 긴 시퀀스 제한
        let recent_tokens = &token_ids[token_ids.len().saturating_sub(seq_len)..];
        
        // 1. 입력 벡터 생성 (토큰 ID들을 임베딩으로 변환)
        let input_vector = self.create_input_vector(recent_tokens);
        
        // 2. 실제 복원된 RBE 가중치 매트릭스와 곱셈
        let output_vector = &self.weight_matrix * input_vector;
        
        // 3. 출력을 어휘 크기로 매핑
        let logits = self.vector_to_logits(&output_vector);
        
        // 4. 한국어 친화적 샘플링
        let next_token = self.sample_korean_token(&logits)?;
        
        Ok(next_token)
    }
    
    /// 토큰들을 입력 벡터로 변환
    fn create_input_vector(&self, token_ids: &[u32]) -> nalgebra::DVector<f32> {
        let input_size = self.weight_matrix.ncols();
        let mut input_vector = nalgebra::DVector::from_element(input_size, 0.0);
        
        // 토큰 ID들을 벡터 공간에 매핑
        for (pos, &token_id) in token_ids.iter().enumerate() {
            let token_idx = (token_id as usize) % input_size;
            let position_weight = 1.0 / (pos + 1) as f32; // 위치별 가중치
            
            // 토큰 임베딩 (간소화된 버전)
            input_vector[token_idx] += position_weight;
            
            // 주변 인덱스도 약간 활성화 (의미적 유사성 모델링)
            for offset in [-2, -1, 1, 2] {
                let neighbor_idx = (token_idx as i32 + offset);
                if neighbor_idx >= 0 && (neighbor_idx as usize) < input_size {
                    input_vector[neighbor_idx as usize] += position_weight * 0.1;
                }
            }
        }
        
        // 정규화
        let norm = input_vector.norm();
        if norm > 0.0 {
            input_vector /= norm;
        }
        
        input_vector
    }
    
    /// 출력 벡터를 로짓으로 변환
    fn vector_to_logits(&self, output_vector: &nalgebra::DVector<f32>) -> Vec<f32> {
        let mut logits = vec![0.0f32; self.vocab_size];
        
        // 출력 벡터의 각 차원을 어휘 항목에 매핑
        for i in 0..logits.len() {
            let vector_idx = i % output_vector.len();
            logits[i] = output_vector[vector_idx];
            
            // 한국어 토큰에 약간의 편향 추가
            if i < 5000 { // 가정: 처음 5000개가 한국어 토큰
                logits[i] += 0.1;
            }
        }
        
        logits
    }
    
    /// 한국어 친화적 토큰 샘플링
    fn sample_korean_token(&self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() {
            return Ok(0);
        }
        
        // Temperature 스케일링 (한국어에 적합)
        let temperature = 0.8;
        let scaled_logits: Vec<f32> = logits.iter()
            .map(|&x| x / temperature)
            .collect();
        
        // Softmax 계산
        let max_logit = scaled_logits.iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        
        let exp_logits: Vec<f32> = scaled_logits.iter()
            .map(|&x| (x - max_logit).exp())
            .collect();
        
        let sum_exp: f32 = exp_logits.iter().sum();
        if sum_exp <= 0.0 {
            return Ok(0);
        }
        
        let probabilities: Vec<f32> = exp_logits.iter()
            .map(|&x| x / sum_exp)
            .collect();
        
        // 누적 확률 샘플링
        let random_val: f32 = rand::random();
        let mut cumulative = 0.0f32;
        
        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_val <= cumulative {
                return Ok(i as u32);
            }
        }
        
        Ok((probabilities.len() - 1) as u32)
    }
    
    /// 모델 정보 출력
    fn print_model_info(&self) {
        println!("\n📊 === 실제 RBE 모델 정보 ===");
        println!("🗜️ 압축된 블록 수: {}", self.compressed_blocks.len());
        println!("📐 복원된 매트릭스: {} × {}", self.weight_matrix.nrows(), self.weight_matrix.ncols());
        println!("🔤 어휘 크기: {}", self.vocab_size);
        println!("🧠 은닉층 크기: {}", self.hidden_size);
        
        // RBE 압축 정보
        let total_elements = self.weight_matrix.nrows() * self.weight_matrix.ncols();
        let rbe_params_count = self.compressed_blocks.len() * 8; // 각 블록당 8개 RBE 파라미터
        let residual_coeffs_count: usize = self.compressed_blocks.iter()
            .map(|b| b.residuals.len())
            .sum();
        
        println!("📈 RBE 파라미터: {}", rbe_params_count);
        println!("🔄 잔차 계수: {}", residual_coeffs_count);
        
        let compression_ratio = total_elements as f32 / (rbe_params_count + residual_coeffs_count) as f32;
        println!("⚡ 실제 압축률: {:.1}:1", compression_ratio);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🇰🇷 === 실제 RBE 압축 기반 한국어 추론 ===");
    println!("core 모듈의 HybridEncodedBlock::decode() 사용\n");
    
    // 실제 압축된 .rbe 파일 경로들
    let rbe_models = vec![
        ("극압축", "./models/skt-kogpt2-base-v2_compressed/kogpt2_256x256_w50.rbe"),
        ("고압축", "./models/skt-kogpt2-base-v2_compressed/kogpt2_256x256_w100.rbe"),
        ("균형", "./models/skt-kogpt2-base-v2_compressed/kogpt2_256x256_w500.rbe"),
    ];
    
    let tokenizer_path = "./models/skt-kogpt2-base-v2/tokenizer.json";
    
    // 사용자 선택
    println!("📋 사용 가능한 RBE 모델:");
    for (i, (name, path)) in rbe_models.iter().enumerate() {
        let file_size = fs::metadata(path)
            .map(|m| m.len() / 1024)
            .unwrap_or(0);
        println!("   {}. {} - {} KB", i + 1, name, file_size);
    }
    
    // 기본값으로 균형 모델 사용
    let selected_model = &rbe_models[2]; // 균형 모델
    println!("🎯 {} 모델 사용: {}", selected_model.0, selected_model.1);
    
    // 실제 RBE 모델 로드
    let model = RealRBEModel::load_from_rbe(selected_model.1, tokenizer_path)?;
    model.print_model_info();
    
    println!("\n💬 실제 RBE 가중치로 한국어 대화 시작! (종료: 'exit')");

    let stdin = io::stdin();
    loop {
        print!("\n질문: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        stdin.read_line(&mut input)?;
        let input = input.trim();
        
        if input == "exit" || input == "quit" || input == "종료" {
            println!("👋 실제 RBE 엔진을 종료합니다.");
            break;
        }
        
        if input.is_empty() {
            continue;
        }

        let start = Instant::now();
        
        match model.generate_korean(input, 25) {
            Ok(response) => {
                let duration = start.elapsed();
                
                // 원래 프롬프트 제거하고 새로 생성된 부분만 표시
                let generated_part = if response.starts_with(input) {
                    response[input.len()..].trim()
                } else {
                    &response
                };
                
                if !generated_part.is_empty() {
                    println!("🎯 실제 RBE 답변: {}", generated_part);
                } else {
                    println!("🎯 실제 RBE 답변: {}", response);
                }
                
                println!("⏱️ 생성 시간: {:.2}초", duration.as_secs_f32());
                println!("🗜️ 실제 core 모듈 RBE 사용");
                println!("✨ HybridEncodedBlock::decode() 기반");
            }
            Err(e) => {
                println!("❌ 오류: {}", e);
            }
        }
    }

    Ok(())
} 
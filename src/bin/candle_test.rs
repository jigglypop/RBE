use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use candle_nn;
use tokenizers::Tokenizer;
use std::io::{self, Write};
use std::time::Instant;
use std::path::Path;

/// Candle을 사용한 간단한 테스트 모델
struct CandleTestModel {
    tokenizer: Tokenizer,
    device: Device,
    vocab_size: usize,
}

impl CandleTestModel {
    /// 간단한 Candle 테스트 모델 생성
    fn new(tokenizer_path: &str) -> Result<Self> {
        println!("🕯️ Candle 테스트 모델 초기화...");
        println!("   - 토크나이저: {}", tokenizer_path);
        
        // 1. 디바이스 설정 (CPU)
        let device = Device::Cpu;
        println!("✅ 디바이스: CPU");
        
        // 2. 토크나이저 로드
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("토크나이저 로드 실패: {:?}", e))?;
        let vocab_size = tokenizer.get_vocab_size(false);
        println!("✅ 토크나이저 로드 완료: {} 어휘", vocab_size);
        
        println!("✅ Candle 테스트 모델 초기화 완료!");
        
        Ok(Self {
            tokenizer,
            device,
            vocab_size,
        })
    }
    
    /// 간단한 텍스트 생성 (기본 Candle 동작 테스트)
    fn generate_text(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        println!("\n💭 Candle 기본 동작 테스트: '{}'", prompt);
        
        // 1. 토크나이징
        let encoding = self.tokenizer.encode(prompt, false)
            .map_err(|e| anyhow::anyhow!("토크나이징 실패: {:?}", e))?;
        let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
        
        println!("🔤 초기 토큰 수: {}", token_ids.len());
        
        // 2. 간단한 텐서 연산 테스트
        for step in 0..max_tokens {
            let next_token = self.simple_forward(&token_ids)?;
            
            // EOS 체크
            if next_token == 1 || next_token == 0 {
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
    
    /// 간단한 forward pass (Candle 텐서 연산 테스트)
    fn simple_forward(&self, token_ids: &[u32]) -> Result<u32> {
        // 기본 텐서 연산으로 다음 토큰 생성 (랜덤)
        let input_len = token_ids.len();
        
        // 간단한 가중치 매트릭스 생성 (임베딩 시뮬레이션) - F32 명시
        let weights = Tensor::randn(0.0f32, 1.0f32, (self.vocab_size, 768), &self.device)?
            .to_dtype(DType::F32)?;
        
        // 입력 토큰을 원핫 벡터로 변환
        let last_token = token_ids[input_len - 1] as usize % self.vocab_size;
        let mut input_vec = vec![0.0f32; self.vocab_size];
        input_vec[last_token] = 1.0;
        
        let input_tensor = Tensor::from_slice(&input_vec, (1, self.vocab_size), &self.device)?
            .to_dtype(DType::F32)?;
        
        // 간단한 매트릭스 곱셈
        let hidden = input_tensor.matmul(&weights)?;
        
        // 출력 레이어 (임시) - F32 명시
        let output_weights = Tensor::randn(0.0f32, 1.0f32, (768, self.vocab_size), &self.device)?
            .to_dtype(DType::F32)?;
        let logits = hidden.matmul(&output_weights)?;
        
        // 소프트맥스 + 샘플링 (간단한 argmax)
        let probs = candle_nn::ops::softmax(&logits, 1)?;
        let next_token_tensor = probs.argmax(1)?;
        
        // 1차원 텐서에서 첫 번째 요소 추출
        let next_token_array = next_token_tensor.to_vec1::<u32>()?;
        let next_token = next_token_array[0];
        
        Ok(next_token % (self.vocab_size as u32))
    }
    
    /// 모델 정보 출력
    fn print_model_info(&self) {
        println!("\n📊 Candle 테스트 모델 정보:");
        println!("  🕯️ 프레임워크: Candle (Rust 네이티브)");
        println!("  🔤 어휘 크기: {}", self.vocab_size);
        println!("  ⚡ 디바이스: {:?}", self.device);
        println!("  🧪 기본 텐서 연산 테스트 모드");
        println!("  ✅ Candle 프레임워크 동작 확인");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🇰🇷 === Candle 프레임워크 기본 동작 테스트 ===");
    println!("Candle의 텐서 연산과 토크나이저 동작 확인\n");
    
    let tokenizer_path = "./models/skt-kogpt2-base-v2/tokenizer.json";
    
    // 파일 존재 확인
    let tokenizer_exists = Path::new(tokenizer_path).exists();
    
    println!("📋 파일 확인:");
    println!("   - 토크나이저: {} ({})", tokenizer_path, 
             if tokenizer_exists { "존재" } else { "❌ 없음" });
    
    if !tokenizer_exists {
        return Err(anyhow::anyhow!("토크나이저 파일이 없습니다."));
    }
    
    // Candle 테스트 모델 초기화
    let model = CandleTestModel::new(tokenizer_path)?;
    model.print_model_info();
    
    println!("\n💬 Candle 기본 동작 테스트 시작! (종료: 'exit')");
    println!("⚠️ 랜덤 가중치로 Candle 텐서 연산 동작 확인");

    let stdin = io::stdin();
    loop {
        print!("\n질문: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        stdin.read_line(&mut input)?;
        let input = input.trim();
        
        if input == "exit" || input == "quit" || input == "종료" {
            println!("👋 Candle 테스트를 종료합니다.");
            break;
        }
        
        if input.is_empty() {
            continue;
        }

        let start = Instant::now();
        
        match model.generate_text(input, 10) {
            Ok(response) => {
                let duration = start.elapsed();
                
                println!("🎯 Candle 테스트 결과: {}", response);
                println!("⏱️ 처리 시간: {:.2}초", duration.as_secs_f32());
                println!("🕯️ Candle 텐서 연산 정상 동작");
            }
            Err(e) => {
                println!("❌ 오류: {}", e);
            }
        }
    }

    Ok(())
} 
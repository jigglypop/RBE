//! 실제 한국어 LLM 채팅 시스템 데모
//! 
//! RBE 압축 시스템을 사용하여 실제로 동작하는 한국어 텍스트 생성을 수행합니다.

use anyhow::Result;
use rbe_llm::nlp::korean_llm::{
    KoreanLLMConfig, KoreanModelLoader, 
    tokenizer::KoreanTokenizer,
    generator::{KoreanTextGenerator, GenerationParams}
};
use std::io::{self, Write};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🇰🇷 실제 한국어 RBE-LLM 채팅 시스템");
    println!("======================================\n");

    // 1. 설정 준비 (실제 생성용 한국어 모델 사용)
    let config = KoreanLLMConfig {
        model_id: "EleutherAI/polyglot-ko-1.3b".to_string(),  // 한국어 GPT 모델 (SafeTensors 지원)
        cache_dir: PathBuf::from("models/korean_cache"),
        enable_compression: true,
        use_rbe_optimization: true,
        temperature: 0.8,
        ..Default::default()
    };

    println!("📋 시스템 설정:");
    println!("  - 모델: {}", config.model_id);
    println!("  - RBE 압축: 활성화");
    println!("  - 캐시 디렉토리: {}", config.cache_dir.display());
    println!();

    // 2. 모델 다운로드 및 로딩
    println!("📥 모델 다운로드 및 로딩 시작...");
    let mut model_loader = KoreanModelLoader::new(&config.model_id, &config.cache_dir);
    
    // 모델 다운로드
    model_loader.download_and_load().await?;
    println!("✅ 모델 다운로드 완료");

    // 3. 실제 가중치 로딩 및 RBE 압축
    println!("\n🔄 실제 모델 가중치 로딩 및 RBE 압축 중...");
    let model_index = model_loader.load_and_compress_weights().await?;
    
    println!("📊 압축 결과:");
    println!("  - 전체 압축률: {:.1}:1", model_index.compression_summary.overall_compression_ratio);
    println!("  - 평균 RMSE: {:.6}", model_index.compression_summary.average_rmse);
    println!("  - 압축된 레이어: {}", model_index.compressed_layers.len());
    println!("  - 총 파라미터: {}", model_index.total_parameters);

    // 4. 토크나이저 로딩
    println!("\n🔤 토크나이저 로딩 중...");
    let mut tokenizer = KoreanTokenizer::new(&config.model_id);
    if let Some(model_path) = model_loader.get_model_path() {
        tokenizer.load_from_model_path(model_path).await?;
    }

    let tokenizer_info = tokenizer.get_info();
    println!("✅ 토크나이저 로딩 완료:");
    println!("  - Vocab 크기: {}", tokenizer_info.vocab_size);
    println!("  - 특수 토큰: {}", tokenizer_info.special_tokens_count);

    // 5. 텍스트 생성기 초기화
    println!("\n🤖 RBE 텍스트 생성기 초기화 중...");
    let mut generator = KoreanTextGenerator::new(config.clone());
    generator.load_model(model_index.clone(), tokenizer).await?;

    // 6. 인터랙티브 채팅 시작
    println!("\n🎉 시스템 초기화 완료! 채팅을 시작합니다.");
    println!("사용법:");
    println!("  - 'exit' 또는 'quit': 종료");
    println!("  - 'stats': 생성 통계 확인");
    println!("  - 'info': 모델 정보 확인");
    println!("  - '/temp X.X': 온도 설정 (예: /temp 0.5)");
    println!("=====================================\n");

    let mut generation_params = GenerationParams::default();

    loop {
        // 사용자 입력 받기
        print!("🗣️  당신: ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        // 명령어 처리
        match input {
            "exit" | "quit" => {
                println!("👋 채팅을 종료합니다. 안녕히 가세요!");
                break;
            }
            "stats" => {
                show_statistics(&generator).await?;
                continue;
            }
            "info" => {
                show_model_info(&model_index);
                continue;
            }
            _ if input.starts_with("/temp ") => {
                if let Ok(temp) = input[6..].parse::<f32>() {
                    if temp >= 0.1 && temp <= 2.0 {
                        generation_params.temperature = temp;
                        println!("🌡️  온도를 {:.1}로 설정했습니다.", temp);
                    } else {
                        println!("❌ 온도는 0.1과 2.0 사이여야 합니다.");
                    }
                } else {
                    println!("❌ 올바른 온도 값을 입력하세요. (예: /temp 0.8)");
                }
                continue;
            }
            "" => continue,
            _ => {}
        }

        // 실제 텍스트 생성
        print!("🤖 RBE-LLM: ");
        io::stdout().flush()?;
        
        match generator.generate_with_params(input, generation_params.clone()).await {
            Ok(response) => {
                // 입력 부분 제거하고 생성된 부분만 추출
                let generated_part = extract_generated_text(input, &response);
                println!("{}\n", generated_part);
            }
            Err(e) => {
                println!("❌ 생성 오류: {}\n", e);
            }
        }
    }

    // 최종 통계
    println!("📊 최종 세션 통계:");
    show_statistics(&generator).await?;

    Ok(())
}

/// 생성 통계 표시
async fn show_statistics(generator: &KoreanTextGenerator) -> Result<()> {
    let stats = generator.get_statistics().await?;
    
    println!("📈 생성 통계:");
    println!("  - 총 생성 횟수: {}", stats.total_generations);
    println!("  - 총 토큰 수: {}", stats.total_tokens);
    println!("  - 평균 생성 시간: {:.2}ms", stats.average_time_ms);
    println!("  - 토큰/초: {:.1}", stats.tokens_per_second);
    
    Ok(())
}

/// 모델 정보 표시
fn show_model_info(model_index: &rbe_llm::nlp::korean_llm::model_loader::ModelIndex) {
    println!("📋 모델 정보:");
    println!("  - 모델 ID: {}", model_index.model_id);
    println!("  - 다운로드 시간: {}", model_index.download_timestamp);
    println!("  - 총 파라미터: {}", model_index.total_parameters);
    println!("  - 원본 크기: {:.1}MB", model_index.compression_summary.total_original_size_mb);
    println!("  - 압축 크기: {:.6}MB", model_index.compression_summary.total_compressed_size_mb);
    println!("  - 압축률: {:.1}:1", model_index.compression_summary.overall_compression_ratio);
    println!("  - 평균 RMSE: {:.6}", model_index.compression_summary.average_rmse);
    println!("  - 압축 시간: {:.1}ms", model_index.compression_summary.compression_time_ms);
    
    println!("  - 파일 구조:");
    for (filename, filetype) in &model_index.file_structure {
        println!("    * {}: {}", filename, filetype);
    }
    
    println!("  - 압축된 레이어 (상위 5개):");
    for (i, layer) in model_index.compressed_layers.iter().take(5).enumerate() {
        println!("    {}. {}: {:?}, 압축률 {:.1}:1, RMSE {:.6}", 
                i + 1, 
                layer.layer_name, 
                layer.original_shape,
                layer.compression_stats.compression_ratio,
                layer.compression_stats.rmse
        );
    }
    
    if model_index.compressed_layers.len() > 5 {
        println!("    ... 및 {} 개 더", model_index.compressed_layers.len() - 5);
    }
}

/// 생성된 텍스트에서 실제 생성 부분만 추출
fn extract_generated_text(input: &str, full_response: &str) -> String {
    // 입력 텍스트 이후의 부분을 생성된 텍스트로 간주
    if let Some(pos) = full_response.find(input) {
        let after_input = &full_response[pos + input.len()..];
        after_input.trim().to_string()
    } else {
        // 입력을 찾을 수 없으면 전체 응답 반환
        full_response.trim().to_string()
    }
}

/// 에러 핸들링을 위한 헬퍼 함수
fn handle_error(e: anyhow::Error) {
    eprintln!("❌ 오류 발생: {}", e);
    eprintln!("가능한 해결책:");
    eprintln!("  1. 인터넷 연결 확인");
    eprintln!("  2. HUGGING_FACE_TOKEN 환경 변수 설정");
    eprintln!("  3. 디스크 공간 확인");
    eprintln!("  4. 프로그램 재실행");
} 
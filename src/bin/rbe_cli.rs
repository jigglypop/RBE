use rbe_llm::sllm::{
    ModelDownloader, SLLMCompressor, RBEInferenceEngine, CompressionConfig, 
    SLLMBenchmark, BenchmarkConfig
};
use clap::{Arg, Command};
use std::path::PathBuf;
use std::process;

#[tokio::main]
async fn main() {
    env_logger::init();
    
    let matches = Command::new("RBE CLI")
        .version("1.0.0")
        .about("푸앵카레 볼 기반 RBE 모델 CLI 도구")
        .subcommand(
            Command::new("download")
                .about("HuggingFace에서 모델 다운로드")
                .arg(
                    Arg::new("model-id")
                        .required(true)
                        .help("HuggingFace 모델 ID (예: skt/kogpt2-base-v2)")
                )
                .arg(
                    Arg::new("output-dir")
                        .long("output")
                        .short('o')
                        .value_name("DIR")
                        .help("출력 디렉토리")
                        .default_value("./models")
                )
        )
        .subcommand(
            Command::new("compress")
                .about("모델 압축")
                .arg(
                    Arg::new("model-path")
                        .required(true)
                        .help("압축할 모델 경로")
                )
                .arg(
                    Arg::new("output-path")
                        .required(true)
                        .help("압축된 모델 출력 경로")
                )
                .arg(
                    Arg::new("compression-level")
                        .long("level")
                        .short('l')
                        .value_name("LEVEL")
                        .help("압축 레벨 (1-5)")
                        .default_value("3")
                )
                .arg(
                    Arg::new("block-size")
                        .long("block-size")
                        .value_name("SIZE")
                        .help("블록 크기")
                        .default_value("32")
                )
                .arg(
                    Arg::new("coefficients")
                        .long("coefficients")
                        .short('c')
                        .value_name("COUNT")
                        .help("웨이블릿 계수 개수")
                        .default_value("500")
                )
        )
        .subcommand(
            Command::new("generate")
                .about("텍스트 생성")
                .arg(
                    Arg::new("compressed-model")
                        .required(true)
                        .help("압축된 모델 경로")
                )
                .arg(
                    Arg::new("original-model")
                        .required(true)
                        .help("원본 모델 경로 (설정용)")
                )
                .arg(
                    Arg::new("prompt")
                        .required(true)
                        .help("생성할 텍스트 프롬프트")
                )
                .arg(
                    Arg::new("max-tokens")
                        .long("max-tokens")
                        .value_name("TOKENS")
                        .help("최대 생성 토큰 수")
                        .default_value("50")
                )
                .arg(
                    Arg::new("temperature")
                        .long("temperature")
                        .value_name("TEMP")
                        .help("Temperature 값 (0.0-2.0)")
                        .default_value("0.7")
                )
                .arg(
                    Arg::new("top-p")
                        .long("top-p")
                        .value_name("TOP_P")
                        .help("Top-p 값 (0.0-1.0)")
                        .default_value("0.9")
                )
        )
        .subcommand(
            Command::new("benchmark")
                .about("성능 벤치마크 실행")
                .arg(
                    Arg::new("iterations")
                        .long("iterations")
                        .short('i')
                        .value_name("COUNT")
                        .help("반복 횟수")
                        .default_value("10")
                )
                .arg(
                    Arg::new("save-results")
                        .long("save")
                        .short('s')
                        .value_name("FILE")
                        .help("결과를 JSON 파일로 저장")
                )
        )
        .subcommand(
            Command::new("info")
                .about("압축된 모델 정보 확인")
                .arg(
                    Arg::new("model-path")
                        .required(true)
                        .help("압축된 모델 파일 경로")
                )
        )
        .get_matches();
    
    let result = match matches.subcommand() {
        Some(("download", sub_matches)) => handle_download(sub_matches).await,
        Some(("compress", sub_matches)) => handle_compress(sub_matches).await,
        Some(("generate", sub_matches)) => handle_generate(sub_matches).await,
        Some(("benchmark", sub_matches)) => handle_benchmark(sub_matches).await,
        Some(("info", sub_matches)) => handle_info(sub_matches).await,
        _ => {
            println!("❌ 명령을 지정해주세요. --help를 참조하세요.");
            process::exit(1);
        }
    };
    
    if let Err(e) = result {
        eprintln!("❌ 오류: {}", e);
        process::exit(1);
    }
}

async fn handle_download(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let model_id = matches.get_one::<String>("model-id").unwrap();
    let output_dir = matches.get_one::<String>("output-dir").unwrap();
    
    println!("📥 모델 다운로드 시작: {}", model_id);
    
    let downloader = ModelDownloader::new(model_id);
    let model_path = downloader.download().await?;
    
    println!("✅ 다운로드 완료: {:?}", model_path);
    Ok(())
}

async fn handle_compress(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = PathBuf::from(matches.get_one::<String>("model-path").unwrap());
    let output_path = PathBuf::from(matches.get_one::<String>("output-path").unwrap());
    let compression_level = matches.get_one::<String>("compression-level").unwrap().parse()?;
    let block_size = matches.get_one::<String>("block-size").unwrap().parse()?;
    let coefficients = matches.get_one::<String>("coefficients").unwrap().parse()?;
    
    println!("🗜️ 모델 압축 시작:");
    println!("   입력: {:?}", model_path);
    println!("   출력: {:?}", output_path);
    println!("   압축 레벨: {}", compression_level);
    println!("   블록 크기: {}", block_size);
    println!("   계수 개수: {}", coefficients);
    
    let config = CompressionConfig {
        wavelet_coefficients: coefficients,
        block_size,
        compression_level,
        num_threads: num_cpus::get(),
        show_progress: true,
    };
    
    let compressor = SLLMCompressor::new(config);
    let compressed_model = compressor.compress_safetensors_model(&model_path, &output_path).await?;
    
    println!("\n🏆 압축 완료!");
    println!("   압축률: {:.1}:1", compressed_model.total_compression_ratio);
    println!("   평균 RMSE: {:.6}", compressed_model.average_rmse);
    println!("   압축 시간: {:.2}초", compressed_model.compression_time);
    
    Ok(())
}

async fn handle_generate(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let compressed_model = PathBuf::from(matches.get_one::<String>("compressed-model").unwrap());
    let original_model = PathBuf::from(matches.get_one::<String>("original-model").unwrap());
    let prompt = matches.get_one::<String>("prompt").unwrap();
    let max_tokens = matches.get_one::<String>("max-tokens").unwrap().parse()?;
    let temperature = matches.get_one::<String>("temperature").unwrap().parse()?;
    let top_p = matches.get_one::<String>("top-p").unwrap().parse()?;
    
    println!("💭 텍스트 생성 시작:");
    println!("   프롬프트: '{}'", prompt);
    println!("   최대 토큰: {}", max_tokens);
    println!("   Temperature: {}", temperature);
    println!("   Top-p: {}", top_p);
    
    let engine = RBEInferenceEngine::from_compressed_model(&compressed_model, &original_model).await?;
    engine.print_model_info();
    
    let generated_text = engine.generate_text(prompt, max_tokens, temperature, top_p)?;
    
    println!("\n📝 생성된 텍스트:");
    println!("─────────────────────────────────");
    println!("{}", generated_text);
    println!("─────────────────────────────────");
    
    Ok(())
}

async fn handle_benchmark(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let iterations = matches.get_one::<String>("iterations").unwrap().parse()?;
    let save_path = matches.get_one::<String>("save-results");
    
    println!("📊 벤치마크 시작 ({}회 반복)", iterations);
    
    let config = BenchmarkConfig {
        iterations,
        warmup_iterations: 3,
        batch_size: 32,
        verbose: true,
    };
    
    let mut benchmark = SLLMBenchmark::new(config);
    
    // RBE 압축 벤치마크
    let matrix_sizes = vec![
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ];
    benchmark.benchmark_rbe_compression(&matrix_sizes);
    
    // 토큰 처리 벤치마크
    let text_lengths = vec![100, 500, 1000, 5000];
    benchmark.benchmark_token_processing(&text_lengths);
    
    // 추론 속도 벤치마크
    let context_lengths = vec![128, 256, 512, 1024];
    benchmark.benchmark_inference_speed(&context_lengths);
    
    // 요약 출력
    benchmark.print_summary();
    
    // 결과 저장
    if let Some(save_path) = save_path {
        benchmark.save_results(save_path)?;
    }
    
    Ok(())
}

async fn handle_info(matches: &clap::ArgMatches) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = matches.get_one::<String>("model-path").unwrap();
    
    println!("📋 모델 정보 조회: {}", model_path);
    
    let content = std::fs::read_to_string(model_path)?;
    let compressed_model: rbe_llm::sllm::CompressedModel = serde_json::from_str(&content)?;
    
    println!("\n🏆 === 압축된 모델 정보 ===");
    println!("모델명: {}", compressed_model.model_name);
    println!("원본 크기: {:.2} MB", compressed_model.original_total_size as f64 / 1_048_576.0);
    println!("압축 크기: {:.2} KB", compressed_model.compressed_total_size as f64 / 1024.0);
    println!("압축률: {:.1}:1", compressed_model.total_compression_ratio);
    println!("평균 RMSE: {:.6}", compressed_model.average_rmse);
    println!("압축 시간: {:.2}초", compressed_model.compression_time);
    
    // 품질 등급
    let quality = if compressed_model.average_rmse < 0.001 { "🥇 S급" }
    else if compressed_model.average_rmse < 0.01 { "🥈 A급" }
    else if compressed_model.average_rmse < 0.05 { "🥉 B급" }
    else { "C급" };
    
    println!("압축 품질: {}", quality);
    
    println!("\n🗜️ 압축된 레이어:");
    for (name, layer) in &compressed_model.layers {
        println!("  {}: {}×{} (RMSE: {:.6})", 
                 name, layer.shape[0], layer.shape[1], layer.rmse);
    }
    
    Ok(())
} 
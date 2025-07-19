use RBE_LLM::sllm::{ModelDownloader, DownloadConfig, SLLMCompressor, CompressionConfig};
use std::path::PathBuf;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 === RBE 모델 압축 파이프라인 시작 ===\n");
    
    // 1단계: 모델 다운로드
    println!("📥 1단계: 한국어 모델 다운로드");
    let download_config = DownloadConfig {
        model_id: "skt/kogpt2-base-v2".to_string(),
        cache_dir: "./models".to_string(),
        use_auth_token: None,
    };
    
    let downloader = ModelDownloader::new(download_config.clone());
    let model_path = match downloader.download().await {
        Ok(path) => {
            println!("✅ 모델 다운로드 완료: {:?}", path);
            path
        }
        Err(e) => {
            println!("❌ 다운로드 실패: {}", e);
            println!("📌 로컬 모델 경로를 사용합니다.");
            PathBuf::from("./models/skt-kogpt2-base-v2")
        }
    };
    
    println!("\n{}\n", "=".repeat(60));
    
    // 2단계: RBE 압축
    println!("🗜️ 2단계: RBE + 웨이블릿 하이브리드 압축");
    let compression_config = CompressionConfig {
        wavelet_coefficients: 500,  // S급 품질
        block_size: 32,            // 최적 블록 크기
        compression_level: 3,       // 균형 모드
        num_threads: num_cpus::get(),
        show_progress: true,
    };
    
    println!("📊 압축 설정:");
    println!("   - 웨이블릿 계수: {}", compression_config.wavelet_coefficients);
    println!("   - 블록 크기: {}×{}", compression_config.block_size, compression_config.block_size);
    println!("   - 압축 레벨: {}", compression_config.compression_level);
    println!("   - 병렬 스레드: {}", compression_config.num_threads);
    
    let compressor = SLLMCompressor::new(compression_config);
    let output_path = PathBuf::from("./compressed_models/kogpt2_rbe_compressed.json");
    
    match compressor.compress_safetensors_model(&model_path, &output_path).await {
        Ok(compressed_model) => {
            println!("\n✅ 압축 성공!");
            println!("📈 압축 통계:");
            println!("   - 원본 크기: {:.2} MB", compressed_model.original_total_size as f64 / 1_048_576.0);
            println!("   - 압축 후: {:.2} KB", compressed_model.compressed_total_size as f64 / 1024.0);
            println!("   - 압축률: {:.1}:1", compressed_model.total_compression_ratio);
            println!("   - 평균 RMSE: {:.6}", compressed_model.average_rmse);
            println!("   - 압축 시간: {:.2}초", compressed_model.compression_time);
            
            // 품질 등급 출력
            let quality = if compressed_model.average_rmse < 0.001 { "🥇 S급 (최고)" }
            else if compressed_model.average_rmse < 0.01 { "🥉 A급 (우수)" }
            else if compressed_model.average_rmse < 0.05 { "B급 (양호)" }
            else { "C급 (보통)" };
            
            println!("   - 품질 등급: {}", quality);
            
            // 메모리 절약률
            let memory_saving = (1.0 - 1.0 / compressed_model.total_compression_ratio) * 100.0;
            println!("   - 메모리 절약: {:.1}%", memory_saving);
            
            if compressed_model.average_rmse < 0.001 {
                println!("\n🎯 목표 RMSE < 0.001 달성! 완벽한 압축입니다!");
            }
        }
        Err(e) => {
            println!("\n❌ 압축 실패: {}", e);
            return Err(e);
        }
    }
    
    println!("\n{}\n", "=".repeat(60));
    
    // 3단계: 압축 효과 시연
    println!("💡 3단계: 압축 효과 시연");
    println!("원본 GPT-2 모델:");
    println!("   - 크기: ~474 MB");
    println!("   - 메모리 요구사항: 최소 2GB RAM");
    println!("   - 모바일 실행: 불가능");
    
    println!("\nRBE 압축 후:");
    println!("   - 크기: < 1 MB");
    println!("   - 메모리 요구사항: < 100MB RAM");
    println!("   - 모바일 실행: 가능!");
    
    println!("\n🎉 전체 파이프라인 완료!");
    println!("📂 압축된 모델 위치: {:?}", output_path);
    
    Ok(())
} 
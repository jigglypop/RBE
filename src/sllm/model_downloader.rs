use std::path::{Path, PathBuf};
use anyhow::{Result};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest;
use serde_json::Value;

/// 한국어 SLLM 모델 다운로더
pub struct ModelDownloader {
    /// 다운로드할 모델 ID
    pub model_id: String,
    /// 로컬 저장 경로
    pub cache_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            cache_dir: PathBuf::from("./models"),
        }
    }
    
    pub async fn download(&self) -> Result<PathBuf> {
        println!("HuggingFace에서 모델 다운로드 시작");
        println!("모델: {}", self.model_id);
        let model_path = self.cache_dir.join(self.model_id.replace('/', "-"));
        if model_path.exists() && self.is_valid_model(&model_path).await {
            println!("모델이 이미 다운로드되어 있습니다: {:?}", model_path);
            return Ok(model_path);
        }
        // 디렉토리 생성
        tokio::fs::create_dir_all(&model_path).await?;
        
        // SKT KoGPT2에 실제로 존재하는 파일들만
        let files = vec![
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
        ];
        
        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("██░"),
        );
        
        for file_name in &files {
            pb.set_message(format!("다운로드 중: {}", file_name));
            let file_path = model_path.join(file_name);
            
            // 파일이 이미 존재하면 스킵
            if file_path.exists() {
                if let Ok(metadata) = tokio::fs::metadata(&file_path).await {
                    // pytorch_model.bin은 큰 파일이므로 크기 체크
                    if file_name == &"pytorch_model.bin" {
                        if metadata.len() > 400_000_000 {  // 400MB 이상
                            pb.inc(1);
                            continue;
                        }
                    } else if metadata.len() > 100 {  // 다른 파일들은 100 bytes 이상
                        pb.inc(1);
                        continue;
                    }
                }
            }
            
            // 실제 다운로드 URL
            let download_url = format!("https://huggingface.co/{}/resolve/main/{}", self.model_id, file_name);
            
            // 실제 다운로드
            match self.download_file(&download_url, &file_path).await {
                Ok(_) => {
                    println!("✓ {} 다운로드 완료", file_name);
                }
                Err(e) => {
                    eprintln!("✗ {} 다운로드 실패: {}", file_name, e);
                    // 모든 파일이 필수
                    return Err(e);
                }
            }
            pb.inc(1);
        }
        pb.finish_with_message("모든 파일 다운로드 완료!");
        println!("모델 다운로드 완료: {:?}", model_path);
        Ok(model_path)
    }
    
    /// 실제 파일 다운로드
    async fn download_file(&self, url: &str, path: &Path) -> Result<()> {
        // 실제 다운로드 URL 출력
        println!("다운로드 시도 중: {}", url);
        
        let client = reqwest::Client::builder()
            .user_agent("RBE-LLM/0.1")
            .timeout(std::time::Duration::from_secs(300))
            .build()?;
            
        let response = client.get(url).send().await?;
        
        if !response.status().is_success() {
            eprintln!("다운로드 실패: {} - URL: {}", response.status(), url);
            return Err(anyhow::anyhow!("Download failed: {} for URL: {}", response.status(), url));
        }
        
        let bytes = response.bytes().await?;
        println!("다운로드 완료: {} bytes", bytes.len());
        tokio::fs::write(path, bytes).await?;
        
        Ok(())
    }
    
    /// 모델 유효성 검증
    async fn is_valid_model(&self, model_path: &Path) -> bool {
        // SKT KoGPT2에 필요한 파일들
        let required_files = vec![
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
        ];
        
        // 모든 필수 파일이 존재하는지 확인
        for file in required_files {
            let file_path = model_path.join(file);
            if !file_path.exists() {
                println!("필수 파일 누락: {}", file);
                return false;
            }
            
            // 파일 크기 확인
            if let Ok(metadata) = tokio::fs::metadata(&file_path).await {
                if file == "pytorch_model.bin" {
                    if metadata.len() < 400_000_000 {  // 400MB 미만이면 불완전
                        println!("pytorch_model.bin 파일이 불완전합니다.");
                        return false;
                    }
                } else if metadata.len() < 100 {  // 다른 파일들은 최소 100 bytes
                    println!("{} 파일이 너무 작습니다.", file);
                    return false;
                }
            } else {
            return false;
            }
        }
        
        // config.json이 유효한 JSON인지 확인
        let config_path = model_path.join("config.json");
        if let Ok(content) = tokio::fs::read_to_string(&config_path).await {
            if serde_json::from_str::<Value>(&content).is_err() {
                println!("config.json이 유효하지 않습니다.");
                return false;
            }
        } else {
            return false;
        }
        
        true
    }
} 
use crate::sllm::{RBEInferenceEngine, SLLMCompressor};
use std::sync::Arc;
use std::path::PathBuf;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use warp::{Filter, Reply, Rejection};
use anyhow::Result;
// use std::collections::HashMap;

/// API 서버 구조체
pub struct RBEApiServer {
    /// 추론 엔진 (스레드 안전)
    engine: Arc<RwLock<Option<RBEInferenceEngine>>>,
    /// 서버 설정
    config: ServerConfig,
    /// 로드된 모델 정보
    model_info: Arc<RwLock<Option<ModelInfo>>>,
}

/// 서버 설정
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_tokens: usize,
    pub default_temperature: f32,
    pub default_top_p: f32,
    pub enable_cors: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            max_tokens: 100,
            default_temperature: 0.7,
            default_top_p: 0.9,
            enable_cors: true,
        }
    }
}

/// 모델 정보
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub compression_ratio: f32,
    pub average_rmse: f32,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub loaded_at: String,
}

/// 텍스트 생성 요청
#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
}

/// 텍스트 생성 응답
#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub generation_time_ms: u64,
    pub tokens_per_second: f32,
}

/// 모델 압축 요청
#[derive(Debug, Deserialize)]
pub struct CompressRequest {
    pub model_path: String,
    pub output_path: String,
    #[serde(default)]
    pub compression_level: Option<u8>,
    #[serde(default)]
    pub block_size: Option<usize>,
}

/// 모델 압축 응답
#[derive(Debug, Serialize)]
pub struct CompressResponse {
    pub success: bool,
    pub message: String,
    pub compression_ratio: Option<f32>,
    pub compression_time_seconds: Option<f64>,
}

/// 모델 로딩 요청
#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    pub compressed_model_path: String,
    pub original_model_path: String,
}

/// 벤치마크 요청
#[derive(Debug, Deserialize)]
pub struct BenchmarkRequest {
    #[serde(default)]
    pub matrix_sizes: Option<Vec<(usize, usize)>>,
    #[serde(default)]
    pub iterations: Option<usize>,
}

/// API 에러 응답
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: u16,
}

impl RBEApiServer {
    /// 새로운 API 서버 생성
    pub fn new(config: ServerConfig) -> Self {
        Self {
            engine: Arc::new(RwLock::new(None)),
            config,
            model_info: Arc::new(RwLock::new(None)),
        }
    }
    
    /// 서버 시작
    pub async fn start(self) -> Result<()> {
        let server = Arc::new(self);
        
        println!("🚀 RBE API 서버 시작 중...");
        println!("   주소: http://{}:{}", server.config.host, server.config.port);
        println!("   최대 토큰: {}", server.config.max_tokens);
        
        // 라우트 설정
        let routes = Self::setup_routes(server.clone());
        
        // CORS 설정 (일단 기본 활성화)
        let routes = routes.with(warp::cors()
            .allow_any_origin()
            .allow_headers(vec!["content-type"])
            .allow_methods(vec!["GET", "POST", "PUT", "DELETE"]));
        
        // 서버 실행
        warp::serve(routes)
            .run((
                server.config.host.parse::<std::net::IpAddr>()?,
                server.config.port
            ))
            .await;
        
        Ok(())
    }
    
    /// 라우트 설정
    fn setup_routes(
        server: Arc<RBEApiServer>,
    ) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone {
        // 헬스체크
        let health = warp::path("health")
            .and(warp::get())
            .map(|| {
                warp::reply::json(&serde_json::json!({
                    "status": "healthy",
                    "service": "RBE API Server"
                }))
            });
        
        // 모델 정보 조회
        let model_info = warp::path("model")
            .and(warp::path("info"))
            .and(warp::get())
            .and(with_server(server.clone()))
            .and_then(Self::get_model_info);
        
        // 모델 로딩
        let load_model = warp::path("model")
            .and(warp::path("load"))
            .and(warp::post())
            .and(warp::body::json())
            .and(with_server(server.clone()))
            .and_then(Self::load_model);
        
        // 모델 압축
        let compress_model = warp::path("model")
            .and(warp::path("compress"))
            .and(warp::post())
            .and(warp::body::json())
            .and(with_server(server.clone()))
            .and_then(Self::compress_model);
        
        // 텍스트 생성
        let generate = warp::path("generate")
            .and(warp::post())
            .and(warp::body::json())
            .and(with_server(server.clone()))
            .and_then(Self::generate_text);
        
        // 벤치마크
        let benchmark = warp::path("benchmark")
            .and(warp::post())
            .and(warp::body::json())
            .and(with_server(server.clone()))
            .and_then(Self::run_benchmark);
        
        // 정적 파일 서빙 (API 문서 등)
        let static_files = warp::path("docs")
            .and(warp::fs::dir("docs/api/"));
        
        health
            .or(model_info)
            .or(load_model)
            .or(compress_model)
            .or(generate)
            .or(benchmark)
            .or(static_files)
    }
    
    /// 모델 정보 조회 핸들러
    async fn get_model_info(server: Arc<RBEApiServer>) -> Result<impl Reply, Rejection> {
        let model_info = server.model_info.read().await;
        
        match model_info.as_ref() {
            Some(info) => Ok(warp::reply::json(info)),
            None => Ok(warp::reply::json(&ErrorResponse {
                error: "모델이 로드되지 않았습니다".to_string(),
                code: 404,
            })),
        }
    }
    
    /// 모델 로딩 핸들러
    async fn load_model(
        request: LoadModelRequest,
        server: Arc<RBEApiServer>,
    ) -> Result<impl Reply, Rejection> {
        println!("📥 모델 로딩 요청: {:?}", request);
        
        let compressed_path = PathBuf::from(&request.compressed_model_path);
        let original_path = PathBuf::from(&request.original_model_path);
        
        match RBEInferenceEngine::from_compressed_model(&compressed_path, &original_path).await {
            Ok(engine) => {
                // 모델 정보 생성
                let info = ModelInfo {
                    name: engine.get_model_name().to_string(),
                    compression_ratio: engine.get_compression_ratio(),
                    average_rmse: engine.get_average_rmse(),
                    vocab_size: engine.get_vocab_size(),
                    hidden_size: engine.get_hidden_size(),
                    num_layers: engine.get_num_layers(),
                    loaded_at: chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                };
                
                // 엔진과 정보 저장
                *server.engine.write().await = Some(engine);
                *server.model_info.write().await = Some(info.clone());
                
                println!("✅ 모델 로딩 완료: {}", info.name);
                
                Ok(warp::reply::json(&serde_json::json!({
                    "success": true,
                    "message": "모델이 성공적으로 로드되었습니다",
                    "model_info": info
                })))
            }
            Err(e) => {
                println!("❌ 모델 로딩 실패: {}", e);
                Ok(warp::reply::json(&ErrorResponse {
                    error: format!("모델 로딩 실패: {}", e),
                    code: 500,
                }))
            }
        }
    }
    
    /// 모델 압축 핸들러
    async fn compress_model(
        request: CompressRequest,
        _server: Arc<RBEApiServer>,
    ) -> Result<impl Reply, Rejection> {
        println!("🗜️ 모델 압축 요청: {:?}", request);
        
        let model_path = PathBuf::from(&request.model_path);
        let output_path = PathBuf::from(&request.output_path);
        
        // 압축 설정
        let mut config = crate::sllm::CompressionConfig::default();
        if let Some(level) = request.compression_level {
            config.compression_level = level;
        }
        if let Some(block_size) = request.block_size {
            config.block_size = block_size;
        }
        
        let compressor = SLLMCompressor::new(config);
        
        match compressor.compress_safetensors_model(&model_path, &output_path).await {
            Ok(compressed_model) => {
                println!("✅ 모델 압축 완료");
                
                Ok(warp::reply::json(&CompressResponse {
                    success: true,
                    message: "모델이 성공적으로 압축되었습니다".to_string(),
                    compression_ratio: Some(compressed_model.total_compression_ratio),
                    compression_time_seconds: Some(compressed_model.compression_time),
                }))
            }
            Err(e) => {
                println!("❌ 모델 압축 실패: {}", e);
                Ok(warp::reply::json(&CompressResponse {
                    success: false,
                    message: format!("모델 압축 실패: {}", e),
                    compression_ratio: None,
                    compression_time_seconds: None,
                }))
            }
        }
    }
    
    /// 텍스트 생성 핸들러
    async fn generate_text(
        request: GenerateRequest,
        server: Arc<RBEApiServer>,
    ) -> Result<impl Reply, Rejection> {
        println!("💭 텍스트 생성 요청: '{}'", request.prompt);
        
        let engine_guard = server.engine.read().await;
        let engine = match engine_guard.as_ref() {
            Some(e) => e,
            None => {
                return Ok(warp::reply::json(&ErrorResponse {
                    error: "모델이 로드되지 않았습니다".to_string(),
                    code: 404,
                }))
            }
        };
        
        // 생성 파라미터 설정
        let max_tokens = request.max_tokens.unwrap_or(server.config.max_tokens);
        let temperature = request.temperature.unwrap_or(server.config.default_temperature);
        let top_p = request.top_p.unwrap_or(server.config.default_top_p);
        
        let start_time = std::time::Instant::now();
        
        match engine.generate_text(&request.prompt, max_tokens, temperature, top_p) {
            Ok(generated_text) => {
                let generation_time = start_time.elapsed();
                let tokens_generated = generated_text.split_whitespace().count(); // 간단한 토큰 카운트
                let tokens_per_second = tokens_generated as f32 / generation_time.as_secs_f32();
                
                println!("✅ 텍스트 생성 완료 ({:.2}초)", generation_time.as_secs_f32());
                
                Ok(warp::reply::json(&GenerateResponse {
                    text: generated_text,
                    tokens_generated,
                    generation_time_ms: generation_time.as_millis() as u64,
                    tokens_per_second,
                }))
            }
            Err(e) => {
                println!("❌ 텍스트 생성 실패: {}", e);
                Ok(warp::reply::json(&ErrorResponse {
                    error: format!("텍스트 생성 실패: {}", e),
                    code: 500,
                }))
            }
        }
    }
    
    /// 벤치마크 실행 핸들러
    async fn run_benchmark(
        request: BenchmarkRequest,
        _server: Arc<RBEApiServer>,
    ) -> Result<impl Reply, Rejection> {
        println!("📊 벤치마크 실행 요청");
        
        // 벤치마크 설정
        let mut benchmark_config = crate::sllm::BenchmarkConfig::default();
        if let Some(iterations) = request.iterations {
            benchmark_config.iterations = iterations;
        }
        
        let mut benchmark = crate::sllm::SLLMBenchmark::new(benchmark_config);
        
        // 행렬 크기 설정
        let matrix_sizes = request.matrix_sizes.unwrap_or_else(|| vec![
            (256, 256),
            (512, 512),
            (1024, 1024),
        ]);
        
        // RBE 압축 벤치마크 실행
        benchmark.benchmark_rbe_compression(&matrix_sizes);
        
        // 토큰 처리 벤치마크 실행
        let text_lengths = vec![100, 500, 1000, 5000];
        benchmark.benchmark_token_processing(&text_lengths);
        
        // 추론 속도 벤치마크 실행
        let context_lengths = vec![128, 256, 512, 1024];
        benchmark.benchmark_inference_speed(&context_lengths);
        
        benchmark.print_summary();
        
        println!("✅ 벤치마크 완료");
        
        Ok(warp::reply::json(&serde_json::json!({
            "success": true,
            "message": "벤치마크가 성공적으로 완료되었습니다"
        })))
    }
}

/// 서버 의존성 주입을 위한 헬퍼
fn with_server(
    server: Arc<RBEApiServer>,
) -> impl Filter<Extract = (Arc<RBEApiServer>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || server.clone())
} 
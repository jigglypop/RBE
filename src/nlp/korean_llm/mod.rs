//! 한국어 LLM 모듈

use std::path::PathBuf;
use serde::{Serialize, Deserialize};

pub mod analyzer;
pub mod generator;
pub mod model_loader;
pub mod tokenizer;

pub use analyzer::*;
pub use generator::*;
pub use model_loader::*;
pub use tokenizer::*;

/// 한국어 LLM 설정
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KoreanLLMConfig {
    /// 모델 ID (HuggingFace)
    pub model_id: String,
    /// 캐시 디렉토리
    pub cache_dir: PathBuf,
    /// RBE 압축 활성화
    pub enable_compression: bool,
    /// RBE 최적화 사용
    pub use_rbe_optimization: bool,
    /// 생성 온도
    pub temperature: f32,
    /// 최대 생성 길이
    pub max_length: usize,
    /// Top-k 샘플링
    pub top_k: usize,
    /// Top-p 샘플링
    pub top_p: f32,
}

impl Default for KoreanLLMConfig {
    fn default() -> Self {
        Self {
            model_id: "EleutherAI/polyglot-ko-1.3b".to_string(),
            cache_dir: PathBuf::from("models/korean_cache"),
            enable_compression: true,
            use_rbe_optimization: true,
            temperature: 0.8,
            max_length: 128,
            top_k: 50,
            top_p: 0.95,
        }
    }
} 
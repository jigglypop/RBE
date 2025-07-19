/// 🇰🇷 한국어 Small Language Model 처리 모듈
/// 
/// HuggingFace에서 한국어 모델을 다운로드하고 RBE로 압축하여
/// 실제 한글 텍스트 생성까지 수행하는 완전한 파이프라인

pub mod model_downloader;
pub mod rbe_compression;
// pub mod inference_engine; // tokenizers 의존성 제거됨
pub mod benchmark;
pub mod korean_llm;
pub mod korean_generator; // 새로운 한국어 생성기

// 모듈 재내보내기
pub use model_downloader::*;
pub use rbe_compression::*;
// pub use inference_engine::*; // tokenizers 의존성 제거됨
pub use benchmark::*;
pub use korean_llm::*;
pub use korean_generator::*; 
// 테스트 모듈 정의
pub mod types_test;
pub mod dct_only_test; // 🆕 DCT Only 순수 성능 테스트
pub mod simple_korean_test; // 🆕 간단한 한글 처리 테스트
pub mod hf_connection_test; // 🆕 HuggingFace 연결 테스트 (시뮬레이션)
pub mod korean_basic_test; // 🆕 한국어 기본 처리 테스트
pub mod korean_llm_demo_test; // 🆕 한국어 LLM 데모 테스트
// pub mod sllm_integration_test; // 🆕 SLLM 전체 파이프라인 테스트 (임시 비활성화)
// pub mod llm_korean_test; // 🆕 LLM 한글 응답 테스트 (임시 비활성화)
pub mod math_test;
pub mod matrix_test;
pub mod hybrid_learning_test;
pub mod encoder_test;
pub mod decoder_test; 
pub mod generator_test;
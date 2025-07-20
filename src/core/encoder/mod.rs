pub mod grid_compressor;
pub mod analysis_results;
pub mod encoder;
pub mod weight_mapper;

// 테스트 모듈
#[cfg(test)]
pub mod __tests__;

// 재수출
pub use encoder::{RBEEncoder, AutoOptimizedEncoder, QualityGrade, CompressionConfig, CompressionProfile};
// HybridEncoder는 RBEEncoder의 별칭으로 호환성 유지
pub use encoder::RBEEncoder as HybridEncoder;
pub use grid_compressor::GridCompressedMatrix;
pub use analysis_results::{FrequencyAnalysisResult, FrequencyType, ContinuousOptimizationResult, ResidualCompressionResult};
pub use weight_mapper::{WeightInfo, ModelLayout, WeightMapper}; 

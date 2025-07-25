//! 실제 모델 가중치 변환 - f32 ↔ Packed128

pub mod loader;
pub mod compress;
pub mod restore;

pub use loader::*;
pub use compress::*; 
pub use restore::*;

/// 변환 통계
#[derive(Debug)]
pub struct TransformStats {
    pub original_size_mb: f64,
    pub compressed_size_mb: f64,
    pub compression_ratio: f64,
    pub rmse: f64,
    pub transform_ms: f64,
    pub restore_ms: f64,
} 
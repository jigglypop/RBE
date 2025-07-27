//! 실제 모델 가중치 변환 - f32 ↔ Packed128

use serde::{Serialize, Deserialize};

pub mod compress;
pub use compress::*;

pub mod loader;
pub use loader::*;

pub mod restore;
pub use restore::*;

/// 변환 통계
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TransformStats {
    pub original_size_mb: f64,
    pub compressed_size_mb: f64,
    pub compression_ratio: f64,
    pub rmse: f64,
    pub transform_ms: f64,
    pub restore_ms: f64,
} 
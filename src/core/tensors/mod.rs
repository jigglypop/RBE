//! 텐서 모듈 - 비트 도메인 푸앵카레볼 구현

pub mod hyperbolic_lut;
pub mod packed_types;
pub mod enhanced_types;  // 새로운 Enhanced128 모듈 추가

// Re-exports
pub use packed_types::{Packed128, Packed64, DecodedParams, BitTensor, BitGradientTracker, AnalyticalGradient};
pub use enhanced_types::{Enhanced128, EnhancedParams, FixedPoint};  // Enhanced128 재-export
pub use hyperbolic_lut::HYPERBOLIC_LUT_DATA;

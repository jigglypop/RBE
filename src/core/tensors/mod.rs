//! 텐서 모듈 - 비트 도메인 푸앵카레볼 구현

pub mod packed_types;
pub mod hyperbolic_lut;
pub mod enhanced_types;
pub mod analytic_grad;

// Re-export core types
pub use packed_types::{Packed128, DecodedParams, AnalyticalGradient, BitTensor, BitGradientTracker, Packed64};
pub use enhanced_types::{Enhanced128, EnhancedParams, FixedPoint};
pub use analytic_grad::{AnalyticGradient, FixedPointMath, get_analytic_gradient};
pub use hyperbolic_lut::HYPERBOLIC_LUT_DATA;

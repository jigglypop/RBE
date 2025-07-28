//! RBE 코어 모듈 - 비트 도메인 푸앵카레볼 구현

pub mod tensors;
pub mod differential;
pub mod optimizers;
pub mod transform;

// Re-exports from tensors
pub use tensors::{
    Packed128, Packed64, DecodedParams, BitTensor, BitGradientTracker, AnalyticalGradient,
    Enhanced128, EnhancedParams, FixedPoint,  // Enhanced128 관련 타입들 추가
    HYPERBOLIC_LUT_DATA
};

// Re-exports from differential
pub use differential::{
    BitForwardPass, BitBackwardPass, DifferentialSystem,
    DifferentialMetrics, ForwardConfig, BackwardConfig,  // 올바른 별칭 사용
};

// Re-exports from optimizers
pub use optimizers::{
    BitAdamState, BitRiemannianAdamState, OptimizerType,
};

// Re-exports from transform
pub use transform::{
    TransformStats, WeightCompressor, WeightDecompressor,
};

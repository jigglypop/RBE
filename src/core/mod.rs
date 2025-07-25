//! # RBE 핵심 라이브러리 모듈
//!
//! 푸앵카레 볼 기반 리만 기저 인코딩의 핵심 구성 요소들

pub mod tensors;
pub mod differential;
pub mod optimizers;

pub use tensors::{
    packed_types::{Packed128, Packed64, CycleState, DecodedParams, BitGradientTracker},
    hyperbolic_lut::HYPERBOLIC_LUT_DATA,
};

pub use differential::{
    BitForwardPass as UnifiedForwardPass, 
    BitBackwardPass as UnifiedBackwardPass,
    DifferentialSystem, DifferentialMetrics,
    OptimizerType,
};

pub use optimizers::{
    BitAdamState, BitRiemannianAdamState,
    OptimizerConfig, OptimizerType as OptType,
};

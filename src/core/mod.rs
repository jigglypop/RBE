//! # RBE 핵심 라이브러리 모듈
//!
//! 푸앵카레 볼 기반 리만 기저 인코딩의 핵심 구성 요소들

pub mod tensors;
pub mod differential;
pub mod optimizers;

// 주요 타입들 재수출
pub use tensors::{
    packed_types::{Packed128, Packed64, CycleState, DecodedParams, BitGradientTracker},
    hyperbolic_lut::HYPERBOLIC_LUT_DATA,
};

pub use differential::{
    cycle_system::UnifiedCycleDifferentialSystem,
    state_transition::StateTransitionEngine,
};

pub use optimizers::{
    BitAdamState, BitRiemannianAdamState,
    OptimizerConfig, OptimizerType,
};

//! # RBE 핵심 라이브러리 모듈
//!
//! 푸앵카레 볼 기반 리만 기저 인코딩의 핵심 구성 요소들

pub mod optimizers;
pub mod differential; // 새로운 통합 미분 시스템
pub mod tensors;

// packed_params is temporarily redirected to tensors for migration
// TODO: Remove this after full migration
pub mod packed_params {
    pub use super::tensors::*;
}

// 주요 타입들 재수출 - 중복 방지를 위해 선택적으로 수출

pub use differential::{
    DifferentialSystem, UnifiedCycleDifferentialSystem, CycleState,
    UnifiedForwardPass, ForwardConfig, ForwardMetrics,
    UnifiedBackwardPass, BackwardConfig, GradientMetrics,
    StateTransitionEngine, TransitionRule, StateTransitionMetrics,
    DifferentialPhase, HyperbolicFunction,
};
pub use optimizers::{
    AdamState, RiemannianAdamState,
    OptimizerConfig, AdamConfig, RiemannianAdamConfig, OptimizerType,
};

// 각 모듈이 자체 테스트를 포함함

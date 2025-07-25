//! RBE-LLM: 리만 기저 인코딩 기반 언어 모델 라이브러리
//!
//! 푸앵카레 볼 기하학과 CORDIC 알고리즘을 결합한 압축 시스템

pub mod core;
// pub mod nlp;

pub use core::{
    // 비트 도메인 텐서 타입들
    Packed128, CycleState, DecodedParams, BitTensor, BitGradientTracker,
    // 비트 도메인 미분 시스템
    // BitForwardPass, BitBackwardPass, DifferentialSystem,
    // 최적화기
    BitAdamState, BitRiemannianAdamState, OptimizerType,
    // 변환 시스템
    // TransformStats, ModelLoader, WeightCompressor, WeightDecompressor,
};

// nlp 모듈 re-export
// pub use nlp::*;

// 편의 타입 별칭
pub type Packed = Packed128;
pub type BitOptimizer = BitAdamState;
pub type RiemannianOptimizer = BitRiemannianAdamState;
// pub type ForwardEngine = BitForwardPass;
// pub type BackwardEngine = BitBackwardPass;
//  
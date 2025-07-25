//! RBE (Riemannian Basis Encoding) 라이브러리
//!
//! 푸앵카레 볼 기반 리만 기저 인코딩으로 극한 압축과 성능을 달성하는 라이브러리

pub mod core;

pub use core::{
    // 텐서 및 데이터 구조
    Packed128, Packed64, CycleState, DecodedParams, BitGradientTracker,
    HYPERBOLIC_LUT_DATA,
    // 비트 도메인 미분 시스템
    UnifiedForwardPass, UnifiedBackwardPass, DifferentialSystem, DifferentialMetrics,
    // 최적화기
    BitAdamState, BitRiemannianAdamState, OptimizerConfig,
};

// 편의 타입 별칭들
pub type Packed = Packed128;
pub type BitOptimizer = BitAdamState;
pub type RiemannianOptimizer = BitRiemannianAdamState;
pub type ForwardEngine = UnifiedForwardPass;
pub type BackwardEngine = UnifiedBackwardPass;
 
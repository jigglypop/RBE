//! RBE 코어 모듈 - 비트 도메인 푸앵카레볼 구현

pub mod tensors;
pub mod optimizers;
pub mod differential;
pub mod transform;

// 핵심 타입들 re-export
pub use tensors::{Packed128, CycleState, DecodedParams, BitTensor, BitGradientTracker};
pub use optimizers::{BitAdamState, BitRiemannianAdamState, OptimizerType};
pub use differential::{DifferentialSystem, BitForwardPass, BitBackwardPass};
pub use transform::{TransformStats, ModelLoader, WeightCompressor, WeightDecompressor};

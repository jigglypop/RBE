//! RBE Library - 리만 비트 인코딩

pub mod core;
pub mod nlp;

// Core exports
pub use core::{
    encoder::*,
    decoder::*,
    tensors::*,
    optimizers::*,
    differential::*,
    transform::{WeightCompressor, WeightDecompressor, TransformStats},
};

// NLP exports  
pub use nlp::*;

// 편의 타입 별칭
pub type Packed = Packed128;
pub type BitOptimizer = BitAdamState;
pub type RiemannianOptimizer = BitRiemannianAdamState;
// pub type ForwardEngine = BitForwardPass;
// pub type BackwardEngine = BitBackwardPass;
//  
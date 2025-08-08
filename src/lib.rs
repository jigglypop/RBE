//! RBE-LLM: 리만 기저 인코딩 기반 언어 모델 라이브러리
//!
//! 푸앵카레 볼 기하학과 CORDIC 알고리즘을 결합한 압축 시스템

pub mod core;
pub mod nlp;

pub use core::{
    tensors::{
        Packed128, DecodedParams, Packed256, Packed256Params, FixedPoint32,
        Enhanced128, EnhancedParams, FixedPoint, AnalyticGradient, FixedPointMath
    },
    optimizers::{BitAdamState, BitRiemannianAdamState},
    transform::{WeightCompressor, TransformStats},
};

// 기본 타입 별칭들
pub type CompressionSeed = Packed128;
pub type HighPrecisionSeed = Packed256; 
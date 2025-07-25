//! # RBE LLM 라이브러리
//!
//! RBE (Riemannian Base Encoding) 기반 LLM 구현체

pub mod core;
// pub mod nlp;  // 임시 비활성화 - tensors, differential, optimizers만 집중

// 주요 타입들 재수출 - 중복 재수출 문제 해결을 위해 선택적으로 수출
pub use core::{
    // tensors
    Packed128, CycleState, TransformType, ResidualCoefficient, 
    HybridEncodedBlock, EncodedBlockGradients, RbeParameters,
    BasisFunction, HyperbolicBasisFunction,
    PoincareMatrix, PoincarePackedBit128, PoincareQuadrant,
    
    // differential  
    DifferentialSystem, UnifiedCycleDifferentialSystem, 
    UnifiedForwardPass, ForwardConfig, ForwardMetrics,
    UnifiedBackwardPass, BackwardConfig, GradientMetrics,
    StateTransitionEngine, TransitionRule, StateTransitionMetrics,
    
    // optimizers
    AdamState, RiemannianAdamState, TransformAnalyzer,
    OptimizerConfig, AdamConfig, RiemannianAdamConfig, OptimizerType,
    
    // encoder/decoder
    RBEEncoder, CompressionConfig, CompressionProfile,
};

// pub use nlp::model_tools::*;  // 임시 비활성화
// pub use nlp::linear::*;  // 임시 비활성화
 
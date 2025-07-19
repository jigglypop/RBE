//! # RBE 핵심 라이브러리 모듈
//!
//! 푸앵카레 볼 기반 리만 기저 인코딩의 핵심 구성 요소들

pub mod packed_params;
pub mod math;
pub mod encoder;
pub mod decoder;
pub mod generator;
pub mod matrix;
pub mod systems;
pub mod optimizers;

pub use packed_params::{
    Packed64, Packed128, DecodedParams, PoincareMatrix, PoincarePackedBit128,
    PoincareQuadrant, HybridEncodedBlock, TransformType, RbeParameters, 
    ResidualCoefficient, EncodedBlockGradients
};

pub use math::*;
pub use encoder::{HybridEncoder, GridCompressedMatrix};
pub use decoder::*;
pub use generator::*;
pub use matrix::*;

// systems 모듈 - config 충돌 방지를 위해 명시적 import
pub use systems::{
    EncodedLayer, FusedEncodedLayer,
    HybridPoincareRBESystem, HybridPoincareLayer, 
    PoincareEncodingLayer, FusionProcessingLayer, HybridLearningLayer,
    SystemConfiguration, LearningParameters, AdaptiveLearningRateConfig,
    LossWeights, HardwareConfiguration,
    PerformanceMonitor, MemoryUsageTracker, ComputationTimeTracker,
    QualityMetricsTracker, EnergyEfficiencyTracker,
    LearningState, LossComponents, ConvergenceStatus,
    StateManager, ParameterManager,
    ResidualCompressor, CORDICEngine, BasisFunctionLUT,
    ParallelGEMMEngine, RiemannianGradientComputer, StateTransitionDifferentiator,
    AdaptiveScheduler, PerformanceAnalyzer, LayerMetrics,
    ActivationStatistics, WeightStatistics, GradientStatistics,
    InternalTransformType, LearningRateStrategy
};

// optimizers 모듈 - config 충돌 방지를 위해 명시적 import  
pub use optimizers::{
    AdamState, RiemannianAdamState, HybridOptimizer, OptimizationPhase, 
    PerformanceMetrics, TransformAnalyzer, OptimizerConfig, 
    AdamConfig, RiemannianAdamConfig, OptimizerType
};

//! # 레이어 시스템 모듈
//!
//! 푸앵카레 RBE 레이어 시스템의 모든 구성요소들

pub mod core_layer;
pub mod hybrid_system;
pub mod config;
pub mod performance;
pub mod state_management;
pub mod compute_engine;

// 핵심 레이어 재수출
pub use core_layer::{EncodedLayer, FusedEncodedLayer};

// 하이브리드 시스템 재수출
pub use hybrid_system::{
    HybridPoincareRBESystem, HybridPoincareLayer, 
    PoincareEncodingLayer, FusionProcessingLayer, HybridLearningLayer
};

// 설정 관련 재수출
pub use config::{
    SystemConfiguration, LearningParameters, AdaptiveLearningRateConfig,
    LossWeights, HardwareConfiguration
};

// 성능 모니터링 재수출  
pub use performance::{
    PerformanceMonitor, MemoryUsageTracker, ComputationTimeTracker,
    QualityMetricsTracker, EnergyEfficiencyTracker
};

// 상태 관리 재수출
pub use state_management::{
    LearningState, LossComponents, ConvergenceStatus,
    StateManager, ParameterManager
};

// 계산 엔진 재수출
pub use compute_engine::{
    ResidualCompressor, CORDICEngine, BasisFunctionLUT,
    ParallelGEMMEngine, RiemannianGradientComputer, StateTransitionDifferentiator,
    AdaptiveScheduler, PerformanceAnalyzer, LayerMetrics,
    ActivationStatistics, WeightStatistics, GradientStatistics,
    InternalTransformType, LearningRateStrategy
};

// 테스트 모듈들
#[cfg(test)]
pub mod __tests__;

#[cfg(test)]
mod tests {
    use super::*;
} 
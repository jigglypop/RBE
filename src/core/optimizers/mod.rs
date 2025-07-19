pub mod adam;
pub mod riemannian_adam;
pub mod hybrid;
pub mod analyzer;
pub mod config;

// 테스트 모듈은 __tests__/mod.rs에 통합되어 있음

// 주요 타입들 재수출
pub use adam::{AdamState};
pub use riemannian_adam::{RiemannianAdamState};
pub use hybrid::{HybridOptimizer, OptimizationPhase, PerformanceMetrics};
pub use analyzer::{TransformAnalyzer};
pub use config::{OptimizerConfig, AdamConfig, RiemannianAdamConfig};

/// 최적화 기법 종류
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizerType {
    Adam,
    RiemannianAdam,
    SGD,
    RMSprop,
} 
pub mod adam;
pub mod riemannian_adam;
pub mod analyzer;
pub mod config;
// 주요 타입들 재수출
pub use adam::{AdamState};
pub use riemannian_adam::{RiemannianAdamState};
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

#[cfg(test)]
pub mod __tests__;

#[cfg(test)]
mod tests {
    use super::*;
} 
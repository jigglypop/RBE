pub mod adam;
pub mod config;
pub mod riemannian_adam;
pub mod gradient_descent;
pub mod momentum;

pub use adam::BitAdamState;
pub use config::OptimizerConfig;
pub use riemannian_adam::BitRiemannianAdamState;
pub use gradient_descent::GradientDescent;
pub use momentum::MomentumOptimizer;

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerType {
    BitAdam,
    BitRiemannianAdam,
    SGD,
    RMSprop,
    Hybrid, // 상황에 따라 자동 선택
} 

#[cfg(test)]
mod tests {
    use super::*;
} 
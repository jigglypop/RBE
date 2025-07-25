pub mod adam;
pub mod riemannian_adam;
pub mod config;

pub use adam::BitAdamState;
pub use riemannian_adam::BitRiemannianAdamState;
pub use config::OptimizerConfig;

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerType {
    BitAdam,
    BitRiemannianAdam,
    SGD,
    RMSprop,
} 

#[cfg(test)]
mod tests {
    use super::*;
} 
//! RBE 코어 모듈 - 비트 도메인 푸앵카레볼 구현

pub mod tensors;
pub mod optimizers;
pub mod differential;
pub mod transform;

// Re-exports
pub use tensors::*;
pub use optimizers::*;
pub use differential::*;
pub use transform::*;

// Convenience type aliases
pub type Enhanced = tensors::Enhanced128;

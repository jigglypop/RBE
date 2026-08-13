pub mod basic_math;
pub mod basis_functions;
pub mod bessel;
pub mod busemann;
pub mod fused_ops;
pub mod gradient;
pub mod lut;
pub mod phase_state;
pub mod poincare;
pub mod state_transition;

// 검증 하네스 기반 (테스트 전용, docs/test/verification_harness.md)
#[cfg(test)]
pub mod verification;

// 테스트 모듈
#[cfg(test)]
mod __tests__;

// 재수출
pub use basic_math::*;
pub use basis_functions::*;
pub use bessel::*;
pub use fused_ops::*;
pub use gradient::*;
pub use poincare::*;
pub use state_transition::StateTransitionGraph;

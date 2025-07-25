//! 텐서 모듈 - 비트 도메인 푸앵카레볼 구현

pub mod packed_types;
pub mod hyperbolic_lut;
pub mod transform_types;
pub mod basis_types;
pub mod poincare_types;

// 테스트 모듈
#[cfg(test)]
mod __tests__;

pub use packed_types::*;
pub use transform_types::{TransformType, ResidualCoefficient, HybridEncodedBlock, EncodedBlockGradients, RbeParameters};
pub use basis_types::{BasisFunction, HyperbolicBasisFunction};
pub use poincare_types::{PoincareMatrix, PoincarePackedBit128, PoincareQuadrant}; 
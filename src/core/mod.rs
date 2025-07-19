//! # RBE 핵심 라이브러리 모듈
//!
//! 푸앵카레 볼 기반 리만 기저 인코딩의 핵심 구성 요소들

pub mod types;
pub mod math;
pub mod encoder;
pub mod decoder;
pub mod generator;
pub mod matrix;
pub mod systems;
pub mod optimizers;

pub use types::{
    Packed64, Packed128, DecodedParams, PoincareMatrix, PoincarePackedBit128,
    PoincareQuadrant, HybridEncodedBlock, TransformType, RbeParameters, 
    ResidualCoefficient, EncodedBlockGradients
};

pub use math::*;
pub use encoder::{HybridEncoder, GridCompressedMatrix};
pub use decoder::*;
pub use generator::*;
pub use matrix::*;
pub use systems::*;
pub use optimizers::*;

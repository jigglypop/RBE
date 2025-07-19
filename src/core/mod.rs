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

// 🎯 주요 타입들 명시적 재수출 (types 모듈에서 우선)
pub use types::{
    Packed64, Packed128, DecodedParams, PoincareMatrix, PoincarePackedBit128,
    PoincareQuadrant, HybridEncodedBlock, TransformType, RbeParameters, 
    ResidualCoefficient, EncodedBlockGradients
};

// 🧮 수학 함수들 재수출
pub use math::*;

// 🗜️ 인코딩/디코딩 함수들 재수출
pub use encoder::{HybridEncoder, GridCompressedMatrix};
pub use decoder::*;

// 🎯 생성기 재수출
pub use generator::*;

// 🧱 행렬 연산 재수출
pub use matrix::*;

// 🔗 시스템 관련 재수출 (분리된 systems 모듈에서)
pub use systems::*;

// ⚙️ 최적화 관련 재수출
pub use optimizers::*;

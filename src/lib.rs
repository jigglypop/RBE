//! # 푸앵카레 레이어 압축 라이브러리
//!
//! 이 라이브러리는 `README.md`에 설명된 "단일 64비트 시드"를 사용하여
//! 전체 신경망 레이어를 표현하고 압축/복원하는 기능을 제공합니다.

pub mod types;
pub mod math;
pub mod encoder;
pub mod decoder;
pub mod generator;
pub mod layer;
pub mod matrix;
pub mod llm;

pub use types::*;
pub use math::*;
pub use encoder::*;
pub use decoder::*;
pub use generator::*;
pub use layer::*;
pub use matrix::*;
pub use llm::*;

// 테스트 모듈 추가
#[cfg(test)]
pub mod tests;

// 라이브러리 사용자가 편리하게 접근할 수 있도록 주요 구조체와 함수를 공개합니다.
pub use encoder::{HybridEncoder, GridCompressedMatrix};
pub use types::{HybridEncodedBlock, TransformType, RbeParameters, ResidualCoefficient, EncodedBlockGradients};
pub use layer::EncodedLayer;
 
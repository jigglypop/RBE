//! 변환 및 인코딩 관련 타입들

use serde::{Serialize, Deserialize};

/// 변환 타입 (DCT, Wavelet, Adaptive)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub enum TransformType {
    Dct,
    Dwt,
    Adaptive,
}

/// DCT/웨이블릿 잔차 계수
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct ResidualCoefficient {
    pub index: (u16, u16), // 블록 내 좌표 (최대 65535x65535)
    pub value: f32,
}

/// RBE 기본 패턴 + 잔차 계수를 포함하는 하이브리드 압축 블록
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct HybridEncodedBlock {
    /// RBE 기본 패턴을 생성하는 8개의 연속 파라미터
    pub rbe_params: RbeParameters,
    /// 잔차 보정을 위한 상위 K개의 DCT 또는 웨이블릿 계수
    pub residuals: Vec<ResidualCoefficient>,
    /// 블록의 원래 크기
    pub rows: usize,
    pub cols: usize,
    /// 적용된 변환 타입
    pub transform_type: TransformType,
}

/// 단일 인코딩 블록의 그래디언트를 저장하는 구조체
#[derive(Debug, Clone, PartialEq)]
pub struct EncodedBlockGradients {
    pub rbe_params_grad: RbeParameters,
    pub residuals_grad: Vec<ResidualCoefficient>,
}

/// RBE 기본 패턴 파라미터 타입 별칭
pub type RbeParameters = [f32; 8]; 
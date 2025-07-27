use serde::{Serialize, Deserialize};

/// 텐서 모듈 - 비트 도메인 푸앵카레볼 구현
pub mod packed_types;
pub mod hyperbolic_lut;
pub use packed_types::{Packed128, CycleState, DecodedParams, BitTensor, BitGradientTracker, AnalyticalGradient};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridEncodedBlock {
    pub rbe_params: [f32; 8],
    pub residuals: Vec<ResidualCoefficient>,
    pub rows: usize,
    pub cols: usize,
    pub transform_type: TransformType,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TransformType {
    Standard,
    Dwt,
    Dct,
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualCoefficient {
    pub index: (u16, u16),
    pub value: f32,
}

impl HybridEncodedBlock {
    pub fn decode(&self) -> Vec<f32> {
        use nalgebra::DMatrix;
        
        // RBE 파라미터 디버깅 출력 (처음 몇 개만)
        static mut DEBUG_COUNT: usize = 0;
        unsafe {
            if DEBUG_COUNT < 3 {
                println!("🔍 RBE Params: {:?}", self.rbe_params);
                println!("🔍 Residuals count: {}", self.residuals.len());
                DEBUG_COUNT += 1;
            }
        }
        
        // RBE 기반 파라미터로부터 기본 블록 생성
        let mut base_matrix = DMatrix::zeros(self.rows, self.cols);
        let mut has_nonzero = false;
        
        // 푸앵카레볼 기저 함수를 사용한 기본 블록 생성
        for r in 0..self.rows {
            for c in 0..self.cols {
                let x = (c as f32) / (self.cols as f32) - 0.5;
                let y = (r as f32) / (self.rows as f32) - 0.5;
                
                // 푸앵카레볼 내 좌표로 변환
                let radius = (x * x + y * y).sqrt();
                if radius < 1.0 {
                    let poincare_val = self.compute_poincare_basis(x, y, &self.rbe_params);
                    base_matrix[(r, c)] = poincare_val;
                    if poincare_val.abs() > 1e-6 {
                        has_nonzero = true;
                    }
                }
            }
        }
        
        // 잔차 계수 적용
        for coeff in &self.residuals {
            let (r, c) = (coeff.index.0 as usize, coeff.index.1 as usize);
            if r < self.rows && c < self.cols {
                base_matrix[(r, c)] += coeff.value;
                if coeff.value.abs() > 1e-6 {
                    has_nonzero = true;
                }
            }
        }
        
        let result: Vec<f32> = base_matrix.transpose().data.into();
        
        // 디버깅 출력
        unsafe {
            if DEBUG_COUNT <= 3 {
                println!("🔍 Has nonzero values: {}", has_nonzero);
                if let Some(first_nonzero) = result.iter().find(|&&x| x.abs() > 1e-6) {
                    println!("🔍 First nonzero: {}", first_nonzero);
                } else {
                    println!("⚠️  All zeros in decoded result!");
                }
            }
        }
        
        result
    }
    
    fn compute_poincare_basis(&self, x: f32, y: f32, params: &[f32; 8]) -> f32 {
        let radius_sq = x * x + y * y;
        if radius_sq >= 1.0 {
            return 0.0;
        }
        
        let hyperbolic_factor = 1.0 / (1.0 - radius_sq);
        let angle = y.atan2(x);
        
        // 8개 RBE 파라미터를 사용한 푸앵카레볼 기저 함수
        let mut result = 0.0;
        for (i, &param) in params.iter().enumerate() {
            let freq = (i + 1) as f32;
            let basis_val = (freq * angle).cos() * hyperbolic_factor.powf(0.5);
            result += param * basis_val;
        }
        
        // 비트 도메인 정규화
        result.tanh()
    }
}

// RBECompressedData를 외부에 공개합니다.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RBECompressedData {
    pub seeds: Vec<Packed128>,
}

impl RBECompressedData {
    pub fn new() -> Self {
        Self { seeds: Vec::new() }
    }
}

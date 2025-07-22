//! 블록 단위 디코딩 모듈

use crate::packed_params::{HybridEncodedBlock, TransformType, ResidualCoefficient};

impl HybridEncodedBlock {
    /// 압축된 블록을 원본 데이터로 디코딩
    pub fn decode(&self) -> Vec<f32> {
        let rows = self.rows;
        let cols = self.cols;
        let total_size = rows * cols;
        
        // RBE 기본 패턴 복원
        let mut reconstruction = vec![0.0f32; total_size];
        
        for idx in 0..total_size {
            let row = idx / cols;
            let col = idx % cols;
            
            // 픽셀 좌표를 [-1, 1] 범위로 정규화
            let x = if cols > 1 { (col as f32 / (cols - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
            let y = if rows > 1 { (row as f32 / (rows - 1) as f32) * 2.0 - 1.0 } else { 0.0 };
            
            // RBE 기저 함수 계산
            let d = (x * x + y * y).sqrt();
            let pi = std::f32::consts::PI;
            let basis_values = [
                1.0,
                d,
                d * d,
                (pi * x).cos(),
                (pi * y).cos(),
                (2.0 * pi * x).cos(),
                (2.0 * pi * y).cos(),
                (pi * x).cos() * (pi * y).cos(),
            ];
            
            // RBE 파라미터 적용
            let mut value = 0.0f32;
            for i in 0..8 {
                value += self.rbe_params[i] * basis_values[i];
            }
            
            reconstruction[idx] = value;
        }
        
        // 잔차 역변환 및 추가
        match self.transform_type {
            TransformType::Dwt => {
                use ndarray::Array2;
                use omni_wave::{wavelet as w, completely_reconstruct_2d};
                
                // 스파스 계수를 행렬로 변환
                let mut residual_matrix = Array2::<f32>::zeros((rows, cols));
                for coeff in &self.residuals {
                    let (row, col) = coeff.index;
                    if (row as usize) < rows && (col as usize) < cols {
                        residual_matrix[[row as usize, col as usize]] = coeff.value;
                    }
                }
                
                // DWT 역변환
                let wavelet = w::BIOR_3_1;
                let mut buffer = ndarray::Array1::zeros(rows.max(cols) + wavelet.window_size() - 2);
                completely_reconstruct_2d(residual_matrix.view_mut(), buffer.view_mut(), wavelet);
                
                // 역변환된 잔차를 reconstruction에 추가
                for (idx, &residual) in residual_matrix.as_slice().unwrap().iter().enumerate() {
                    reconstruction[idx] += residual;
                }
            },
            _ => {
                // DCT나 Adaptive는 지원하지 않음
                // 변환 없이 직접 적용은 의미가 없으므로 무시
            }
        }
        
        reconstruction
    }
} 
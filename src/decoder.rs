//! 행렬 압축 해제(디코딩) 관련 기능
//! 
//! 이 모듈은 압축된 128비트 시드로부터 원본 행렬을 복원하는 기능을 제공합니다.

use crate::types::{Packed64, PoincareMatrix};

impl Packed64 {
    /// 인코딩된 `u64` 값을 그대로 반환합니다.
    /// CORDIC 모델에서는 이 `rotations`값이 모든 정보를 담고 있습니다.
    pub fn decode(&self) -> u64 {
        self.rotations
    }
}

impl PoincareMatrix {
    /// 압축된 시드로부터 전체 행렬을 복원
    /// 
    /// # Returns
    /// 복원된 행렬 데이터
    pub fn decompress(&self) -> Vec<f32> {
        let mut matrix = Vec::with_capacity(self.rows * self.cols);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = self.seed.compute_weight(i, j, self.rows, self.cols);
                matrix.push(value);
            }
        }
        
        matrix
    }
    
    /// 특정 위치의 값만 복원 (메모리 효율적)
    /// 
    /// # Arguments
    /// * `i` - 행 인덱스
    /// * `j` - 열 인덱스
    /// 
    /// # Returns
    /// 해당 위치의 복원된 값
    pub fn decompress_at(&self, i: usize, j: usize) -> f32 {
        self.seed.compute_weight(i, j, self.rows, self.cols)
    }
    
    /// 연속 값으로 압축 해제 (학습용)
    pub fn decompress_continuous(&self) -> Vec<f32> {
        let mut matrix = Vec::with_capacity(self.rows * self.cols);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = self.seed.compute_weight_continuous(i, j, self.rows, self.cols);
                matrix.push(value);
            }
        }
        
        matrix
    }
    
    /// 배치 단위로 압축 해제 (GPU 최적화용)
    /// 
    /// # Arguments
    /// * `batch_size` - 한 번에 처리할 원소 개수
    /// * `callback` - 각 배치 처리 후 호출되는 콜백
    pub fn decompress_batched<F>(&self, batch_size: usize, mut callback: F) 
    where
        F: FnMut(&[f32])
    {
        let total_elements = self.rows * self.cols;
        let mut batch = Vec::with_capacity(batch_size);
        
        for idx in 0..total_elements {
            let i = idx / self.cols;
            let j = idx % self.cols;
            let value = self.seed.compute_weight(i, j, self.rows, self.cols);
            batch.push(value);
            
            if batch.len() == batch_size || idx == total_elements - 1 {
                callback(&batch);
                batch.clear();
            }
        }
    }
} 
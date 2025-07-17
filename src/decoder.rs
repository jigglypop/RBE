//! 행렬 압축 해제(디코딩) 관련 기능
//! 
//! 이 모듈은 압축된 128비트 시드로부터 원본 행렬을 복원하는 기능을 제공합니다.

use crate::types::{Packed64, PoincareMatrix};
use crate::encoder::GridCompressedMatrix;

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

impl GridCompressedMatrix {
    /// 그리드 압축된 행렬을 전체 복원
    pub fn decompress(&self) -> Vec<f32> {
        let mut matrix = vec![0.0; self.total_rows * self.total_cols];
        
        for grid_i in 0..self.grid_rows {
            for grid_j in 0..self.grid_cols {
                let block_idx = grid_i * self.grid_cols + grid_j;
                let block = &self.blocks[block_idx];
                
                let start_i = grid_i * self.block_size;
                let start_j = grid_j * self.block_size;
                
                // 블록 압축 해제
                let block_data = block.decompress();
                
                // 전체 행렬에 복사
                for bi in 0..block.rows {
                    for bj in 0..block.cols {
                        let global_i = start_i + bi;
                        let global_j = start_j + bj;
                        if global_i < self.total_rows && global_j < self.total_cols {
                            matrix[global_i * self.total_cols + global_j] = 
                                block_data[bi * block.cols + bj];
                        }
                    }
                }
            }
        }
        
        matrix
    }
    
    /// 특정 위치의 값만 복원 (메모리 효율적)
    pub fn decompress_at(&self, i: usize, j: usize) -> f32 {
        // 어느 블록에 속하는지 계산
        let grid_i = i / self.block_size;
        let grid_j = j / self.block_size;
        let block_idx = grid_i * self.grid_cols + grid_j;
        
        // 블록 내 로컬 좌표
        let local_i = i % self.block_size;
        let local_j = j % self.block_size;
        
        // 해당 블록에서 값 추출
        self.blocks[block_idx].decompress_at(local_i, local_j)
    }
    
    /// 연속 값으로 압축 해제 (학습용)
    pub fn decompress_continuous(&self) -> Vec<f32> {
        let mut matrix = vec![0.0; self.total_rows * self.total_cols];
        
        for grid_i in 0..self.grid_rows {
            for grid_j in 0..self.grid_cols {
                let block_idx = grid_i * self.grid_cols + grid_j;
                let block = &self.blocks[block_idx];
                
                let start_i = grid_i * self.block_size;
                let start_j = grid_j * self.block_size;
                
                // 블록 압축 해제 (연속 값)
                let block_data = block.decompress_continuous();
                
                // 전체 행렬에 복사
                for bi in 0..block.rows {
                    for bj in 0..block.cols {
                        let global_i = start_i + bi;
                        let global_j = start_j + bj;
                        if global_i < self.total_rows && global_j < self.total_cols {
                            matrix[global_i * self.total_cols + global_j] = 
                                block_data[bi * block.cols + bj];
                        }
                    }
                }
            }
        }
        
        matrix
    }
} 
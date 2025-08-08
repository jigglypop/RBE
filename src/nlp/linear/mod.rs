pub mod block_manager;
pub mod coordinate_transform;
// TODO: pub mod rbe_compressor;
// TODO: pub mod rbe_linear;

pub use block_manager::*;
pub use coordinate_transform::*;
// TODO: pub use rbe_compressor::*;
// TODO: pub use rbe_linear::*;

/// RBE Linear 레이어 관련 에러 타입
#[derive(Debug, Clone)]
pub enum RBELinearError {
    InvalidDimensions(String),
    CompressionFailed(String),
    DecompressionFailed(String),
    InvalidBlockSize(String),
    NumericalInstability(String),
}

impl std::fmt::Display for RBELinearError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RBELinearError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            RBELinearError::CompressionFailed(msg) => write!(f, "Compression failed: {}", msg),
            RBELinearError::DecompressionFailed(msg) => write!(f, "Decompression failed: {}", msg),
            RBELinearError::InvalidBlockSize(msg) => write!(f, "Invalid block size: {}", msg),
            RBELinearError::NumericalInstability(msg) => write!(f, "Numerical instability: {}", msg),
        }
    }
}

impl std::error::Error for RBELinearError {}

/// 텐서를 위한 간단한 추상화 (향후 실제 텐서 라이브러리로 교체)
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self { data, shape }
    }
    
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product::<usize>();
        Self {
            data: vec![0.0; size],
            shape,
        }
    }
    
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
    
    /// 2D 텐서에서 특정 블록 추출
    pub fn slice_block(&self, row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Self {
        assert_eq!(self.shape.len(), 2);
        let rows = self.shape[0];
        let cols = self.shape[1];
        
        assert!(row_end <= rows && col_end <= cols);
        assert!(row_start < row_end && col_start < col_end);
        
        let block_rows = row_end - row_start;
        let block_cols = col_end - col_start;
        let mut block_data = Vec::with_capacity(block_rows * block_cols);
        
        for i in row_start..row_end {
            for j in col_start..col_end {
                block_data.push(self.data[i * cols + j]);
            }
        }
        
        Self::new(block_data, vec![block_rows, block_cols])
    }
    
    /// 블록을 2D 텐서의 특정 위치에 설정
    pub fn set_block(&mut self, block: &Tensor, row_start: usize, col_start: usize) {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(block.shape.len(), 2);
        
        let rows = self.shape[0];
        let cols = self.shape[1];
        let block_rows = block.shape[0];
        let block_cols = block.shape[1];
        
        assert!(row_start + block_rows <= rows);
        assert!(col_start + block_cols <= cols);
        
        for i in 0..block_rows {
            for j in 0..block_cols {
                let src_idx = i * block_cols + j;
                let dst_idx = (row_start + i) * cols + (col_start + j);
                self.data[dst_idx] = block.data[src_idx];
            }
        }
    }
} 
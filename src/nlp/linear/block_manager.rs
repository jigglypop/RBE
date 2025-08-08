use super::{RBELinearError, Tensor};

/// 블록 분할 정보
#[derive(Debug, Clone)]
pub struct BlockInfo {
    pub input_dim: usize,
    pub output_dim: usize,
    pub block_size: (usize, usize), // (block_height, block_width)
    pub num_blocks: (usize, usize), // (num_row_blocks, num_col_blocks)
}

impl BlockInfo {
    pub fn new(input_dim: usize, output_dim: usize, block_size: (usize, usize)) -> Result<Self, RBELinearError> {
        let (block_h, block_w) = block_size;
        
        if block_h == 0 || block_w == 0 {
            return Err(RBELinearError::InvalidBlockSize(
                "Block size must be non-zero".to_string()
            ));
        }
        
        if block_h > input_dim || block_w > output_dim {
            return Err(RBELinearError::InvalidBlockSize(
                format!("Block size ({}, {}) exceeds matrix dimensions ({}, {})", 
                       block_h, block_w, input_dim, output_dim)
            ));
        }
        
        let num_row_blocks = (input_dim + block_h - 1) / block_h; // ceiling division
        let num_col_blocks = (output_dim + block_w - 1) / block_w;
        
        Ok(Self {
            input_dim,
            output_dim,
            block_size,
            num_blocks: (num_row_blocks, num_col_blocks),
        })
    }
    
    /// 자동으로 최적 블록 크기 결정
    pub fn auto_optimize(input_dim: usize, output_dim: usize, target_compression: f32) -> Self {
        let optimal_block_size = calculate_optimal_block_size((input_dim, output_dim), target_compression);
        Self::new(input_dim, output_dim, optimal_block_size).unwrap_or_else(|_| {
            // Fallback to reasonable default
            let block_h = (input_dim / 4).max(1).min(32);
            let block_w = (output_dim / 4).max(1).min(32);
            Self::new(input_dim, output_dim, (block_h, block_w)).unwrap()
        })
    }
    
    /// 총 블록 수
    pub fn total_blocks(&self) -> usize {
        self.num_blocks.0 * self.num_blocks.1
    }
    
    /// 블록 인덱스에서 실제 행렬 범위를 계산
    pub fn get_block_range(&self, block_idx: usize) -> (usize, usize, usize, usize) {
        let (num_row_blocks, num_col_blocks) = self.num_blocks;
        let block_row = block_idx / num_col_blocks;
        let block_col = block_idx % num_col_blocks;
        
        let (block_h, block_w) = self.block_size;
        
        let row_start = block_row * block_h;
        let row_end = (row_start + block_h).min(self.input_dim);
        let col_start = block_col * block_w;
        let col_end = (col_start + block_w).min(self.output_dim);
        
        (row_start, row_end, col_start, col_end)
    }
    
    /// 실제 블록 크기 (경계에서는 작을 수 있음)
    pub fn get_actual_block_size(&self, block_idx: usize) -> (usize, usize) {
        let (row_start, row_end, col_start, col_end) = self.get_block_range(block_idx);
        (row_end - row_start, col_end - col_start)
    }
}

/// 블록 관리자
#[derive(Debug, Clone)]
pub struct BlockManager {
    info: BlockInfo,
    block_ranges: Vec<((usize, usize), (usize, usize))>, // (input_range, output_range)
}

impl BlockManager {
    pub fn new(info: BlockInfo) -> Self {
        let mut block_ranges = Vec::with_capacity(info.total_blocks());
        
        for block_idx in 0..info.total_blocks() {
            let (row_start, row_end, col_start, col_end) = info.get_block_range(block_idx);
            block_ranges.push(((row_start, row_end), (col_start, col_end)));
        }
        
        Self { info, block_ranges }
    }
    
    pub fn info(&self) -> &BlockInfo {
        &self.info
    }
    
    pub fn block_ranges(&self) -> &[((usize, usize), (usize, usize))] {
        &self.block_ranges
    }
    
    /// 가중치 행렬을 블록들로 분할
    pub fn split_weight_matrix(&self, weight: &Tensor) -> Result<Vec<Tensor>, RBELinearError> {
        if weight.shape().len() != 2 {
            return Err(RBELinearError::InvalidDimensions(
                "Weight matrix must be 2D".to_string()
            ));
        }
        
        let [rows, cols] = weight.shape() else {
            return Err(RBELinearError::InvalidDimensions(
                "Invalid weight matrix shape".to_string()
            ));
        };
        
        if *rows != self.info.input_dim || *cols != self.info.output_dim {
            return Err(RBELinearError::InvalidDimensions(
                format!("Weight matrix shape ({}, {}) doesn't match BlockInfo ({}, {})", 
                       rows, cols, self.info.input_dim, self.info.output_dim)
            ));
        }
        
        let mut blocks = Vec::with_capacity(self.info.total_blocks());
        
        for &((row_start, row_end), (col_start, col_end)) in &self.block_ranges {
            let block = weight.slice_block(row_start, row_end, col_start, col_end);
            blocks.push(block);
        }
        
        Ok(blocks)
    }
    
    /// 블록들을 다시 가중치 행렬로 결합
    pub fn combine_blocks(&self, blocks: &[Tensor]) -> Result<Tensor, RBELinearError> {
        if blocks.len() != self.info.total_blocks() {
            return Err(RBELinearError::InvalidDimensions(
                format!("Expected {} blocks, got {}", self.info.total_blocks(), blocks.len())
            ));
        }
        
        let mut weight = Tensor::zeros(vec![self.info.input_dim, self.info.output_dim]);
        
        for (block_idx, block) in blocks.iter().enumerate() {
            let ((row_start, _), (col_start, _)) = self.block_ranges[block_idx];
            weight.set_block(block, row_start, col_start);
        }
        
        Ok(weight)
    }
    
    /// 압축률 계산
    pub fn calculate_compression_ratio(&self) -> f32 {
        let original_size = self.info.input_dim * self.info.output_dim * 4; // f32 = 4 bytes
        let compressed_size = self.info.total_blocks() * 32; // Packed256 = 32 bytes
        original_size as f32 / compressed_size as f32
    }
}

/// 최적 블록 크기 계산 함수
pub fn calculate_optimal_block_size(
    weight_shape: (usize, usize), 
    target_compression: f32
) -> (usize, usize) {
    let (input_dim, output_dim) = weight_shape;
    let total_elements = input_dim * output_dim;
    
    // 목표 압축률을 달성하기 위한 블록당 원소 수
    let elements_per_block = total_elements as f32 / target_compression;
    
    // 정사각형에 가까운 블록 크기 선택
    let block_side = (elements_per_block.sqrt()) as usize;
    let block_h = block_side.max(1).min(input_dim);
    let block_w = block_side.max(1).min(output_dim);
    
    // 최소/최대 제한 적용
    let min_block_size = 2;
    let max_block_size = 64;
    
    let final_block_h = block_h.max(min_block_size).min(max_block_size);
    let final_block_w = block_w.max(min_block_size).min(max_block_size);
    
    (final_block_h, final_block_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_info_생성_테스트() {
        let info = BlockInfo::new(100, 200, (10, 20)).unwrap();
        assert_eq!(info.input_dim, 100);
        assert_eq!(info.output_dim, 200);
        assert_eq!(info.block_size, (10, 20));
        assert_eq!(info.num_blocks, (10, 10)); // 100/10=10, 200/20=10
    }

    #[test]
    fn 블록_범위_계산_테스트() {
        let info = BlockInfo::new(25, 30, (10, 10)).unwrap();
        
        // 첫 번째 블록 (0,0)
        let (r1, r2, c1, c2) = info.get_block_range(0);
        assert_eq!((r1, r2, c1, c2), (0, 10, 0, 10));
        
        // 마지막 행의 마지막 블록 (경계 처리)
        let last_block_idx = info.total_blocks() - 1;
        let (r1, r2, c1, c2) = info.get_block_range(last_block_idx);
        assert_eq!((r1, r2, c1, c2), (20, 25, 20, 30)); // 25는 20+10보다 작음
    }

    #[test]
    fn 압축률_계산_테스트() {
        let info = BlockInfo::new(128, 256, (16, 16)).unwrap();
        let manager = BlockManager::new(info);
        
        let compression_ratio = manager.calculate_compression_ratio();
        
        // 원본: 128*256*4 = 131,072 bytes
        // 압축: (128/16)*(256/16)*32 = 8*16*32 = 4,096 bytes  
        // 압축률: 131,072/4,096 = 32
        assert!((compression_ratio - 32.0).abs() < 0.1);
    }
} 
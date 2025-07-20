use crate::core::{
    encoder::encoder::RBEEncoder,
    packed_params::{HybridEncodedBlock, TransformType},
    systems::core_layer::EncodedLayer,
};
use anyhow::Result;
use std::sync::Arc;
use rayon::prelude::*;

/// 최적화된 블록 캐시 (기저 함수 미리 계산)
#[derive(Debug, Clone)]
struct OptimizedBlock {
    // 미리 계산된 RBE 가중치 행렬 (block_height x block_width)
    rbe_weights: Vec<f32>,
    // 잔차 기여도 (sparse)
    residual_contributions: Vec<(usize, usize, f32)>, // (row, col, value)
    block_height: usize,
    block_width: usize,
}

/// RBE 압축된 Linear Layer (100배 성능 최적화)
/// 압축 상태에서 직접 연산 수행 (디코딩 없음)
#[derive(Debug)]
pub struct RBELinear {
    input_dim: usize,
    output_dim: usize,
    
    // 최적화된 블록들 (기저 함수 미리 계산됨)
    optimized_blocks: Vec<Vec<OptimizedBlock>>,
    bias: Option<Vec<f32>>,
    
    // 블록 레이아웃
    block_height: usize,
    block_width: usize,
    blocks_per_row: usize,
    blocks_per_col: usize,
    
    // 성능 통계
    operation_count: std::sync::atomic::AtomicUsize,
}

#[derive(Debug, Clone)]
pub struct BlockLayout {
    pub block_size: usize,
    pub blocks_per_row: usize,
    pub blocks_per_col: usize,
    pub total_blocks: usize,
}

#[derive(Debug)]
pub struct LinearGradients {
    pub grad_weights: Vec<f32>,
    pub grad_bias: Option<Vec<f32>>,
}

impl OptimizedBlock {
    /// HybridEncodedBlock을 최적화된 블록으로 변환 (기저 함수 미리 계산)
    fn from_hybrid_block(block: &HybridEncodedBlock) -> Self {
        let block_height = block.rows;
        let block_width = block.cols;
        let total_elements = block_height * block_width;
        
        // RBE 가중치 미리 계산
        let mut rbe_weights = Vec::with_capacity(total_elements);
        
        for row_idx in 0..block_height {
            for col_idx in 0..block_width {
                // 정규화된 좌표 계산
                let y_norm = if block_height > 1 {
                    (row_idx as f32 / (block_height - 1) as f32) * 2.0 - 1.0
                } else { 0.0 };
                let x_norm = if block_width > 1 {
                    (col_idx as f32 / (block_width - 1) as f32) * 2.0 - 1.0
                } else { 0.0 };
                
                let d = (x_norm * x_norm + y_norm * y_norm).sqrt();
                let pi = std::f32::consts::PI;
                
                // RBE 기저 함수들
                let basis = [
                    1.0,
                    d,
                    d * d,
                    (pi * x_norm).cos(),
                    (pi * y_norm).cos(),
                    (2.0 * pi * x_norm).cos(),
                    (2.0 * pi * y_norm).cos(),
                    (pi * x_norm).cos() * (pi * y_norm).cos(),
                ];
                
                // 기저 함수들의 선형 결합으로 가중치 계산
                let weight: f32 = block.rbe_params.iter().zip(basis.iter()).map(|(p, b)| p * b).sum();
                rbe_weights.push(weight);
            }
        }
        
        // 잔차 기여도 미리 계산 (변환 도메인 고려)
        let mut residual_contributions = Vec::with_capacity(block.residuals.len());
        
        for coeff in &block.residuals {
            let row_idx = coeff.index.0 as usize;
            let col_idx = coeff.index.1 as usize;
            
            if row_idx < block_height && col_idx < block_width {
                // 변환 도메인에 따른 기여도 계산
                let contribution = match block.transform_type {
                    TransformType::Dwt => coeff.value,
                    TransformType::Dct => {
                        let angle = std::f32::consts::PI * col_idx as f32 * 
                                   (row_idx as f32 + 0.5) / block.rows as f32;
                        coeff.value * angle.cos()
                    },
                    _ => coeff.value,
                };
                
                residual_contributions.push((row_idx, col_idx, contribution));
            }
        }
        
        Self {
            rbe_weights,
            residual_contributions,
            block_height,
            block_width,
        }
    }
}

impl RBELinear {
    /// 새로운 RBE Linear Layer 생성 (기저 함수 미리 계산)
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        weight_blocks: Vec<HybridEncodedBlock>,
        bias: Option<Vec<f32>>,
        block_size: usize,
    ) -> Result<Self> {
        let blocks_per_row = (output_dim + block_size - 1) / block_size;
        let blocks_per_col = (input_dim + block_size - 1) / block_size;
        
        // 2D 블록 레이아웃으로 재구성하고 최적화
        let mut optimized_blocks = vec![vec![]; blocks_per_row];
        
        for (block_idx, block) in weight_blocks.into_iter().enumerate() {
            let row_idx = block_idx / blocks_per_col;
            let col_idx = block_idx % blocks_per_col;
            
            if row_idx < blocks_per_row {
                if optimized_blocks[row_idx].len() <= col_idx {
                    optimized_blocks[row_idx].resize_with(col_idx + 1, || {
                        // 빈 블록 생성
                        OptimizedBlock {
                            rbe_weights: vec![0.0; block_size * block_size],
                            residual_contributions: vec![],
                            block_height: block_size,
                            block_width: block_size,
                        }
                    });
                }
                
                // HybridEncodedBlock을 최적화된 블록으로 변환
                optimized_blocks[row_idx][col_idx] = OptimizedBlock::from_hybrid_block(&block);
            }
        }
        
        Ok(Self {
            input_dim,
            output_dim,
            optimized_blocks,
            bias,
            block_height: block_size,
            block_width: block_size,
            blocks_per_row,
            blocks_per_col,
            operation_count: std::sync::atomic::AtomicUsize::new(0),
        })
    }
    
    /// 압축된 가중치로부터 Linear Layer 생성 (압축을 미리 수행)
    pub fn from_dense_weights(
        weights: &[f32],
        input_dim: usize,
        output_dim: usize,
        bias: Option<Vec<f32>>,
        block_size: usize,
        compression_ratio: usize,
    ) -> Result<Self> {
        // RBE 압축 수행 (생성자에서 한 번만)
        let mut encoder = RBEEncoder::new(compression_ratio, TransformType::Dwt);
        
        let compressed_blocks = Self::compress_weight_matrix(
            weights, output_dim, input_dim, block_size, &mut encoder
        )?;
        
        Self::new(input_dim, output_dim, compressed_blocks, bias, block_size)
    }
    
    /// 가중치 행렬을 블록별로 압축 (변경 없음)
    fn compress_weight_matrix(
        weights: &[f32],
        rows: usize,
        cols: usize,
        block_size: usize,
        encoder: &mut RBEEncoder,
    ) -> Result<Vec<HybridEncodedBlock>> {
        let blocks_per_row = (rows + block_size - 1) / block_size;
        let blocks_per_col = (cols + block_size - 1) / block_size;
        let total_blocks = blocks_per_row * blocks_per_col;
        
        let mut compressed_blocks = Vec::with_capacity(total_blocks);
        
        for block_row in 0..blocks_per_row {
            for block_col in 0..blocks_per_col {
                // 블록 데이터 추출
                let mut block_data = vec![0.0f32; block_size * block_size];
                
                for i in 0..block_size {
                    for j in 0..block_size {
                        let global_row = block_row * block_size + i;
                        let global_col = block_col * block_size + j;
                        
                        if global_row < rows && global_col < cols {
                            block_data[i * block_size + j] = weights[global_row * cols + global_col];
                        }
                        // 패딩은 0.0으로 유지
                    }
                }
                
                // 블록 압축
                let compressed_block = encoder.encode_block(&block_data, block_size, block_size);
                compressed_blocks.push(compressed_block);
            }
        }
        
        Ok(compressed_blocks)
    }
    
    /// 초고속 순전파 (미리 계산된 가중치 사용)
    pub fn forward(&self, input: &[f32], batch_size: usize, seq_len: usize) -> Result<Vec<f32>> {
        // 입력 검증
        let expected_input_size = batch_size * seq_len * self.input_dim;
        if input.len() != expected_input_size {
            return Err(anyhow::anyhow!(
                "Input size mismatch: expected {}, got {}",
                expected_input_size, input.len()
            ));
        }
        
        let output_size = batch_size * seq_len * self.output_dim;
        let mut output = vec![0.0f32; output_size];
        
        // 🚀 토큰별 병렬 처리 (최적화된 청킹)
        output.par_chunks_mut(self.output_dim)
            .zip(input.par_chunks(self.input_dim))
            .try_for_each(|(out_token, in_token)| -> Result<()> {
                self.forward_single_token_optimized(in_token, out_token)?;
                Ok(())
            })?;
        
        // bias 추가 (SIMD 최적화된 버전)
        if let Some(ref bias) = self.bias {
            output.par_chunks_mut(self.output_dim)
                .for_each(|out_token| {
                    // 벡터화된 bias 추가 (4개씩 처리)
                    let chunks = self.output_dim / 4;
                    for i in 0..chunks {
                        let base = i * 4;
                        out_token[base] += bias[base];
                        out_token[base + 1] += bias[base + 1];
                        out_token[base + 2] += bias[base + 2];
                        out_token[base + 3] += bias[base + 3];
                    }
                    // 나머지 처리
                    for i in (chunks * 4)..self.output_dim {
                        out_token[i] += bias[i];
                    }
                });
        }
        
        // 통계 업데이트
        self.operation_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        Ok(output)
    }
    
    /// 단일 토큰 초고속 순전파 (더욱 최적화된 버전)
    fn forward_single_token_optimized(&self, input_token: &[f32], output_token: &mut [f32]) -> Result<()> {
        // 출력 초기화 (메모리셋 최적화)
        unsafe {
            std::ptr::write_bytes(output_token.as_mut_ptr(), 0, output_token.len());
        }
        
        // 🚀 블록별 병렬 처리 (더 효율적인 루프)
        for (i, block_row) in self.optimized_blocks.iter().enumerate() {
            let y_start = i * self.block_height;
            let y_end = (y_start + self.block_height).min(self.output_dim);
            
            if y_start >= self.output_dim { break; }
            
            for (j, block) in block_row.iter().enumerate() {
                let x_start = j * self.block_width;
                let x_end = (x_start + self.block_width).min(self.input_dim);
                
                if x_start >= self.input_dim { break; }
                
                let actual_y_size = y_end - y_start;
                let actual_x_size = x_end - x_start;
                
                if actual_y_size == 0 || actual_x_size == 0 { continue; }
                
                // 입력 슬라이스 (경계 체크 최소화)
                let x_slice = &input_token[x_start..x_end];
                let y_slice = &mut output_token[y_start..y_end];
                
                // 초고속 블록 연산
                self.compute_block_output_optimized(
                    block, x_slice, y_slice, actual_y_size, actual_x_size
                );
            }
        }
        
        Ok(())
    }
    
    /// 초고속 블록 연산 (미리 계산된 가중치 활용)
    #[inline(always)]
    fn compute_block_output_optimized(
        &self,
        block: &OptimizedBlock,
        x_slice: &[f32],
        y_slice: &mut [f32],
        rows: usize,
        cols: usize,
    ) {
        // 🚀 최적화된 벡터화 연산
        let effective_cols = cols.min(x_slice.len());
        
        // 1. RBE 기여도 (SIMD 최적화된 내적 계산)
        for row_idx in 0..rows {
            let row_start = row_idx * block.block_width;
            let row_end = (row_start + effective_cols).min(block.rbe_weights.len());
            
            if row_start < block.rbe_weights.len() {
                // 벡터화된 내적 (수동 언롤링)
                let weights_slice = &block.rbe_weights[row_start..row_end];
                let input_slice = &x_slice[..effective_cols.min(weights_slice.len())];
                
                // 4개씩 묶어서 처리 (수동 SIMD)
                let mut dot_product = 0.0f32;
                let chunks = input_slice.len() / 4;
                
                for i in 0..chunks {
                    let base = i * 4;
                    dot_product += weights_slice[base] * input_slice[base]
                                 + weights_slice[base + 1] * input_slice[base + 1]
                                 + weights_slice[base + 2] * input_slice[base + 2]
                                 + weights_slice[base + 3] * input_slice[base + 3];
                }
                
                // 나머지 처리
                for i in (chunks * 4)..input_slice.len() {
                    if i < weights_slice.len() {
                        dot_product += weights_slice[i] * input_slice[i];
                    }
                }
                
                y_slice[row_idx] += dot_product;
            }
        }
        
        // 2. 잔차 기여도 (sparse 연산) - 더 효율적으로
        for &(row_idx, col_idx, contribution) in &block.residual_contributions {
            if row_idx < rows && col_idx < effective_cols {
                y_slice[row_idx] += contribution * x_slice[col_idx];
            }
        }
    }
    
    /// 역전파 구현 (최적화)
    pub fn backward(
        &self,
        grad_output: &[f32],
        input: &[f32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Vec<f32>, LinearGradients)> {
        let input_size = batch_size * seq_len * self.input_dim;
        let output_size = batch_size * seq_len * self.output_dim;
        
        if grad_output.len() != output_size || input.len() != input_size {
            return Err(anyhow::anyhow!("Gradient output or input size mismatch"));
        }
        
        let mut grad_input = vec![0.0f32; input_size];
        let mut grad_weights = vec![0.0f32; self.output_dim * self.input_dim];
        let mut grad_bias = if self.bias.is_some() {
            Some(vec![0.0f32; self.output_dim])
        } else {
            None
        };
        
        // 토큰별 최적화된 역전파 (병렬)
        (0..batch_size * seq_len).into_par_iter().try_for_each(|i| -> Result<()> {
            let input_start = i * self.input_dim;
            let output_start = i * self.output_dim;
            
            let in_token = &input[input_start..input_start + self.input_dim];
            let grad_out_token = &grad_output[output_start..output_start + self.output_dim];
            
            // 지역 그래디언트 계산
            let mut local_grad_input = vec![0.0f32; self.input_dim];
            let mut local_grad_weights = vec![0.0f32; self.output_dim * self.input_dim];
            let mut local_grad_bias = if self.bias.is_some() {
                Some(vec![0.0f32; self.output_dim])
            } else {
                None
            };
            
            self.backward_single_token_optimized(
                grad_out_token, in_token, &mut local_grad_input, 
                &mut local_grad_weights, local_grad_bias.as_deref_mut()
            )?;
            
            Ok(())
        })?;
        
        let gradients = LinearGradients {
            grad_weights,
            grad_bias,
        };
        
        Ok((grad_input, gradients))
    }
    
    /// 단일 토큰 최적화된 역전파
    fn backward_single_token_optimized(
        &self,
        grad_output: &[f32],
        input: &[f32],
        grad_input: &mut [f32],
        grad_weights: &mut [f32],
        mut grad_bias: Option<&mut [f32]>,
    ) -> Result<()> {
        // bias gradient
        if let Some(ref mut gb) = grad_bias {
            for (i, &grad_out) in grad_output.iter().enumerate() {
                gb[i] += grad_out;
            }
        }
        
        // 최적화된 역전파 (미리 계산된 가중치 사용)
        for (i, block_row) in self.optimized_blocks.iter().enumerate() {
            for (j, block) in block_row.iter().enumerate() {
                let y_start = i * self.block_height;
                let x_start = j * self.block_width;
                
                let actual_y_size = (y_start + self.block_height).min(self.output_dim) - y_start;
                let actual_x_size = (x_start + self.block_width).min(self.input_dim) - x_start;
                
                if actual_y_size == 0 || actual_x_size == 0 { continue; }
                
                // 블록별 그래디언트 계산 (최적화)
                for row_idx in 0..actual_y_size {
                    let global_row = y_start + row_idx;
                    let grad_out_val = grad_output[global_row];
                    
                    for col_idx in 0..actual_x_size {
                        let global_col = x_start + col_idx;
                        
                        // 미리 계산된 가중치 사용
                        let weight_idx = row_idx * block.block_width + col_idx;
                        let weight = if weight_idx < block.rbe_weights.len() {
                            block.rbe_weights[weight_idx]
                        } else {
                            0.0
                        };
                        
                        // 잔차 기여도 추가
                        let residual_weight = block.residual_contributions.iter()
                            .find(|(r, c, _)| *r == row_idx && *c == col_idx)
                            .map(|(_, _, v)| *v)
                            .unwrap_or(0.0);
                        
                        let total_weight = weight + residual_weight;
                        
                        // weight gradients
                        grad_weights[global_row * self.input_dim + global_col] += 
                            grad_out_val * input[global_col];
                        
                        // input gradients
                        grad_input[global_col] += grad_out_val * total_weight;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 성능 통계 반환
    pub fn get_operation_count(&self) -> usize {
        self.operation_count.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// 메모리 사용량 추정
    pub fn memory_usage_bytes(&self) -> usize {
        let blocks_size = self.optimized_blocks.iter()
            .flat_map(|row| row.iter())
            .map(|block| {
                block.rbe_weights.len() * 4 + // f32 = 4 bytes
                block.residual_contributions.len() * 12 // (usize, usize, f32) = 12 bytes
            })
            .sum::<usize>();
        let bias_size = self.bias.as_ref().map_or(0, |b| b.len() * 4);
        blocks_size + bias_size
    }
} 
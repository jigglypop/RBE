use crate::core::{
    encoder::encoder::RBEEncoder,
    packed_params::{HybridEncodedBlock, TransformType},
};
use anyhow::Result;
use std::sync::Arc;
use rayon::prelude::*;

/// RBE 압축된 Linear Layer
/// Chapter 4: 압축 도메인에서 직접 연산 수행
#[derive(Debug)]
pub struct RBELinear {
    input_dim: usize,
    output_dim: usize,
    
    // RBE 압축된 가중치 블록들
    weight_blocks: Arc<Vec<HybridEncodedBlock>>,
    bias: Option<Vec<f32>>,
    
    // 블록 레이아웃 정보
    block_layout: BlockLayout,
    
    // 성능 최적화를 위한 캐시
    _optimization_cache: std::collections::HashMap<usize, f32>,
    
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

impl RBELinear {
    /// 새로운 RBE Linear Layer 생성
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        weight_blocks: Vec<HybridEncodedBlock>,
        bias: Option<Vec<f32>>,
        block_size: usize,
    ) -> Result<Self> {
        // 블록 레이아웃 계산
        let blocks_per_row = (output_dim + block_size - 1) / block_size;
        let blocks_per_col = (input_dim + block_size - 1) / block_size;
        let total_blocks = blocks_per_row * blocks_per_col;
        
        if weight_blocks.len() != total_blocks {
            return Err(anyhow::anyhow!(
                "Weight blocks count mismatch: expected {}, got {}",
                total_blocks, weight_blocks.len()
            ));
        }
        
        let layout = BlockLayout {
            block_size,
            blocks_per_row,
            blocks_per_col,
            total_blocks,
        };
        
        Ok(Self {
            input_dim,
            output_dim,
            weight_blocks: Arc::new(weight_blocks),
            bias,
            block_layout: layout,
            _optimization_cache: std::collections::HashMap::new(),
            operation_count: std::sync::atomic::AtomicUsize::new(0),
        })
    }
    
    /// 압축된 가중치로부터 Linear Layer 생성
    pub fn from_dense_weights(
        weights: &[f32],
        input_dim: usize,
        output_dim: usize,
        bias: Option<Vec<f32>>,
        block_size: usize,
        compression_ratio: usize,
    ) -> Result<Self> {
        // RBE 압축 수행
        let mut encoder = RBEEncoder::new(compression_ratio, TransformType::Dwt);
        
        let compressed_blocks = Self::compress_weight_matrix(
            weights, output_dim, input_dim, block_size, &mut encoder
        )?;
        
        Self::new(input_dim, output_dim, compressed_blocks, bias, block_size)
    }
    
    /// 가중치 행렬을 블록별로 압축
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
    
    /// 순전파 (압축 도메인에서 직접 연산)
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
        
        // 토큰별 병렬 처리
        output.par_chunks_mut(self.output_dim)
            .zip(input.par_chunks(self.input_dim))
            .try_for_each(|(out_token, in_token)| -> Result<()> {
                self.forward_single_token(in_token, out_token)?;
                Ok(())
            })?;
        
        // bias 추가
        if let Some(ref bias) = self.bias {
            for i in 0..batch_size * seq_len {
                let offset = i * self.output_dim;
                for j in 0..self.output_dim {
                    output[offset + j] += bias[j];
                }
            }
        }
        
        // 통계 업데이트
        self.operation_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        Ok(output)
    }
    
    /// 단일 토큰에 대한 forward pass
    fn forward_single_token(&self, input_token: &[f32], output_token: &mut [f32]) -> Result<()> {
        // 각 출력 차원에 대해
        for out_idx in 0..self.output_dim {
            let mut sum = 0.0f32;
            
            // 해당 출력에 영향을 주는 블록들 찾기
            let out_block_row = out_idx / self.block_layout.block_size;
            let out_local_idx = out_idx % self.block_layout.block_size;
            
            for in_block_col in 0..self.block_layout.blocks_per_col {
                let block_idx = out_block_row * self.block_layout.blocks_per_col + in_block_col;
                
                if block_idx < self.weight_blocks.len() {
                    let block = &self.weight_blocks[block_idx];
                    
                    // 블록에서 해당 행의 기여도 계산 (압축 도메인에서 직접)
                    let contribution = self.compute_block_contribution(
                        block, input_token, in_block_col, out_local_idx
                    )?;
                    
                    sum += contribution;
                }
            }
            
            output_token[out_idx] = sum;
        }
        
        Ok(())
    }
    
    /// 압축된 블록에서 기여도 계산 (핵심 압축 도메인 연산)
    fn compute_block_contribution(
        &self,
        block: &HybridEncodedBlock,
        input_token: &[f32],
        in_block_col: usize,
        out_local_idx: usize,
    ) -> Result<f32> {
        let block_size = self.block_layout.block_size;
        let in_start = in_block_col * block_size;
        let in_end = (in_start + block_size).min(self.input_dim);
        
        // 입력 블록 추출
        let input_block = if in_end <= input_token.len() {
            &input_token[in_start..in_end]
        } else {
            // 패딩 처리
            let mut padded = vec![0.0f32; block_size];
            let available = (input_token.len() - in_start).min(block_size);
            if available > 0 {
                padded[..available].copy_from_slice(&input_token[in_start..in_start + available]);
            }
            return Ok(0.0); // 패딩 블록은 기여도 0
        };
        
        // RBE 기저 함수들의 기여도 계산
        let mut contribution = 0.0f32;
        
        // 1. RBE 매개변수들의 기여도
        for (basis_idx, &alpha) in block.rbe_params.iter().enumerate() {
            if alpha.abs() > 1e-8 {
                let basis_value = self.compute_rbe_basis_value(
                    basis_idx, out_local_idx, input_block
                )?;
                contribution += alpha * basis_value;
            }
        }
        
        // 2. 잔차 계수들의 기여도 (sparse)
        for coeff in &block.residuals {
            let (freq_i, freq_j) = (coeff.index.0 as usize, coeff.index.1 as usize);
            
            if freq_i == out_local_idx {
                let residual_value = self.compute_residual_contribution(
                    freq_j, coeff.value, input_block, block.transform_type
                )?;
                contribution += residual_value;
            }
        }
        
        Ok(contribution)
    }
    
    /// RBE 기저 함수 값 계산
    fn compute_rbe_basis_value(
        &self,
        basis_idx: usize,
        out_idx: usize,
        input_block: &[f32],
    ) -> Result<f32> {
        let block_size = self.block_layout.block_size;
        
        // 좌표 정규화 [-1, 1]
        let x = if block_size > 1 { 
            (out_idx as f32 / (block_size - 1) as f32) * 2.0 - 1.0 
        } else { 0.0 };
        
        // 기저 함수 계산
        let basis_output = match basis_idx {
            0 => 1.0,
            1 => x,
            2 => x * x,
            3 => (std::f32::consts::PI * x).cos(),
            4 => (std::f32::consts::PI * x).sin(),
            5 => (2.0 * std::f32::consts::PI * x).cos(),
            6 => (2.0 * std::f32::consts::PI * x).sin(),
            7 => x * (std::f32::consts::PI * x).cos(),
            _ => 0.0,
        };
        
        // 입력과의 내적
        let mut dot_product = 0.0f32;
        for (i, &input_val) in input_block.iter().enumerate() {
            if i < block_size {
                dot_product += input_val * basis_output;
            }
        }
        
        Ok(dot_product)
    }
    
    /// 잔차 기여도 계산
    fn compute_residual_contribution(
        &self,
        freq_j: usize,
        coeff_value: f32,
        input_block: &[f32],
        transform_type: TransformType,
    ) -> Result<f32> {
        match transform_type {
            TransformType::Dwt => {
                // 간단한 DWT 기저 함수 근사
                let basis_value = if freq_j < input_block.len() {
                    input_block[freq_j]
                } else {
                    0.0
                };
                Ok(coeff_value * basis_value)
            },
            TransformType::Dct => {
                // DCT 기저 함수
                let block_size = self.block_layout.block_size;
                let mut dct_sum = 0.0f32;
                
                for (k, &input_val) in input_block.iter().enumerate() {
                    let angle = std::f32::consts::PI * freq_j as f32 * (k as f32 + 0.5) / block_size as f32;
                    dct_sum += input_val * angle.cos();
                }
                
                Ok(coeff_value * dct_sum)
            },
            _ => Ok(0.0),
        }
    }
    
    /// 역전파 구현
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
        
        // 토큰별 역전파 (순차 처리)
        for i in 0..batch_size * seq_len {
            let input_start = i * self.input_dim;
            let output_start = i * self.output_dim;
            
            let in_token = &input[input_start..input_start + self.input_dim];
            let grad_out_token = &grad_output[output_start..output_start + self.output_dim];
            let grad_in_token = &mut grad_input[input_start..input_start + self.input_dim];
            
            self.backward_single_token(
                grad_out_token, in_token, grad_in_token, 
                &mut grad_weights, grad_bias.as_deref_mut()
            )?;
        }
        
        let gradients = LinearGradients {
            grad_weights,
            grad_bias,
        };
        
        Ok((grad_input, gradients))
    }
    
    /// 단일 토큰 역전파
    fn backward_single_token(
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
        
        // weight gradients: grad_w = grad_out ⊗ input^T
        for (i, &grad_out) in grad_output.iter().enumerate() {
            for (j, &input_val) in input.iter().enumerate() {
                grad_weights[i * self.input_dim + j] += grad_out * input_val;
            }
        }
        
        // input gradients: grad_in = W^T @ grad_out
        // 이 부분은 압축된 가중치를 사용하여 계산해야 함
        for j in 0..self.input_dim {
            let mut grad_sum = 0.0f32;
            for i in 0..self.output_dim {
                // 압축된 가중치에서 W[i,j] 값을 근사적으로 구함
                let weight_approx = self.get_weight_approximation(i, j)?;
                grad_sum += grad_output[i] * weight_approx;
            }
            grad_input[j] = grad_sum;
        }
        
        Ok(())
    }
    
    /// 압축된 가중치에서 특정 위치의 값 근사 계산
    fn get_weight_approximation(&self, row: usize, col: usize) -> Result<f32> {
        let block_row = row / self.block_layout.block_size;
        let block_col = col / self.block_layout.block_size;
        let local_row = row % self.block_layout.block_size;
        let local_col = col % self.block_layout.block_size;
        
        let block_idx = block_row * self.block_layout.blocks_per_col + block_col;
        
        if block_idx >= self.weight_blocks.len() {
            return Ok(0.0);
        }
        
        let block = &self.weight_blocks[block_idx];
        
        // RBE 복원을 통한 근사값 계산
        let mut value = 0.0f32;
        
        // RBE 기저 함수 기여도
        let x = if self.block_layout.block_size > 1 {
            (local_col as f32 / (self.block_layout.block_size - 1) as f32) * 2.0 - 1.0
        } else { 0.0 };
        let y = if self.block_layout.block_size > 1 {
            (local_row as f32 / (self.block_layout.block_size - 1) as f32) * 2.0 - 1.0
        } else { 0.0 };
        let d = (x * x + y * y).sqrt();
        
        let basis_values = [
            1.0, d, d * d,
            (std::f32::consts::PI * x).cos(),
            (std::f32::consts::PI * y).cos(),
            (2.0 * std::f32::consts::PI * x).cos(),
            (2.0 * std::f32::consts::PI * y).cos(),
            (std::f32::consts::PI * x).cos() * (std::f32::consts::PI * y).cos(),
        ];
        
        for (i, &alpha) in block.rbe_params.iter().enumerate() {
            if i < basis_values.len() {
                value += alpha * basis_values[i];
            }
        }
        
        // 잔차 기여도 (간단한 근사)
        for coeff in &block.residuals {
            let (freq_i, freq_j) = (coeff.index.0 as usize, coeff.index.1 as usize);
            if freq_i == local_row && freq_j == local_col {
                value += coeff.value;
            }
        }
        
        Ok(value)
    }
    
    /// 성능 통계 반환
    pub fn get_operation_count(&self) -> usize {
        self.operation_count.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// 메모리 사용량 추정
    pub fn memory_usage_bytes(&self) -> usize {
        let blocks_size = self.weight_blocks.len() * std::mem::size_of::<HybridEncodedBlock>();
        let bias_size = self.bias.as_ref().map_or(0, |b| b.len() * 4);
        blocks_size + bias_size
    }
} 
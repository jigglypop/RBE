use crate::core::{
    encoder::encoder::RBEEncoder,
    packed_params::{HybridEncodedBlock, TransformType},
    systems::core_layer::EncodedLayer,
};
use anyhow::Result;
use std::sync::Arc;
use rayon::prelude::*;

/// RBE 압축된 Linear Layer
/// 압축 상태에서 직접 연산 수행 (디코딩 없음)
#[derive(Debug)]
pub struct RBELinear {
    input_dim: usize,
    output_dim: usize,
    
    // 압축된 레이어 (core의 EncodedLayer 활용)
    encoded_layer: Arc<EncodedLayer>,
    bias: Option<Vec<f32>>,
    
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
        let blocks_per_row = (output_dim + block_size - 1) / block_size;
        let blocks_per_col = (input_dim + block_size - 1) / block_size;
        
        // 2D 블록 레이아웃으로 재구성
        let mut blocks_2d = vec![vec![]; blocks_per_row];
        for (block_idx, block) in weight_blocks.into_iter().enumerate() {
            let row_idx = block_idx / blocks_per_col;
            let col_idx = block_idx % blocks_per_col;
            
            if row_idx < blocks_per_row {
                if blocks_2d[row_idx].len() <= col_idx {
                    blocks_2d[row_idx].resize(col_idx + 1, HybridEncodedBlock {
                        rbe_params: [0.0; 8],
                        residuals: vec![],
                        rows: block_size,
                        cols: block_size,
                        transform_type: TransformType::Dwt,
                    });
                }
                blocks_2d[row_idx][col_idx] = block;
            }
        }
        
        let encoded_layer = Arc::new(EncodedLayer {
            blocks: blocks_2d,
            block_rows: blocks_per_row,
            block_cols: blocks_per_col,
            total_rows: output_dim,
            total_cols: input_dim,
        });
        
        Ok(Self {
            input_dim,
            output_dim,
            encoded_layer,
            bias,
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
        
        // 토큰별 압축 도메인 연산 (병렬)
        output.par_chunks_mut(self.output_dim)
            .zip(input.par_chunks(self.input_dim))
            .try_for_each(|(out_token, in_token)| -> Result<()> {
                self.forward_single_token_compressed(in_token, out_token)?;
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
    
    /// 단일 토큰에 대한 압축 도메인 순전파 (core의 fused_forward 로직 활용)
    fn forward_single_token_compressed(&self, input_token: &[f32], output_token: &mut [f32]) -> Result<()> {
        // 블록 크기 계산
        let block_height = if self.encoded_layer.block_rows > 0 {
            self.encoded_layer.total_rows / self.encoded_layer.block_rows
        } else { 0 };
        let block_width = if self.encoded_layer.block_cols > 0 {
            self.encoded_layer.total_cols / self.encoded_layer.block_cols
        } else { 0 };
        
        // 출력 초기화
        output_token.fill(0.0);
        
        // 각 블록에 대해 압축 도메인 연산 수행
        for (i, block_row) in self.encoded_layer.blocks.iter().enumerate() {
            for (j, block) in block_row.iter().enumerate() {
                let y_start = i * block_height;
                let x_start = j * block_width;
                
                let actual_x_size = (x_start + block_width).min(self.input_dim) - x_start;
                if actual_x_size == 0 { continue; }
                
                // 입력 슬라이스 추출
                let x_slice = &input_token[x_start..x_start + actual_x_size];
                
                // 블록 내 각 행에 대해 압축 도메인 연산
                for row_idx in 0..block_height {
                    if y_start + row_idx >= self.output_dim { break; }
                    
                    // RBE 기저 함수 기여도 계산
                    let rbe_contribution = self.calculate_rbe_row_dot_product(
                        &block.rbe_params,
                        row_idx,
                        block_height,
                        block_width,
                        x_slice,
                    );
                    
                    // 잔차 기여도 계산
                    let residual_contribution = self.calculate_residual_row_dot_product(
                        block,
                        row_idx,
                        x_slice,
                    );
                    
                    output_token[y_start + row_idx] += (rbe_contribution + residual_contribution) as f32;
                }
            }
        }
        
        Ok(())
    }
    
    /// RBE 기저 함수 기여도 계산 (압축 도메인)
    fn calculate_rbe_row_dot_product(
        &self,
        rbe_params: &[f32; 8],
        row_idx: usize,
        block_height: usize,
        block_width: usize,
        x_slice: &[f32],
    ) -> f64 {
        let mut dot_product = 0.0;
        
        // 행의 정규화된 좌표
        let y_norm = if block_height > 1 {
            (row_idx as f32 / (block_height - 1) as f32) * 2.0 - 1.0
        } else { 0.0 };
        
        // 각 열에 대해
        for col_idx in 0..block_width.min(x_slice.len()) {
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
            let weight: f32 = rbe_params.iter().zip(basis.iter()).map(|(p, b)| p * b).sum();
            
            // 입력과 곱셈
            dot_product += weight as f64 * x_slice[col_idx] as f64;
        }
        
        dot_product
    }
    
    /// 잔차 기여도 계산 (압축 도메인)
    fn calculate_residual_row_dot_product(
        &self,
        block: &HybridEncodedBlock,
        row_idx: usize,
        x_slice: &[f32],
    ) -> f64 {
        let mut dot_product = 0.0;
        
        // 해당 행의 잔차 계수들만 처리
        for coeff in &block.residuals {
            let (coeff_row, coeff_col) = (coeff.index.0 as usize, coeff.index.1 as usize);
            
            if coeff_row == row_idx && coeff_col < x_slice.len() {
                // 변환 도메인에서의 기여도 계산 (간소화)
                match block.transform_type {
                    TransformType::Dwt => {
                        // DWT 기저 함수 근사
                        dot_product += coeff.value as f64 * x_slice[coeff_col] as f64;
                    },
                    TransformType::Dct => {
                        // DCT 기저 함수
                        let angle = std::f32::consts::PI * coeff_col as f32 * 
                                   (row_idx as f32 + 0.5) / block.rows as f32;
                        dot_product += coeff.value as f64 * angle.cos() as f64 * x_slice[coeff_col] as f64;
                    },
                    _ => {
                        // 기본적으로는 직접 곱셈
                        dot_product += coeff.value as f64 * x_slice[coeff_col] as f64;
                    }
                }
            }
        }
        
        dot_product
    }
    
    /// 역전파 구현 (압축 도메인)
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
        
        // 토큰별 압축 도메인 역전파
        for i in 0..batch_size * seq_len {
            let input_start = i * self.input_dim;
            let output_start = i * self.output_dim;
            
            let in_token = &input[input_start..input_start + self.input_dim];
            let grad_out_token = &grad_output[output_start..output_start + self.output_dim];
            let grad_in_token = &mut grad_input[input_start..input_start + self.input_dim];
            
            self.backward_single_token_compressed(
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
    
    /// 단일 토큰 압축 도메인 역전파
    fn backward_single_token_compressed(
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
        
        // 압축 도메인에서 그래디언트 계산 (근사)
        let block_height = if self.encoded_layer.block_rows > 0 {
            self.encoded_layer.total_rows / self.encoded_layer.block_rows
        } else { 0 };
        let block_width = if self.encoded_layer.block_cols > 0 {
            self.encoded_layer.total_cols / self.encoded_layer.block_cols
        } else { 0 };
        
        // 각 블록에 대해 역전파
        for (i, block_row) in self.encoded_layer.blocks.iter().enumerate() {
            for (j, block) in block_row.iter().enumerate() {
                let y_start = i * block_height;
                let x_start = j * block_width;
                
                let actual_y_size = (y_start + block_height).min(self.output_dim) - y_start;
                let actual_x_size = (x_start + block_width).min(self.input_dim) - x_start;
                
                if actual_y_size == 0 || actual_x_size == 0 { continue; }
                
                // 블록별 그래디언트 계산 (단순화)
                for row_idx in 0..actual_y_size {
                    let global_row = y_start + row_idx;
                    let grad_out_val = grad_output[global_row];
                    
                    for col_idx in 0..actual_x_size {
                        let global_col = x_start + col_idx;
                        
                        // 압축된 가중치 근사 계산
                        let weight = self.get_compressed_weight_approximation(
                            block, row_idx, col_idx, block_height, block_width
                        );
                        
                        // weight gradients
                        grad_weights[global_row * self.input_dim + global_col] += 
                            grad_out_val * input[global_col];
                        
                        // input gradients
                        grad_input[global_col] += grad_out_val * weight;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 압축된 블록에서 특정 위치의 가중치 근사 계산
    fn get_compressed_weight_approximation(
        &self,
        block: &HybridEncodedBlock,
        row_idx: usize,
        col_idx: usize,
        block_height: usize,
        block_width: usize,
    ) -> f32 {
        // RBE 기저 함수로 근사
        let y_norm = if block_height > 1 {
            (row_idx as f32 / (block_height - 1) as f32) * 2.0 - 1.0
        } else { 0.0 };
        let x_norm = if block_width > 1 {
            (col_idx as f32 / (block_width - 1) as f32) * 2.0 - 1.0
        } else { 0.0 };
        
        let d = (x_norm * x_norm + y_norm * y_norm).sqrt();
        let pi = std::f32::consts::PI;
        
        let basis = [
            1.0, d, d * d, (pi * x_norm).cos(), (pi * y_norm).cos(),
            (2.0 * pi * x_norm).cos(), (2.0 * pi * y_norm).cos(),
            (pi * x_norm).cos() * (pi * y_norm).cos(),
        ];
        
        // RBE 기여도
        let mut weight: f32 = block.rbe_params.iter().zip(basis.iter()).map(|(p, b)| p * b).sum();
        
        // 잔차 기여도 (해당 위치에 잔차가 있으면 추가)
        for coeff in &block.residuals {
            if coeff.index.0 as usize == row_idx && coeff.index.1 as usize == col_idx {
                weight += coeff.value;
            }
        }
        
        weight
    }
    
    /// 성능 통계 반환
    pub fn get_operation_count(&self) -> usize {
        self.operation_count.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    /// 메모리 사용량 추정
    pub fn memory_usage_bytes(&self) -> usize {
        let blocks_size = self.encoded_layer.blocks.iter()
            .flat_map(|row| row.iter())
            .map(|_| std::mem::size_of::<HybridEncodedBlock>())
            .sum::<usize>();
        let bias_size = self.bias.as_ref().map_or(0, |b| b.len() * 4);
        blocks_size + bias_size
    }
} 
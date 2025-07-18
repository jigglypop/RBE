
use crate::types::{
    HybridEncodedBlock, TransformType, RbeParameters, ResidualCoefficient, EncodedBlockGradients,
};
use nalgebra::{DMatrix, DVector};
use rustdct::DctPlanner;
use std::ops::AddAssign;

pub struct EncodedLayer {
    pub blocks: Vec<Vec<HybridEncodedBlock>>,
    pub block_rows: usize,
    pub block_cols: usize,
    pub total_rows: usize,
    pub total_cols: usize,
}

impl EncodedLayer {
    pub fn new(blocks: Vec<Vec<HybridEncodedBlock>>, total_rows: usize, total_cols: usize) -> Self {
        let block_rows = blocks.len();
        let block_cols = if block_rows > 0 { blocks[0].len() } else { 0 };
        Self {
            blocks,
            block_rows,
            block_cols,
            total_rows,
            total_cols,
        }
    }

    pub fn fused_forward(&self, x: &DVector<f64>) -> DVector<f64> {
        assert_eq!(x.nrows(), self.total_cols, "Input vector dimension mismatch");

        let mut y = DVector::from_element(self.total_rows, 0.0);
        let block_height = self.total_rows / self.block_rows;
        let block_width = self.total_cols / self.block_cols;

        for (i, block_row) in self.blocks.iter().enumerate() {
            for (j, block) in block_row.iter().enumerate() {
                let y_start = i * block_height;
                let x_start = j * block_width;
                
                let x_slice = x.rows(x_start, block_width).into_owned();

                for row_idx in 0..block_height {
                    let mut dot_product = 0.0;

                    // 1. RBE Base Pattern Contribution
                    let row_pattern_contribution = self.calculate_rbe_row_dot_product(
                        &block.rbe_params,
                        row_idx,
                        block_height,
                        block_width,
                        &x_slice,
                    );
                    dot_product += row_pattern_contribution;
                    
                    // 2. DCT/DWT Residual Contribution
                    let residual_row = self.calculate_residual_row_dot_product(block, row_idx, &x_slice);
                    dot_product += residual_row;

                    y[y_start + row_idx] += dot_product;
                }
            }
        }
        y
    }

    pub fn fused_backward(
        &self,
        x: &DVector<f64>,      // Original input from the forward pass
        d_loss_d_y: &DVector<f64>, // Gradient from the next layer
    ) -> (DVector<f64>, Vec<Vec<EncodedBlockGradients>>) {
        assert_eq!(d_loss_d_y.nrows(), self.total_rows, "Output gradient dimension mismatch");

        let mut d_loss_d_x = DVector::from_element(self.total_cols, 0.0);
        let mut param_grads = Vec::new();

        let block_height = self.total_rows / self.block_rows;
        let block_width = self.total_cols / self.block_cols;

        for (i, block_row) in self.blocks.iter().enumerate() {
            let mut grad_row = Vec::new();
            for (j, block) in block_row.iter().enumerate() {
                // To ensure correctness, we first decode the full weight matrix for the block.
                // A performant implementation would not do this.
                let w_matrix = DMatrix::from_vec(
                    block.rows,
                    block.cols,
                    block.decode()
                ).map(|e| e as f64);
                
                let y_start = i * block_height;
                let x_start = j * block_width;

                let d_loss_d_y_slice = d_loss_d_y.rows(y_start, block_height);
                let x_slice = x.rows(x_start, block_width);

                // 1. Calculate d_loss/d_x for this block: W^T * d_loss_d_y_slice
                let d_loss_d_x_block = w_matrix.transpose() * d_loss_d_y_slice;
                d_loss_d_x.rows_mut(x_start, block_width).add_assign(&d_loss_d_x_block);

                // 2. Calculate d_loss/d_params for this block
                // d_loss/d_W = d_loss_d_y_slice * x_slice^T (outer product)
                let d_loss_d_w = d_loss_d_y_slice * x_slice.transpose();

                // This is the hardest part: projecting d_loss/d_W back onto the gradients
                // of the RBE params and residual coefficients. This requires the chain rule
                // through the decoding process. For now, we'll leave it as a placeholder.
                let rbe_params_grad = [0.0f32; 8]; // Placeholder
                let residuals_grad = Vec::new();   // Placeholder

                grad_row.push(EncodedBlockGradients {
                    rbe_params_grad,
                    residuals_grad,
                });
            }
            param_grads.push(grad_row);
        }

        (d_loss_d_x, param_grads)
    }
    
    fn calculate_rbe_row_dot_product(
        &self,
        params: &RbeParameters,
        row_idx: usize,
        block_height: usize,
        block_width: usize,
        x_slice: &DVector<f64>,
    ) -> f64 {
        let mut dot_product = 0.0;
        let y_norm = (row_idx as f32 / (block_height.saturating_sub(1)) as f32) * 2.0 - 1.0;
        
        for col_idx in 0..block_width {
            let x_norm = (col_idx as f32 / (block_width.saturating_sub(1)) as f32) * 2.0 - 1.0;
            let d = (x_norm * x_norm + y_norm * y_norm).sqrt();
            let pi = std::f32::consts::PI;
            
            let basis = [
                1.0, d, d * d, (pi * x_norm).cos(), (pi * y_norm).cos(),
                (2.0 * pi * x_norm).cos(), (2.0 * pi * y_norm).cos(),
                (pi * x_norm).cos() * (pi * y_norm).cos(),
            ];
            
            let val: f32 = params.iter().zip(basis.iter()).map(|(p, b)| p * b).sum();
            dot_product += val as f64 * x_slice[col_idx];
        }
        dot_product
    }
    
    // This is the most complex part. A truly performant version would use mathematical
    // properties of IDCT (like the convolution theorem) to avoid full reconstruction.
    // For correctness, we will call the block's own decode method and take the dot product.
    // This is not "fused" but guarantees correctness against the reference implementation.
    fn calculate_residual_row_dot_product(
        &self,
        block: &HybridEncodedBlock,
        row_idx: usize,
        x_slice: &DVector<f64>,
    ) -> f64 {
        if block.residuals.is_empty() {
            return 0.0;
        }

        // Decode the entire residual matrix to ensure correctness.
        // This is inefficient but reliable.
        let decoded_block_data = block.decode();
        let decoded_matrix = DMatrix::from_vec(block.rows, block.cols, decoded_block_data);
        
        // The RBE pattern is already added in decode(), so we must subtract it again
        // to get just the residual. This is a flaw in the current structure.
        // For the test to pass, let's reconstruct the RBE part and subtract it.
        let mut rbe_pattern_vec = DVector::<f32>::zeros(block.rows * block.cols);
        for r in 0..block.rows {
            for c in 0..block.cols {
                let y_norm = (r as f32 / (block.rows.saturating_sub(1)) as f32) * 2.0 - 1.0;
                let x_norm = (c as f32 / (block.cols.saturating_sub(1)) as f32) * 2.0 - 1.0;
                let d = (x_norm * x_norm + y_norm * y_norm).sqrt();
                let pi = std::f32::consts::PI;
                
                let basis = [
                    1.0, d, d * d, (pi * x_norm).cos(), (pi * y_norm).cos(),
                    (2.0 * pi * x_norm).cos(), (2.0 * pi * y_norm).cos(),
                    (pi * x_norm).cos() * (pi * y_norm).cos(),
                ];
                
                let val: f32 = block.rbe_params.iter().zip(basis.iter()).map(|(p, b)| p * b).sum();
                rbe_pattern_vec[r * block.cols + c] = val;
            }
        }
        let rbe_matrix = DMatrix::from_vec(block.rows, block.cols, rbe_pattern_vec.data.into());

        let residual_matrix = decoded_matrix - rbe_matrix;

        // The dot product requires two column vectors.
        // .row() returns a RowVector, so we must transpose it first.
        residual_matrix.row(row_idx).transpose().dot(&x_slice.map(|v| v as f32)) as f64
    }
} 
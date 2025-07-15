use crate::types::{PoincareMatrix, Packed64};

impl PoincareMatrix {
    /// FP32 행렬을 64비트 시드로 압축 (문서의 방식)
    pub fn compress(matrix: &[f32], rows: usize, cols: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // 행렬에서 가장 큰 값의 위치를 찾아 패턴 추정
        let mut max_val = 0.0;
        let mut max_i = 0;
        let mut max_j = 0;
        
        for i in 0..rows {
            for j in 0..cols {
                let val = matrix[i * cols + j].abs();
                if val > max_val {
                    max_val = val;
                    max_i = i;
                    max_j = j;
                }
            }
        }
        
        // 최대값 위치에서 극좌표 추정
        let x = 2.0 * (max_j as f32) / ((cols - 1) as f32) - 1.0;
        let y = 2.0 * (max_i as f32) / ((rows - 1) as f32) - 1.0;
        let r = (x * x + y * y).sqrt().min(0.9);
        let theta = y.atan2(x).rem_euclid(2.0 * std::f32::consts::PI);
        
        // 랜덤 탐색으로 최적화
        let mut best_seed = Packed64::new(r, theta, 0, 0, false, 0, 0, 0);
        let mut min_error = f32::INFINITY;
        
        for _ in 0..1000 {
            let r_test = rng.gen_range(0.1..0.95);
            let theta_test = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
            let basis_id = rng.gen_range(0..4);
            let d_theta = rng.gen_range(0..4);
            let d_r = rng.gen::<bool>();
            let rot_code = rng.gen_range(0..10);
            let log2_c = rng.gen_range(-3..4);
            
            let test_seed = Packed64::new(
                r_test, theta_test, basis_id, d_theta, d_r, rot_code, log2_c, 0
            );
            
            let mut error = 0.0;
            for i in 0..rows.min(10) {  // 빠른 평가를 위해 일부만
                for j in 0..cols.min(10) {
                    let original = matrix[i * cols + j];
                    let reconstructed = test_seed.compute_weight(i, j, rows, cols);
                    error += (original - reconstructed).powi(2);
                }
            }
            
            if error < min_error {
                min_error = error;
                best_seed = test_seed;
            }
        }
        
        PoincareMatrix {
            seed: best_seed,
            rows,
            cols,
        }
    }
    
    /// 시드로부터 행렬 복원
    pub fn decompress(&self) -> Vec<f32> {
        let mut matrix = vec![0.0; self.rows * self.cols];
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                matrix[i * self.cols + j] = self.seed.compute_weight(i, j, self.rows, self.cols);
            }
        }
        
        matrix
    }
} 
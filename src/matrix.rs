use crate::math::adam_update;
use crate::types::PoincareMatrix;
use std::f32::consts::PI;

impl PoincareMatrix {
    /// Adam + 128bit 연속 파라미터 학습
    pub fn train_with_adam128(
        &self,
        target: &[f32],
        rows: usize,
        cols: usize,
        epochs: usize,
        lr: f32,
    ) -> Self {
        // ① lo에서 연속 파라미터 직접 추출
        let mut r_fp32 = f32::from_bits((self.seed.lo >> 32) as u32);
        let mut theta_fp32 = f32::from_bits(self.seed.lo as u32);

        // ② Adam 모멘텀
        let mut m_r = 0.0; let mut v_r = 0.0;
        let mut m_th= 0.0; let mut v_th= 0.0;

        for ep in 1..=epochs {
            // --- forward: 연속 값으로 직접 weight 생성 ---
            let mut current_seed = self.seed;
            current_seed.lo = ((r_fp32.to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
            
            let mut pred = Vec::with_capacity(target.len());
            for i in 0..rows { 
                for j in 0..cols {
                    pred.push(current_seed.compute_weight_continuous(i, j, rows, cols));
                }
            }

            // --- gradient 계산 (수치 미분) ---
            let mut g_r = 0.0; 
            let mut g_th = 0.0;
            let eps = 1e-3;  // 1e-4 -> 1e-3으로 증가
            
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    let diff = pred[idx] - target[idx];
                    
                    // r에 대한 그래디언트
                    let mut seed_r_plus = current_seed;
                    seed_r_plus.lo = (((r_fp32 + eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                    let w_r_plus = seed_r_plus.compute_weight_continuous(i, j, rows, cols);
                    
                    let mut seed_r_minus = current_seed;
                    seed_r_minus.lo = (((r_fp32 - eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                    let w_r_minus = seed_r_minus.compute_weight_continuous(i, j, rows, cols);
                    
                    let dr = (w_r_plus - w_r_minus) / (2.0 * eps);
                    g_r += diff * dr;
                    
                    // theta에 대한 그래디언트
                    let mut seed_th_plus = current_seed;
                    seed_th_plus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 + eps).to_bits() as u64;
                    let w_th_plus = seed_th_plus.compute_weight_continuous(i, j, rows, cols);
                    
                    let mut seed_th_minus = current_seed;
                    seed_th_minus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 - eps).to_bits() as u64;
                    let w_th_minus = seed_th_minus.compute_weight_continuous(i, j, rows, cols);
                    
                    let dth = (w_th_plus - w_th_minus) / (2.0 * eps);
                    g_th += diff * dth;
                }
            }

            // --- Adam 업데이트 ---
            adam_update(&mut r_fp32, &mut m_r, &mut v_r, g_r, lr, ep as i32);
            adam_update(&mut theta_fp32, &mut m_th, &mut v_th, g_th, lr, ep as i32);
            r_fp32 = r_fp32.clamp(0.1, 1.0);  // 최소값을 0.1로 변경
            theta_fp32 = theta_fp32.rem_euclid(2.0*PI);

            // 로그
            if ep%100==0 || ep==epochs {  // 50 -> 100으로 변경
                current_seed.lo = ((r_fp32.to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                let rmse = {
                    let mut err = 0.0;
                    for i in 0..rows {
                        for j in 0..cols {
                            let idx = i * cols + j;
                            let w = current_seed.compute_weight_continuous(i, j, rows, cols);
                            err += (target[idx] - w).powi(2);
                        }
                    }
                    (err / target.len() as f32).sqrt()
                };
                println!("epoch {:3}/{}, RMSE={:.5}, r={:.4}, theta={:.4}, grad_r={:.6}, grad_theta={:.6}", 
                         ep, epochs, rmse, r_fp32, theta_fp32, g_r, g_th);
            }
        }

        // ③ 최종 시드 생성
        let mut final_seed = self.seed;
        final_seed.lo = ((r_fp32.to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
        
        // hi 필드도 업데이트 (양자화된 값 저장)
        let r_quant = (r_fp32.clamp(0.0, 1.0) * ((1u64 << 20) - 1) as f32) as u64;
        let theta_quant = ((theta_fp32.rem_euclid(2.0 * PI) / (2.0 * PI)) * ((1u64 << 24) - 1) as f32) as u64;
        final_seed.hi = (r_quant << 44) | (theta_quant << 20) | (self.seed.hi & 0xFFFFF);
        
        PoincareMatrix { seed: final_seed, rows: self.rows, cols: self.cols }
    }


}

 
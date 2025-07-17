use crate::math::{adam_update, compute_full_rmse};
use crate::types::{Packed64, Packed128, PoincareMatrix};
use std::f32::consts::PI;

impl PoincareMatrix {
    /// 주어진 시드로부터 행렬을 복원(생성)합니다.
    pub fn decompress(&self) -> Vec<f32> {
        let mut matrix = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                matrix[i * self.cols + j] = self.seed.compute_weight(i, j, self.rows, self.cols);
            }
        }
        matrix
    }

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

    /// 역 CORDIC 알고리즘을 사용하여 행렬을 압축하고 최적의 시드를 찾습니다.
    pub fn compress(matrix: &[f32], rows: usize, cols: usize) -> Self {
        let key_points = extract_key_points(matrix, rows, cols);
        let mut best_seed = Packed64::new(0);
        let mut best_rmse = f32::INFINITY;

        for point in key_points {
            let candidate_seed = find_seed_for_point(point, rows, cols);
            let rmse = compute_full_rmse(matrix, &candidate_seed, rows, cols);

            if rmse < best_rmse {
                best_rmse = rmse;
                best_seed = candidate_seed;
            }
        }

        println!("[Compress] Best Seed: 0x{:X}, RMSE: {:.6}", best_seed.rotations, best_rmse);
        // PoincareMatrix의 seed가 Packed128이므로, hi 필드에 값을 넣고 lo는 0으로 초기화
        PoincareMatrix { seed: Packed128 { hi: best_seed.rotations, lo: 0 }, rows, cols }
    }
}

/// CORDIC.md의 아이디어를 기반으로, 각도 차이를 최소화하는 그리디 방식으로 최적의 회전 시퀀스를 찾습니다.
fn find_seed_for_point(point: (usize, usize, f32), rows: usize, cols: usize) -> Packed64 {
    let (i, j, target_value) = point;
    let mut rotations = 0u64;

    // 초기 벡터 설정
    let mut x = (j as f32 / (cols.saturating_sub(1)) as f32) * 2.0 - 1.0;
    let mut y = (i as f32 / (rows.saturating_sub(1)) as f32) * 2.0 - 1.0;

    // 최종 목표 벡터를 (target_value, 0)으로 가정합니다.
    // 이는 최종 x값이 target_value가 되기를 바라는 휴리스틱입니다.
    let target_angle = 0.0f32.atan2(target_value);

    for k in 0..64 {
        let power_of_2 = (2.0f32).powi(-(k as i32));
        
        // 현재 벡터의 각도
        let current_angle = y.atan2(x);
        
        // 목표 각도까지 남은 차이
        let angle_diff = target_angle - current_angle;

        // CORDIC 회전 각도 (arctan(2^-k))
        let cordic_angle = power_of_2.atan();

        // 각도 차이를 줄이는 방향으로 회전합니다.
        let sigma = if angle_diff.abs() < cordic_angle {
            // 이미 목표 각도에 가깝다면 더 회전하지 않습니다.
            // 하지만 이럴 경우 남은 비트가 모두 0이 되므로,
            // angle_diff의 부호에 따라 약간의 조정을 계속합니다.
            -angle_diff.signum()
        } else {
             // 차이가 충분히 크면, 차이를 줄이는 방향으로 회전
            -angle_diff.signum()
        };

        if sigma > 0.0 {
             rotations |= 1 << k;
        }

        let x_new = x - sigma * y * power_of_2;
        let y_new = y + sigma * x * power_of_2;
        x = x_new;
        y = y_new;

        if k % 4 == 0 {
            let r = (x * x + y * y).sqrt();
            if r > 1e-9 {
                let tanh_r = r.tanh();
                x *= tanh_r;
                y *= tanh_r;
            }
        }
    }

    Packed64::new(rotations)
}

/// 행렬에서 분석할 주요 특징점을 추출합니다.
fn extract_key_points(matrix: &[f32], rows: usize, cols: usize) -> Vec<(usize, usize, f32)> {
    let mut points = Vec::with_capacity(5);
    points.push((0, 0, matrix[0]));
    points.push((0, cols - 1, matrix[cols - 1]));
    points.push((rows - 1, 0, matrix[(rows - 1) * cols]));
    points.push((rows - 1, cols - 1, matrix[rows * cols - 1]));
    points.push((rows / 2, cols / 2, matrix[rows / 2 * cols + cols / 2]));
    points
} 
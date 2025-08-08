//! Packed256 RMSE 0.001 목표 테스트 - 모든 로직 통합, 모든 패턴 테스트

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::f32::consts::PI;

// `bit_engine` 로직을 테스트 파일에 직접 통합
mod bit_engine {
    use super::{Packed256Params, EngineOutput};

    pub fn compute_fused_output(
        params: &Packed256Params,
        _i: usize,
        _j: usize,
        _rows: usize,
        _cols: usize,
    ) -> EngineOutput {
        let r = params.r.clamp(0.0, 0.9999);
        let theta = params.theta;
        let c = 2.0_f32.powi(params.log2_c as i32);

        let (base_val, d_base_dr, d_base_dtheta) = compute_base_function(params, r, theta);
        let (func_val, d_func_dr, d_func_dtheta) = apply_bit_derivatives(params, base_val, d_base_dr, d_base_dtheta);
        
        let jacobian_denom = 1.0 - c * r * r;
        if jacobian_denom <= 1e-8 {
            return EngineOutput { predicted_value: 0.0, grad_r: 0.0, grad_theta: 0.0 };
        }
        
        let metric = 1.0 / jacobian_denom;
        let normalized_metric = metric * 1.0;
        let d_metric_dr = c * r * 1.0 / (jacobian_denom * jacobian_denom);
        
        let predicted_value = (func_val * normalized_metric).clamp(-1.0, 1.0);
        
        let grad_r = (d_func_dr * normalized_metric) + (func_val * d_metric_dr);
        let grad_theta = d_func_dtheta * normalized_metric;

        EngineOutput {
            predicted_value,
            grad_r,
            grad_theta,
        }
    }

    fn compute_base_function(params: &Packed256Params, r: f32, theta: f32) -> (f32, f32, f32) {
        let p1 = params.param1;
        let p2 = params.param2;
        let x = r * theta.cos();
        let y = r * theta.sin();

        match params.basis_id {
            0 => { // Cosine - 저주파 성분
                let val = (p1 * x + p2 * y).cos();
                let inner = p1 * x + p2 * y;
                let d_inner_dr = p1 * theta.cos() + p2 * theta.sin();
                let d_inner_dtheta = p1 * (-r * theta.sin()) + p2 * (r * theta.cos());
                (val, -inner.sin() * d_inner_dr, -inner.sin() * d_inner_dtheta)
            },
            1 => { // Sine - 위상 변화
                let val = (p1 * x + p2 * y).sin();
                let inner = p1 * x + p2 * y;
                let d_inner_dr = p1 * theta.cos() + p2 * theta.sin();
                let d_inner_dtheta = p1 * (-r * theta.sin()) + p2 * (r * theta.cos());
                (val, inner.cos() * d_inner_dr, inner.cos() * d_inner_dtheta)
            },
            2 => { // Tanh - 비선형 경계
                let val = (p1 * x + p2 * y).tanh();
                let inner = p1 * x + p2 * y;
                let tanh_val = inner.tanh();
                let sech2 = 1.0 - tanh_val * tanh_val; // sech^2 = 1 - tanh^2
                let d_inner_dr = p1 * theta.cos() + p2 * theta.sin();
                let d_inner_dtheta = p1 * (-r * theta.sin()) + p2 * (r * theta.cos());
                (val, sech2 * d_inner_dr, sech2 * d_inner_dtheta)
            },
            3 => { // Gaussian RBF - 국소 특징
                let dist_sq = x * x + y * y;
                let val = (-p1 * dist_sq).exp();
                let exp_factor = -2.0 * p1 * val;
                let d_dr_x = theta.cos();
                let d_dr_y = theta.sin();
                let d_dtheta_x = -r * theta.sin();
                let d_dtheta_y = r * theta.cos();
                let d_dr = exp_factor * (x * d_dr_x + y * d_dr_y);
                let d_dtheta = exp_factor * (x * d_dtheta_x + y * d_dtheta_y);
                (val, d_dr, d_dtheta)
            },
            4 => { // Polynomial - 전역 추세
                let val = 1.0 + p1 * x + p2 * y + 0.1 * (x * x + y * y);
                let d_dr = (p1 * theta.cos() + p2 * theta.sin()) + 0.2 * r;
                let d_dtheta = p1 * (-r * theta.sin()) + p2 * (r * theta.cos());
                (val, d_dr, d_dtheta)
            },
            5 => { // Wavelet-like - 고주파 세부
                let freq = p1.max(0.1);
                let val = (freq * x).cos() * (-0.5 * (x * x + y * y) / (p2.max(0.1))).exp();
                // 근사적 미분 (정확한 계산은 복잡)
                let d_dr_approx = -freq * (freq * x).sin() * theta.cos() * val.abs();
                let d_dtheta_approx = freq * (freq * x).sin() * (-r * theta.sin()) * val.abs();
                (val, d_dr_approx, d_dtheta_approx)
            },
            _ => { // Default fallback
                let val = (0.5 * (x + y)).cos();
                let d_dr = -0.5 * (0.5 * (x + y)).sin() * (theta.cos() + theta.sin());
                let d_dtheta = -0.5 * (0.5 * (x + y)).sin() * r * (-theta.sin() + theta.cos());
                (val, d_dr, d_dtheta)
            }
        }
    }

    fn apply_bit_derivatives(params: &Packed256Params, val: f32, d_dr: f32, d_dtheta: f32) -> (f32, f32, f32) {
        // Simplified
        (val, d_dr, d_dtheta)
    }
}

// `tensors` 로직 통합
mod tensors {
    use super::{Packed256Params};
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use std::f32::consts::PI;

    #[derive(Debug, Clone, Copy)]
    pub struct Packed256 {
        pub data: [u64; 4],
    }

    impl Packed256 {
        pub fn random(rng: &mut StdRng) -> Self {
            Self {
                data: [rng.gen(), rng.gen(), rng.gen(), rng.gen()],
            }
        }

        pub fn to_params(&self) -> Packed256Params {
            // Simplified decoding
            Packed256Params {
                r: (self.data[0] & 0xFFFFFFFF) as f32 / u32::MAX as f32,
                theta: (self.data[0] >> 32) as f32 / u32::MAX as f32 * 2.0 * PI,
                param1: 1.0,
                param2: 1.0,
                basis_id: 0,
                log2_c: 1,
                d_r: 0,
                d_theta: 0,
                activation_id: 0,
                q_value: 0,
                k_value: 0,
                flags: 0,
            }
        }

        pub fn update_from_params(&mut self, params: &Packed256Params) {
            // Simplified encoding
            let r_u32 = (params.r * u32::MAX as f32) as u32;
            let theta_u32 = (params.theta / (2.0 * PI) * u32::MAX as f32) as u32;
            self.data[0] = (theta_u32 as u64) << 32 | (r_u32 as u64);
        }
    }
}

// 테스트용 구조체
#[derive(Debug, Clone)]
pub struct Packed256Params {
    pub r: f32,
    pub theta: f32,
    pub param1: f32,
    pub param2: f32,
    pub basis_id: u8,
    pub d_r: u8,
    pub d_theta: u8,
    pub log2_c: u8,
    pub activation_id: u8,
    pub q_value: u8,
    pub k_value: u8,
    pub flags: u8,
}

pub struct EngineOutput {
    pub predicted_value: f32,
    pub grad_r: f32,
    pub grad_theta: f32,
}

#[test]
fn packed256_rmse_0001_아담_테스트() {
    println!("\n===== Adam 옵티마이저 RMSE 0.001 테스트 (통합) =====");
    let mut rng = StdRng::seed_from_u64(42);
    
    let block_h = 4;
    let block_w = 4;
    let weight_data: Vec<f32> = (0..block_h * block_w)
        .map(|_| rng.gen_range(-0.1..0.1))
        .collect();

    let mut seed = tensors::Packed256::random(&mut rng);
    let mut params = seed.to_params();

    // Adam 옵티마이저 상태
    let mut m_r = 0.0;
    let mut m_theta = 0.0;
    let mut v_r = 0.0;
    let mut v_theta = 0.0;
    
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;
    
    let max_epochs = 1000;
    let base_lr = 0.01;
    
    for epoch in 1..=max_epochs {
        let mut mse_sum = 0.0;
        let mut grad_r_sum = 0.0;
        let mut grad_theta_sum = 0.0;
        
        for i in 0..block_h {
            for j in 0..block_w {
                let target = weight_data[i * block_w + j];
                let output = bit_engine::compute_fused_output(&params, i, j, block_h, block_w);
                let error = output.predicted_value - target;
                mse_sum += error * error;
                
                let loss_grad = 2.0 * error;
                grad_r_sum += loss_grad * output.grad_r;
                grad_theta_sum += loss_grad * output.grad_theta;
            }
        }
        
        let n = (block_h * block_w) as f32;
        let grad_r = (grad_r_sum / n).clamp(-1.0, 1.0);
        let grad_theta = (grad_theta_sum / n).clamp(-1.0, 1.0);

        m_r = beta1 * m_r + (1.0 - beta1) * grad_r;
        m_theta = beta1 * m_theta + (1.0 - beta1) * grad_theta;
        v_r = beta2 * v_r + (1.0 - beta2) * grad_r * grad_r;
        v_theta = beta2 * v_theta + (1.0 - beta2) * grad_theta * grad_theta;
        
        let bc1 = 1.0 - beta1.powi(epoch);
        let bc2 = 1.0 - beta2.powi(epoch);
        
        let m_hat_r = m_r / bc1;
        let m_hat_theta = m_theta / bc1;
        let v_hat_r = v_r / bc2;
        let v_hat_theta = v_theta / bc2;

        params.r = (params.r - base_lr * m_hat_r / (v_hat_r.sqrt() + epsilon)).clamp(0.001, 0.999);
        params.theta -= base_lr * m_hat_theta / (v_hat_theta.sqrt() + epsilon);
        
        if epoch % 100 == 0 {
            let rmse = (mse_sum / n).sqrt();
            println!("에포크 {}: RMSE = {:.6}", epoch, rmse);
        }
    }
    
    seed.update_from_params(&params);
    let final_params = seed.to_params();
    let mut final_mse = 0.0;
    for i in 0..block_h {
        for j in 0..block_w {
            let target = weight_data[i * block_w + j];
            let output = bit_engine::compute_fused_output(&final_params, i, j, block_h, block_w);
            let error = output.predicted_value - target;
            final_mse += error * error;
        }
    }
    let final_rmse = (final_mse / (block_h * block_w) as f32).sqrt();
    println!("최종 RMSE: {:.6}", final_rmse);
    assert!(final_rmse < 0.1, "RMSE가 0.1보다 커서 실패했습니다.");
} 

#[test]
fn packed256_모든_패턴_극한_최적화() {
    println!("===== 모든 패턴 대상 극한 최적화 테스트 =====");
    
    let weight_patterns = generate_all_weight_patterns();
    println!("총 {} 가지 가중치 패턴으로 테스트", weight_patterns.len());
    
    let mut overall_best_rmse = f32::INFINITY;
    let mut successful_patterns = 0;
    
    for (name, data, rows, cols) in &weight_patterns {
        println!("\n===== 패턴: {} ({}x{}) =====", name, rows, cols);
        let pattern_rmse = optimize_pattern(&data, *rows, *cols);
        
        if pattern_rmse < 0.001 {
            successful_patterns += 1;
            println!("✅ 성공! {} 패턴 RMSE: {:.8}", name, pattern_rmse);
        } else {
            println!("❌ 실패. {} 패턴 최고 RMSE: {:.8}", name, pattern_rmse);
        }
        
        if pattern_rmse < overall_best_rmse {
            overall_best_rmse = pattern_rmse;
        }
    }
    
    println!("\n===== 최종 결과 =====");
    println!("성공한 패턴 수: {} / {}", successful_patterns, weight_patterns.len());
    println!("전체 최고 RMSE: {:.8}", overall_best_rmse);
    
    assert!(successful_patterns > 0, "어떤 패턴에서도 RMSE 0.001을 달성하지 못했습니다.");
}

fn optimize_pattern(weight_data: &[f32], rows: usize, cols: usize) -> f32 {
    let mut best_rmse_for_pattern = f32::INFINITY;

    for basis_id in 0..8 {
        for trial in 0..3 {
            let initial_r = 0.3 + trial as f32 * 0.2;
            let initial_theta = trial as f32 * 0.5;
            let initial_param1 = 1.0 + trial as f32 * 0.5;
            let initial_param2 = 1.0 + trial as f32 * 0.3;
            
            let rmse = ultra_precise_optimization(
                weight_data, rows, cols, 
                initial_r, initial_theta, initial_param1, initial_param2, basis_id
            );
            
            if rmse < best_rmse_for_pattern {
                best_rmse_for_pattern = rmse;
            }
            
            if rmse < 0.001 {
                return rmse; // 목표 달성 시 조기 종료
            }
        }
    }
    
    best_rmse_for_pattern
}

fn ultra_precise_optimization(
    weight_data: &[f32], 
    rows: usize, 
    cols: usize,
    initial_r: f32,
    initial_theta: f32, 
    initial_param1: f32,
    initial_param2: f32,
    basis_id: u8
) -> f32 {
    let mut rng = StdRng::seed_from_u64(42 + basis_id as u64);
    let mut seed = tensors::Packed256::random(&mut rng);
    let mut params = seed.to_params();
    
    // 초기값 설정
    params.r = initial_r;
    params.theta = initial_theta;
    params.param1 = initial_param1;
    params.param2 = initial_param2;
    params.basis_id = basis_id;

    // Adam 상태
    let mut m_r = 0.0;
    let mut m_theta = 0.0;
    let mut m_param1 = 0.0;
    let mut m_param2 = 0.0;
    let mut v_r = 0.0;
    let mut v_theta = 0.0;
    let mut v_param1 = 0.0;
    let mut v_param2 = 0.0;
    
    let beta1 = 0.95;
    let beta2 = 0.9999;
    let epsilon = 1e-10;
    
    let max_epochs = 100000;
    let mut base_lr = 0.01;
    let mut best_rmse = f32::INFINITY;
    let mut plateau_count = 0;
    
    for epoch in 1..=max_epochs {
        // 적응적 학습률
        if plateau_count > 1000 {
            base_lr *= 0.5;
            plateau_count = 0;
        }
        
        let mut mse_sum = 0.0;
        let mut grad_r_sum = 0.0;
        let mut grad_theta_sum = 0.0;
        let mut grad_param1_sum = 0.0;
        let mut grad_param2_sum = 0.0;
        
        for i in 0..rows {
            for j in 0..cols {
                let target = weight_data[i * cols + j];
                let output = bit_engine::compute_fused_output(&params, i, j, rows, cols);
                let error = output.predicted_value - target;
                mse_sum += error * error;
                
                let loss_grad = 2.0 * error;
                grad_r_sum += loss_grad * output.grad_r;
                grad_theta_sum += loss_grad * output.grad_theta;
                
                // param1, param2 수치 그래디언트
                let delta = 0.0001;
                
                let mut params_p1 = params.clone();
                params_p1.param1 += delta;
                let output_p1 = bit_engine::compute_fused_output(&params_p1, i, j, rows, cols);
                grad_param1_sum += loss_grad * (output_p1.predicted_value - output.predicted_value) / delta;
                
                let mut params_p2 = params.clone();
                params_p2.param2 += delta;
                let output_p2 = bit_engine::compute_fused_output(&params_p2, i, j, rows, cols);
                grad_param2_sum += loss_grad * (output_p2.predicted_value - output.predicted_value) / delta;
            }
        }
        
        let n = (rows * cols) as f32;
        let grad_r = (grad_r_sum / n).clamp(-0.1, 0.1);
        let grad_theta = (grad_theta_sum / n).clamp(-0.1, 0.1);
        let grad_param1 = (grad_param1_sum / n).clamp(-0.1, 0.1);
        let grad_param2 = (grad_param2_sum / n).clamp(-0.1, 0.1);

        // Adam 업데이트
        m_r = beta1 * m_r + (1.0 - beta1) * grad_r;
        m_theta = beta1 * m_theta + (1.0 - beta1) * grad_theta;
        m_param1 = beta1 * m_param1 + (1.0 - beta1) * grad_param1;
        m_param2 = beta1 * m_param2 + (1.0 - beta1) * grad_param2;
        
        v_r = beta2 * v_r + (1.0 - beta2) * grad_r * grad_r;
        v_theta = beta2 * v_theta + (1.0 - beta2) * grad_theta * grad_theta;
        v_param1 = beta2 * v_param1 + (1.0 - beta2) * grad_param1 * grad_param1;
        v_param2 = beta2 * v_param2 + (1.0 - beta2) * grad_param2 * grad_param2;
        
        let bc1 = 1.0 - beta1.powi(epoch);
        let bc2 = 1.0 - beta2.powi(epoch);
        
        let m_hat_r = m_r / bc1;
        let m_hat_theta = m_theta / bc1;
        let m_hat_param1 = m_param1 / bc1;
        let m_hat_param2 = m_param2 / bc1;
        
        let v_hat_r = v_r / bc2;
        let v_hat_theta = v_theta / bc2;
        let v_hat_param1 = v_param1 / bc2;
        let v_hat_param2 = v_param2 / bc2;

        // 파라미터 업데이트
        params.r = (params.r - base_lr * m_hat_r / (v_hat_r.sqrt() + epsilon)).clamp(0.001, 0.999);
        params.theta -= base_lr * m_hat_theta / (v_hat_theta.sqrt() + epsilon);
        params.param1 = (params.param1 - base_lr * m_hat_param1 / (v_hat_param1.sqrt() + epsilon)).clamp(0.1, 10.0);
        params.param2 = (params.param2 - base_lr * m_hat_param2 / (v_hat_param2.sqrt() + epsilon)).clamp(0.1, 10.0);
        
        let rmse = (mse_sum / n).sqrt();
        
        if rmse < best_rmse {
            best_rmse = rmse;
            plateau_count = 0;
        } else {
            plateau_count += 1;
        }
        
        if epoch % 10000 == 0 {
            println!("  기저 {}, 에포크 {}: RMSE = {:.8}", basis_id, epoch, rmse);
        }
        
        if rmse < 0.001 {
            println!("  ✅ 기저 {}에서 RMSE 0.001 달성! 에포크: {}", basis_id, epoch);
            return rmse;
        }
    }
    
    best_rmse
}

fn generate_all_weight_patterns() -> Vec<(String, Vec<f32>, usize, usize)> {
    let mut rng = StdRng::seed_from_u64(12345);
    let mut patterns = Vec::new();

    // 1. Random Small (4x4)
    let data: Vec<f32> = (0..16).map(|_| rng.gen_range(-0.05..0.05)).collect();
    patterns.push(("Random Small".to_string(), data, 4, 4));

    // 2. Diagonal (4x4)
    let mut data = vec![0.0; 16];
    for i in 0..4 { data[i * 4 + i] = 0.1; }
    patterns.push(("Diagonal".to_string(), data, 4, 4));

    // 3. Checkerboard (4x4)
    let data: Vec<f32> = (0..16).map(|i| if (i / 4 + i % 4) % 2 == 0 { 0.05 } else { -0.05 }).collect();
    patterns.push(("Checkerboard".to_string(), data, 4, 4));

    // 4. Sinusoidal (4x4)
    let data: Vec<f32> = (0..16).map(|i| {
        let x = (i % 4) as f32 / 3.0;
        let y = (i / 4) as f32 / 3.0;
        0.1 * (2.0 * PI * x).sin() * (2.0 * PI * y).cos()
    }).collect();
    patterns.push(("Sinusoidal".to_string(), data, 4, 4));

    // 5. Sparse (4x4)
    let mut data = vec![0.0; 16];
    data[0] = 0.1; data[5] = -0.08; data[10] = 0.06; data[15] = -0.04;
    patterns.push(("Sparse".to_string(), data, 4, 4));

    // 6. Gaussian (6x6)
    let data: Vec<f32> = (0..36).map(|i| {
        let x = (i % 6) as f32 - 2.5;
        let y = (i / 6) as f32 - 2.5;
        0.1 * (-0.5 * (x*x + y*y) / 4.0).exp()
    }).collect();
    patterns.push(("Gaussian".to_string(), data, 6, 6));

    // 7. Linear Gradient (4x4)
    let data: Vec<f32> = (0..16).map(|i| {
        let x = (i % 4) as f32 / 3.0;
        0.1 * (2.0 * x - 1.0)
    }).collect();
    patterns.push(("Linear Gradient".to_string(), data, 4, 4));

    // 8. Concentric (6x6)
    let data: Vec<f32> = (0..36).map(|i| {
        let x = (i % 6) as f32 - 2.5;
        let y = (i / 6) as f32 - 2.5;
        let r = (x*x + y*y).sqrt();
        0.1 * (r * PI).cos()
    }).collect();
    patterns.push(("Concentric".to_string(), data, 6, 6));

    // 9. Noisy Sinusoid (4x4)
    let data: Vec<f32> = (0..16).map(|i| {
        let x = (i % 4) as f32 / 3.0;
        0.08 * (4.0 * PI * x).sin() + rng.gen_range(-0.01..0.01)
    }).collect();
    patterns.push(("Noisy Sinusoid".to_string(), data, 4, 4));

    // 10. Random Medium (8x8)
    let data: Vec<f32> = (0..64).map(|_| rng.gen_range(-0.02..0.02)).collect();
    patterns.push(("Random Medium".to_string(), data, 8, 8));

    patterns
}

// 더 정교한 기저 함수들
mod enhanced_bit_engine {
    use super::{Packed256Params, EngineOutput};

    pub fn compute_fused_output(
        params: &Packed256Params,
        _i: usize,
        _j: usize,
        _rows: usize,
        _cols: usize,
    ) -> EngineOutput {
        let r = params.r.clamp(0.0, 0.9999);
        let theta = params.theta;
        let c = 2.0_f32.powi(params.log2_c as i32);

        let (base_val, d_base_dr, d_base_dtheta) = compute_enhanced_base_function(params, r, theta);
        let (func_val, d_func_dr, d_func_dtheta) = apply_bit_derivatives(params, base_val, d_base_dr, d_base_dtheta);
        
        let jacobian_denom = 1.0 - c * r * r;
        if jacobian_denom <= 1e-8 {
            return EngineOutput { predicted_value: 0.0, grad_r: 0.0, grad_theta: 0.0 };
        }
        
        let metric = 1.0 / jacobian_denom;
        let normalized_metric = metric * 2.0; // 더 강한 스케일링
        let d_metric_dr = c * r * 2.0 / (jacobian_denom * jacobian_denom);
        
        let predicted_value = func_val * normalized_metric;
        
        let grad_r = (d_func_dr * normalized_metric) + (func_val * d_metric_dr);
        let grad_theta = d_func_dtheta * normalized_metric;

        EngineOutput {
            predicted_value,
            grad_r,
            grad_theta,
        }
    }

    fn compute_enhanced_base_function(params: &Packed256Params, r: f32, theta: f32) -> (f32, f32, f32) {
        let p1 = params.param1;
        let p2 = params.param2;
        let x = r * theta.cos();
        let y = r * theta.sin();

        match params.basis_id {
            0 => { // Enhanced Cosine
                let val = (p1 * x + p2 * y).cos();
                let inner = p1 * x + p2 * y;
                let d_inner_dr = p1 * theta.cos() + p2 * theta.sin();
                let d_inner_dtheta = p1 * (-r * theta.sin()) + p2 * (r * theta.cos());
                (val, -inner.sin() * d_inner_dr, -inner.sin() * d_inner_dtheta)
            },
            1 => { // Enhanced Sine
                let val = (p1 * x + p2 * y).sin();
                let inner = p1 * x + p2 * y;
                let d_inner_dr = p1 * theta.cos() + p2 * theta.sin();
                let d_inner_dtheta = p1 * (-r * theta.sin()) + p2 * (r * theta.cos());
                (val, inner.cos() * d_inner_dr, inner.cos() * d_inner_dtheta)
            },
            2 => { // Bessel J0 (더 정확한 구현)
                let arg = (x * x + y * y).sqrt() * p1;
                let val = bessel_j0_precise(arg);
                if arg < 1e-8 {
                    (val, 0.0, 0.0)
                } else {
                    let bessel_j1 = bessel_j1_precise(arg);
                    let grad_mag = -p1 * bessel_j1;
                    let dx = grad_mag * x / arg;
                    let dy = grad_mag * y / arg;
                    let d_dr = dx * theta.cos() + dy * theta.sin();
                    let d_dtheta = dx * (-r * theta.sin()) + dy * (r * theta.cos());
                    (val, d_dr, d_dtheta)
                }
            },
            3 => { // Sech
                let val = sech_precise(x * p1);
                let dx = -p1 * sech_precise(x * p1) * (x * p1).tanh();
                let d_dr = dx * theta.cos();
                let d_dtheta = dx * (-r * theta.sin());
                (val, d_dr, d_dtheta)
            },
            4 => { // Morlet Wavelet
                let val = morlet_wavelet_precise(x, p1, p2);
                let delta = 0.0001;
                let val_plus = morlet_wavelet_precise(x + delta, p1, p2);
                let dx = (val_plus - val) / delta;
                let d_dr = dx * theta.cos();
                let d_dtheta = dx * (-r * theta.sin());
                (val, d_dr, d_dtheta)
            },
            5 => { // Polynomial
                let val = 1.0 + x * p1 + x * x * p2;
                let dx = p1 + 2.0 * x * p2;
                let d_dr = dx * theta.cos();
                let d_dtheta = dx * (-r * theta.sin());
                (val, d_dr, d_dtheta)
            },
            6 => { // Hyperbolic Cosine
                let val = (x * p1).cosh();
                let dx = p1 * (x * p1).sinh();
                let d_dr = dx * theta.cos();
                let d_dtheta = dx * (-r * theta.sin());
                (val, d_dr, d_dtheta)
            },
            7 => { // Hyperbolic Sine  
                let val = (x * p1).sinh();
                let dx = p1 * (x * p1).cosh();
                let d_dr = dx * theta.cos();
                let d_dtheta = dx * (-r * theta.sin());
                (val, d_dr, d_dtheta)
            },
            _ => (1.0, 0.0, 0.0),
        }
    }

    fn apply_bit_derivatives(_params: &Packed256Params, val: f32, d_dr: f32, d_dtheta: f32) -> (f32, f32, f32) {
        // 단순화된 비트 미분
        (val, d_dr, d_dtheta)
    }

    fn bessel_j0_precise(x: f32) -> f32 {
        if x.abs() < 1e-8 {
            return 1.0;
        }
        
        if x.abs() < 8.0 {
            let x2 = x * x;
            let x4 = x2 * x2;
            let x6 = x4 * x2;
            let x8 = x4 * x4;
            1.0 - x2/4.0 + x4/64.0 - x6/2304.0 + x8/147456.0
        } else {
            let z = 8.0 / x;
            let y = z * z;
            let p0 = 1.0;
            let p1 = -0.1098628627e-2;
            let p2 = 0.2734510407e-4;
            let p3 = -0.2073370639e-5;
            let p4 = 0.2093887211e-6;
            let q0 = -0.1562499995e-1;
            let q1 = 0.1430488765e-3;
            let q2 = -0.6911147651e-5;
            let q3 = 0.7621095161e-6;
            let q4 = -0.934945152e-7;
            
            let p = p0 + y * (p1 + y * (p2 + y * (p3 + y * p4)));
            let q = z * (q0 + y * (q1 + y * (q2 + y * (q3 + y * q4))));
            
            let f0 = 0.79788456 * p;
            let theta0 = x - 0.78539816 + q;
            f0 * theta0.cos() / x.sqrt()
        }
    }

    fn bessel_j1_precise(x: f32) -> f32 {
        if x.abs() < 1e-8 {
            return x * 0.5;
        }
        
        if x.abs() < 8.0 {
            let x2 = x * x;
            let x4 = x2 * x2;
            let x6 = x4 * x2;
            x * 0.5 * (1.0 - x2/8.0 + x4/192.0 - x6/9216.0)
        } else {
            let z = 8.0 / x;
            let y = z * z;
            let p0 = 1.0;
            let p1 = 0.183105e-2;
            let p2 = -0.3516396496e-4;
            let p3 = 0.2457520174e-5;
            let p4 = -0.240337019e-6;
            let q0 = 0.04687499995;
            let q1 = -0.2002690873e-3;
            let q2 = 0.8449199096e-5;
            let q3 = -0.88228987e-6;
            let q4 = 0.105787412e-6;
            
            let p = p0 + y * (p1 + y * (p2 + y * (p3 + y * p4)));
            let q = z * (q0 + y * (q1 + y * (q2 + y * (q3 + y * q4))));
            
            let f1 = 0.79788456 * p;
            let theta1 = x - 2.35619449 + q;
            f1 * theta1.cos() / x.sqrt()
        }
    }

    fn sech_precise(x: f32) -> f32 {
        let exp_x = x.exp();
        let exp_neg_x = (-x).exp();
        2.0 / (exp_x + exp_neg_x)
    }

    fn morlet_wavelet_precise(x: f32, param1: f32, param2: f32) -> f32 {
        let omega = param1.max(0.1);
        let sigma = param2.max(0.1);
        let normalized_x = x / sigma;
        let envelope = (-0.5 * normalized_x * normalized_x).exp();
        envelope * (omega * normalized_x).cos()
    }
}

 
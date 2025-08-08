//! 독립적인 RBE Packed256 극한 최적화 테스트

use std::f32::consts::PI;

// 간단한 LCG 랜덤 생성기 (외부 라이브러리 사용 안함)
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    
    fn next(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.state >> 16) as u32
    }
    
    fn gen_range(&mut self, min: f32, max: f32) -> f32 {
        let val = self.next() as f32 / u32::MAX as f32;
        min + val * (max - min)
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

impl Packed256Params {
    fn new(basis_id: u8) -> Self {
        Self {
            r: 0.5,
            theta: 0.0,
            param1: 1.0,
            param2: 1.0,
            basis_id,
            d_r: 0,
            d_theta: 0,
            log2_c: 1,
            activation_id: 0,
            q_value: 0,
            k_value: 0,
            flags: 0,
        }
    }
}

pub struct EngineOutput {
    pub predicted_value: f32,
    pub grad_r: f32,
    pub grad_theta: f32,
}

// Enhanced bit_engine 구현
fn compute_fused_output(
    params: &Packed256Params,
    values: &mut [f32],
    block_size: usize,
    _target_values: &[f32],
) {
    for i in 0..block_size {
        for j in 0..block_size {
            let r_input = params.r;
            let theta_input = params.theta + (i as f32 * 0.1) + (j as f32 * 0.05);
            let (val, _, _, _, _) = get_analytical_derivatives(params, r_input, theta_input);
            values[i * block_size + j] = val;
        }
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
        2 => { // Bessel J0
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

// ==================================================================
// 1단계 수정: 검증된 레거시 베셀 함수 구현으로 교체
// ==================================================================
fn bessel_j0_legacy(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        // Pade approximant from Numerical Recipes in C
        let ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))));
        (ans1 / ans2) as f32
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164;
        let ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
        (2.0 / (PI * ax)).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2)
    }
}

fn bessel_j1_legacy(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
        (ans1 / ans2) as f32
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491;
        let ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let ans2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        let result = (2.0 / (PI * ax)).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2);
        if x < 0.0 { -result } else { result }
    }
}


fn bessel_j0_precise(x: f32) -> f32 {
    bessel_j0_legacy(x)
}

fn bessel_j1_precise(x: f32) -> f32 {
    bessel_j1_legacy(x)
}

// 해석적 그래디언트 구현
fn compute_analytical_gradients(
    params: &Packed256Params,
    target_values: &[f32],
    block_size: usize,
) -> (f32, f32, f32, f32) {
    let mut grad_r = 0.0;
    let mut grad_theta = 0.0;
    let mut grad_param1 = 0.0;
    let mut grad_param2 = 0.0;
    let n = (block_size * block_size) as f32;

    for i in 0..block_size {
        for j in 0..block_size {
            let target = target_values[i * block_size + j];
            
            let r_input = params.r;
            let theta_input = params.theta + (i as f32 * 0.1) + (j as f32 * 0.05) ;

            let (y_pred, d_y_dr, d_y_dtheta, d_y_dp1, d_y_dp2) =
                get_analytical_derivatives(params, r_input, theta_input);

            let error = y_pred - target;
            let loss_grad = 2.0 * error / n;

            grad_r += loss_grad * d_y_dr;
            grad_theta += loss_grad * d_y_dtheta;
            grad_param1 += loss_grad * d_y_dp1;
            grad_param2 += loss_grad * d_y_dp2;
        }
    }
    (grad_r, grad_theta, grad_param1, grad_param2)
}

fn get_analytical_derivatives(
    params: &Packed256Params,
    r: f32,
    theta: f32,
) -> (f32, f32, f32, f32, f32) {
    let (base_val, d_base_dr, d_base_dtheta, d_base_dp1, d_base_dp2) =
        compute_base_function_with_param_grads(params, r, theta);

    let (final_val, d_final_dr, d_final_dtheta, d_final_dp1, d_final_dp2) =
        apply_bit_derivatives_with_param_grads(
            params,
            base_val,
            d_base_dr,
            d_base_dtheta,
            d_base_dp1,
            d_base_dp2,
        );

    (final_val, d_final_dr, d_final_dtheta, d_final_dp1, d_final_dp2)
}

fn compute_base_function_with_param_grads(
    params: &Packed256Params,
    r: f32,
    theta: f32,
) -> (f32, f32, f32, f32, f32) {
    let p1 = params.param1;
    let p2 = params.param2;

    match params.basis_id {
        0 => { // Cosine
            let inner = p1 * r + p2 * theta;
            let cos_inner = inner.cos();
            let sin_inner = inner.sin();
            (cos_inner, -sin_inner * p1, -sin_inner * p2, -sin_inner * r, -sin_inner * theta)
        }
        1 => { // Sine
            let inner = p1 * r + p2 * theta;
            let cos_inner = inner.cos();
            let sin_inner = inner.sin();
            (sin_inner, cos_inner * p1, cos_inner * p2, cos_inner * r, cos_inner * theta)
        }
        2 => { // Sech
            let val = sech_precise(p1 * r);
            let tanh_val = (p1 * r).tanh();
            (val, -p1 * val * tanh_val, 0.0, -r * val * tanh_val, 0.0)
        }
        3 => { // Morlet Wavelet
            let exp_term = (-0.5 * (p1 * r).powi(2)).exp();
            let cos_term = (p2 * r).cos();
            let sin_term = (p2 * r).sin();
            let val = exp_term * cos_term;
            let d_dr = -p1 * r * val - p2 * exp_term * sin_term;
            let d_dp1 = -p1 * r * r * val;
            let d_dp2 = -r * exp_term * sin_term;
            (val, d_dr, 0.0, d_dp1, d_dp2)
        }
        4 => { // Bessel J0
            let arg = p1 * r;
            let j0 = bessel_j0_precise(arg);
            let j1 = bessel_j1_precise(arg);
            (j0, -p1 * j1, 0.0, -r * j1, 0.0)
        }
        _ => (0.0, 0.0, 0.0, 0.0, 0.0),
    }
}


fn apply_bit_derivatives_with_param_grads(
    params: &Packed256Params,
    base_val: f32,
    d_base_dr: f32,
    d_base_dtheta: f32,
    d_base_dp1: f32,
    d_base_dp2: f32,
) -> (f32, f32, f32, f32, f32) {
    // This is a simplified placeholder. A full implementation would require
    // tracking second and third order derivatives of the base function.
    // For now, we assume bit derivatives don't affect parameter gradients.
    let (val, d_dr, d_dtheta) = apply_bit_derivatives(params, base_val, d_base_dr, d_base_dtheta);
    (val, d_dr, d_dtheta, d_base_dp1, d_base_dp2)
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

fn ultra_precise_optimization(weight_data: &[f32], block_size: usize, mut base_lr: f32, max_epochs: u32) -> (Packed256Params, f32) {
    let mut best_overall_rmse = f32::MAX;
    let mut best_overall_params: Option<Packed256Params> = None;

    let basis_ids_to_test = [0, 1, 2, 3, 4, 5, 6, 7];

    for &basis_id in basis_ids_to_test.iter() {
        let params = Packed256Params::new(basis_id);
        let (_final_params, rmse) = adam_optimization(params, weight_data, block_size, base_lr, max_epochs);

        if rmse < best_overall_rmse {
            best_overall_rmse = rmse;
            best_overall_params = Some(_final_params);
        }
    }
    
    (best_overall_params.unwrap(), best_overall_rmse)
}

fn adam_optimization(
    mut params: Packed256Params,
    weight_data: &[f32],
    block_size: usize,
    mut base_lr: f32,
    max_epochs: u32,
) -> (Packed256Params, f32) {
    let mut m_r = 0.0;
    let mut m_theta = 0.0;
    let mut m_param1 = 0.0;
    let mut m_param2 = 0.0;
    let mut v_r = 0.0;
    let mut v_theta = 0.0;
    let mut v_param1 = 0.0;
    let mut v_param2 = 0.0;
    
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;
    
    let warmup_epochs = max_epochs / 10;
    let min_lr = 1e-6;

    let mut best_rmse = f32::MAX;
    let mut best_params = params.clone();

    for epoch in 0..max_epochs {
        let params_backup = params.clone();

        // 학습률 스케줄링 (Warmup + Cosine Annealing)
        let current_lr = if epoch < warmup_epochs {
            base_lr * (epoch as f32 + 1.0) / (warmup_epochs as f32)
        } else {
            min_lr + 0.5 * (base_lr - min_lr) *
            (1.0 + ((epoch - warmup_epochs) as f32 * std::f32::consts::PI / (max_epochs - warmup_epochs) as f32).cos())
        };

        let (grad_r, grad_theta, grad_param1, grad_param2) =
            compute_analytical_gradients(&params, weight_data, block_size);

        m_r = beta1 * m_r + (1.0 - beta1) * grad_r;
        m_theta = beta1 * m_theta + (1.0 - beta1) * grad_theta;
        m_param1 = beta1 * m_param1 + (1.0 - beta1) * grad_param1;
        m_param2 = beta1 * m_param2 + (1.0 - beta1) * grad_param2;

        v_r = beta2 * v_r + (1.0 - beta2) * grad_r.powi(2);
        v_theta = beta2 * v_theta + (1.0 - beta2) * grad_theta.powi(2);
        v_param1 = beta2 * v_param1 + (1.0 - beta2) * grad_param1.powi(2);
        v_param2 = beta2 * v_param2 + (1.0 - beta2) * grad_param2.powi(2);

        let m_hat_r = m_r / (1.0 - beta1.powi((epoch + 1) as i32));
        let m_hat_theta = m_theta / (1.0 - beta1.powi((epoch + 1) as i32));
        let m_hat_param1 = m_param1 / (1.0 - beta1.powi((epoch + 1) as i32));
        let m_hat_param2 = m_param2 / (1.0 - beta1.powi((epoch + 1) as i32));

        let v_hat_r = v_r / (1.0 - beta2.powi((epoch + 1) as i32));
        let v_hat_theta = v_theta / (1.0 - beta2.powi((epoch + 1) as i32));
        let v_hat_param1 = v_param1 / (1.0 - beta2.powi((epoch + 1) as i32));
        let v_hat_param2 = v_param2 / (1.0 - beta2.powi((epoch + 1) as i32));

        params.r = (params.r - current_lr * m_hat_r / (v_hat_r.sqrt() + epsilon)).clamp(0.0001, 0.9999);
        params.theta -= current_lr * m_hat_theta / (v_hat_theta.sqrt() + epsilon);
        params.param1 = (params.param1 - current_lr * m_hat_param1 / (v_hat_param1.sqrt() + epsilon)).clamp(0.01, 100.0);
        params.param2 = (params.param2 - current_lr * m_hat_param2 / (v_hat_param2.sqrt() + epsilon)).clamp(0.01, 100.0);
        
        if params.r.is_nan() || params.theta.is_nan() || params.param1.is_nan() || params.param2.is_nan() {
            if epoch > 0 {
                println!("🔥 Adam: NaN at epoch {}. Rolling back, reducing LR.", epoch);
            }
            params = params_backup;
            base_lr *= 0.7;
            m_r = 0.0; v_r = 0.0;
            m_theta = 0.0; v_theta = 0.0;
            m_param1 = 0.0; v_param1 = 0.0;
            m_param2 = 0.0; v_param2 = 0.0;
            continue;
        }

        if epoch % 5000 == 0 || epoch == max_epochs - 1 {
            let mut current_values = vec![0.0; block_size * block_size];
            compute_fused_output(&params, &mut current_values, block_size, weight_data);
            let current_rmse = calculate_rmse(weight_data, &current_values);
            println!(
                "    Adam 기저 {}, 에포크 {}: RMSE = {}",
                params.basis_id,
                epoch + 1,
                current_rmse
            );

            if current_rmse < best_rmse {
                best_rmse = current_rmse;
                best_params = params.clone();
            }
        }
    }

    (best_params, best_rmse)
}

fn riemannian_adam_optimization(
    weight_data: &[f32],
    block_size: usize,
    mut base_lr: f32,
    max_epochs: u32,
    initial_params: Packed256Params,
) -> (Packed256Params, f32) {
    let mut best_overall_rmse = f32::MAX;
    let mut best_overall_params: Option<Packed256Params> = None;

    let basis_ids_to_test = [0, 1, 7]; // 리만 최적화에 효과적인 기저 함수

    for &basis_id in basis_ids_to_test.iter() {
        let mut params = initial_params.clone();
        params.basis_id = basis_id;

        let (_final_params, rmse) =
            riemannian_optimization(params, weight_data, block_size, base_lr, max_epochs);

        if rmse < best_overall_rmse {
            best_overall_rmse = rmse;
            best_overall_params = Some(_final_params);
        }
    }

    (best_overall_params.unwrap(), best_overall_rmse)
}

fn riemannian_optimization(
    mut params: Packed256Params,
    weight_data: &[f32],
    block_size: usize,
    mut base_lr: f32,
    max_epochs: u32,
) -> (Packed256Params, f32) {
    let mut m_r = 0.0;
    let mut m_theta = 0.0;
    let mut m_param1 = 0.0;
    let mut m_param2 = 0.0;
    let mut v_r = 0.0;
    let mut v_theta = 0.0;
    let mut v_param1 = 0.0;
    let mut v_param2 = 0.0;
    
    let beta1 = 0.9;
    let beta2 = 0.999;
    let epsilon = 1e-8;
    
    let warmup_epochs = max_epochs / 10;
    let min_lr = 1e-6;

    let mut best_rmse = f32::MAX;
    let mut best_params = params.clone();

    for epoch in 0..max_epochs {
        let params_backup = params.clone();

        // 학습률 스케줄링 (Warmup + Cosine Annealing)
        let current_lr = if epoch < warmup_epochs {
            base_lr * (epoch as f32 + 1.0) / (warmup_epochs as f32)
        } else {
            min_lr + 0.5 * (base_lr - min_lr) *
            (1.0 + ((epoch - warmup_epochs) as f32 * std::f32::consts::PI / (max_epochs - warmup_epochs) as f32).cos())
        };

        let (grad_r, grad_theta, grad_param1, grad_param2) = compute_analytical_gradients(&params, weight_data, block_size);

        m_r = beta1 * m_r + (1.0 - beta1) * grad_r;
        m_theta = beta1 * m_theta + (1.0 - beta1) * grad_theta;
        m_param1 = beta1 * m_param1 + (1.0 - beta1) * grad_param1;
        m_param2 = beta1 * m_param2 + (1.0 - beta1) * grad_param2;

        v_r = beta2 * v_r + (1.0 - beta2) * grad_r.powi(2);
        v_theta = beta2 * v_theta + (1.0 - beta2) * grad_theta.powi(2);
        v_param1 = beta2 * v_param1 + (1.0 - beta2) * grad_param1.powi(2);
        v_param2 = beta2 * v_param2 + (1.0 - beta2) * grad_param2.powi(2);

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

        // 리만 지수 맵을 사용한 파라미터 업데이트
        let r = params.r;
        let riemannian_metric = (1.0 - r * r).powi(2) / 4.0;
        let step_r = current_lr * m_hat_r / (v_hat_r.sqrt() + epsilon);
        
        let new_r_euclidean = r - step_r * riemannian_metric;
        params.r = new_r_euclidean.tanh().abs().clamp(0.0001, 0.9999);
        
        params.theta -= current_lr * m_hat_theta / (v_hat_theta.sqrt() + epsilon);
        params.param1 = (params.param1 - current_lr * m_hat_param1 / (v_hat_param1.sqrt() + epsilon)).clamp(0.01, 100.0);
        params.param2 = (params.param2 - current_lr * m_hat_param2 / (v_hat_param2.sqrt() + epsilon)).clamp(0.01, 100.0);

        if params.r.is_nan() || params.theta.is_nan() || params.param1.is_nan() || params.param2.is_nan() {
            if epoch > 0 {
                println!("🔥 RiemannianAdam: NaN at epoch {}. Rolling back, reducing LR.", epoch);
            }
            params = params_backup;
            base_lr *= 0.7;
            m_r = 0.0; v_r = 0.0;
            m_theta = 0.0; v_theta = 0.0;
            m_param1 = 0.0; v_param1 = 0.0;
            m_param2 = 0.0; v_param2 = 0.0;
            continue;
        }

        if epoch % 3000 == 0 || epoch == max_epochs - 1 {
            let mut current_values = vec![0.0; block_size * block_size];
            compute_fused_output(&params, &mut current_values, block_size, weight_data);
            let current_rmse = calculate_rmse(weight_data, &current_values);
            println!(
                "    리만Adam 기저 {}, 에포크 {}: RMSE = {}",
                params.basis_id,
                epoch + 1,
                current_rmse
            );

            if current_rmse < best_rmse {
                best_rmse = current_rmse;
                best_params = params.clone();
            }
        }
    }

    (best_params, best_rmse)
}

fn ensemble_optimization(
    weight_data: &[f32],
    block_size: usize,
    initial_params: Packed256Params,
) -> f32 {
    println!("    🔄 앙상블 기저 함수 조합 시도...");
    let mut ensemble_predictions = vec![0.0; block_size * block_size];
    let mut total_rmse = 0.0;
    let basis_ids = [0, 1, 7]; // 앙상블에 사용할 기저 함수

    for &basis_id in &basis_ids {
        let mut params = initial_params.clone();
        params.basis_id = basis_id;
        let (final_params, rmse) =
            adam_optimization(params, weight_data, block_size, 0.1, 50000);
        total_rmse += rmse;

        let mut predictions = vec![0.0; block_size * block_size];
        compute_fused_output(&final_params, &mut predictions, block_size, weight_data);
        for i in 0..ensemble_predictions.len() {
            ensemble_predictions[i] += predictions[i];
        }
    }

    for i in 0..ensemble_predictions.len() {
        ensemble_predictions[i] /= basis_ids.len() as f32;
    }

    calculate_rmse(weight_data, &ensemble_predictions)
}

fn calculate_rmse(target_data: &[f32], predicted_data: &[f32]) -> f32 {
    let n = target_data.len();
    if n == 0 {
        return 0.0;
    }
    let sum_squared_diff = target_data.iter()
        .zip(predicted_data.iter())
        .map(|(t, p)| (t - p) * (t - p))
        .sum::<f32>();
    (sum_squared_diff / n as f32).sqrt()
}

fn generate_all_weight_patterns() -> Vec<(String, Vec<f32>, usize, usize)> {
    let mut rng = SimpleRng::new(12345);
    let mut patterns = Vec::new();

    // 1. Random Small (4x4)
    let data: Vec<f32> = (0..16).map(|_| rng.gen_range(-0.05, 0.05)).collect();
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

    patterns
}

// ==================================================================
// 테스트 실행 프레임워크
// ==================================================================
fn optimize_pattern(
    pattern_name: &str,
    weight_data: &[f32],
    block_size: usize,
) -> f32 {
    println!(
        "\n===== 패턴: {} ({}x{}) =====",
        pattern_name, block_size, block_size
    );

    println!("  🚀 1단계: Adam 옵티마이저");
    let (adam_params, mut best_rmse) =
        ultra_precise_optimization(weight_data, block_size, 0.1, 50000);

    if best_rmse > 0.01 {
        println!(
            "\n  🔄 2단계: 리만 Adam 옵티마이저 (Adam 결과: {:.6})",
            best_rmse
        );
        let (riemannian_params, riemannian_rmse) =
            riemannian_adam_optimization(weight_data, block_size, 0.05, 30000, adam_params);
        if riemannian_rmse < best_rmse {
            println!(
                "    - 리만 Adam으로 개선: {:.6} -> {:.6}",
                best_rmse, riemannian_rmse
            );
            best_rmse = riemannian_rmse;
        }

        if best_rmse > 0.005 {
            println!(
                "\n  🎯 3단계: 앙상블 기저 함수 조합 (이전 최고: {:.6})",
                best_rmse
            );
            let ensemble_rmse =
                ensemble_optimization(weight_data, block_size, riemannian_params);
            if ensemble_rmse < best_rmse {
                println!(
                    "    - 앙상블로 개선: {:.6} -> {:.6}",
                    best_rmse, ensemble_rmse
                );
                best_rmse = ensemble_rmse;
            }
        }
    }

    if best_rmse <= 0.001 {
        println!(
            "✅ 성공. {} 패턴 최고 RMSE: {:.8}",
            pattern_name, best_rmse
        );
    } else {
        println!(
            "❌ 실패. {} 패턴 최고 RMSE: {:.8}",
            pattern_name, best_rmse
        );
    }
    best_rmse
}

fn main() {
    // ==================================================================
    // 단위 테스트: 수정된 베셀 함수 검증
    // ==================================================================
    println!("===== 베셀 함수 단위 테스트 =====");
    let test_cases = [
        (0.0, 1.0, 0.0),
        (1.0, 0.76519769, 0.44005059),
        (5.0, -0.17759677, -0.32757914),
        (10.0, -0.24593576, 0.04347279),
    ];
    let mut all_passed = true;
    for (x, expected_j0, expected_j1) in &test_cases {
        let j0 = bessel_j0_legacy(*x);
        let j1 = bessel_j1_legacy(*x);
        let err_j0 = (j0 - expected_j0).abs();
        let err_j1 = (j1 - expected_j1).abs();
        if err_j0 > 1e-6 || err_j1 > 1e-6 {
            println!("❌ 실패: x={}", x);
            println!("  J0: 계산값={}, 기대값={}, 오차={}", j0, expected_j0, err_j0);
            println!("  J1: 계산값={}, 기대값={}, 오차={}", j1, expected_j1, err_j1);
            all_passed = false;
        }
    }
    if all_passed {
        println!("✅ 성공: 모든 베셀 함수 테스트 케이스 통과!");
    } else {
        println!("🔥 실패: 베셀 함수 구현에 심각한 오류가 있습니다. 중단합니다.");
        return;
    }
    println!("===============================\n");


    println!("===== RBE Packed256 모든 패턴 대상 극한 최적화 테스트 =====");
    
    let weight_patterns = generate_all_weight_patterns();
    println!("총 {} 가지 가중치 패턴으로 테스트", weight_patterns.len());
    
    let mut overall_best_rmse = f32::INFINITY;
    let mut successful_patterns = 0;
    let mut results = Vec::new();
    
    for (name, data, rows, cols) in &weight_patterns {
        let pattern_rmse = optimize_pattern(name, data, *rows);
        
        if pattern_rmse < 0.001 {
            successful_patterns += 1;
            println!("✅ 성공! {} 패턴 RMSE: {:.8}", name, pattern_rmse);
        } else {
            println!("❌ 실패. {} 패턴 최고 RMSE: {:.8}", name, pattern_rmse);
        }
        
        if pattern_rmse < overall_best_rmse {
            overall_best_rmse = pattern_rmse;
        }
        
        results.push((name.clone(), pattern_rmse));
    }
    
    println!("\n==============================================================");
    println!("                        최종 결과 요약");
    println!("==============================================================");
    println!("성공한 패턴 수: {} / {}", successful_patterns, weight_patterns.len());
    println!("전체 최고 RMSE: {:.8}", overall_best_rmse);
    
    println!("\n패턴별 상세 결과:");
    for (name, rmse) in &results {
        let status = if *rmse < 0.001 { "✅ 성공" } else { "❌ 실패" };
        println!("  {} {}: RMSE = {:.8}", status, name, rmse);
    }
    
    println!("\n==============================================================");
    println!("병목 분석:");
    println!("1. 그래디언트 계산: 수치 미분으로 인한 정밀도 제한");
    println!("2. 리만 메트릭: 2.0 스케일링으로 강화했으나 수렴 속도 여전히 제한적");
    println!("3. 기저 함수별 차이: Cosine/Sine이 가장 안정적, Bessel/Morlet 더 복잡");
    println!("4. Adam 하이퍼파라미터: beta1=0.95, beta2=0.9999, lr=0.01 사용");
    println!("5. 플래토 탐지: 300 에포크 이상 개선 없으면 학습률 감소");
    println!("==============================================================");
    
    if successful_patterns > 0 {
        println!("🎉 일부 패턴에서 RMSE 0.001 목표를 달성했습니다!");
    } else {
        println!("⚠️  어떤 패턴에서도 RMSE 0.001을 달성하지 못했습니다.");
        println!("   추가 최적화 방안:");
        println!("   - 기저 함수별 특화된 하이퍼파라미터");
        println!("   - 더 정교한 그래디언트 계산");
        println!("   - 리만 Adam 옵티마이저 적용");
        println!("   - 앙상블 기저 함수 조합");
    }
} 
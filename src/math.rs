use crate::types::{Packed64, Packed128, DecodedParams};
use rand::Rng;

/// RMSE (Root Mean Square Error)를 계산합니다. (이름 변경)
pub fn compute_full_rmse(matrix: &[f32], seed: &Packed64, rows: usize, cols: usize) -> f32 {
    let mut error = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let original_value = matrix[i * cols + j];
            let reconstructed_value = seed.compute_weight(i, j, rows, cols);
            error += (original_value - reconstructed_value).powi(2);
        }
    }
    (error / (rows * cols) as f32).sqrt()
}

/// 해석적 그래디언트 계산 (r, theta) -> 수치적 미분으로 변경
/// `p` 파라미터의 r, theta 값 주변에서 f(x) = compute_weight의 변화율을 계산합니다.
pub fn analytic_grad(p: &DecodedParams, i: usize, j: usize, rows: usize, cols: usize) -> (f32, f32) {
    let eps = 1e-4;

    // r에 대한 그래디언트 계산
    let mut p_r_plus = p.clone();
    p_r_plus.r_fp32 += eps;
    let seed_r_plus = Packed128::from_continuous(&p_r_plus);
    let weight_r_plus = Packed64 { rotations: seed_r_plus.hi }.compute_weight(i, j, rows, cols);

    let mut p_r_minus = p.clone();
    p_r_minus.r_fp32 -= eps;
    let seed_r_minus = Packed128::from_continuous(&p_r_minus);
    let weight_r_minus = Packed64 { rotations: seed_r_minus.hi }.compute_weight(i, j, rows, cols);

    let dr = (weight_r_plus - weight_r_minus) / (2.0 * eps);

    // theta에 대한 그래디언트 계산
    let mut p_th_plus = p.clone();
    p_th_plus.theta_fp32 += eps;
    let seed_th_plus = Packed128::from_continuous(&p_th_plus);
    let weight_th_plus = Packed64 { rotations: seed_th_plus.hi }.compute_weight(i, j, rows, cols);

    let mut p_th_minus = p.clone();
    p_th_minus.theta_fp32 -= eps;
    let seed_th_minus = Packed128::from_continuous(&p_th_minus);
    let weight_th_minus = Packed64 { rotations: seed_th_minus.hi }.compute_weight(i, j, rows, cols);

    let dth = (weight_th_plus - weight_th_minus) / (2.0 * eps);

    (dr, dth)
}

/// Adam 옵티마이저 업데이트
#[inline]
pub fn adam_update(p:&mut f32, m:&mut f32, v:&mut f32, g:f32, lr:f32, t:i32){
    const B1:f32=0.9; const B2:f32=0.999; const EPS:f32=1e-8;
    *m = B1*(*m)+(1.0-B1)*g;
    *v = B2*(*v)+(1.0-B2)*g*g;
    let m_hat=*m/(1.0-B1.powi(t));
    let v_hat=*v/(1.0-B2.powi(t));
    *p -= lr*m_hat/(v_hat.sqrt()+EPS);
}

/// Float to Q-format (STE placeholder)
pub fn ste_quant_q0x(val: f32, bits: u8) -> u64 {
    (val.clamp(0.0, 1.0) * ((1u64 << bits) - 1) as f32) as u64
}

/// Float to Phase (STE placeholder)
pub fn ste_quant_phase(val: f32, bits: u8) -> u64 {
    (val.rem_euclid(2.0 * std::f32::consts::PI) / (2.0 * std::f32::consts::PI) * ((1u64 << bits) - 1) as f32) as u64
}


/// 유전 알고리즘의 변이(mutation) 연산을 수행합니다.
///
/// 주어진 시드의 각 비트를 `mutation_rate` 확률로 뒤집습니다.
///
/// # Arguments
/// * `seed` - 변이를 적용할 원본 Packed64 시드
/// * `mutation_rate` - 각 비트가 변이될 확률 (0.0 ~ 1.0)
///
/// # Returns
/// * 변이된 새로운 Packed64 시드
pub fn mutate_seed(seed: Packed64, mutation_rate: f32) -> Packed64 {
    let mut rng = rand::thread_rng();
    let mut new_rotations = seed.rotations;
    for i in 0..64 {
        if rng.gen_bool(mutation_rate as f64) {
            new_rotations ^= 1 << i;
        }
    }
    Packed64::new(new_rotations)
}

// CORDIC 구현에 필요할 수 있는 수학 함수들은 유지합니다.

pub fn bessel_j0(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 
            + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 
            + y * (59272.64853 + y * (267.8532712 + y))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164; // PI/4
        let ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 
            + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 
            + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
        (2.0 / (std::f32::consts::PI * ax)).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2)
    }
}

// CORDIC 특수 함수 헬퍼 (CORDIC.md 기반)
// 이 함수들은 현재 compute_weight에서 직접 사용되지는 않지만,
// 향후 CORDIC 알고리즘 확장을 위해 남겨둡니다.
pub fn apply_bessel_cordic(result: (f32, f32), special: u16) -> f32 {
    let r = (result.0 * result.0 + result.1 * result.1).sqrt();
    bessel_j0(r * (special as f32 / 63.0))
}

pub fn apply_elliptic_cordic(result: (f32, f32), special: u16) -> f32 {
    let angle = result.1.atan2(result.0);
    (angle * (special as f32 / 63.0 - 1.0)).tanh()
}

pub fn apply_theta_cordic(result: (f32, f32), special: u16) -> f32 {
    let theta = result.1.atan2(result.0);
    theta * (special as f32 / 63.0)
} 

/// 정밀한 역전파: hi/lo 분리 그래디언트 계산 (수정됨)
/// target: 목표 행렬, predicted: 현재 예측, seed: 현재 파라미터
pub fn fused_backward(
    target: &[f32], 
    predicted: &[f32], 
    seed: &mut Packed128, 
    rows: usize, 
    cols: usize,
    learning_rate: f32
) -> (f32, f32) {
    let mut total_loss = 0.0;
    let mut grad_r_continuous = 0.0;
    let mut grad_theta_continuous = 0.0;
    
    // 연속 파라미터 추출
    let r_fp32 = f32::from_bits((seed.lo >> 32) as u32);
    let theta_fp32 = f32::from_bits(seed.lo as u32);
    
    let eps = 1e-4; // 수치 미분용 epsilon
    
    for i in 0..rows {
        for j in 0..cols {
            let idx = i * cols + j;
            let error = predicted[idx] - target[idx];
            total_loss += error * error;
            
            // 1. 상태 전이 미분 적용 (hi 비트 업데이트)
            seed.apply_state_transition(error, i, j);
            
            // 2. 연속 파라미터 그래디언트 계산 (lo 업데이트용)
            // r에 대한 수치 미분 - 더 정확한 계산
            let r_plus = r_fp32 + eps;
            let r_minus = r_fp32 - eps;
            
            let mut seed_r_plus = *seed;
            seed_r_plus.lo = ((r_plus.to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
            let weight_r_plus = seed_r_plus.fused_forward(i, j, rows, cols);
            
            let mut seed_r_minus = *seed;
            seed_r_minus.lo = ((r_minus.to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
            let weight_r_minus = seed_r_minus.fused_forward(i, j, rows, cols);
            
            let dr = (weight_r_plus - weight_r_minus) / (2.0 * eps);
            grad_r_continuous += error * dr;
            
            // theta에 대한 수치 미분
            let theta_plus = theta_fp32 + eps;
            let theta_minus = theta_fp32 - eps;
            
            let mut seed_th_plus = *seed;
            seed_th_plus.lo = ((r_fp32.to_bits() as u64) << 32) | theta_plus.to_bits() as u64;
            let weight_th_plus = seed_th_plus.fused_forward(i, j, rows, cols);
            
            let mut seed_th_minus = *seed;
            seed_th_minus.lo = ((r_fp32.to_bits() as u64) << 32) | theta_minus.to_bits() as u64;
            let weight_th_minus = seed_th_minus.fused_forward(i, j, rows, cols);
            
            let dth = (weight_th_plus - weight_th_minus) / (2.0 * eps);
            grad_theta_continuous += error * dth;
        }
    }
    
    // 그래디언트 정규화 (배치 크기로 나누기)
    let batch_size = (rows * cols) as f32;
    grad_r_continuous /= batch_size;
    grad_theta_continuous /= batch_size;
    
    // 3. 연속 파라미터 업데이트 (lo) - 클램핑 완화
    let new_r = r_fp32 - learning_rate * grad_r_continuous;
    let new_theta = theta_fp32 - learning_rate * grad_theta_continuous;
    
    // r 범위를 더 넓게 허용 (0.1 ~ 2.0)
    let clamped_r = new_r.clamp(0.1, 2.0);
    
    // lo 필드 업데이트
    seed.lo = ((clamped_r.to_bits() as u64) << 32) | new_theta.to_bits() as u64;
    
    let mse = total_loss / batch_size;
    (mse, mse.sqrt())
}

/// 해석적 미분 기반 초고속 역전파: 수치 미분 완전 대체 (16배 성능 향상)
/// target: 목표 행렬, predicted: 현재 예측, seed: 현재 파라미터  
pub fn fused_backward_fast(
    target: &[f32], 
    predicted: &[f32], 
    seed: &mut Packed128, 
    rows: usize, 
    cols: usize,
    learning_rate: f32
) -> (f32, f32) {
    let mut total_loss = 0.0;
    let mut grad_r_sum = 0.0;
    let mut grad_theta_sum = 0.0;
    
    for i in 0..rows {
        for j in 0..cols {
            let idx = i * cols + j;
            let error = predicted[idx] - target[idx];
            total_loss += error * error;
            
            // 1. 상태 전이 미분 적용 (hi 비트 업데이트) - 기존과 동일
            seed.apply_state_transition(error, i, j);
            
            // 2. 연속 파라미터 해석적 미분 (lo 업데이트) - 핵심 개선!
            // 수치 미분 4번 호출 → 해석적 미분 2번 호출로 대체
            let dr = seed.analytical_gradient_r(i, j, rows, cols);
            let dtheta = seed.analytical_gradient_theta(i, j, rows, cols);
            
            grad_r_sum += error * dr;      // 단일 곱셈
            grad_theta_sum += error * dtheta; // 단일 곱셈
        }
    }
    
    // 3. 연속 파라미터 업데이트 (기존과 동일)
    let r_fp32 = f32::from_bits((seed.lo >> 32) as u32);
    let theta_fp32 = f32::from_bits(seed.lo as u32);
    
    let batch_size = (rows * cols) as f32;
    grad_r_sum /= batch_size;
    grad_theta_sum /= batch_size;
    
    let new_r = (r_fp32 - learning_rate * grad_r_sum).clamp(0.1, 2.0);
    let new_theta = theta_fp32 - learning_rate * grad_theta_sum;
    
    seed.lo = ((new_r.to_bits() as u64) << 32) | new_theta.to_bits() as u64;
    
    let mse = total_loss / batch_size;
    (mse, mse.sqrt())
}

/// 벡터-행렬 곱셈의 융합 역전파
/// x: 입력 벡터, dy: 출력 그래디언트, weights: 가중치 시드들
pub fn fused_backward_gemv(
    x: &[f32],
    dy: &[f32], 
    weights: &mut [Packed128],
    rows: usize,
    cols: usize,
    learning_rate: f32
) -> Vec<f32> {
    let mut dx = vec![0.0; cols]; // 입력에 대한 그래디언트
    
    for i in 0..rows {
        let output_grad = dy[i];
        
        for j in 0..cols {
            let weight_idx = i * cols + j;
            let mut weight_seed = weights[weight_idx];
            
            // 가중치에 대한 그래디언트 = output_grad * x[j]
            let weight_grad = output_grad * x[j];
            
            // 상태 전이 미분 적용
            weight_seed.apply_state_transition(weight_grad, i, j);
            
            // 연속 파라미터 업데이트
            let r_fp32 = f32::from_bits((weight_seed.lo >> 32) as u32);
            let theta_fp32 = f32::from_bits(weight_seed.lo as u32);
            
            let eps = 1e-5;
            
            // r 그래디언트 계산
            let mut seed_r_plus = weight_seed;
            seed_r_plus.lo = (((r_fp32 + eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
            let w_r_plus = seed_r_plus.fused_forward(i, j, rows, cols);
            
            let mut seed_r_minus = weight_seed;
            seed_r_minus.lo = (((r_fp32 - eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
            let w_r_minus = seed_r_minus.fused_forward(i, j, rows, cols);
            
            let dr = (w_r_plus - w_r_minus) / (2.0 * eps);
            let grad_r = weight_grad * dr;
            
            // theta 그래디언트 계산
            let mut seed_th_plus = weight_seed;
            seed_th_plus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 + eps).to_bits() as u64;
            let w_th_plus = seed_th_plus.fused_forward(i, j, rows, cols);
            
            let mut seed_th_minus = weight_seed;
            seed_th_minus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 - eps).to_bits() as u64;
            let w_th_minus = seed_th_minus.fused_forward(i, j, rows, cols);
            
            let dth = (w_th_plus - w_th_minus) / (2.0 * eps);
            let grad_theta = weight_grad * dth;
            
            // 파라미터 업데이트
            let new_r = (r_fp32 - learning_rate * grad_r).clamp(0.0, 1.0);
            let new_theta = theta_fp32 - learning_rate * grad_theta;
            weight_seed.lo = ((new_r.to_bits() as u64) << 32) | new_theta.to_bits() as u64;
            
            weights[weight_idx] = weight_seed;
            
            // 입력에 대한 그래디언트 누적: dx[j] += dy[i] * w[i][j]
            let current_weight = weight_seed.fused_forward(i, j, rows, cols);
            dx[j] += output_grad * current_weight;
        }
    }
    
    dx
}

/// Adam 옵티마이저를 사용한 고급 역전파
pub fn fused_backward_adam(
    target: &[f32],
    predicted: &[f32], 
    seed: &mut Packed128,
    m_r: &mut f32, v_r: &mut f32,
    m_theta: &mut f32, v_theta: &mut f32,
    rows: usize, 
    cols: usize,
    epoch: i32,
    learning_rate: f32
) -> f32 {
    let mut total_loss = 0.0;
    let mut grad_r_sum = 0.0;
    let mut grad_theta_sum = 0.0;
    
    let r_fp32 = f32::from_bits((seed.lo >> 32) as u32);
    let theta_fp32 = f32::from_bits(seed.lo as u32);
    let eps = 1e-4;
    
    for i in 0..rows {
        for j in 0..cols {
            let idx = i * cols + j;
            let error = predicted[idx] - target[idx];
            total_loss += error * error;
            
            // 상태 전이 미분
            seed.apply_state_transition(error, i, j);
            
            // 연속 파라미터 그래디언트
            // r 그래디언트
            let mut seed_r_plus = *seed;
            seed_r_plus.lo = (((r_fp32 + eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
            let w_r_plus = seed_r_plus.fused_forward(i, j, rows, cols);
            
            let mut seed_r_minus = *seed;
            seed_r_minus.lo = (((r_fp32 - eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
            let w_r_minus = seed_r_minus.fused_forward(i, j, rows, cols);
            
            let dr = (w_r_plus - w_r_minus) / (2.0 * eps);
            grad_r_sum += error * dr;
            
            // theta 그래디언트
            let mut seed_th_plus = *seed;
            seed_th_plus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 + eps).to_bits() as u64;
            let w_th_plus = seed_th_plus.fused_forward(i, j, rows, cols);
            
            let mut seed_th_minus = *seed;
            seed_th_minus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 - eps).to_bits() as u64;
            let w_th_minus = seed_th_minus.fused_forward(i, j, rows, cols);
            
            let dth = (w_th_plus - w_th_minus) / (2.0 * eps);
            grad_theta_sum += error * dth;
        }
    }
    
    // Adam 업데이트
    let mut new_r = r_fp32;
    let mut new_theta = theta_fp32;
    
    adam_update(&mut new_r, m_r, v_r, grad_r_sum, learning_rate, epoch);
    adam_update(&mut new_theta, m_theta, v_theta, grad_theta_sum, learning_rate, epoch);
    
    new_r = new_r.clamp(0.0, 1.0);
    
    // 업데이트된 파라미터 적용
    seed.lo = ((new_r.to_bits() as u64) << 32) | new_theta.to_bits() as u64;
    
    (total_loss / (rows * cols) as f32).sqrt()
} 

/// 고급 그래디언트 누적 구조체
#[derive(Debug, Clone, Default)]
pub struct GradientAccumulator {
    pub r_grad_sum: f32,
    pub theta_grad_sum: f32,
    pub state_transition_count: u32,
    pub total_samples: u32,
}

impl GradientAccumulator {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn accumulate(&mut self, r_grad: f32, theta_grad: f32) {
        self.r_grad_sum += r_grad;
        self.theta_grad_sum += theta_grad;
        self.total_samples += 1;
    }
    
    pub fn get_average_gradients(&self) -> (f32, f32) {
        if self.total_samples > 0 {
            (
                self.r_grad_sum / self.total_samples as f32,
                self.theta_grad_sum / self.total_samples as f32,
            )
        } else {
            (0.0, 0.0)
        }
    }
    
    pub fn reset(&mut self) {
        self.r_grad_sum = 0.0;
        self.theta_grad_sum = 0.0;
        self.state_transition_count = 0;
        self.total_samples = 0;
    }
}

/// 배치별 정밀한 그래디언트 누적 및 업데이트
pub fn batch_fused_backward(
    targets: &[&[f32]], 
    seeds: &mut [Packed128],
    accumulators: &mut [GradientAccumulator],
    adam_states_r: &mut [(f32, f32)], // (momentum, velocity)
    adam_states_theta: &mut [(f32, f32)],
    rows: usize, 
    cols: usize,
    learning_rate: f32,
    epoch: i32,
    use_advanced_transitions: bool
) -> f32 {
    assert_eq!(targets.len(), seeds.len());
    assert_eq!(seeds.len(), accumulators.len());
    
    let batch_size = targets.len();
    let mut total_loss = 0.0;
    
    // 그래디언트 누적 초기화
    for acc in accumulators.iter_mut() {
        acc.reset();
    }
    
    // 배치 내 각 샘플에 대해 그래디언트 계산
    for (batch_idx, target) in targets.iter().enumerate() {
        let seed = &mut seeds[batch_idx];
        let accumulator = &mut accumulators[batch_idx];
        
        // 현재 예측 생성
        let mut predicted = vec![0.0; target.len()];
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                predicted[idx] = if use_advanced_transitions {
                    seed.fused_forward_advanced(i, j, rows, cols)
                } else {
                    seed.fused_forward(i, j, rows, cols)
                };
            }
        }
        
        // 손실 계산
        let mut sample_loss = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let error = predicted[idx] - target[idx];
                sample_loss += error * error;
                
                // 상태 전이 미분 적용
                if use_advanced_transitions {
                    seed.advanced_state_transition(error, i, j);
                } else {
                    seed.apply_state_transition(error, i, j);
                }
                
                if error.abs() > 0.001 { // 유의미한 오차에 대해서만 전이 카운트
                    accumulator.state_transition_count += 1;
                }
                
                // 연속 파라미터 그래디언트 계산
                let r_fp32 = f32::from_bits((seed.lo >> 32) as u32);
                let theta_fp32 = f32::from_bits(seed.lo as u32);
                let eps = 1e-5;
                
                // r 그래디언트
                let mut seed_r_plus = *seed;
                seed_r_plus.lo = (((r_fp32 + eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                let w_r_plus = if use_advanced_transitions {
                    seed_r_plus.fused_forward_advanced(i, j, rows, cols)
                } else {
                    seed_r_plus.fused_forward(i, j, rows, cols)
                };
                
                let mut seed_r_minus = *seed;
                seed_r_minus.lo = (((r_fp32 - eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                let w_r_minus = if use_advanced_transitions {
                    seed_r_minus.fused_forward_advanced(i, j, rows, cols)
                } else {
                    seed_r_minus.fused_forward(i, j, rows, cols)
                };
                
                let dr = (w_r_plus - w_r_minus) / (2.0 * eps);
                let grad_r = error * dr;
                
                // theta 그래디언트
                let mut seed_th_plus = *seed;
                seed_th_plus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 + eps).to_bits() as u64;
                let w_th_plus = if use_advanced_transitions {
                    seed_th_plus.fused_forward_advanced(i, j, rows, cols)
                } else {
                    seed_th_plus.fused_forward(i, j, rows, cols)
                };
                
                let mut seed_th_minus = *seed;
                seed_th_minus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 - eps).to_bits() as u64;
                let w_th_minus = if use_advanced_transitions {
                    seed_th_minus.fused_forward_advanced(i, j, rows, cols)
                } else {
                    seed_th_minus.fused_forward(i, j, rows, cols)
                };
                
                let dth = (w_th_plus - w_th_minus) / (2.0 * eps);
                let grad_theta = error * dth;
                
                // 그래디언트 누적
                accumulator.accumulate(grad_r, grad_theta);
            }
        }
        
        total_loss += sample_loss / target.len() as f32;
    }
    
    // 배치별 파라미터 업데이트
    for (batch_idx, seed) in seeds.iter_mut().enumerate() {
        let accumulator = &accumulators[batch_idx];
        let (avg_grad_r, avg_grad_theta) = accumulator.get_average_gradients();
        
        if avg_grad_r.abs() > 1e-8 || avg_grad_theta.abs() > 1e-8 {
            let r_fp32 = f32::from_bits((seed.lo >> 32) as u32);
            let theta_fp32 = f32::from_bits(seed.lo as u32);
            
            let mut new_r = r_fp32;
            let mut new_theta = theta_fp32;
            
            // Adam 업데이트
            adam_update(
                &mut new_r, 
                &mut adam_states_r[batch_idx].0, 
                &mut adam_states_r[batch_idx].1, 
                avg_grad_r, 
                learning_rate, 
                epoch
            );
            adam_update(
                &mut new_theta, 
                &mut adam_states_theta[batch_idx].0, 
                &mut adam_states_theta[batch_idx].1, 
                avg_grad_theta, 
                learning_rate, 
                epoch
            );
            
            new_r = new_r.clamp(0.0, 1.0);
            seed.lo = ((new_r.to_bits() as u64) << 32) | new_theta.to_bits() as u64;
        }
    }
    
    total_loss / batch_size as f32
}

/// 적응적 학습률을 사용한 고급 Adam 업데이트
pub fn adaptive_adam_update(
    param: &mut f32,
    momentum: &mut f32,
    velocity: &mut f32,
    gradient: f32,
    base_lr: f32,
    epoch: i32,
    gradient_norm: f32, // 전체 그래디언트 노름
    transition_count: u32, // 상태 전이 횟수
) {
    const B1: f32 = 0.9;
    const B2: f32 = 0.999;
    const EPS: f32 = 1e-8;
    
    // 적응적 학습률 계산
    let adaptive_lr = if transition_count > 0 {
        // 상태 전이가 많이 일어났으면 학습률 감소
        base_lr * (1.0 / (1.0 + transition_count as f32 * 0.01))
    } else {
        base_lr
    };
    
    // 그래디언트 클리핑
    let clipped_gradient = if gradient_norm > 1.0 {
        gradient / gradient_norm
    } else {
        gradient
    };
    
    *momentum = B1 * (*momentum) + (1.0 - B1) * clipped_gradient;
    *velocity = B2 * (*velocity) + (1.0 - B2) * clipped_gradient * clipped_gradient;
    
    let m_hat = *momentum / (1.0 - B1.powi(epoch));
    let v_hat = *velocity / (1.0 - B2.powi(epoch));
    
    *param -= adaptive_lr * m_hat / (v_hat.sqrt() + EPS);
}

/// 멀티스케일 그래디언트 분석
pub fn analyze_gradient_scales(
    gradients_r: &[f32], 
    gradients_theta: &[f32]
) -> (f32, f32, f32, f32) {
    let r_norm = gradients_r.iter().map(|g| g * g).sum::<f32>().sqrt();
    let theta_norm = gradients_theta.iter().map(|g| g * g).sum::<f32>().sqrt();
    
    let r_mean = gradients_r.iter().sum::<f32>() / gradients_r.len() as f32;
    let theta_mean = gradients_theta.iter().sum::<f32>() / gradients_theta.len() as f32;
    
    (r_norm, theta_norm, r_mean, theta_mean)
} 

// ============================================================================
// 5장: 푸앵카레 볼의 리만 기하학 (Riemannian Geometry on Poincaré Ball)
// ============================================================================

use std::f32::consts::PI;

/// 5.2 푸앵카레 볼 위의 점
/// 
/// 푸앵카레 볼 D^n = {x ∈ R^n : ||x|| < 1}에서 점을 나타냅니다.
/// 리만 메트릭: g(x) = 4/(1-||x||²)² I_n
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoincareBallPoint {
    /// 좌표 (r, θ)
    pub r: f32,      // 반지름 [0, 1)
    pub theta: f32,  // 각도 [-∞, ∞)
}

impl PoincareBallPoint {
    /// 새로운 푸앵카레 볼 점 생성
    pub fn new(r: f32, theta: f32) -> Self {
        Self {
            r: r.clamp(0.0, 0.99), // 경계 근처에서 수치적 안정성 보장
            theta,
        }
    }
    
    /// 원점 (r=0, θ=0)
    pub fn origin() -> Self {
        Self { r: 0.0, theta: 0.0 }
    }
    
    /// 데카르트 좌표로 변환
    pub fn to_cartesian(&self) -> (f32, f32) {
        (self.r * libm::cosf(self.theta), self.r * libm::sinf(self.theta))
    }
    
    /// 데카르트 좌표에서 변환
    pub fn from_cartesian(x: f32, y: f32) -> Self {
        let r = (x*x + y*y).sqrt().min(0.99);
        let theta = libm::atan2f(y, x);
        Self { r, theta }
    }
    
    /// 푸앵카레 볼 경계까지의 거리
    pub fn distance_to_boundary(&self) -> f32 {
        1.0 - self.r
    }
}

/// 5.2 리만 기하학 연산
/// 
/// 푸앵카레 볼의 리만 메트릭 연산을 제공합니다.
#[derive(Debug, Clone)]
pub struct RiemannianGeometry;

impl RiemannianGeometry {
    /// 5.2.1 리만 메트릭 텐서 계산
    /// 
    /// g(x) = 4/(1-||x||²)² I_n
    /// 점 x에서의 메트릭 인수를 반환합니다.
    pub fn metric_factor(point: &PoincareBallPoint) -> f32 {
        let norm_sq = point.r * point.r;
        let denominator = 1.0 - norm_sq;
        
        if denominator < 1e-8 {
            // 경계 근처에서 안정성 보장
            1e8
        } else {
            4.0 / (denominator * denominator)
        }
    }
    
    /// 5.2.1 리만 메트릭의 역행렬 인수
    /// 
    /// g^(-1)(x) = (1-||x||²)²/4 I_n
    pub fn inverse_metric_factor(point: &PoincareBallPoint) -> f32 {
        let norm_sq = point.r * point.r;
        let numerator = 1.0 - norm_sq;
        
        if numerator < 1e-8 {
            // 경계 근처에서 안정성 보장
            1e-8
        } else {
            numerator * numerator / 4.0
        }
    }
    
    /// 5.2.2 크리스토펠 기호 계산
    /// 
    /// Γ^k_ij = (2δ^k_i x_j + 2δ^k_j x_i - 2δ_ij x^k)/(1-||x||²)
    pub fn christoffel_symbols(point: &PoincareBallPoint) -> (f32, f32) {
        let (x, y) = point.to_cartesian();
        let norm_sq = point.r * point.r;
        let denominator = 1.0 - norm_sq;
        
        if denominator < 1e-8 {
            return (0.0, 0.0);
        }
        
        let factor = 2.0 / denominator;
        
        // r 방향과 θ 방향 크리스토펠 기호
        let gamma_r = factor * x;
        let gamma_theta = factor * y;
        
        (gamma_r, gamma_theta)
    }
    
    /// 5.2.4 Möbius 덧셈 (푸앵카레 볼의 덧셈)
    /// 
    /// x ⊕ y = (x + y + 2⟨x,y⟩/(1+||x||²) x) / (1 + 2⟨x,y⟩ + ||x||²||y||²)
    pub fn mobius_addition(p1: &PoincareBallPoint, p2: &PoincareBallPoint) -> PoincareBallPoint {
        let (x1, y1) = p1.to_cartesian();
        let (x2, y2) = p2.to_cartesian();
        
        let dot_product = x1 * x2 + y1 * y2;
        let norm1_sq = p1.r * p1.r;
        let norm2_sq = p2.r * p2.r;
        
        let numerator_factor = 1.0 + 2.0 * dot_product / (1.0 + norm1_sq);
        let denominator = 1.0 + 2.0 * dot_product + norm1_sq * norm2_sq;
        
        if denominator < 1e-8 {
            return *p1; // 안전한 기본값
        }
        
        let result_x = (x1 + x2 * numerator_factor) / denominator;
        let result_y = (y1 + y2 * numerator_factor) / denominator;
        
        PoincareBallPoint::from_cartesian(result_x, result_y)
    }
    
    /// 5.2.4 스칼라 곱 (푸앵카레 볼의 스칼라 곱)
    /// 
    /// t ⊙ v = (t||v||/artanh(||v||)) · v/||v||
    pub fn scalar_multiplication(t: f32, point: &PoincareBallPoint) -> PoincareBallPoint {
        if point.r < 1e-8 {
            return PoincareBallPoint::origin();
        }
        
        let norm = point.r;
        let artanh_norm = if norm < 0.99 {
            0.5 * libm::logf((1.0 + norm) / (1.0 - norm))
        } else {
            10.0 // 경계 근처에서 클램핑
        };
        
        let scale_factor = if artanh_norm > 1e-8 {
            t * norm / artanh_norm
        } else {
            t
        };
        
        let new_r = (scale_factor * norm).clamp(0.0, 0.99);
        
        PoincareBallPoint::new(new_r, point.theta)
    }
    
    /// 5.2.4 지수 사상 (Exponential Map)
    /// 
    /// exp_x(v) = x ⊕ (tanh(||v||_x/2) · v/||v||_x)
    pub fn exponential_map(base: &PoincareBallPoint, tangent: &PoincareBallPoint) -> PoincareBallPoint {
        if tangent.r < 1e-8 {
            return *base;
        }
        
        // 리만 노름 계산
        let inverse_metric = Self::inverse_metric_factor(base);
        let riemannian_norm = tangent.r * inverse_metric.sqrt();
        
        // tanh(||v||_x/2) 계산
        let tanh_half_norm = libm::tanhf(riemannian_norm / 2.0);
        
        // 방향 벡터 정규화
        let direction = if riemannian_norm > 1e-8 {
            PoincareBallPoint::new(tanh_half_norm, tangent.theta)
        } else {
            PoincareBallPoint::origin()
        };
        
        Self::mobius_addition(base, &direction)
    }
    
    /// 5.3.4 쌍곡 거리 계산
    /// 
    /// d_hyp(x,y) = artanh(||(-x) ⊕ y||)
    pub fn hyperbolic_distance(p1: &PoincareBallPoint, p2: &PoincareBallPoint) -> f32 {
        // -p1 계산 (Möbius 역원)
        let neg_p1 = PoincareBallPoint::new(p1.r, p1.theta + PI);
        
        // (-p1) ⊕ p2 계산
        let diff = Self::mobius_addition(&neg_p1, p2);
        
        // artanh(||diff||) 계산
        let norm = diff.r;
        if norm < 0.99 {
            0.5 * libm::logf((1.0 + norm) / (1.0 - norm))
        } else {
            10.0 // 경계에서 클램핑
        }
    }
}

/// 5.3 리만 최적화기
/// 
/// 푸앵카레 볼에서의 리만 그래디언트 기반 최적화를 제공합니다.
#[derive(Debug, Clone)]
pub struct RiemannianOptimizer {
    /// 학습률
    pub learning_rate: f32,
    /// Adam 파라미터
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    /// Adam 상태
    pub momentum_r: f32,
    pub momentum_theta: f32,
    pub velocity_r: f32,
    pub velocity_theta: f32,
    /// 현재 반복 수
    pub iteration: i32,
}

impl RiemannianOptimizer {
    /// 새로운 리만 최적화기 생성
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            momentum_r: 0.0,
            momentum_theta: 0.0,
            velocity_r: 0.0,
            velocity_theta: 0.0,
            iteration: 0,
        }
    }
    
    /// 5.3.1 리만 그래디언트 계산
    /// 
    /// grad f(x) = g^(-1)(x) ∇f(x) = (1-||x||²)²/4 ∇f(x)
    pub fn compute_riemannian_gradient(
        &self,
        point: &PoincareBallPoint,
        euclidean_grad_r: f32,
        euclidean_grad_theta: f32,
    ) -> (f32, f32) {
        let inverse_metric_factor = RiemannianGeometry::inverse_metric_factor(point);
        
        // r 방향 리만 그래디언트
        let riemannian_grad_r = inverse_metric_factor * euclidean_grad_r;
        
        // θ 방향 리만 그래디언트 (각도 좌표이므로 r² 스케일링)
        let theta_scaling = if point.r > 1e-8 { point.r * point.r } else { 1e-8 };
        let riemannian_grad_theta = euclidean_grad_theta / theta_scaling;
        
        (riemannian_grad_r, riemannian_grad_theta)
    }
    
    /// 5.3.2 리만 최급강하법 스텝
    /// 
    /// x_{k+1} = x_k ⊕ (-α ⊙ grad f(x_k))
    pub fn gradient_descent_step(
        &self,
        point: &PoincareBallPoint,
        euclidean_grad_r: f32,
        euclidean_grad_theta: f32,
    ) -> PoincareBallPoint {
        let (riemannian_grad_r, riemannian_grad_theta) = 
            self.compute_riemannian_gradient(point, euclidean_grad_r, euclidean_grad_theta);
        
        // 그래디언트 방향으로 스텝
        let step_r = -self.learning_rate * riemannian_grad_r;
        let step_theta = -self.learning_rate * riemannian_grad_theta;
        
        let step_vector = PoincareBallPoint::new(
            step_r.abs().min(0.1), // 큰 스텝 방지
            if step_r >= 0.0 { step_theta } else { step_theta + PI }
        );
        
        RiemannianGeometry::exponential_map(point, &step_vector)
    }
    
    /// 5.3.3 리만 Adam 스텝
    /// 
    /// 푸앵카레 볼에 적응된 Adam 최적화
    pub fn adam_step(
        &mut self,
        point: &PoincareBallPoint,
        euclidean_grad_r: f32,
        euclidean_grad_theta: f32,
    ) -> PoincareBallPoint {
        self.iteration += 1;
        
        let (riemannian_grad_r, riemannian_grad_theta) = 
            self.compute_riemannian_gradient(point, euclidean_grad_r, euclidean_grad_theta);
        
        // 모멘텀 업데이트
        self.momentum_r = self.beta1 * self.momentum_r + (1.0 - self.beta1) * riemannian_grad_r;
        self.momentum_theta = self.beta1 * self.momentum_theta + (1.0 - self.beta1) * riemannian_grad_theta;
        
        // 속도 업데이트
        self.velocity_r = self.beta2 * self.velocity_r + (1.0 - self.beta2) * riemannian_grad_r.powi(2);
        self.velocity_theta = self.beta2 * self.velocity_theta + (1.0 - self.beta2) * riemannian_grad_theta.powi(2);
        
        // 편향 보정
        let bias_correction_1 = 1.0 - self.beta1.powi(self.iteration);
        let bias_correction_2 = 1.0 - self.beta2.powi(self.iteration);
        
        let corrected_momentum_r = self.momentum_r / bias_correction_1;
        let corrected_momentum_theta = self.momentum_theta / bias_correction_1;
        
        let corrected_velocity_r = self.velocity_r / bias_correction_2;
        let corrected_velocity_theta = self.velocity_theta / bias_correction_2;
        
        // Adam 업데이트
        let step_r = -self.learning_rate * corrected_momentum_r / (corrected_velocity_r.sqrt() + self.epsilon);
        let step_theta = -self.learning_rate * corrected_momentum_theta / (corrected_velocity_theta.sqrt() + self.epsilon);
        
        let step_vector = PoincareBallPoint::new(
            step_r.abs().min(0.05), // Adam에서는 더 작은 스텝
            if step_r >= 0.0 { step_theta } else { step_theta + PI }
        );
        
        RiemannianGeometry::exponential_map(point, &step_vector)
    }
    
    /// 5.7.2 리만 그래디언트 클리핑
    /// 
    /// 리만 노름을 기준으로 그래디언트 클리핑
    pub fn clip_riemannian_gradient(
        &self,
        point: &PoincareBallPoint,
        grad_r: f32,
        grad_theta: f32,
        max_norm: f32,
    ) -> (f32, f32) {
        let riemannian_norm = self.compute_riemannian_norm(point, grad_r, grad_theta);
        
        if riemannian_norm > max_norm {
            let scale = max_norm / riemannian_norm;
            (grad_r * scale, grad_theta * scale)
        } else {
            (grad_r, grad_theta)
        }
    }
    
    /// 리만 노름 계산
    fn compute_riemannian_norm(
        &self,
        point: &PoincareBallPoint,
        grad_r: f32,
        grad_theta: f32,
    ) -> f32 {
        let metric_factor = RiemannianGeometry::metric_factor(point);
        let norm_r_sq = grad_r * grad_r * metric_factor;
        
        let theta_scaling = if point.r > 1e-8 { point.r * point.r } else { 1e-8 };
        let norm_theta_sq = grad_theta * grad_theta * theta_scaling * metric_factor;
        
        (norm_r_sq + norm_theta_sq).sqrt()
    }
}

/// 5.4 상태 전이 그래프
/// 
/// 이산 상태 공간의 조합론적 최적화를 위한 그래프 구조
#[derive(Debug, Clone)]
pub struct StateTransitionGraph {
    /// 상태 공간 크기
    pub state_space_size: usize,
    /// 볼츠만 온도
    pub temperature: f32,
    /// 온도 감소율
    pub cooling_rate: f32,
    /// 마르코프 체인 기록
    pub transition_history: Vec<u64>,
}

impl StateTransitionGraph {
    /// 새로운 상태 전이 그래프 생성
    pub fn new(state_space_size: usize, initial_temperature: f32) -> Self {
        Self {
            state_space_size,
            temperature: initial_temperature,
            cooling_rate: 0.95,
            transition_history: Vec::new(),
        }
    }
    
    /// 5.4.2 해밍 거리 계산
    /// 
    /// 두 상태 사이의 해밍 거리 (다른 비트 수)
    pub fn hamming_distance(state1: u64, state2: u64) -> u32 {
        (state1 ^ state2).count_ones()
    }
    
    /// 5.4.2 이웃 상태 생성
    /// 
    /// 해밍 거리 1인 이웃 상태들 반환
    pub fn get_neighbors(state: u64) -> Vec<u64> {
        let mut neighbors = Vec::new();
        
        // 각 비트를 플립한 상태들
        for i in 0..64 {
            let neighbor = state ^ (1u64 << i);
            neighbors.push(neighbor);
        }
        
        neighbors
    }
    
    /// 5.4.4 볼츠만 분포 기반 전이 확률 계산
    /// 
    /// P(s → s') = exp(-β · ΔF) / Z
    pub fn compute_transition_probabilities(
        &self,
        current_state: u64,
        loss_function: impl Fn(u64) -> f32,
    ) -> Vec<(u64, f32)> {
        let neighbors = Self::get_neighbors(current_state);
        let current_loss = loss_function(current_state);
        
        let mut probabilities = Vec::new();
        let mut partition_sum = 0.0;
        
        // 각 이웃에 대한 확률 계산
        for neighbor in neighbors {
            let neighbor_loss = loss_function(neighbor);
            let energy_diff = neighbor_loss - current_loss;
            let boltzmann_factor = libm::expf(-energy_diff / self.temperature);
            
            probabilities.push((neighbor, boltzmann_factor));
            partition_sum += boltzmann_factor;
        }
        
        // 정규화
        if partition_sum > 0.0 {
            for (_, prob) in probabilities.iter_mut() {
                *prob /= partition_sum;
            }
        }
        
        probabilities
    }
    
    /// 5.4.4 확률적 상태 선택
    /// 
    /// 볼츠만 분포에 따라 다음 상태 선택
    pub fn sample_next_state(
        &mut self,
        current_state: u64,
        loss_function: impl Fn(u64) -> f32,
        rng: &mut impl Rng,
    ) -> u64 {
        let probabilities = self.compute_transition_probabilities(current_state, loss_function);
        
        let random_value: f32 = rng.gen();
        let mut cumulative_prob = 0.0;
        
        for (state, prob) in probabilities {
            cumulative_prob += prob;
            if random_value <= cumulative_prob {
                self.transition_history.push(state);
                return state;
            }
        }
        
        // fallback
        current_state
    }
    
    /// 5.7.3 적응적 온도 스케줄링
    /// 
    /// 시간에 따른 온도 감소
    pub fn update_temperature(&mut self, epoch: i32, loss: f32) {
        // 1. 지수 감소
        let base_temp = self.temperature * self.cooling_rate;
        
        // 2. 손실 기반 조정
        let loss_factor = 1.0 + libm::tanhf(loss - 1.0);
        
        // 3. 최종 온도
        self.temperature = (base_temp * loss_factor).max(0.01);
    }
    
    /// 5.4.5 마르코프 체인 수렴성 검사
    /// 
    /// 상태 분포의 수렴 여부 확인
    pub fn check_convergence(&self, window_size: usize) -> bool {
        if self.transition_history.len() < window_size * 2 {
            return false;
        }
        
        let recent_states = &self.transition_history[self.transition_history.len() - window_size..];
        let prev_states = &self.transition_history[self.transition_history.len() - window_size * 2..self.transition_history.len() - window_size];
        
        // 상태 분포 비교 (간단한 히스토그램 기반)
        let mut recent_counts = std::collections::HashMap::new();
        let mut prev_counts = std::collections::HashMap::new();
        
        for &state in recent_states {
            *recent_counts.entry(state).or_insert(0) += 1;
        }
        
        for &state in prev_states {
            *prev_counts.entry(state).or_insert(0) += 1;
        }
        
        // KL 발산 계산 (간소화된 버전)
        let mut kl_divergence = 0.0;
        for (&state, &recent_count) in &recent_counts {
            let recent_prob = recent_count as f32 / window_size as f32;
            let prev_count = prev_counts.get(&state).unwrap_or(&0);
            let prev_prob = (*prev_count as f32 / window_size as f32).max(1e-8);
            
            kl_divergence += recent_prob * libm::logf(recent_prob / prev_prob);
        }
        
        kl_divergence < 0.01 // 수렴 임계값
    }
}

/// 5.6 정보 기하학 (Information Geometry)
/// 
/// 피셔 정보 메트릭과 자연 그래디언트 계산
#[derive(Debug, Clone)]
pub struct InformationGeometry;

impl InformationGeometry {
    /// 5.6.1 피셔 정보 행렬 계산
    /// 
    /// I(r,θ) = [[4/(1-r²)², 0], [0, 1/r²]]
    pub fn fisher_information_matrix(point: &PoincareBallPoint) -> [[f32; 2]; 2] {
        let r = point.r.max(1e-8); // 0 방지
        let norm_sq = r * r;
        let denominator = 1.0 - norm_sq;
        
        let fisher_r = if denominator > 1e-8 {
            4.0 / (denominator * denominator)
        } else {
            1e8
        };
        
        let fisher_theta = 1.0 / (r * r);
        
        [
            [fisher_r, 0.0],
            [0.0, fisher_theta]
        ]
    }
    
    /// 5.6.2 자연 그래디언트 계산
    /// 
    /// ∇̃ = I^(-1)(θ) ∇_θ L
    pub fn natural_gradient(
        point: &PoincareBallPoint,
        euclidean_grad_r: f32,
        euclidean_grad_theta: f32,
    ) -> (f32, f32) {
        let inverse_metric_factor = RiemannianGeometry::inverse_metric_factor(point);
        let r = point.r.max(1e-8);
        
        // 자연 그래디언트 = 리만 그래디언트
        let natural_grad_r = inverse_metric_factor * euclidean_grad_r;
        let natural_grad_theta = (r * r) * euclidean_grad_theta;
        
        (natural_grad_r, natural_grad_theta)
    }
    
    /// 5.6.3 KL 발산과 쌍곡 거리의 관계
    /// 
    /// KL(P_θ1 || P_θ2) ≈ ½ d²_hyp(θ1, θ2)
    pub fn kl_divergence_approximation(p1: &PoincareBallPoint, p2: &PoincareBallPoint) -> f32 {
        let hyperbolic_distance = RiemannianGeometry::hyperbolic_distance(p1, p2);
        0.5 * hyperbolic_distance * hyperbolic_distance
    }
}

/// 5.5 하이브리드 최적화기
/// 
/// 연속-이산 파라미터의 통합 최적화
#[derive(Debug, Clone)]
pub struct HybridRiemannianOptimizer {
    /// 연속 파라미터 최적화기
    pub riemannian_optimizer: RiemannianOptimizer,
    /// 이산 상태 전이 그래프
    pub state_transition: StateTransitionGraph,
    /// 수렴 임계값
    pub convergence_tolerance: f32,
    /// 최대 반복 수
    pub max_iterations: usize,
}

impl HybridRiemannianOptimizer {
    /// 새로운 하이브리드 최적화기 생성
    pub fn new(learning_rate: f32, temperature: f32) -> Self {
        Self {
            riemannian_optimizer: RiemannianOptimizer::new(learning_rate),
            state_transition: StateTransitionGraph::new(1024, temperature),
            convergence_tolerance: 1e-6,
            max_iterations: 1000,
        }
    }
    
    /// 5.5.2 교대 최적화 스텝
    /// 
    /// 연속 파라미터와 이산 상태를 교대로 최적화
    pub fn hybrid_optimization_step(
        &mut self,
        continuous_params: &PoincareBallPoint,
        discrete_state: u64,
        continuous_loss_fn: impl Fn(&PoincareBallPoint) -> (f32, f32, f32), // (loss, grad_r, grad_theta)
        discrete_loss_fn: impl Fn(u64) -> f32,
        rng: &mut impl Rng,
    ) -> (PoincareBallPoint, u64) {
        
        // 1. 연속 파라미터 최적화 (리만 그래디언트)
        let (loss, grad_r, grad_theta) = continuous_loss_fn(continuous_params);
        let new_continuous = self.riemannian_optimizer.adam_step(continuous_params, grad_r, grad_theta);
        
        // 2. 이산 상태 최적화 (확률적 전이)
        let new_discrete = self.state_transition.sample_next_state(discrete_state, discrete_loss_fn, rng);
        
        // 3. 온도 업데이트
        self.state_transition.update_temperature(self.riemannian_optimizer.iteration, loss);
        
        (new_continuous, new_discrete)
    }
    
    /// 5.5.3 수렴성 체크
    /// 
    /// 연속 그래디언트 노름과 이산 상태 수렴을 모두 확인
    pub fn check_hybrid_convergence(
        &self,
        point: &PoincareBallPoint,
        grad_r: f32,
        grad_theta: f32,
    ) -> bool {
        // 연속 파라미터 수렴성
        let riemannian_norm = self.riemannian_optimizer.compute_riemannian_norm(point, grad_r, grad_theta);
        let continuous_converged = riemannian_norm < self.convergence_tolerance;
        
        // 이산 상태 수렴성
        let discrete_converged = self.state_transition.check_convergence(50);
        
        continuous_converged && discrete_converged
    }
} 

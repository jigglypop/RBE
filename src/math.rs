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

use crate::packed_params::Packed128;
use super::basic_math::adam_update;

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
    epoch: i32
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
                predicted[idx] = seed.fused_forward(i, j, rows, cols);
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
                seed.apply_state_transition(error, i, j);
                
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
                let w_r_plus = seed_r_plus.fused_forward(i, j, rows, cols);
                
                let mut seed_r_minus = *seed;
                seed_r_minus.lo = (((r_fp32 - eps).to_bits() as u64) << 32) | theta_fp32.to_bits() as u64;
                let w_r_minus = seed_r_minus.fused_forward(i, j, rows, cols);
                
                let dr = (w_r_plus - w_r_minus) / (2.0 * eps);
                let grad_r = error * dr;
                
                // theta 그래디언트
                let mut seed_th_plus = *seed;
                seed_th_plus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 + eps).to_bits() as u64;
                let w_th_plus = seed_th_plus.fused_forward(i, j, rows, cols);
                
                let mut seed_th_minus = *seed;
                seed_th_minus.lo = ((r_fp32.to_bits() as u64) << 32) | (theta_fp32 - eps).to_bits() as u64;
                let w_th_minus = seed_th_minus.fused_forward(i, j, rows, cols);
                
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
 
/// 해석적 그래디언트 계산을 위한 Packed128 확장 트레이트
pub trait AnalyticalGradient {
    /// R 파라미터에 대한 해석적 그래디언트 (안정적 비트 연산)
    fn analytical_gradient_r(&self, i: usize, j: usize, _rows: usize, _cols: usize) -> f32;
    
    /// Theta 파라미터에 대한 해석적 그래디언트 (안정적 비트 연산)
    fn analytical_gradient_theta(&self, i: usize, j: usize, _rows: usize, _cols: usize) -> f32;
}

impl AnalyticalGradient for Packed128 {
    /// R 파라미터에 대한 해석적 그래디언트 (안정적 비트 연산)
    fn analytical_gradient_r(&self, i: usize, j: usize, _rows: usize, _cols: usize) -> f32 {
        // 단순하지만 안정적인 접근법
        
        let r_bits = (self.lo >> 32) as u32;
        
        // 1. 안전한 perturbation
        let r_plus_bits = r_bits.saturating_add(1);
        let r_minus_bits = r_bits.saturating_sub(1);
        
        // 2. 안전한 차분 계산 (오버플로우 방지)
        let bit_diff = r_plus_bits.wrapping_sub(r_minus_bits) as i64;
        
        // 3. 상태 기반 가중치 (간단하게)
        let position_hash = ((i * 31 + j * 17) & 0x3F) as u64;
        let state_bits = (self.hi >> (position_hash % 64)) & 0xFF;
        
        // 4. 정규화된 가중치 [-1, 1]
        let state_weight = (state_bits as f32 - 128.0) / 128.0;
        
        // 5. 위치 가중치
        let pos_weight = ((i ^ j) as f32 / 31.0) - 0.5;  // [-0.5, 0.5]
        
        // 6. 결합된 가중치
        let combined_weight = state_weight * (1.0 + pos_weight);
        
        // 7. 안전한 스케일링 (고정값 사용)
        let gradient = (bit_diff as f32) * combined_weight * 1e-7;
        
        // 8. 안전한 범위 제한
        gradient.clamp(-1.0, 1.0)
    }

    /// Theta 파라미터에 대한 해석적 그래디언트 (안정적 비트 연산)
    fn analytical_gradient_theta(&self, i: usize, j: usize, _rows: usize, _cols: usize) -> f32 {
        // 동일한 안정적 접근법
        
        let theta_bits = self.lo as u32;
        
        // 1. 안전한 perturbation
        let theta_plus_bits = theta_bits.saturating_add(1);
        let theta_minus_bits = theta_bits.saturating_sub(1);
        
        // 2. 안전한 차분
        let bit_diff = theta_plus_bits.wrapping_sub(theta_minus_bits) as i64;
        
        // 3. 다른 해시 패턴
        let position_hash = ((i * 23 + j * 41) & 0x3F) as u64;
        let state_bits = (self.hi >> ((position_hash + 32) % 64)) & 0xFF;
        
        // 4. 정규화된 가중치
        let state_weight = (state_bits as f32 - 128.0) / 128.0;
        
        // 5. 각도 특성 가중치
        let angle_weight = ((i + j) as f32 / 31.0) - 0.5;
        
        // 6. 결합
        let combined_weight = state_weight * (1.0 + angle_weight * 0.5);
        
        // 7. 더 작은 스케일링 (theta는 더 민감)
        let gradient = (bit_diff as f32) * combined_weight * 1e-8;
        
        // 8. 안전한 범위
        gradient.clamp(-1.0, 1.0)
    }
}
 
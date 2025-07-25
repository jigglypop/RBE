//! 모멘텀(Momentum) 옵티마이저 - 정밀 수학적 구현

use crate::core::tensors::{Packed128, DecodedParams};

/// 모멘텀 옵티마이저
/// 정확한 수학적 그래디언트와 속도(velocity) 항을 사용하여 수렴 가속
#[derive(Debug, Clone)]
pub struct MomentumOptimizer {
    beta: f32,               // 모멘텀 계수 (일반적으로 0.9)
    velocity_r: f32,         // r 방향 속도
    velocity_theta: f32,     // θ 방향 속도
    use_riemannian: bool,    // 리만 기하학 적용 여부
}

impl MomentumOptimizer {
    pub fn new(beta: f32) -> Self {
        Self {
            beta,
            velocity_r: 0.0,
            velocity_theta: 0.0,
            use_riemannian: false,
        }
    }
    
    pub fn with_riemannian(beta: f32, use_riemannian: bool) -> Self {
        Self {
            beta,
            velocity_r: 0.0,
            velocity_theta: 0.0,
            use_riemannian,
        }
    }
    
    /// 정확한 수학적 그래디언트를 사용한 모멘텀 업데이트
    pub fn update(
        &mut self,
        packed: &mut Packed128,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
        target: f32,
        learning_rate: f32,
    ) {
        // 1. 정확한 그래디언트 계산
        let (grad_r, grad_theta) = if self.use_riemannian {
            // 리만 자연 그래디언트
            packed.compute_riemannian_gradients(i, j, rows, cols, target, false) // L2 손실 사용
        } else {
            // 유클리드 그래디언트
            let (gr, gt, _) = packed.compute_gradients(i, j, rows, cols, target, false); // L2 손실 사용
            (gr, gt)
        };
        
        // 2. 모멘텀 업데이트: v = β*v + g
        self.velocity_r = self.beta * self.velocity_r + grad_r;
        self.velocity_theta = self.beta * self.velocity_theta + grad_theta;
        
        // 3. 현재 파라미터
        let mut params = packed.decode();
        
        // 4. 파라미터 업데이트: θ = θ - α*v
        params.r_fp32 -= learning_rate * self.velocity_r;
        params.theta_fp32 -= learning_rate * self.velocity_theta;
        
        // 5. 범위 제약
        params.r_fp32 = params.r_fp32.clamp(0.0, 0.999999);
        params.theta_fp32 = params.theta_fp32.rem_euclid(2.0 * std::f32::consts::PI);
        
        // 6. 업데이트된 파라미터 적용
        packed.update_from_continuous(&params);
    }
    
    /// 간단한 인터페이스 (이전 버전과의 호환성)
    pub fn update_simple(
        &mut self,
        packed: &mut Packed128,
        predicted: f32,
        target: f32,
        learning_rate: f32,
    ) {
        // 더미 좌표로 호출 (실제로는 사용하지 않는 것을 권장)
        self.update(packed, 0, 0, 1, 1, target, learning_rate);
    }
    
    /// 속도 상태 초기화
    pub fn reset_velocity(&mut self) {
        self.velocity_r = 0.0;
        self.velocity_theta = 0.0;
    }
    
    /// 현재 속도 상태 확인
    pub fn get_velocity(&self) -> (f32, f32) {
        (self.velocity_r, self.velocity_theta)
    }
} 
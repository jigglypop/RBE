//! 기본 경사하강법(Gradient Descent) 옵티마이저 - 정밀 수학적 구현

use crate::core::tensors::{Packed128, DecodedParams};

/// 기본 경사하강법 옵티마이저
/// 수학적으로 정확한 그래디언트를 사용하여 파라미터 업데이트
#[derive(Debug, Clone)]
pub struct GradientDescent {
    // 상태가 없는 단순 옵티마이저
    use_riemannian: bool,  // 리만 기하학 적용 여부
}

impl GradientDescent {
    pub fn new() -> Self {
        Self {
            use_riemannian: false,  // 기본적으로 유클리드 그래디언트 사용
        }
    }
    
    pub fn with_riemannian(use_riemannian: bool) -> Self {
        Self {
            use_riemannian,
        }
    }
    
    /// 정확한 수학적 그래디언트를 사용한 파라미터 업데이트
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
        
        // 2. 현재 파라미터
        let mut params = packed.decode();
        
        // 3. 경사하강법 업데이트
        params.r_fp32 -= learning_rate * grad_r;
        params.theta_fp32 -= learning_rate * grad_theta;
        
        // 4. 범위 제약
        params.r_fp32 = params.r_fp32.clamp(0.0, 0.999999);
        params.theta_fp32 = params.theta_fp32.rem_euclid(2.0 * std::f32::consts::PI);
        
        // 5. 업데이트된 파라미터 적용
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
} 
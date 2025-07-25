//! BitAdam 옵티마이저 - 정밀 수학적 구현
//! 
//! Adaptive Moment Estimation (Adam)을 RBE 시스템에 적용
//! 정확한 수학적 그래디언트와 적응적 학습률을 사용

use crate::core::tensors::{Packed128, DecodedParams};

/// BitAdam 옵티마이저 상태
/// Adam 알고리즘의 1차/2차 모멘트를 유지하며 적응적 학습률 제공
#[derive(Debug, Clone)]
pub struct BitAdamState {
    // Adam 하이퍼파라미터
    beta1: f32,         // 1차 모멘트 지수이동평균 계수 (기본 0.9)
    beta2: f32,         // 2차 모멘트 지수이동평균 계수 (기본 0.999)
    epsilon: f32,       // 수치 안정성을 위한 작은 값 (기본 1e-8)
    
    // 모멘트 상태
    m_r: f32,           // r에 대한 1차 모멘트
    v_r: f32,           // r에 대한 2차 모멘트
    m_theta: f32,       // θ에 대한 1차 모멘트
    v_theta: f32,       // θ에 대한 2차 모멘트
    
    // 시간 스텝
    t: u32,             // 업데이트 횟수
    
    // 옵션
    use_riemannian: bool,   // 리만 기하학 적용 여부
    use_amsgrad: bool,      // AMSGrad 변형 사용 여부
    vmax_r: f32,            // AMSGrad용 최대 2차 모멘트 (r)
    vmax_theta: f32,        // AMSGrad용 최대 2차 모멘트 (θ)
}

impl BitAdamState {
    pub fn new() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m_r: 0.0,
            v_r: 0.0,
            m_theta: 0.0,
            v_theta: 0.0,
            t: 0,
            use_riemannian: false,
            use_amsgrad: false,
            vmax_r: 0.0,
            vmax_theta: 0.0,
        }
    }
    
    pub fn with_config(beta1: f32, beta2: f32, epsilon: f32, use_riemannian: bool) -> Self {
        Self {
            beta1,
            beta2,
            epsilon,
            m_r: 0.0,
            v_r: 0.0,
            m_theta: 0.0,
            v_theta: 0.0,
            t: 0,
            use_riemannian,
            use_amsgrad: false,
            vmax_r: 0.0,
            vmax_theta: 0.0,
        }
    }
    
    /// 정확한 수학적 그래디언트를 사용한 Adam 업데이트
    pub fn bit_update(
        &mut self,
        packed: &mut Packed128,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
        target: f32,
        learning_rate: f32,
    ) {
        // 고정소수점 버전 사용
        self.bit_update_fixed_point(packed, i, j, rows, cols, target, learning_rate);
    }

    /// 고정소수점 연산을 사용한 정밀한 Adam 업데이트
    pub fn bit_update_fixed_point(
        &mut self,
        packed: &mut Packed128,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
        target: f32,
        learning_rate: f32,
    ) {
        self.t += 1;
        
        // 1. 정확한 그래디언트 계산
        let (grad_r, grad_theta, predicted) = if self.use_riemannian {
            // 리만 자연 그래디언트
            let (gr, gt) = packed.compute_riemannian_gradients(i, j, rows, cols, target, false);
            let pred = packed.fused_forward(i, j, rows, cols);
            (gr, gt, pred)
        } else {
            // 유클리드 그래디언트
            packed.compute_gradients(i, j, rows, cols, target, false)
        };
        
        // 2. 1차 모멘트 업데이트 (지수이동평균)
        self.m_r = self.beta1 * self.m_r + (1.0 - self.beta1) * grad_r;
        self.m_theta = self.beta1 * self.m_theta + (1.0 - self.beta1) * grad_theta;
        
        // 3. 2차 모멘트 업데이트 (지수이동평균)
        self.v_r = self.beta2 * self.v_r + (1.0 - self.beta2) * grad_r.powi(2);
        self.v_theta = self.beta2 * self.v_theta + (1.0 - self.beta2) * grad_theta.powi(2);
        
        // 4. AMSGrad 변형 (선택적)
        let (v_r_used, v_theta_used) = if self.use_amsgrad {
            self.vmax_r = self.vmax_r.max(self.v_r);
            self.vmax_theta = self.vmax_theta.max(self.v_theta);
            (self.vmax_r, self.vmax_theta)
        } else {
            (self.v_r, self.v_theta)
        };
        
        // 5. 편향 보정 (Bias correction)
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
        
        let m_hat_r = self.m_r / bias_correction1;
        let m_hat_theta = self.m_theta / bias_correction1;
        
        let v_hat_r = v_r_used / bias_correction2;
        let v_hat_theta = v_theta_used / bias_correction2;
        
        // 6. Adam 그래디언트 계산
        let adam_grad_r = m_hat_r / (v_hat_r.sqrt() + self.epsilon);
        let adam_grad_theta = m_hat_theta / (v_hat_theta.sqrt() + self.epsilon);
        
        // 7. 고정소수점 업데이트 (정밀도 손실 없음)
        packed.update_gradients_fixed_point(adam_grad_r, adam_grad_theta, learning_rate);
        
        // 8. 11비트 사이클 업데이트
        let total_gradient_magnitude = (grad_r.abs() + grad_theta.abs()) / 2.0;
        packed.apply_cycle_gradient(total_gradient_magnitude);
        
        // 디버깅 정보 (선택적)
        if self.t % 100 == 0 {
            let params = packed.decode();
            println!("Epoch {}: r={:.6}, theta={:.6}, pred={:.6}, target={:.6}, loss={:.6}",
                     self.t, params.r_fp32, params.theta_fp32, predicted, target, (predicted - target).abs());
        }
    }

    /// 정확한 수학적 그래디언트를 사용한 Adam 업데이트 (이전 버전)
    pub fn bit_update_old(
        &mut self,
        packed: &mut Packed128,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
        target: f32,
        learning_rate: f32,
    ) {
        self.t += 1;
        
        // 1. 정확한 그래디언트 계산
        let (grad_r, grad_theta) = if self.use_riemannian {
            // 리만 자연 그래디언트
            packed.compute_riemannian_gradients(i, j, rows, cols, target, false) // L2 손실 사용
        } else {
            // 유클리드 그래디언트
            let (gr, gt, _) = packed.compute_gradients(i, j, rows, cols, target, false); // L2 손실 사용
            (gr, gt)
        };
        
        // 2. 1차 모멘트 업데이트 (지수이동평균)
        self.m_r = self.beta1 * self.m_r + (1.0 - self.beta1) * grad_r;
        self.m_theta = self.beta1 * self.m_theta + (1.0 - self.beta1) * grad_theta;
        
        // 3. 2차 모멘트 업데이트 (지수이동평균)
        self.v_r = self.beta2 * self.v_r + (1.0 - self.beta2) * grad_r.powi(2);
        self.v_theta = self.beta2 * self.v_theta + (1.0 - self.beta2) * grad_theta.powi(2);
        
        // 4. AMSGrad 변형 (선택적)
        let (v_r_used, v_theta_used) = if self.use_amsgrad {
            self.vmax_r = self.vmax_r.max(self.v_r);
            self.vmax_theta = self.vmax_theta.max(self.v_theta);
            (self.vmax_r, self.vmax_theta)
        } else {
            (self.v_r, self.v_theta)
        };
        
        // 5. 편향 보정 (Bias correction)
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
        
        let m_hat_r = self.m_r / bias_correction1;
        let m_hat_theta = self.m_theta / bias_correction1;
        
        let v_hat_r = v_r_used / bias_correction2;
        let v_hat_theta = v_theta_used / bias_correction2;
        
        // 6. Adam 업데이트 규칙
        let mut params = packed.decode();
        
        params.r_fp32 -= learning_rate * m_hat_r / (v_hat_r.sqrt() + self.epsilon);
        params.theta_fp32 -= learning_rate * m_hat_theta / (v_hat_theta.sqrt() + self.epsilon);
        
        // 7. 범위 제약
        params.r_fp32 = params.r_fp32.clamp(0.0, 0.999999);
        params.theta_fp32 = params.theta_fp32.rem_euclid(2.0 * std::f32::consts::PI);
        
        // 8. 업데이트된 파라미터 적용
        packed.update_from_continuous(&params);
    }
    
    /// 간단한 인터페이스 (이전 버전과의 호환성)
    pub fn bit_update_simple(
        &mut self,
        packed: &mut Packed128,
        predicted: f32,
        target: f32,
        learning_rate: f32,
    ) {
        // 더미 좌표로 호출 (실제로는 사용하지 않는 것을 권장)
        self.bit_update(packed, 0, 0, 1, 1, target, learning_rate);
    }
    
    /// 옵티마이저 상태 초기화
    pub fn reset(&mut self) {
        self.m_r = 0.0;
        self.v_r = 0.0;
        self.m_theta = 0.0;
        self.v_theta = 0.0;
        self.t = 0;
        self.vmax_r = 0.0;
        self.vmax_theta = 0.0;
    }
    
    /// AMSGrad 변형 활성화/비활성화
    pub fn set_amsgrad(&mut self, use_amsgrad: bool) {
        self.use_amsgrad = use_amsgrad;
    }
    
    /// 현재 옵티마이저 상태 정보 반환
    pub fn get_state_info(&self) -> (u32, f32, f32, f32, f32) {
        (self.t, self.m_r, self.v_r, self.m_theta, self.v_theta)
    }
    
    /// 적응적 학습률 계산 (디버깅용)
    pub fn get_adaptive_lr(&self, base_lr: f32) -> (f32, f32) {
        let bias_correction1 = 1.0 - self.beta1.powi(self.t.max(1) as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t.max(1) as i32);
        
        let m_hat_r = self.m_r / bias_correction1;
        let m_hat_theta = self.m_theta / bias_correction1;
        let v_hat_r = self.v_r / bias_correction2;
        let v_hat_theta = self.v_theta / bias_correction2;
        
        let lr_r = base_lr * m_hat_r.abs() / (v_hat_r.sqrt() + self.epsilon);
        let lr_theta = base_lr * m_hat_theta.abs() / (v_hat_theta.sqrt() + self.epsilon);
        
        (lr_r, lr_theta)
    }
}

impl Default for BitAdamState {
    fn default() -> Self {
        Self::new()
    }
} 
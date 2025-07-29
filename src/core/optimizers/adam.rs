//! BitAdam 옵티마이저 - 정밀 수학적 구현 (Enhanced128 통합)
//! 
//! Adaptive Moment Estimation (Adam)을 RBE 시스템에 적용
//! Packed128 및 Enhanced128 모두 지원

use crate::core::tensors::{
    Packed128, Enhanced128, DecodedParams, EnhancedParams, AnalyticalGradient,
    Packed256, Packed256Params
};
use crate::core::differential::bit_engine;

/// RBE 압축 시드 공통 트레이트 (Adam 통합용)
pub trait RBESeed: Clone {
    type Params;
    
    /// 그래디언트 계산
    fn compute_gradients(&self, i: usize, j: usize, rows: usize, cols: usize, target: f32, use_riemannian: bool) -> (f32, f32, f32);
    /// 순전파
    fn fused_forward(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32;
    /// 파라미터 디코딩
    fn decode(&self) -> Self::Params;
    /// 파라미터 업데이트
    fn update_from_params(&mut self, params: &Self::Params);
    /// Adam 업데이트 (r, theta 파라미터)
    fn adam_update(&mut self, m_hat_r: f32, m_hat_theta: f32, v_hat_r: f32, v_hat_theta: f32, learning_rate: f32, epsilon: f32);
}

/// Packed128에 대한 RBESeed 구현
impl RBESeed for Packed128 {
    type Params = DecodedParams;
    
    fn compute_gradients(&self, i: usize, j: usize, rows: usize, cols: usize, target: f32, use_riemannian: bool) -> (f32, f32, f32) {
        if use_riemannian {
            let (gr, gt) = self.compute_riemannian_gradients(i, j, rows, cols, target, false);
            let pred = self.fused_forward(i, j, rows, cols);
            (gr, gt, pred)
        } else {
            self.compute_gradients(i, j, rows, cols, target, false)
        }
    }
    
    fn fused_forward(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        self.fused_forward(i, j, rows, cols)
    }
    
    fn decode(&self) -> Self::Params {
        self.decode()
    }
    
    fn update_from_params(&mut self, params: &Self::Params) {
        self.update_from_continuous(params);
    }
    
    fn adam_update(&mut self, m_hat_r: f32, m_hat_theta: f32, v_hat_r: f32, v_hat_theta: f32, learning_rate: f32, epsilon: f32) {
        let mut params = self.decode();
        
        // Adam 업데이트 규칙
        params.r_fp32 -= learning_rate * m_hat_r / (v_hat_r.sqrt() + epsilon);
        params.theta_fp32 -= learning_rate * m_hat_theta / (v_hat_theta.sqrt() + epsilon);
        
        // 범위 제약
        params.r_fp32 = params.r_fp32.clamp(0.0, 0.999999);
        params.theta_fp32 = params.theta_fp32.rem_euclid(2.0 * std::f32::consts::PI);
        
        self.update_from_continuous(&params);
    }
}

/// Enhanced128에 대한 RBESeed 구현
impl RBESeed for Enhanced128 {
    type Params = EnhancedParams;
    
    fn compute_gradients(&self, i: usize, j: usize, rows: usize, cols: usize, target: f32, _use_riemannian: bool) -> (f32, f32, f32) {
        // Enhanced128은 이미 리만 기하학이 내장됨
        let predicted = self.fused_forward_enhanced(i, j, rows, cols);
        let error = predicted - target;
        
        // 수치 미분으로 그래디언트 계산
        let grad_r = self.analytical_gradient_r(i, j, rows, cols) * error;
        let grad_theta = self.analytical_gradient_theta(i, j, rows, cols) * error;
        
        (grad_r, grad_theta, predicted)
    }
    
    fn fused_forward(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        self.fused_forward_enhanced(i, j, rows, cols)
    }
    
    fn decode(&self) -> Self::Params {
        self.decode_enhanced()
    }
    
    fn update_from_params(&mut self, params: &Self::Params) {
        // Enhanced128을 새로 생성하여 업데이트
        *self = Enhanced128::from_legacy_params(
            params.r_fp32,
            params.theta_fp32,
            params.basis_id,
            params.d_theta,
            params.d_r,
            params.rot_code,
            params.log2_c,
        );
    }
    
    fn adam_update(&mut self, m_hat_r: f32, m_hat_theta: f32, v_hat_r: f32, v_hat_theta: f32, learning_rate: f32, epsilon: f32) {
        let mut params = self.decode_enhanced();
        
        // Adam 업데이트 규칙
        params.r_fp32 -= learning_rate * m_hat_r / (v_hat_r.sqrt() + epsilon);
        params.theta_fp32 -= learning_rate * m_hat_theta / (v_hat_theta.sqrt() + epsilon);
        
        // 범위 제약
        params.r_fp32 = params.r_fp32.clamp(0.0, 0.999999);
        params.theta_fp32 = params.theta_fp32.rem_euclid(2.0 * std::f32::consts::PI);
        
        // Enhanced128 재생성
        *self = Enhanced128::from_legacy_params(
            params.r_fp32,
            params.theta_fp32,
            params.basis_id,
            params.d_theta,
            params.d_r,
            params.rot_code,
            params.log2_c,
        );
    }
}

/// Packed256에 대한 RBESeed 구현
impl RBESeed for Packed256 {
    type Params = Packed256Params;

    fn compute_gradients(&self, i: usize, j: usize, rows: usize, cols: usize, target: f32, _use_riemannian: bool) -> (f32, f32, f32) {
        let params = self.decode();
        let output = bit_engine::compute_fused_output(&params, i, j, rows, cols);
        
        let loss_grad = 2.0 * (output.predicted_value - target);

        let final_grad_r = (loss_grad * output.grad_r).clamp(-1.0, 1.0);
        let final_grad_theta = (loss_grad * output.grad_theta).clamp(-1.0, 1.0);

        (final_grad_r, final_grad_theta, output.predicted_value)
    }

    fn fused_forward(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        let params = self.decode();
        bit_engine::compute_fused_output(&params, i, j, rows, cols).predicted_value
    }

    fn decode(&self) -> Self::Params {
        // Packed256에 이미 구현된 getter를 사용
        Packed256Params {
            r: self.get_r(),
            theta: self.get_theta(),
            param1: self.get_param1(),
            param2: self.get_param2(),
            basis_id: self.get_basis_id(),
            d_r: self.get_d_r(),
            d_theta: self.get_d_theta(),
            log2_c: self.get_log2_c(),
            activation_id: self.get_activation_id(),
            q_value: self.get_q_value(),
            k_value: self.get_k_value(),
            flags: self.get_flags(),
        }
    }

    fn update_from_params(&mut self, params: &Self::Params) {
        // Packed256에 이미 구현된 setter를 사용
        self.set_r(params.r);
        self.set_theta(params.theta);
        self.set_param1(params.param1);
        self.set_param2(params.param2);
        self.set_basis_id(params.basis_id);
        self.set_d_r(params.d_r);
        self.set_d_theta(params.d_theta);
        self.set_log2_c(params.log2_c);
        self.set_activation_id(params.activation_id);
        self.set_q_value(params.q_value);
        self.set_k_value(params.k_value);
        self.set_flags(params.flags);
    }

    fn adam_update(&mut self, m_hat_r: f32, m_hat_theta: f32, v_hat_r: f32, v_hat_theta: f32, learning_rate: f32, epsilon: f32) {
        let mut params = self.decode();
        
        // Adam 업데이트 규칙
        params.r -= learning_rate * m_hat_r / (v_hat_r.sqrt() + epsilon);
        params.theta -= learning_rate * m_hat_theta / (v_hat_theta.sqrt() + epsilon);
        
        // 범위 제약
        params.r = params.r.clamp(0.0, 0.9999);
        params.theta = params.theta.rem_euclid(2.0 * std::f32::consts::PI);
        
        self.update_from_params(&params);
    }
}


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
    
    pub fn set_learning_rate(&mut self, _learning_rate: f32) {
        // BitAdamState는 learning_rate를 내부적으로 저장하지 않고 bit_update에서 받음
        // 호환성을 위한 빈 메서드
    }

    /// 정확한 수학적 그래디언트를 사용한 Adam 업데이트 (Enhanced128 지원)
    pub fn bit_update<T: RBESeed>(
        &mut self,
        seed: &mut T,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
        target: f32,
        learning_rate: f32,
    ) {
        self.t += 1;
        
        // 1. 정확한 그래디언트 계산
        let (grad_r, grad_theta, _predicted) = seed.compute_gradients(i, j, rows, cols, target, self.use_riemannian);
        
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
        
        // 6. 파라미터 업데이트 (제네릭)
        seed.adam_update(m_hat_r, m_hat_theta, v_hat_r, v_hat_theta, learning_rate, self.epsilon);
    }

    /// 기존 Packed128 전용 버전 (호환성 유지)
    pub fn bit_update_packed128(
        &mut self,
        packed: &mut Packed128,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
        target: f32,
        learning_rate: f32,
    ) {
        self.bit_update(packed, i, j, rows, cols, target, learning_rate);
    }

    /// Enhanced128을 위한 Adam 업데이트 (편의 메서드)
    pub fn bit_update_enhanced(
        &mut self,
        enhanced: &mut Enhanced128,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
        target: f32,
        learning_rate: f32,
    ) {
        self.bit_update(enhanced, i, j, rows, cols, target, learning_rate);
    }
    
    /// 간단한 인터페이스 (이전 버전과의 호환성)
    pub fn bit_update_simple(
        &mut self,
        packed: &mut Packed128,
        _predicted: f32,
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
    pub fn get_adaptive_lr(&self) -> (f32, f32) {
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
        
        let m_hat_r = self.m_r / bias_correction1;
        let m_hat_theta = self.m_theta / bias_correction1;
        let v_hat_r = self.v_r / bias_correction2;
        let v_hat_theta = self.v_theta / bias_correction2;
        
        let adaptive_lr_r = 1.0 / (v_hat_r.sqrt() + self.epsilon);
        let adaptive_lr_theta = 1.0 / (v_hat_theta.sqrt() + self.epsilon);
        
        (adaptive_lr_r, adaptive_lr_theta)
    }
}

impl Default for BitAdamState {
    fn default() -> Self {
        Self::new()
    }
} 
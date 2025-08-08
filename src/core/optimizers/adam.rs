//! BitAdam 옵티마이저 - 정밀 수학적 구현 (Enhanced128 통합)
//! 
//! Adaptive Moment Estimation (Adam)을 RBE 시스템에 적용
//! Packed128 및 Enhanced128 모두 지원

use crate::core::tensors::{
    Packed128, Enhanced128, DecodedParams, EnhancedParams, AnalyticalGradient,
    Packed256, Packed256Params
};
use crate::core::differential::bit_engine;

/// RBE 시드 제네릭 트레이트 (Packed128/Enhanced128/Packed256 지원)
pub trait RBESeed: Copy {
    fn get_r(&self) -> f32;
    fn get_theta(&self) -> f32;
    fn set_r(&mut self, r: f32);
    fn set_theta(&mut self, theta: f32);
    fn fused_forward_generic(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32;
    // 선택적: 해석 그라디언트 지원 (예: Packed256→bit_engine)
    fn analytic_grads(&self, _i: usize, _j: usize, _rows: usize, _cols: usize) -> Option<(f32, f32, f32, f32)> {
        None
    }
    // 선택적: 진폭(amp = param2) 접근자
    fn get_amp(&self) -> f32 { 0.0 }
    fn set_amp(&mut self, _amp: f32) {}
    // 선택적: 주파수(param1) 접근자 (기본 비활성)
    fn get_param1(&self) -> f32 { 0.0 }
    fn set_param1(&mut self, _v: f32) {}
}

impl RBESeed for Packed128 {
    fn get_r(&self) -> f32 { self.decode().r_fp32 }
    fn get_theta(&self) -> f32 { self.decode().theta_fp32 }
    fn set_r(&mut self, r: f32) {
        let mut p = self.decode();
        p.r_fp32 = r;
        self.update_from_continuous(&p);
    }
    fn set_theta(&mut self, theta: f32) {
        let mut p = self.decode();
        p.theta_fp32 = theta;
        self.update_from_continuous(&p);
    }
    fn fused_forward_generic(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        self.fused_forward(i, j, rows, cols)
    }
    fn analytic_grads(&self, _i: usize, _j: usize, _rows: usize, _cols: usize) -> Option<(f32, f32, f32, f32)> { None }
}

impl RBESeed for Enhanced128 {
    fn get_r(&self) -> f32 { self.decode_enhanced().r_fp32 }
    fn get_theta(&self) -> f32 { self.decode_enhanced().theta_fp32 }
    fn set_r(&mut self, r: f32) {
        let mut p = self.decode_enhanced();
        p.r_fp32 = r;
        *self = Enhanced128::from_legacy_params(
            p.r_fp32, p.theta_fp32, p.basis_id, p.d_theta, p.d_r, p.rot_code, p.log2_c
        );
    }
    fn set_theta(&mut self, theta: f32) {
        let mut p = self.decode_enhanced();
        p.theta_fp32 = theta;
        *self = Enhanced128::from_legacy_params(
            p.r_fp32, p.theta_fp32, p.basis_id, p.d_theta, p.d_r, p.rot_code, p.log2_c
        );
    }
    fn fused_forward_generic(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        self.fused_forward_enhanced(i, j, rows, cols)
    }
    fn analytic_grads(&self, _i: usize, _j: usize, _rows: usize, _cols: usize) -> Option<(f32, f32, f32, f32)> { None }
}

impl RBESeed for Packed256 {
    fn get_r(&self) -> f32 { self.get_r() }
    fn get_theta(&self) -> f32 { self.get_theta() }
    fn set_r(&mut self, r: f32) { self.set_r(r) }
    fn set_theta(&mut self, theta: f32) { self.set_theta(theta) }
    fn fused_forward_generic(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        let params = Packed256Params {
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
        };
        bit_engine::compute_fused_output(&params, i, j, rows, cols).predicted_value
    }
    fn analytic_grads(&self, i: usize, j: usize, rows: usize, cols: usize) -> Option<(f32, f32, f32, f32)> {
        let params = Packed256Params {
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
        };
        let out = bit_engine::compute_fused_output(&params, i, j, rows, cols);
        Some((out.predicted_value, out.grad_r, out.grad_theta, out.grad_p2))
    }
    fn get_amp(&self) -> f32 { self.get_param2() }
    fn set_amp(&mut self, amp: f32) { self.set_param2(amp) }
    fn get_param1(&self) -> f32 { self.get_param1() }
    fn set_param1(&mut self, v: f32) { self.set_param1(v) }
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
    m_amp: f32,         // amp(param2)에 대한 1차 모멘트
    v_amp: f32,         // amp(param2)에 대한 2차 모멘트
    m_p1: f32,          // p1(param1) 1차 모멘트
    v_p1: f32,          // p1(param1) 2차 모멘트
    
    // 시간 스텝
    t: u32,             // 업데이트 횟수
    
    // 옵션
    use_riemannian: bool,   // 리만 기하학 적용 여부
    use_amsgrad: bool,      // AMSGrad 변형 사용 여부
    vmax_r: f32,            // AMSGrad용 최대 2차 모멘트 (r)
    vmax_theta: f32,        // AMSGrad용 최대 2차 모멘트 (θ)
    vmax_amp: f32,          // AMSGrad용 최대 2차 모멘트 (amp)
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
            m_amp: 0.0,
            v_amp: 0.0,
            m_p1: 0.0,
            v_p1: 0.0,
            t: 0,
            use_riemannian: false,
            use_amsgrad: false,
            vmax_r: 0.0,
            vmax_theta: 0.0,
            vmax_amp: 0.0,
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
            m_amp: 0.0,
            v_amp: 0.0,
            m_p1: 0.0,
            v_p1: 0.0,
            t: 0,
            use_riemannian,
            use_amsgrad: false,
            vmax_r: 0.0,
            vmax_theta: 0.0,
            vmax_amp: 0.0,
        }
    }
    
    pub fn set_learning_rate(&mut self, _learning_rate: f32) {
        // BitAdamState는 learning_rate를 내부적으로 저장하지 않고 bit_update에서 받음
        // 호환성을 위한 빈 메서드
    }

    /// 제네릭 RBE 시드용 Adam 업데이트
    pub fn bit_update<S: RBESeed>(&mut self, seed: &mut S, i: usize, j: usize, rows: usize, cols: usize, target: f32, learning_rate: f32) {
        self.t += 1;
        
        // 기본 예측값 (해석 경로 사용 시 덮어씀)
        let mut predicted = seed.fused_forward_generic(i, j, rows, cols);
        let mut loss_grad = predicted - target; // d(0.5*(p-t)^2)/dp

        // 해석 그라디언트가 있으면 우선 사용
        if let Some((p_pred, grad_r_analytic, grad_theta_analytic, grad_amp_analytic)) = seed.analytic_grads(i, j, rows, cols) {
            predicted = p_pred;
            loss_grad = predicted - target;
            // 자연 그래디언트 스케일링
            let r0 = seed.get_r();
            let th0 = seed.get_theta();
            let one_minus_r2 = (1.0 - r0 * r0).max(1e-6);
            let scale_r = (one_minus_r2 * one_minus_r2) / 4.0;
            let scale_theta = (one_minus_r2 * one_minus_r2) / (4.0 * (r0 * r0 + 1e-9));

            let grad_r = loss_grad * grad_r_analytic * scale_r;
            let grad_theta = loss_grad * grad_theta_analytic * scale_theta;
            let grad_amp = 0.0; // amp는 폐형해로 에폭 말에만 갱신

            // 1차/2차 모멘트 업데이트 및 파라미터 갱신 (동일)
            self.m_r = self.beta1 * self.m_r + (1.0 - self.beta1) * grad_r;
            self.m_theta = self.beta1 * self.m_theta + (1.0 - self.beta1) * grad_theta;
            self.m_amp = self.beta1 * self.m_amp + (1.0 - self.beta1) * grad_amp;
            self.v_r = self.beta2 * self.v_r + (1.0 - self.beta2) * grad_r.powi(2);
            self.v_theta = self.beta2 * self.v_theta + (1.0 - self.beta2) * grad_theta.powi(2);
            self.v_amp = self.beta2 * self.v_amp + (1.0 - self.beta2) * grad_amp.powi(2);

            let (v_r_used, v_theta_used) = if self.use_amsgrad {
                self.vmax_r = self.vmax_r.max(self.v_r);
                self.vmax_theta = self.vmax_theta.max(self.v_theta);
                (self.vmax_r, self.vmax_theta)
            } else { (self.v_r, self.v_theta) };
            let v_amp_used = if self.use_amsgrad {
                self.vmax_amp = self.vmax_amp.max(self.v_amp);
                self.vmax_amp
            } else { self.v_amp };

            let bc1 = 1.0 - self.beta1.powi(self.t as i32);
            let bc2 = 1.0 - self.beta2.powi(self.t as i32);
            let m_hat_r = self.m_r / bc1;
            let m_hat_theta = self.m_theta / bc1;
            let v_hat_r = v_r_used / bc2;
            let v_hat_theta = v_theta_used / bc2;
            let m_hat_amp = self.m_amp / bc1;
            let v_hat_amp = v_amp_used / bc2;

            let step_r = learning_rate * m_hat_r / (v_hat_r.sqrt() + self.epsilon);
            let step_theta = learning_rate * m_hat_theta / (v_hat_theta.sqrt() + self.epsilon);
            let step_amp = 0.0;

            let new_r = (r0 - step_r).clamp(0.0, 0.9999);
            let new_theta = th0 - step_theta;
            seed.set_r(new_r);
            seed.set_theta(new_theta);
            // amp는 여기서 갱신하지 않음 (폐형해로 처리)
            // p1 보조 업데이트 (수치미분)
            let p1_0 = seed.get_param1();
            let eps_p1 = 1e-4_f32.max(1e-3 * p1_0.abs());
            let dp_dp1 = {
                let mut s_plus = *seed; s_plus.set_param1((p1_0 + eps_p1).clamp(-16.0, 16.0));
                let mut s_minus = *seed; s_minus.set_param1((p1_0 - eps_p1).clamp(-16.0, 16.0));
                let p_plus = s_plus.fused_forward_generic(i, j, rows, cols);
                let p_minus = s_minus.fused_forward_generic(i, j, rows, cols);
                (p_plus - p_minus) / (2.0 * eps_p1)
            };
            let grad_p1 = loss_grad * dp_dp1;
            self.m_p1 = self.beta1 * self.m_p1 + (1.0 - self.beta1) * grad_p1;
            self.v_p1 = self.beta2 * self.v_p1 + (1.0 - self.beta2) * grad_p1.powi(2);
            let bc1 = 1.0 - self.beta1.powi(self.t as i32);
            let bc2 = 1.0 - self.beta2.powi(self.t as i32);
            let m_hat_p1 = self.m_p1 / bc1;
            let v_hat_p1 = self.v_p1 / bc2;
            let step_p1 = learning_rate * m_hat_p1 / (v_hat_p1.sqrt() + self.epsilon);
            seed.set_param1((p1_0 - step_p1).clamp(-16.0, 16.0));
            return;
        }

        // 수치미분으로 ∂p/∂r, ∂p/∂θ 근사 (중심차분 + 적응 eps)
        let r0 = seed.get_r();
        let th0 = seed.get_theta();
        let eps_r = (1e-4_f32).max(1e-2 * (1.0 - r0).abs());
        let eps_th = 1e-4_f32;

        // 중앙차분 + 적응 eps(에스컬레이션)
        fn central_diff<S: RBESeed>(seed: &S, i: usize, j: usize, rows: usize, cols: usize, mut getter: impl FnMut(&S) -> f32, mut setter: impl FnMut(S, f32) -> S, base: f32, base_pred: f32, eps0: f32) -> f32 {
            let mut eps = eps0;
            let mut best = 0.0f32;
            for _ in 0..3 {
                let mut s_plus = *seed;
                s_plus = setter(s_plus, base + eps);
                let p_plus = s_plus.fused_forward_generic(i, j, rows, cols);

                let mut s_minus = *seed;
                s_minus = setter(s_minus, base - eps);
                let p_minus = s_minus.fused_forward_generic(i, j, rows, cols);

                let grad = (p_plus - p_minus) / (2.0 * eps);
                if grad.abs() > best.abs() {
                    best = grad;
                }
                if best.abs() < 1e-6 { eps *= 10.0; } else { break; }
            }
            best
        }

        let dp_dr = central_diff(
            seed, i, j, rows, cols,
            |s| s.get_r(),
            |mut s, v| { s.set_r(v.clamp(0.0, 0.9999)); s },
            r0, predicted, eps_r,
        );

        let dp_dtheta = central_diff(
            seed, i, j, rows, cols,
            |s| s.get_theta(),
            |mut s, v| { s.set_theta(v); s },
            th0, predicted, eps_th,
        );

        // 자연 그래디언트 스케일링 (푸앵카레 볼)
        let one_minus_r2 = (1.0 - r0 * r0).max(1e-6);
        let scale_r = (one_minus_r2 * one_minus_r2) / 4.0; // for dr component
        let scale_theta = (one_minus_r2 * one_minus_r2) / (4.0 * (r0 * r0 + 1e-9));

        let grad_r = loss_grad * dp_dr * scale_r;
        let grad_theta = loss_grad * dp_dtheta * scale_theta;

        // 1차/2차 모멘트
        self.m_r = self.beta1 * self.m_r + (1.0 - self.beta1) * grad_r;
        self.m_theta = self.beta1 * self.m_theta + (1.0 - self.beta1) * grad_theta;
        self.m_amp = self.beta1 * self.m_amp + (1.0 - self.beta1) * 0.0; // 수치 경로에선 amp 미갱신
        // p1 수치 그라디언트
        let p1_0 = seed.get_param1();
        let eps_p1 = 1e-4_f32.max(1e-3 * p1_0.abs());
        let dp_dp1 = {
            let mut s_plus = *seed; s_plus.set_param1((p1_0 + eps_p1).clamp(-16.0, 16.0));
            let mut s_minus = *seed; s_minus.set_param1((p1_0 - eps_p1).clamp(-16.0, 16.0));
            let p_plus = s_plus.fused_forward_generic(i, j, rows, cols);
            let p_minus = s_minus.fused_forward_generic(i, j, rows, cols);
            (p_plus - p_minus) / (2.0 * eps_p1)
        };
        let grad_p1 = loss_grad * dp_dp1;
        self.m_p1 = self.beta1 * self.m_p1 + (1.0 - self.beta1) * grad_p1;

        self.v_r = self.beta2 * self.v_r + (1.0 - self.beta2) * grad_r.powi(2);
        self.v_theta = self.beta2 * self.v_theta + (1.0 - self.beta2) * grad_theta.powi(2);
        self.v_amp = self.beta2 * self.v_amp + (1.0 - self.beta2) * 0.0;
        self.v_p1 = self.beta2 * self.v_p1 + (1.0 - self.beta2) * grad_p1.powi(2);
        
        let (v_r_used, v_theta_used) = if self.use_amsgrad {
            self.vmax_r = self.vmax_r.max(self.v_r);
            self.vmax_theta = self.vmax_theta.max(self.v_theta);
            (self.vmax_r, self.vmax_theta)
        } else {
            (self.v_r, self.v_theta)
        };
        let _v_amp_used = if self.use_amsgrad { self.vmax_amp = self.vmax_amp.max(self.v_amp); self.vmax_amp } else { self.v_amp };

        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        let m_hat_r = self.m_r / bc1;
        let m_hat_theta = self.m_theta / bc1;
        let v_hat_r = v_r_used / bc2;
        let v_hat_theta = v_theta_used / bc2;
        let m_hat_p1 = self.m_p1 / bc1;
        let v_hat_p1 = self.v_p1 / bc2;

        // 적응 스텝 (자연스텝 보정 포함)
        let step_r = learning_rate * m_hat_r / (v_hat_r.sqrt() + self.epsilon);
        let step_theta = learning_rate * m_hat_theta / (v_hat_theta.sqrt() + self.epsilon);
        let step_p1 = learning_rate * m_hat_p1 / (v_hat_p1.sqrt() + self.epsilon);

        let new_r = (r0 - step_r).clamp(0.0, 0.9999);
        let new_theta = th0 - step_theta;
        seed.set_r(new_r);
        seed.set_theta(new_theta);
        seed.set_param1((p1_0 - step_p1).clamp(-16.0, 16.0));
    }

    /// Packed128용 업데이트
    pub fn bit_update_packed128(
        &mut self,
        seed: &mut Packed128,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
        target: f32,
        learning_rate: f32,
    ) {
        // 제네릭 경로 재사용
        self.bit_update(seed, i, j, rows, cols, target, learning_rate);
    }


    
    /// 옵티마이저 상태 초기화
    pub fn reset(&mut self) {
        self.m_r = 0.0;
        self.v_r = 0.0;
        self.m_theta = 0.0;
        self.v_theta = 0.0;
        self.m_amp = 0.0;
        self.v_amp = 0.0;
        self.m_p1 = 0.0;
        self.v_p1 = 0.0;
        self.t = 0;
        self.vmax_r = 0.0;
        self.vmax_theta = 0.0;
        self.vmax_amp = 0.0;
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
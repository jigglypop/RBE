//! 비트 도메인 푸앵카레볼 리만 Adam 최적화기
//! 정확한 수학적 구현과 비트 정밀도 보장

use crate::core::tensors::packed_types::*;
use std::f32::consts::PI;

/// 비트 도메인 리만 Adam 상태
/// 푸앵카레볼에서의 정확한 리만 기하학 구현
#[derive(Debug, Clone)]
pub struct BitRiemannianAdamState {
    /// Adam 하이퍼파라미터
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    
    /// 1차 모멘트 (운동량)
    pub m_r: f32,
    pub m_theta: f32,
    
    /// 2차 모멘트 (분산)
    pub v_r: f32,
    pub v_theta: f32,
    
    /// AMSGrad를 위한 최대값 추적
    pub vmax_r: f32,
    pub vmax_theta: f32,
    
    /// 시간 스텝
    pub t: u32,
    
    /// 옵션
    pub use_amsgrad: bool,
    pub clip_grad: f32,
}

impl Default for BitRiemannianAdamState {
    fn default() -> Self {
        Self::new()
    }
}

impl BitRiemannianAdamState {
    pub fn new() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m_r: 0.0,
            m_theta: 0.0,
            v_r: 0.0,
            v_theta: 0.0,
            vmax_r: 0.0,
            vmax_theta: 0.0,
            t: 0,
            use_amsgrad: false,
            clip_grad: 10.0,
        }
    }
    
    pub fn with_config(beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            beta1,
            beta2,
            epsilon,
            ..Self::new()
        }
    }
    
    /// 푸앵카레볼 메트릭 텐서 계산
    /// g = 4/(1-||x||²)² I_n
    fn compute_metric_factor(&self, r: f32) -> f32 {
        let r_clamped = r.min(0.999);
        let one_minus_r2 = (1.0 - r_clamped * r_clamped).max(1e-6);
        4.0 / (one_minus_r2 * one_minus_r2)
    }
    
    /// 지수 사상 (Exponential map)
    /// exp_x(v) = x ⊕ (tanh(λ_x||v||/2) v/||v||)
    fn exponential_map(&self, x: f32, v: f32) -> f32 {
        if v.abs() < 1e-8 {
            return x;
        }
        
        // λ_x = 2/(1-||x||²)
        let lambda_x = 2.0 / (1.0 - x * x).max(1e-6);
        
        // tanh(λ_x|v|/2)
        let arg = lambda_x * v.abs() / 2.0;
        let tanh_term = arg.tanh();
        
        // 뫼비우스 덧셈
        let direction = v.signum();
        let update = tanh_term * direction;
        
        // x ⊕ update
        let numerator = x + update;
        let denominator = 1.0 + x * update;
        
        (numerator / denominator).clamp(-0.999, 0.999)
    }
    
    /// 로그 사상 (Logarithmic map) - 역변환
    /// log_x(y) = (2/λ_x) atanh(||x ⊖ y||) (x ⊖ y)/||x ⊖ y||
    fn logarithmic_map(&self, x: f32, y: f32) -> f32 {
        // 뫼비우스 뺄셈: x ⊖ y = (x - y)/(1 - xy)
        let diff = (x - y) / (1.0 - x * y).max(1e-6);
        
        if diff.abs() < 1e-8 {
            return 0.0;
        }
        
        let lambda_x = 2.0 / (1.0 - x * x).max(1e-6);
        let atanh_term = diff.abs().min(0.999).atanh();
        
        (2.0 / lambda_x) * atanh_term * diff.signum()
    }
    
    /// 평행 이동 (Parallel transport)
    /// 벡터 v를 x에서 y로 이동
    fn parallel_transport(&self, x: f32, y: f32, v: f32) -> f32 {
        // 간단한 경우들 처리
        if (x - y).abs() < 1e-8 || v.abs() < 1e-8 {
            return v;
        }
        
        // 쌍곡선 코사인 법칙 사용
        let lambda_x = 2.0 / (1.0 - x * x).max(1e-6);
        let lambda_y = 2.0 / (1.0 - y * y).max(1e-6);
        
        v * (lambda_x / lambda_y).sqrt()
    }
    
    /// 리만 그래디언트 계산
    fn compute_riemannian_gradient(&self, euclidean_grad: f32, r: f32, is_r_component: bool) -> f32 {
        let metric_factor = self.compute_metric_factor(r);
        
        // 리만 그래디언트 = g^{-1} * 유클리드 그래디언트
        // g^{-1} = (1-||x||²)²/4 I_n
        let inv_metric = 1.0 / metric_factor;
        
        if is_r_component {
            // r 방향 성분
            euclidean_grad * inv_metric
        } else {
            // θ 방향 성분 (추가 r² 스케일링)
            euclidean_grad * inv_metric / (r * r).max(1e-6)
        }
    }
    
    /// **핵심: 정밀한 리만 Adam 업데이트**
    pub fn bit_riemannian_update(
        &mut self,
        packed: &mut Packed128,
        i: usize,
        j: usize,
        target: f32,
        learning_rate: f32,
        rows: usize,
        cols: usize,
    ) {
        self.t += 1;
        
        // 1. 안정화된 리만 그래디언트를 직접 계산 (핵심 변경)
        // 기존의 불안정한 유클리드->리만 변환 대신, packed_types에 구현된 안정적인 함수를 직접 호출
        let (grad_r, grad_theta) =
            packed.compute_riemannian_gradients(i, j, rows, cols, target, false);

        // 2. 그래디언트 클리핑
        // packed_types에서 이미 동적 클리핑이 적용되었지만, 추가적인 안전장치로 최대값을 제한할 수 있음
        let grad_r_clipped = grad_r.clamp(-self.clip_grad, self.clip_grad);
        let grad_theta_clipped = grad_theta.clamp(-self.clip_grad, self.clip_grad);
        
        // 3. Adam 모멘트 업데이트
        self.m_r = self.beta1 * self.m_r + (1.0 - self.beta1) * grad_r_clipped;
        self.m_theta = self.beta1 * self.m_theta + (1.0 - self.beta1) * grad_theta_clipped;
        
        self.v_r = self.beta2 * self.v_r + (1.0 - self.beta2) * grad_r_clipped.powi(2);
        self.v_theta = self.beta2 * self.v_theta + (1.0 - self.beta2) * grad_theta_clipped.powi(2);
        
        // 4. AMSGrad (선택적)
        if self.use_amsgrad {
            self.vmax_r = self.vmax_r.max(self.v_r);
            self.vmax_theta = self.vmax_theta.max(self.v_theta);
        }
        
        // 5. 편향 보정
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
        
        let m_hat_r = self.m_r / bias_correction1;
        let m_hat_theta = self.m_theta / bias_correction1;
        
        let v_to_use_r = if self.use_amsgrad { self.vmax_r } else { self.v_r };
        let v_to_use_theta = if self.use_amsgrad { self.vmax_theta } else { self.v_theta };
        
        let v_hat_r = v_to_use_r / bias_correction2;
        let v_hat_theta = v_to_use_theta / bias_correction2;
        
        // 6. Adam 업데이트 방향 계산
        let update_r = learning_rate * m_hat_r / (v_hat_r.sqrt() + self.epsilon);
        let update_theta = learning_rate * m_hat_theta / (v_hat_theta.sqrt() + self.epsilon);
        
        // 7. 새로운 직접 업데이트 방식 적용
        packed.update_with_riemannian_grad(update_r, update_theta, learning_rate);
        
        // 디버깅 (40 스텝마다)
        if self.t % 40 == 39 {
            let predicted = packed.fused_forward_poincare(i, j, rows, cols);
            let params = packed.decode();
            println!("RiemannianAdam[{}]: r={:.6}, θ={:.6}, pred={:.6}, target={:.6}, loss={:.6}",
                     self.t + 1, params.r_fp32, params.theta_fp32, predicted, target, (predicted - target).abs());
        }
    }
    
    /// 수렴 확인
    pub fn is_converged(&self, threshold: f32) -> bool {
        self.m_r.abs() < threshold && 
        self.m_theta.abs() < threshold
    }
    
    /// 상태 초기화
    pub fn reset(&mut self) {
        self.m_r = 0.0;
        self.v_r = 0.0;
        self.m_theta = 0.0;
        self.v_theta = 0.0;
        self.vmax_r = 0.0;
        self.vmax_theta = 0.0;
        self.t = 0;
    }
    
    /// 현재 상태 정보
    pub fn get_state_info(&self) -> (u32, f32, f32, f32, f32) {
        (self.t, self.m_r, self.v_r, self.m_theta, self.v_theta)
    }
    
    /// 적응적 학습률 계산
    pub fn get_adaptive_lr(&self, base_lr: f32) -> (f32, f32) {
        if self.t == 0 {
            return (base_lr, base_lr);
        }
        
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);
        
        let m_hat_r = self.m_r / bias_correction1;
        let m_hat_theta = self.m_theta / bias_correction1;
        let v_hat_r = self.v_r / bias_correction2;
        let v_hat_theta = self.v_theta / bias_correction2;
        
        let lr_r = base_lr / (v_hat_r.sqrt() + self.epsilon);
        let lr_theta = base_lr / (v_hat_theta.sqrt() + self.epsilon);
        
        (lr_r, lr_theta)
    }
    
    /// Q16.16 고정소수점 변환 (호환성)
    pub fn f32_to_q16(val: f32) -> u32 {
        (val * 65536.0) as i32 as u32
    }
    
    pub fn q16_to_f32(bits: u32) -> f32 {
        (bits as i32) as f32 / 65536.0
    }
    
    /// 비트 메트릭 텐서 계산 (호환성)
    pub fn compute_bit_metric_tensor(&mut self, r_bits: u32) -> (u32, u32) {
        let r = Self::q16_to_f32(r_bits);
        let metric = self.compute_metric_factor(r);
        let metric_bits = Self::f32_to_q16(metric);
        (metric_bits, metric_bits)
    }
    
    /// 뫼비우스 변환 (호환성)
    pub fn mobius_transform(&self, x: f32, v: f32, _c: f32) -> f32 {
        self.exponential_map(x, v)
    }
    
    /// 벤치마크 함수
    pub fn benchmark_riemannian_operations(iterations: usize) {
        use std::time::Instant;
        
        println!("\n=== 리만 Adam 성능 벤치마크 ===");
        
        let mut optimizer = Self::new();
        let mut packed = Packed128::from_continuous(&DecodedParams {
            r_fp32: 0.5,
            theta_fp32: PI / 4.0,
        });
        
        let start = Instant::now();
        
        for i in 0..iterations {
            let target = ((i as f32) % 100.0) / 100.0;
            let row = i % 10;
            let col = (i * 7) % 10;
            
            optimizer.bit_riemannian_update(&mut packed, row, col, target, 0.01, 10, 10);
        }
        
        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;
        
        println!("업데이트 속도: {:.1} ns/op ({:.1} MHz)", 
                ns_per_op, 1000.0 / ns_per_op);
        
        let final_params = packed.decode();
        println!("최종 파라미터: r={:.6}, θ={:.6}", 
                final_params.r_fp32, final_params.theta_fp32);
        
        let (t, m_r, v_r, m_theta, v_theta) = optimizer.get_state_info();
        println!("최종 상태: t={}, m_r={:.6}, v_r={:.6}, m_θ={:.6}, v_θ={:.6}",
                t, m_r, v_r, m_theta, v_theta);
    }
} 
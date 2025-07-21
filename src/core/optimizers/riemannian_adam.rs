use std::f32::consts::PI;

// f32 정밀도에 맞는 푸앵카레볼 경계값
const POINCARE_BOUNDARY_F32: f32 = 0.9999999;

/// 푸앵카레 볼에서의 리만 아담 최적화기
/// 
/// 6.1 리만 아담 알고리즘
/// 하이퍼볼릭 공간에서의 기울기 하강법
#[derive(Debug, Clone)]
pub struct RiemannianAdamState {
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub m_r: f32,      // r에 대한 1차 모멘텀
    pub v_r: f32,      // r에 대한 2차 모멘텀
    pub m_theta: f32,  // θ에 대한 1차 모멘텀
    pub v_theta: f32,  // θ에 대한 2차 모멘텀
    pub t: i32,        // 시간 스텝
}

impl Default for RiemannianAdamState {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m_r: 0.0,
            v_r: 0.0,
            m_theta: 0.0,
            v_theta: 0.0,
            t: 0,
        }
    }
}

impl RiemannianAdamState {
    pub fn new() -> Self {
        Self {
            m_r: 0.0,
            v_r: 0.0,
            m_theta: 0.0,
            v_theta: 0.0,
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
    
    pub fn with_config(beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            m_r: 0.0,
            v_r: 0.0,
            m_theta: 0.0,
            v_theta: 0.0,
            t: 0,
            beta1,
            beta2,
            epsilon,
        }
    }
    
    /// 푸앵카레 볼에서의 메트릭 텐서 계산
    pub fn compute_metric_tensor(&self, r: f32) -> (f32, f32) {
        let factor = 4.0 / (1.0 - r * r).powi(2);
        let g_rr = factor;
        let g_theta_theta = factor * r * r;
        (g_rr, g_theta_theta)
    }
    
    /// 뫼비우스 덧셈
    pub fn mobius_add(&self, x: f32, y: f32) -> f32 {
        let numerator = x + y;
        let denominator = 1.0 + x * y;
        numerator / denominator.max(1e-8)
    }
    
    /// 푸앵카레 볼에서의 지수 사상
    pub fn exponential_map(&self, x: f32, v: f32) -> f32 {
        let norm_v = v.abs();
        if norm_v < 1e-8 {
            return x;
        }
        
        let tanh_factor = (norm_v / 2.0).tanh();
        let direction = v / norm_v;
        self.mobius_add(x, tanh_factor * direction)
    }
    
    /// Riemannian Adam 업데이트
    pub fn update(&mut self, r: &mut f32, theta: &mut f32, grad_r: f32, grad_theta: f32, learning_rate: f32) {
        self.t += 1;
        
        // 메트릭 텐서 계산
        let (g_rr, g_theta_theta) = self.compute_metric_tensor(*r);
        
        // 리만 그래디언트 계산
        let riem_grad_r = grad_r / g_rr;
        let riem_grad_theta = grad_theta / g_theta_theta;
        
        // 모멘텀 업데이트 (벡터 전송 포함)
        self.m_r = self.beta1 * self.m_r + (1.0 - self.beta1) * riem_grad_r;
        self.v_r = self.beta2 * self.v_r + (1.0 - self.beta2) * riem_grad_r * riem_grad_r;
        
        self.m_theta = self.beta1 * self.m_theta + (1.0 - self.beta1) * riem_grad_theta;
        self.v_theta = self.beta2 * self.v_theta + (1.0 - self.beta2) * riem_grad_theta * riem_grad_theta;
        
        // 편향 보정
        let m_r_hat = self.m_r / (1.0 - self.beta1.powi(self.t));
        let v_r_hat = self.v_r / (1.0 - self.beta2.powi(self.t));
        
        let m_theta_hat = self.m_theta / (1.0 - self.beta1.powi(self.t));
        let v_theta_hat = self.v_theta / (1.0 - self.beta2.powi(self.t));
        
        // 업데이트 벡터 계산
        let update_r = -learning_rate * m_r_hat / (v_r_hat.sqrt() + self.epsilon);
        let update_theta = -learning_rate * m_theta_hat / (v_theta_hat.sqrt() + self.epsilon);
        
        // 지수 사상을 통한 업데이트
        *r = self.exponential_map(*r, update_r).clamp(0.0, POINCARE_BOUNDARY_F32);
        *theta = ((*theta + update_theta) % (2.0 * PI) + 2.0 * PI) % (2.0 * PI);
    }
    
    /// 상태 초기화
    pub fn reset(&mut self) {
        self.m_r = 0.0;
        self.v_r = 0.0;
        self.m_theta = 0.0;
        self.v_theta = 0.0;
        self.t = 0;
    }
} 
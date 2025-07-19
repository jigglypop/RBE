/// Adam 최적화기 상태
#[derive(Debug, Clone)]
pub struct AdamState {
    pub m: f32,  // 1차 모멘트
    pub v: f32,  // 2차 모멘트
    pub t: i32,  // 시간 스텝
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

impl AdamState {
    pub fn new() -> Self {
        Self {
            m: 0.0,
            v: 0.0,
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
    
    pub fn with_config(beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            m: 0.0,
            v: 0.0,
            t: 0,
            beta1,
            beta2,
            epsilon,
        }
    }
    
    /// Adam 업데이트 수행
    pub fn update(&mut self, param: &mut f32, gradient: f32, learning_rate: f32) {
        self.t += 1;
        
        // 모멘텀 업데이트
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient;
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * gradient * gradient;
        
        // 편향 보정
        let m_hat = self.m / (1.0 - self.beta1.powi(self.t));
        let v_hat = self.v / (1.0 - self.beta2.powi(self.t));
        
        // 파라미터 업데이트
        *param -= learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
    }
    
    /// 상태 초기화
    pub fn reset(&mut self) {
        self.m = 0.0;
        self.v = 0.0;
        self.t = 0;
    }
} 
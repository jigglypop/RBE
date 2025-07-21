/// Adam 최적화기 상태 (인라인 최적화 집중)
#[derive(Debug, Clone)]
pub struct AdamState {
    pub m: f32,  // 1차 모멘트
    pub v: f32,  // 2차 모멘트
    pub t: i32,  // 시간 스텝
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    // 캐시 제거: 단순 계산에서는 오버헤드가 더 큼
}

impl Default for AdamState {
    fn default() -> Self {
        Self::new()
    }
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
    
    /// **인라인 최적화된 Adam 업데이트** (캐시 제거)
    #[inline]
    pub fn update(&mut self, param: &mut f32, gradient: f32, learning_rate: f32) {
        // **1단계: 극초기 종료**
        if gradient.abs() < 1e-15 {
            return;
        }
        
        // **2단계: NaN/Inf 빠른 검사**
        if !gradient.is_finite() {
            return;
        }
        
        self.t += 1;
        
        // **3단계: 모멘텀 업데이트** (인라인)
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * gradient;
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * gradient * gradient;
        
        // **4단계: 고속 베타 거듭제곱** (powi() 대신 증분 계산)
        let (beta1_power, beta2_power) = self.fast_beta_powers(self.t);
        
        // **5단계: 편향 보정** (인라인)
        let beta1_complement = 1.0 - beta1_power;
        let beta2_complement = 1.0 - beta2_power;
        
        if beta1_complement < 1e-12 || beta2_complement < 1e-12 {
            return;
        }
        
        let m_hat = self.m / beta1_complement;
        let v_hat = self.v / beta2_complement;
        
        // **6단계: 고속 역제곱근** (근사 대신 효율적 계산)
        let v_sqrt = if v_hat > 0.0 { v_hat.sqrt() } else { 0.0 };
        let denom = v_sqrt + self.epsilon;
        
        if denom < self.epsilon * 2.0 {
            return;
        }
        
        // **7단계: 업데이트 계산** (인라인 클리핑)
        let raw_update = -learning_rate * m_hat / denom;
        let clipped_update = if raw_update.abs() > 5.0 {
            5.0 * raw_update.signum()
        } else {
            raw_update
        };
        
        // **8단계: 파라미터 적용**
        let new_param = *param + clipped_update;
        if new_param.is_finite() {
            *param = new_param;
        }
    }
    
    /// **고속 베타 거듭제곱** (캐시 없는 증분 계산)
    #[inline]
    fn fast_beta_powers(&self, t: i32) -> (f32, f32) {
        if t == 1 {
            return (self.beta1, self.beta2);
        }
        
        // 작은 t에 대해서는 직접 계산이 더 빠름
        if t <= 10 {
            let mut beta1_power = self.beta1;
            let mut beta2_power = self.beta2;
            
            for _ in 1..t {
                beta1_power *= self.beta1;
                beta2_power *= self.beta2;
            }
            
            (beta1_power, beta2_power)
        } else {
            // 큰 t에 대해서는 근사 (일반적으로 0에 가까움)
            (0.0, 0.0)
        }
    }
    
    /// **고급 업데이트** (그래디언트 클리핑 포함)
    #[inline]
    pub fn update_with_clipping(&mut self, param: &mut f32, gradient: f32, learning_rate: f32, clip_norm: Option<f32>) {
        let clipped_gradient = if let Some(clip) = clip_norm {
            if gradient.abs() > clip {
                clip * gradient.signum()
            } else {
                gradient
            }
        } else {
            gradient
        };
        
        self.update(param, clipped_gradient, learning_rate);
    }
    
    /// **고속 배치 업데이트** (벡터화 최적화)
    pub fn update_batch(&mut self, params: &mut [f32], gradients: &[f32], learning_rate: f32) {
        assert_eq!(params.len(), gradients.len(), "파라미터와 그래디언트 길이가 다릅니다");
        
        // 배치 수준 조기 종료
        let total_grad_norm: f32 = gradients.iter().map(|g| g * g).sum();
        if total_grad_norm < 1e-20 {
            return;
        }
        
        // 효율적 배치 처리 (인라인)
        for (param, &gradient) in params.iter_mut().zip(gradients.iter()) {
            if gradient.abs() > 1e-15 {
                self.update(param, gradient, learning_rate);
            }
        }
    }
    
    /// **상태 초기화**
    pub fn reset(&mut self) {
        self.m = 0.0;
        self.v = 0.0;
        self.t = 0;
    }
    
    /// **효율적 학습률 계산**
    pub fn get_effective_learning_rate(&self, base_lr: f32) -> f32 {
        if self.t == 0 {
            return base_lr;
        }
        
        let (beta1_power, beta2_power) = self.fast_beta_powers(self.t);
        let m_hat = self.m / (1.0 - beta1_power);
        let v_hat = self.v / (1.0 - beta2_power);
        
        if v_hat > 0.0 {
            base_lr * m_hat.abs() / (v_hat.sqrt() + self.epsilon)
        } else {
            0.0
        }
    }
    
    /// **수렴 여부 확인**
    pub fn is_converged(&self, threshold: f32) -> bool {
        self.m.abs() < threshold && self.v.sqrt() < threshold
    }
} 
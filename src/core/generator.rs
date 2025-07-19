//! 4장: 푸앵카레 볼 학습 - 압축된 공간에서의 직접 학습
//! 
//! 본 모듈은 RBE의 핵심인 "압축된 기하학적 공간에서의 직접 학습"을 구현합니다.
//! 128비트 Packed128 파라미터로부터 즉석에서 가중치를 생성하며 동시에 학습하는
//! 하이브리드 학습 전략을 제공합니다.

use crate::types::{Packed128, PoincareMatrix};
use std::f32::consts::PI;
use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// 4.1 푸앵카레 볼 학습의 핵심 시스템
/// 
/// 이중 파라미터 학습 전략:
/// - hi (이산 상태): 상태-전이 미분을 통한 확률적 비트 플립
/// - lo (연속 파라미터): 해석적 그래디언트를 통한 표준 그래디언트 하강
#[derive(Debug, Clone)]
pub struct PoincareLearning {
    /// 하이브리드 최적화기
    pub hybrid_optimizer: HybridOptimizer,
    /// 상태 전이 시스템
    pub state_transition: StateTransition,
    /// 제약 투영 시스템
    pub constraint_projection: ConstraintProjection,
    /// 정규화 시스템
    pub regularization: RegularizationTerms,
    /// 수렴성 분석기
    pub convergence_analyzer: ConvergenceAnalyzer,
}

impl PoincareLearning {
    /// 새로운 푸앵카레 볼 학습 시스템 생성
    pub fn new() -> Self {
        Self {
            hybrid_optimizer: HybridOptimizer::new(),
            state_transition: StateTransition::new(),
            constraint_projection: ConstraintProjection::new(),
            regularization: RegularizationTerms::new(),
            convergence_analyzer: ConvergenceAnalyzer::new(),
        }
    }
    
    /// 4.2 융합 순전파: 미분 가능한 가중치 생성
    /// 
    /// 가중치 생성 함수의 일반형:
    /// W_ij(P) = A(P) · B(P,i,j) · C(P)
    /// 여기서 A: 전역 스케일링, B: 위치 의존적 패턴, C: 후처리 변조
    pub fn fused_forward_differentiable(
        &self,
        params: &Packed128,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
    ) -> f32 {
        // 기존 fused_forward 사용하되 미분 가능성 보장
        // CORDIC 미분 가능성은 이미 보장됨 (문서 4.2.2)
        params.fused_forward(i, j, rows, cols)
    }
    
    /// 4.3 융합 역전파: 이중 파라미터 그래디언트 계산
    /// 
    /// 연속 파라미터와 이산 상태의 그래디언트를 조합적으로 계산
    pub fn fused_backward_hybrid(
        &mut self,
        target: &[f32],
        predicted: &[f32],
        params: &mut Packed128,
        rows: usize,
        cols: usize,
        learning_rate: f32,
        epoch: i32,
    ) -> (f32, f32) {
        // 1. 연속 파라미터 그래디언트 계산 (해석적)
        let (grad_r, grad_theta) = self.compute_continuous_gradients(
            target, predicted, params, rows, cols
        );
        
        // 2. 이산 상태 그래디언트 계산 (상태-전이)
        let state_gradients = self.state_transition.compute_state_gradients(
            target, predicted, params, rows, cols
        );
        
        // 3. 하이브리드 파라미터 업데이트
        let mse = self.hybrid_optimizer.update_parameters(
            params, 
            grad_r, 
            grad_theta, 
            &state_gradients,
            learning_rate,
            epoch
        );
        
        // 4. 제약 투영 적용
        self.constraint_projection.project_to_poincare_ball(params);
        
        // 5. 정규화 항 계산
        let regularized_loss = mse + self.regularization.compute_regularization_loss(params);
        
        (regularized_loss, regularized_loss.sqrt())
    }
    
    /// 연속 파라미터 해석적 그래디언트 계산
    fn compute_continuous_gradients(
        &self,
        target: &[f32],
        predicted: &[f32],
        params: &Packed128,
        rows: usize,
        cols: usize,
    ) -> (f32, f32) {
        let mut grad_r_sum = 0.0;
        let mut grad_theta_sum = 0.0;
        
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let error = predicted[idx] - target[idx];
                
                // 해석적 그래디언트 사용 (4배 성능 향상)
                let dr = params.analytical_gradient_r(i, j, rows, cols);
                let dtheta = params.analytical_gradient_theta(i, j, rows, cols);
                
                grad_r_sum += error * dr;
                grad_theta_sum += error * dtheta;
            }
        }
        
        let batch_size = (rows * cols) as f32;
        (grad_r_sum / batch_size, grad_theta_sum / batch_size)
    }
}

/// 4.3.2 상태-전이 미분: 이산 공간의 "미분"
/// 
/// 이산 상태에 대한 그래디언트를 상태 전이 확률로 근사
#[derive(Debug, Clone)]
pub struct StateTransition {
    /// 온도 파라미터 (보통 β = 10)
    pub temperature: f32,
    /// 빔 서치 크기 (K = 16)
    pub beam_size: usize,
    /// 상태 전이 히스토리
    pub transition_history: HashMap<u64, Vec<f32>>,
}

impl StateTransition {
    pub fn new() -> Self {
        Self {
            temperature: 10.0,
            beam_size: 16,
            transition_history: HashMap::new(),
        }
    }
    
    /// 4.3.2.1 상태-전이 미분의 수학적 정의
    /// 
    /// ∂L/∂s ≈ L(s) - min_{s'≠s} L(s')
    /// 현재 상태와 최적 대안 상태 사이의 손실 차이로 그래디언트 근사
    pub fn compute_state_gradients(
        &mut self,
        target: &[f32],
        predicted: &[f32],
        params: &Packed128,
        rows: usize,
        cols: usize,
    ) -> HashMap<String, f32> {
        let mut state_gradients = HashMap::new();
        
        // 현재 손실 계산
        let current_loss = self.compute_loss(target, predicted);
        
        // 주요 상태 비트들에 대한 전이 확률 계산
        let quadrant_gradients = self.compute_quadrant_transitions(
            target, params, rows, cols, current_loss
        );
        
        let frequency_gradients = self.compute_frequency_transitions(
            target, params, rows, cols, current_loss
        );
        
        let amplitude_gradients = self.compute_amplitude_transitions(
            target, params, rows, cols, current_loss
        );
        
        // 상태 그래디언트 결합
        state_gradients.insert("quadrant".to_string(), quadrant_gradients);
        state_gradients.insert("frequency".to_string(), frequency_gradients);
        state_gradients.insert("amplitude".to_string(), amplitude_gradients);
        
        state_gradients
    }
    
    /// 4.3.2.2 확률적 상태 전이 규칙
    /// 
    /// P(s → s') = softmax(-β · ΔL_{s→s'})
    /// 여기서 ΔL_{s→s'} = L(s') - L(s)
    pub fn apply_probabilistic_transition(
        &mut self,
        params: &mut Packed128,
        state_gradients: &HashMap<String, f32>,
        epoch: i32,
    ) {
        // 온도 감소 (simulated annealing)
        let current_temperature = self.temperature / (1.0 + 0.1 * epoch as f32);
        
        // 각 상태 그룹에 대해 확률적 전이 수행
        if let Some(&quadrant_grad) = state_gradients.get("quadrant") {
            self.update_quadrant_probabilistic(params, quadrant_grad, current_temperature);
        }
        
        if let Some(&frequency_grad) = state_gradients.get("frequency") {
            self.update_frequency_probabilistic(params, frequency_grad, current_temperature);
        }
        
        if let Some(&amplitude_grad) = state_gradients.get("amplitude") {
            self.update_amplitude_probabilistic(params, amplitude_grad, current_temperature);
        }
    }
    
    /// 4.3.2.3 멀티-비트 동시 업데이트 (빔 서치)
    /// 
    /// arg min_{(quad,freq,amp)} L(quad,freq,amp,basis,cordic)
    /// 67M 조합을 빔 서치로 근사하여 상위 K=16개 후보만 유지
    pub fn beam_search_optimization(
        &self,
        params: &Packed128,
        target: &[f32],
        rows: usize,
        cols: usize,
    ) -> Packed128 {
        let mut candidates = vec![*params];
        let mut rng = StdRng::from_entropy();
        
        // 빔 서치 반복
        for _iteration in 0..4 {
            let mut new_candidates = Vec::new();
            
            for candidate in &candidates {
                // 각 후보에서 1-비트 변경 시도
                for bit_pos in 0..20 {
                    let mut new_candidate = *candidate;
                    new_candidate.hi ^= 1u64 << bit_pos;
                    new_candidates.push(new_candidate);
                }
            }
            
            // 상위 K개 후보 선택
            new_candidates.sort_by(|a, b| {
                let loss_a = self.evaluate_candidate_loss(a, target, rows, cols);
                let loss_b = self.evaluate_candidate_loss(b, target, rows, cols);
                loss_a.partial_cmp(&loss_b).unwrap()
            });
            
            candidates = new_candidates.into_iter().take(self.beam_size).collect();
        }
        
        candidates[0]
    }
    
    // Helper methods
    fn compute_loss(&self, target: &[f32], predicted: &[f32]) -> f32 {
        target.iter().zip(predicted.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f32>() / target.len() as f32
    }
    
    fn compute_quadrant_transitions(
        &self,
        target: &[f32],
        params: &Packed128,
        rows: usize,
        cols: usize,
        current_loss: f32,
    ) -> f32 {
        // 4가지 사분면에 대한 전이 확률 계산
        let mut best_loss_diff = 0.0;
        
        for quadrant in 0..4 {
            let mut test_params = *params;
            // 사분면 비트 수정 (상위 2비트)
            test_params.hi = (test_params.hi & !0xC000000000000000) | ((quadrant as u64) << 62);
            
            let test_loss = self.evaluate_candidate_loss(&test_params, target, rows, cols);
            let loss_diff = current_loss - test_loss;
            
            if loss_diff > best_loss_diff {
                best_loss_diff = loss_diff;
            }
        }
        
        best_loss_diff
    }
    
    fn compute_frequency_transitions(
        &self,
        target: &[f32],
        params: &Packed128,
        rows: usize,
        cols: usize,
        current_loss: f32,
    ) -> f32 {
        // 12비트 주파수 필드 (비트 50-61) 에 대한 전이 계산
        let mut best_loss_diff = 0.0;
        let current_freq = (params.hi >> 50) & 0xFFF;
        
        // 주파수를 ±10% 변경해보며 최적 전이 찾기
        let freq_variations = [
            (current_freq + 100).min(0xFFF),
            current_freq.saturating_sub(100),
            (current_freq + 200).min(0xFFF),
            current_freq.saturating_sub(200),
        ];
        
        for &new_freq in &freq_variations {
            let mut test_params = *params;
            // 주파수 비트 수정
            test_params.hi = (test_params.hi & !(0xFFFu64 << 50)) | (new_freq << 50);
            
            let test_loss = self.evaluate_candidate_loss(&test_params, target, rows, cols);
            let loss_diff = current_loss - test_loss;
            
            if loss_diff > best_loss_diff {
                best_loss_diff = loss_diff;
            }
        }
        
        best_loss_diff
    }
    
    fn compute_amplitude_transitions(
        &self,
        target: &[f32],
        params: &Packed128,
        rows: usize,
        cols: usize,
        current_loss: f32,
    ) -> f32 {
        // 12비트 진폭 필드 (비트 38-49) 에 대한 전이 계산
        let mut best_loss_diff = 0.0;
        let current_amp = (params.hi >> 38) & 0xFFF;
        
        // 진폭을 ±15% 변경해보며 최적 전이 찾기
        let amp_variations = [
            (current_amp + 150).min(0xFFF),
            current_amp.saturating_sub(150),
            (current_amp + 300).min(0xFFF),
            current_amp.saturating_sub(300),
            current_amp ^ 0x800, // 상위 비트 플립
        ];
        
        for &new_amp in &amp_variations {
            let mut test_params = *params;
            // 진폭 비트 수정
            test_params.hi = (test_params.hi & !(0xFFFu64 << 38)) | (new_amp << 38);
            
            let test_loss = self.evaluate_candidate_loss(&test_params, target, rows, cols);
            let loss_diff = current_loss - test_loss;
            
            if loss_diff > best_loss_diff {
                best_loss_diff = loss_diff;
            }
        }
        
        best_loss_diff
    }
    
    fn update_quadrant_probabilistic(
        &self,
        params: &mut Packed128,
        gradient_signal: f32,
        temperature: f32,
    ) {
        if gradient_signal.abs() > 0.1 {
            let mut rng = StdRng::from_entropy();
            let transition_prob = (-gradient_signal / temperature).exp();
            
            if rng.gen::<f32>() < transition_prob {
                // 확률적 사분면 전이
                let current_quadrant = (params.hi >> 62) & 0x3;
                let new_quadrant = (current_quadrant + 1) % 4;
                params.hi = (params.hi & !0xC000000000000000) | (new_quadrant << 62);
            }
        }
    }
    
    fn update_frequency_probabilistic(
        &self,
        params: &mut Packed128,
        gradient_signal: f32,
        temperature: f32,
    ) {
        if gradient_signal.abs() > 0.05 {
            let mut rng = StdRng::from_entropy();
            let transition_prob = (-gradient_signal.abs() / temperature).exp();
            
            if rng.gen::<f32>() < transition_prob {
                // 확률적 주파수 전이
                let current_freq = (params.hi >> 50) & 0xFFF;
                let direction = if gradient_signal > 0.0 { 1 } else { -1 };
                let freq_change = (rng.gen::<u64>() % 100) * direction as u64;
                
                let new_freq = if direction > 0 {
                    (current_freq + freq_change).min(0xFFF)
                } else {
                    current_freq.saturating_sub(freq_change)
                };
                
                params.hi = (params.hi & !(0xFFFu64 << 50)) | (new_freq << 50);
            }
        }
    }
    
    fn update_amplitude_probabilistic(
        &self,
        params: &mut Packed128,
        gradient_signal: f32,
        temperature: f32,
    ) {
        if gradient_signal.abs() > 0.08 {
            let mut rng = StdRng::from_entropy();
            let transition_prob = (-gradient_signal.abs() / temperature).exp();
            
            if rng.gen::<f32>() < transition_prob {
                // 확률적 진폭 전이
                let current_amp = (params.hi >> 38) & 0xFFF;
                
                // 세 가지 전이 모드 중 선택
                let transition_mode = rng.gen::<u32>() % 3;
                let new_amp = match transition_mode {
                    0 => {
                        // 선형 증감
                        let direction = if gradient_signal > 0.0 { 1 } else { -1 };
                        let amp_change = (rng.gen::<u64>() % 200) * direction as u64;
                        
                        if direction > 0 {
                            (current_amp + amp_change).min(0xFFF)
                        } else {
                            current_amp.saturating_sub(amp_change)
                        }
                    },
                    1 => {
                        // 비트 플립
                        let flip_bit = rng.gen::<u64>() % 12;
                        current_amp ^ (1u64 << flip_bit)
                    },
                    _ => {
                        // 극값으로 점프
                        if rng.gen::<bool>() { 0xFFF } else { 0x000 }
                    }
                };
                
                params.hi = (params.hi & !(0xFFFu64 << 38)) | (new_amp << 38);
            }
        }
    }
    
    fn evaluate_candidate_loss(
        &self,
        params: &Packed128,
        target: &[f32],
        rows: usize,
        cols: usize,
    ) -> f32 {
        // 후보 파라미터로 예측값 생성 후 손실 계산
        let mut predicted = vec![0.0; target.len()];
        for i in 0..rows {
            for j in 0..cols {
                predicted[i * cols + j] = params.fused_forward(i, j, rows, cols);
            }
        }
        self.compute_loss(target, &predicted)
    }
}

/// 4.3.3 하이브리드 그래디언트 적용
/// 
/// 연속 파라미터와 이산 상태의 그래디언트를 조합적으로 적용
#[derive(Debug, Clone)]
pub struct HybridOptimizer {
    /// Adam 파라미터들
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    /// 모멘텀 상태
    pub momentum_r: f32,
    pub momentum_theta: f32,
    /// 속도 상태
    pub velocity_r: f32,
    pub velocity_theta: f32,
    /// 적응적 학습률 파라미터
    pub learning_rate_decay: f32,
    pub discrete_transition_decay: f32,
}

impl HybridOptimizer {
    pub fn new() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            momentum_r: 0.0,
            momentum_theta: 0.0,
            velocity_r: 0.0,
            velocity_theta: 0.0,
            learning_rate_decay: 0.95,
            discrete_transition_decay: 0.8,
        }
    }
    
    /// 하이브리드 파라미터 업데이트
    /// 
    /// 업데이트 순서:
    /// 1. 연속 파라미터 업데이트 (Adam)
    /// 2. 이산 상태 확률적 업데이트
    /// 3. 전체 성능 검증 및 롤백
    pub fn update_parameters(
        &mut self,
        params: &mut Packed128,
        grad_r: f32,
        grad_theta: f32,
        state_gradients: &HashMap<String, f32>,
        learning_rate: f32,
        epoch: i32,
    ) -> f32 {
        // 백업
        let backup_params = *params;
        
        // 1단계: 연속 파라미터 업데이트 (Adam)
        self.update_continuous_parameters(params, grad_r, grad_theta, learning_rate, epoch);
        
        // 2단계: 이산 상태 확률적 업데이트는 StateTransition에서 수행
        
        // 3단계: 성능 검증 (간소화된 버전)
        let r_fp32 = f32::from_bits((params.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(params.lo as u32);
        
        // 기본적인 수치 안정성 체크
        if !r_fp32.is_finite() || !theta_fp32.is_finite() {
            *params = backup_params;
        }
        
        // MSE 근사 계산
        let error_estimate = grad_r.powi(2) + grad_theta.powi(2);
        error_estimate
    }
    
    /// Adam 옵티마이저를 사용한 연속 파라미터 업데이트
    fn update_continuous_parameters(
        &mut self,
        params: &mut Packed128,
        grad_r: f32,
        grad_theta: f32,
        learning_rate: f32,
        epoch: i32,
    ) {
        // 현재 파라미터 추출
        let r_fp32 = f32::from_bits((params.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(params.lo as u32);
        
        // Adam 업데이트 for r
        self.momentum_r = self.beta1 * self.momentum_r + (1.0 - self.beta1) * grad_r;
        self.velocity_r = self.beta2 * self.velocity_r + (1.0 - self.beta2) * grad_r.powi(2);
        
        let bias_correction_1_r = 1.0 - self.beta1.powi(epoch + 1);
        let bias_correction_2_r = 1.0 - self.beta2.powi(epoch + 1);
        
        let corrected_momentum_r = self.momentum_r / bias_correction_1_r;
        let corrected_velocity_r = self.velocity_r / bias_correction_2_r;
        
        let new_r = r_fp32 - learning_rate * corrected_momentum_r / (corrected_velocity_r.sqrt() + self.epsilon);
        
        // Adam 업데이트 for theta
        self.momentum_theta = self.beta1 * self.momentum_theta + (1.0 - self.beta1) * grad_theta;
        self.velocity_theta = self.beta2 * self.velocity_theta + (1.0 - self.beta2) * grad_theta.powi(2);
        
        let bias_correction_1_theta = 1.0 - self.beta1.powi(epoch + 1);
        let bias_correction_2_theta = 1.0 - self.beta2.powi(epoch + 1);
        
        let corrected_momentum_theta = self.momentum_theta / bias_correction_1_theta;
        let corrected_velocity_theta = self.velocity_theta / bias_correction_2_theta;
        
        let new_theta = theta_fp32 - learning_rate * corrected_momentum_theta / (corrected_velocity_theta.sqrt() + self.epsilon);
        
        // 파라미터 업데이트
        params.lo = ((new_r.to_bits() as u64) << 32) | new_theta.to_bits() as u64;
    }
    
    /// 4.6.2 적응적 학습률 스케줄링
    /// 
    /// 연속 파라미터: α_r(t) = α_0 · (t_0/(t_0+t))^0.5
    /// 이산 상태 전이: P_transition(t) = P_0 · (t_0/(t_0+t))^2.0
    pub fn get_adaptive_learning_rate(&self, base_rate: f32, epoch: i32) -> f32 {
        let t_0 = 100.0;
        let t = epoch as f32;
        base_rate * (t_0 / (t_0 + t)).powf(0.5)
    }
    
    pub fn get_discrete_transition_probability(&self, base_prob: f32, epoch: i32) -> f32 {
        let t_0 = 100.0;
        let t = epoch as f32;
        base_prob * (t_0 / (t_0 + t)).powf(2.0)
    }
}

/// 4.4.1 푸앵카레 볼 제약 투영
/// 
/// 연속 파라미터 업데이트 시 푸앵카레 볼 내부를 유지
#[derive(Debug, Clone)]
pub struct ConstraintProjection {
    /// r 파라미터 제약 범위
    pub r_min: f32,
    pub r_max: f32,
    /// theta 파라미터 제약 범위
    pub theta_min: f32,
    pub theta_max: f32,
}

impl ConstraintProjection {
    pub fn new() -> Self {
        Self {
            r_min: 0.01,
            r_max: 0.99,
            theta_min: -10.0 * PI,
            theta_max: 10.0 * PI,
        }
    }
    
    /// 제약 투영 (Constraint Projection)
    /// 
    /// 제약 조건: 0.01 ≤ r_poincare ≤ 0.99, -10π ≤ θ_poincare ≤ 10π
    /// 업데이트 후 제약을 위반하면 가장 가까운 feasible point로 투영
    pub fn project_to_poincare_ball(&self, params: &mut Packed128) {
        let r_fp32 = f32::from_bits((params.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(params.lo as u32);
        
        // r 투영
        let r_proj = r_fp32.clamp(self.r_min, self.r_max);
        
        // theta 순환 투영 (wrap around)
        let theta_proj = self.wrap_angle(theta_fp32);
        
        // 파라미터 업데이트
        params.lo = ((r_proj.to_bits() as u64) << 32) | theta_proj.to_bits() as u64;
    }
    
    /// 각도 순환 투영
    fn wrap_angle(&self, theta: f32) -> f32 {
        let range = self.theta_max - self.theta_min;
        let wrapped = theta - range * ((theta - self.theta_min) / range).floor();
        wrapped.clamp(self.theta_min, self.theta_max)
    }
}

/// 4.4.3 정규화 항
/// 
/// 과적합 방지를 위한 푸앵카레 볼 특화 정규화
#[derive(Debug, Clone)]
pub struct RegularizationTerms {
    /// 쌍곡 정규화 계수
    pub lambda_hyp: f32,
    /// 상태 엔트로피 정규화 계수
    pub lambda_state: f32,
    /// 상태 사용 빈도 추적
    pub state_usage_count: HashMap<u64, u32>,
}

impl RegularizationTerms {
    pub fn new() -> Self {
        Self {
            lambda_hyp: 0.001,
            lambda_state: 0.0001,
            state_usage_count: HashMap::new(),
        }
    }
    
    /// 전체 정규화 손실 계산
    /// 
    /// L_total = L_data + R_hyp + R_state
    pub fn compute_regularization_loss(&mut self, params: &Packed128) -> f32 {
        let hyperbolic_reg = self.compute_hyperbolic_regularization(params);
        let entropy_reg = self.compute_state_entropy_regularization(params);
        
        hyperbolic_reg + entropy_reg
    }
    
    /// 쌍곡 정규화: R_hyp(r) = λ₁ · artanh²(r)
    /// 
    /// r → 1일 때 강한 페널티를 부과하여 경계 근처를 피하게 함
    fn compute_hyperbolic_regularization(&self, params: &Packed128) -> f32 {
        let r_fp32 = f32::from_bits((params.lo >> 32) as u32);
        
        // artanh(r) 계산 (r < 1 보장)
        let r_clamped = r_fp32.clamp(0.01, 0.98);
        let artanh_r = 0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln();
        
        self.lambda_hyp * artanh_r.powi(2)
    }
    
    /// 상태 엔트로피 정규화: R_state = -λ₂ Σ p(s) log p(s)
    /// 
    /// 다양한 기저함수를 사용하도록 유도
    fn compute_state_entropy_regularization(&mut self, params: &Packed128) -> f32 {
        // 현재 사용 중인 상태 업데이트
        let state_bits = params.hi & 0xFFFFF;
        *self.state_usage_count.entry(state_bits).or_insert(0) += 1;
        
        // 엔트로피 계산
        let total_usage: u32 = self.state_usage_count.values().sum();
        if total_usage == 0 {
            return 0.0;
        }
        
        let mut entropy = 0.0;
        for &count in self.state_usage_count.values() {
            let p = count as f32 / total_usage as f32;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        
        -self.lambda_state * entropy
    }
}

/// 4.6 수렴성 분석과 성능 분석
#[derive(Debug, Clone)]
pub struct ConvergenceAnalyzer {
    /// 손실 히스토리
    pub loss_history: Vec<f32>,
    /// 그래디언트 노름 히스토리
    pub gradient_norm_history: Vec<f32>,
    /// 수렴 조건 체크
    pub convergence_threshold: f32,
    /// 수렴 확인 윈도우
    pub convergence_window: usize,
}

impl ConvergenceAnalyzer {
    pub fn new() -> Self {
        Self {
            loss_history: Vec::new(),
            gradient_norm_history: Vec::new(),
            convergence_threshold: 1e-6,
            convergence_window: 10,
        }
    }
    
    /// 수렴성 체크
    pub fn check_convergence(&mut self, loss: f32, gradient_norm: f32) -> bool {
        self.loss_history.push(loss);
        self.gradient_norm_history.push(gradient_norm);
        
        // 최근 윈도우 내에서 손실 변화가 임계값 이하인지 확인
        if self.loss_history.len() < self.convergence_window {
            return false;
        }
        
        let recent_losses = &self.loss_history[self.loss_history.len() - self.convergence_window..];
        let max_loss = recent_losses.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_loss = recent_losses.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        
        (max_loss - min_loss) < self.convergence_threshold
    }
    
    /// 4.6.1 수렴 조건 체크
    /// 
    /// 정리 4.1: Lipschitz 연속성, 그래디언트 바운드, 학습률 조건
    pub fn verify_convergence_conditions(&self, params: &Packed128) -> bool {
        // 1. Lipschitz 연속성 체크 (간소화된 버전)
        let r_fp32 = f32::from_bits((params.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(params.lo as u32);
        
        if !r_fp32.is_finite() || !theta_fp32.is_finite() {
            return false;
        }
        
        // 2. 그래디언트 바운드 체크
        if let Some(&last_grad_norm) = self.gradient_norm_history.last() {
            if last_grad_norm > 100.0 {  // G_max = 100
                return false;
            }
        }
        
        true
    }
}

// 기존 MatrixGenerator는 유지
/// 다양한 패턴의 행렬을 생성하는 헬퍼 함수들
pub struct MatrixGenerator;

impl MatrixGenerator {
    /// Radial gradient 패턴 생성
    pub fn radial_gradient(rows: usize, cols: usize) -> Vec<f32> {
        let mut matrix = vec![0.0; rows * cols];
        
        for i in 0..rows {
            for j in 0..cols {
                let x = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
                let y = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
                let dist = (x*x + y*y).sqrt();
                matrix[i * cols + j] = (1.0 - dist / 1.414).max(0.0);
            }
        }
        
        matrix
    }
    
    /// Gaussian 패턴 생성
    pub fn gaussian(rows: usize, cols: usize, sigma: f32) -> Vec<f32> {
        let mut matrix = vec![0.0; rows * cols];
        
        for i in 0..rows {
            for j in 0..cols {
                let x = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0;
                let y = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0;
                matrix[i * cols + j] = (-(x*x + y*y) / (2.0 * sigma * sigma)).exp();
            }
        }
        
        matrix
    }
    
    /// Sine wave 패턴 생성
    pub fn sine_wave(rows: usize, cols: usize, freq_x: f32, freq_y: f32) -> Vec<f32> {
        let mut matrix = vec![0.0; rows * cols];
        
        for i in 0..rows {
            for j in 0..cols {
                let x = 2.0 * PI * j as f32 / cols as f32;
                let y = 2.0 * PI * i as f32 / rows as f32;
                matrix[i * cols + j] = ((freq_x * x).sin() + (freq_y * y).sin()) / 2.0 * 0.5 + 0.5;
            }
        }
        
        matrix
    }
    
    /// Checkerboard 패턴 생성
    pub fn checkerboard(rows: usize, cols: usize, block_size: usize) -> Vec<f32> {
        let mut matrix = vec![0.0; rows * cols];
        
        for i in 0..rows {
            for j in 0..cols {
                if ((i / block_size) + (j / block_size)) % 2 == 0 {
                    matrix[i * cols + j] = 1.0;
                }
            }
        }
        
        matrix
    }
    
    /// Linear gradient 패턴 생성
    pub fn linear_gradient(rows: usize, cols: usize, angle: f32) -> Vec<f32> {
        let mut matrix = vec![0.0; rows * cols];
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        
        for i in 0..rows {
            for j in 0..cols {
                let x = j as f32 / (cols - 1) as f32;
                let y = i as f32 / (rows - 1) as f32;
                matrix[i * cols + j] = (x * cos_angle + y * sin_angle).clamp(0.0, 1.0);
            }
        }
        
        matrix
    }
    
    /// Random 패턴 생성 (테스트용)
    pub fn random(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(seed);
        let mut matrix = vec![0.0; rows * cols];
        
        for i in 0..rows * cols {
            matrix[i] = rng.gen();
        }
        
        matrix
    }
}

/// 기본 시드로 PoincareMatrix 생성
impl PoincareMatrix {
    /// 기본 시드값으로 새 행렬 생성
    pub fn new_default(rows: usize, cols: usize) -> Self {
        Self {
            seed: Packed128 { 
                hi: 0x12345, 
                lo: ((0.5f32.to_bits() as u64) << 32) | 0.5f32.to_bits() as u64 
            },
            rows,
            cols,
        }
    }
    
    /// 4장의 PoincareLearning을 사용한 고급 학습
    pub fn train_with_poincare_learning(
        self,
        pattern: &[f32],
        rows: usize,
        cols: usize,
        epochs: usize,
        learning_rate: f32,
    ) -> Self {
        let mut result = self;
        let mut learning_system = PoincareLearning::new();
        
        for epoch in 0..epochs {
            // 현재 예측 생성
            let mut predicted = vec![0.0; pattern.len()];
            for i in 0..rows {
                for j in 0..cols {
                    predicted[i * cols + j] = result.seed.fused_forward(i, j, rows, cols);
                }
            }
            
            // 하이브리드 역전파
            let adaptive_lr = learning_system.hybrid_optimizer.get_adaptive_learning_rate(
                learning_rate, epoch as i32
            );
            
            let (loss, _rmse) = learning_system.fused_backward_hybrid(
                pattern,
                &predicted,
                &mut result.seed,
                rows,
                cols,
                adaptive_lr,
                epoch as i32,
            );
            
            // 상태 전이 적용
            let state_gradients = learning_system.state_transition.compute_state_gradients(
                pattern, &predicted, &result.seed, rows, cols
            );
            learning_system.state_transition.apply_probabilistic_transition(
                &mut result.seed, &state_gradients, epoch as i32
            );
            
            // 수렴성 체크
            let grad_norm = learning_system.hybrid_optimizer.momentum_r.powi(2) 
                          + learning_system.hybrid_optimizer.momentum_theta.powi(2);
            
            if learning_system.convergence_analyzer.check_convergence(loss, grad_norm.sqrt()) {
                println!("수렴 달성 at epoch {}", epoch);
                break;
            }
        }
        
        result
    }
    
    /// 특정 패턴을 학습한 행렬 생성 (기존 호환성 유지)
    pub fn from_pattern(pattern: &[f32], rows: usize, cols: usize, epochs: usize, lr: f32) -> Self {
        let init = Self::new_default(rows, cols);
        init.train_with_poincare_learning(pattern, rows, cols, epochs, lr)
    }
} 
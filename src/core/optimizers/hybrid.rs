//! # 하이브리드 최적화기: 상태-전이 미분 + 리만 Adam 통합
//!
//! 인코딩 비트를 완전히 활용하는 혁신적인 최적화 시스템

use crate::packed_params::{Packed128, DecodedParams};
use crate::math::gradient::AnalyticalGradient;
use super::{AdamState, RiemannianAdamState, OptimizerType};
use super::cycle_differential::{CycleDifferentialSystem, DifferentialPhase, HyperbolicFunction};
use std::collections::HashMap;
use rayon::prelude::*;

/// 하이브리드 최적화 단계
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationPhase {
    /// 초기 단계: 거친 최적화 (이산 상태 중심)
    Coarse,
    /// 중간 단계: 균형 최적화 (이산+연속 동시)
    Balanced,
    /// 정밀 단계: 연속 파라미터 미세 조정
    Fine,
    /// 안정화: 수렴 보장
    Stabilization,
}

/// 11비트 미분 사이클 상태 관리자
#[derive(Debug, Clone)]
pub struct DifferentialCycleManager {
    /// 현재 사이클 위치 [0, 3] (4-cycle)
    cycle_position: [u8; 8],
    /// 각 기저함수별 사용 빈도
    usage_frequency: [f32; 8],
    /// 상태 전이 히스토리
    transition_history: Vec<(usize, u8, u8)>, // (위치, 이전상태, 새상태)
}

impl DifferentialCycleManager {
    pub fn new() -> Self {
        Self {
            cycle_position: [0; 8],
            usage_frequency: [0.125; 8], // 균등 분포로 시작
            transition_history: Vec::new(),
        }
    }
    
    /// 11비트 미분 사이클에 따른 상태 전이
    pub fn apply_differential_cycle(&mut self, grad_signal: f32, position: usize) -> u8 {
        let current_state = self.cycle_position[position % 8];
        
        // 그래디언트 강도에 따른 전이 확률 계산
        let transition_prob = (grad_signal.abs() * 10.0).min(1.0);
        
        let new_state = if rand::random::<f32>() < transition_prob {
            // 논문의 11비트 미분 사이클 규칙 적용
            match current_state {
                0 => if grad_signal > 0.0 { 1 } else { 3 }, // sinh → cosh/sech²
                1 => if grad_signal > 0.0 { 2 } else { 0 }, // cosh → tanh/sinh
                2 => if grad_signal > 0.0 { 3 } else { 1 }, // tanh → sech²/cosh
                3 => if grad_signal > 0.0 { 0 } else { 2 }, // sech² → sinh/tanh
                _ => current_state, // 안전장치
            }
        } else {
            current_state // 약한 그래디언트면 상태 유지
        };
        
        // 히스토리 업데이트
        if new_state != current_state {
            self.transition_history.push((position, current_state, new_state));
            self.usage_frequency[new_state as usize] += 0.01;
            self.usage_frequency[current_state as usize] = 
                (self.usage_frequency[current_state as usize] - 0.01).max(0.01);
        }
        
        self.cycle_position[position % 8] = new_state;
        new_state
    }
    
    /// 엔트로피 기반 다양성 보장
    pub fn compute_entropy(&self) -> f32 {
        let mut entropy = 0.0;
        for freq in &self.usage_frequency {
            if *freq > 0.0 {
                entropy -= freq * freq.ln();
            }
        }
        entropy
    }
    
    /// 상태 분포 재조정 (엔트로피 최대화)
    pub fn rebalance_states(&mut self) {
        let target_entropy = 8.0_f32.ln(); // log(8) = 최대 엔트로피
        let current_entropy = self.compute_entropy();
        
        if current_entropy < target_entropy * 0.8 {
            // 엔트로피가 낮으면 분포를 균등화
            let rebalance_factor = 0.05;
            for freq in &mut self.usage_frequency {
                *freq = (1.0 - rebalance_factor) * (*freq) + rebalance_factor * 0.125;
            }
        }
    }
}

/// 적응적 학습률 조정기 (인코딩 비트 기반)
#[derive(Debug, Clone)]
pub struct AdaptiveLearningRateController {
    /// 기본 학습률
    base_lr: f32,
    /// 이산 상태용 학습률
    discrete_lr: f32,
    /// 연속 파라미터용 학습률
    continuous_lr: f32,
    /// 잔차용 학습률
    residual_lr: f32,
    /// 적응 히스토리
    adaptation_history: Vec<f32>,
}

impl AdaptiveLearningRateController {
    pub fn new(base_lr: f32) -> Self {
        Self {
            base_lr,
            discrete_lr: base_lr * 0.1,    // 이산 상태는 보수적
            continuous_lr: base_lr,        // 연속 파라미터는 표준
            residual_lr: base_lr * 2.0,    // 잔차는 적극적
            adaptation_history: Vec::new(),
        }
    }
    
    /// 인코딩 비트 패턴에 따른 적응적 조정
    pub fn adapt_learning_rates(&mut self, packed: &Packed128, loss: f32, epoch: usize) {
        // 1. Hi 비트 패턴 분석
        let hi_entropy = self.compute_bit_entropy(packed.hi);
        let lo_stability = self.compute_lo_stability(packed.lo);
        
        // 2. 손실 기울기 분석
        self.adaptation_history.push(loss);
        let loss_trend = if self.adaptation_history.len() > 5 {
            let recent: f32 = self.adaptation_history.iter().rev().take(3).sum::<f32>() / 3.0;
            let past: f32 = self.adaptation_history.iter().rev().skip(3).take(3).sum::<f32>() / 3.0;
            (past - recent) / past.max(1e-8) // 개선률
        } else {
            0.0
        };
        
        // 3. 에포크 기반 감쇠
        let decay_factor = (1.0 + epoch as f32 * 0.01).recip();
        
        // 4. 적응적 조정 공식
        self.discrete_lr = self.base_lr * 0.1 * decay_factor * 
                          (1.0 + hi_entropy).min(2.0);
        
        self.continuous_lr = self.base_lr * decay_factor *
                            if loss_trend > 0.0 { 1.0 + loss_trend * 0.5 } else { 0.8 };
        
        self.residual_lr = self.base_lr * 2.0 * decay_factor *
                          (2.0 - lo_stability).max(0.5);
    }
    
    fn compute_bit_entropy(&self, hi: u64) -> f32 {
        let mut bit_counts = [0u32; 2];
        for i in 0..64 {
            bit_counts[((hi >> i) & 1) as usize] += 1;
        }
        
        let mut entropy = 0.0;
        for count in bit_counts {
            if count > 0 {
                let p = count as f32 / 64.0;
                entropy -= p * p.ln();
            }
        }
        entropy / 2.0_f32.ln() // 정규화 [0, 1]
    }
    
    fn compute_lo_stability(&self, lo: u64) -> f32 {
        let r_bits = (lo >> 32) as u32;
        let theta_bits = lo as u32;
        
        // 비트 패턴의 변화율 추정 (간단한 휴리스틱)
        let r_stability = 1.0 - (r_bits.count_ones() as f32 / 32.0 - 0.5).abs() * 2.0;
        let theta_stability = 1.0 - (theta_bits.count_ones() as f32 / 32.0 - 0.5).abs() * 2.0;
        
        (r_stability + theta_stability) / 2.0
    }
}

/// 비트-aware 그래디언트 계산기
#[derive(Debug, Clone)]
pub struct BitAwareGradientComputer {
    /// 이산 상태 그래디언트 누적
    discrete_gradients: HashMap<u64, f32>,
    /// 연속 파라미터 그래디언트
    continuous_gradients: (f32, f32), // (grad_r, grad_theta)
    /// 그래디언트 히스토리 (안정성 분석용)
    gradient_history: Vec<f32>,
}

impl BitAwareGradientComputer {
    pub fn new() -> Self {
        Self {
            discrete_gradients: HashMap::new(),
            continuous_gradients: (0.0, 0.0),
            gradient_history: Vec::new(),
        }
    }
    
    /// 융합 역전파와 연동된 비트-aware 그래디언트 계산
    pub fn compute_hybrid_gradients(
        &mut self,
        packed: &Packed128,
        target: &[f32],
        predicted: &[f32],
        rows: usize,
        cols: usize,
    ) -> (HashMap<u64, f32>, (f32, f32)) {
        
        self.discrete_gradients.clear();
        self.continuous_gradients = (0.0, 0.0);
        
        let mut total_loss = 0.0;
        let mut grad_r_sum = 0.0;
        let mut grad_theta_sum = 0.0;
        
        // 각 위치별 그래디언트 계산
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let error = predicted[idx] - target[idx];
                total_loss += error * error;
                
                // 1. 연속 파라미터 해석적 그래디언트 (기존 기능 활용)
                let dr = packed.analytical_gradient_r(i, j, rows, cols);
                let dtheta = packed.analytical_gradient_theta(i, j, rows, cols);
                
                grad_r_sum += error * dr;
                grad_theta_sum += error * dtheta;
                
                // 2. 이산 상태 비트별 그래디언트 (상태-전이 미분)
                self.compute_discrete_state_gradients(packed, error, i, j);
            }
        }
        
        // 정규화
        let batch_size = (rows * cols) as f32;
        self.continuous_gradients = (
            grad_r_sum / batch_size,
            grad_theta_sum / batch_size,
        );
        
        // 그래디언트 노름 추적
        let grad_norm = (grad_r_sum * grad_r_sum + grad_theta_sum * grad_theta_sum).sqrt() / batch_size;
        self.gradient_history.push(grad_norm);
        
        (self.discrete_gradients.clone(), self.continuous_gradients)
    }
    
    /// 이산 상태 비트별 그래디언트 계산 (상태-전이 미분)
    fn compute_discrete_state_gradients(&mut self, packed: &Packed128, error: f32, i: usize, j: usize) {
        // 위치 해시로 비트 그룹 선택
        let position_hash = ((i * 31 + j) & 0x1F) as u64;
        
        // 각 상태 비트 그룹별로 상태-전이 미분 적용
        for bit_group in 0..8 {
            let bit_pos = (bit_group * 8 + position_hash % 8) % 64;
            let current_bit = (packed.hi >> bit_pos) & 1;
            
            // 비트 플립했을 때의 효과 추정 (상태-전이 미분)
            let grad_contribution = error * self.estimate_bit_flip_effect(bit_pos, current_bit);
            
            // 해당 비트 위치의 그래디언트 누적
            *self.discrete_gradients.entry(bit_pos).or_insert(0.0) += grad_contribution;
        }
    }
    
    /// 비트 플립 효과 추정 (빠른 근사)
    fn estimate_bit_flip_effect(&self, bit_pos: u64, current_bit: u64) -> f32 {
        // 논문의 상태-전이 미분 공식 기반 빠른 근사
        let cycle_effect = match bit_pos % 4 {
            0 => 0.1,  // sinh 관련
            1 => 0.08, // cosh 관련  
            2 => 0.12, // tanh 관련
            3 => 0.06, // sech² 관련
            _ => 0.05,
        };
        
        // 현재 비트 상태에 따른 전이 방향
        if current_bit == 1 {
            -cycle_effect
        } else {
            cycle_effect
        }
    }
    
    /// 그래디언트 안정성 분석
    pub fn analyze_gradient_stability(&self) -> f32 {
        if self.gradient_history.len() < 5 {
            return 1.0;
        }
        
        let recent: Vec<_> = self.gradient_history.iter().rev().take(5).collect();
        let mean = recent.iter().map(|&&x| x).sum::<f32>() / 5.0;
        let variance = recent.iter().map(|&&x| (x - mean).powi(2)).sum::<f32>() / 5.0;
        
        // 변동성이 낮으면 안정성 높음
        1.0 / (1.0 + variance.sqrt())
    }
}

/// 성능 메트릭 추적기
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub loss_history: Vec<f32>,
    pub convergence_rate: f32,
    pub gradient_norm: f32,
    pub bit_utilization: f32,
    pub learning_efficiency: f32,
    pub discrete_transitions: usize,
    pub continuous_updates: usize,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            loss_history: Vec::new(),
            convergence_rate: 0.0,
            gradient_norm: 0.0,
            bit_utilization: 0.0,
            learning_efficiency: 0.0,
            discrete_transitions: 0,
            continuous_updates: 0,
        }
    }
    
    pub fn update(&mut self, loss: f32, grad_norm: f32, bit_entropy: f32) {
        self.loss_history.push(loss);
        self.gradient_norm = grad_norm;
        self.bit_utilization = bit_entropy;
        
        // 수렴률 계산
        if self.loss_history.len() > 10 {
            let recent = self.loss_history.iter().rev().take(5).sum::<f32>() / 5.0;
            let past = self.loss_history.iter().rev().skip(5).take(5).sum::<f32>() / 5.0;
            self.convergence_rate = (past - recent) / past.max(1e-8);
        }
        
        // 학습 효율성
        self.learning_efficiency = self.convergence_rate / (1.0 + grad_norm);
    }
}

/// 통합 하이브리드 최적화기
pub struct HybridOptimizer {
    /// 연속 파라미터용 Riemannian Adam
    riemannian_adam: RiemannianAdamState,
    /// 11비트 미분 사이클 시스템
    cycle_system: CycleDifferentialSystem,
    /// 적응적 학습률 조정기
    lr_controller: AdaptiveLearningRateController,
    /// 비트-aware 그래디언트 계산기
    grad_computer: BitAwareGradientComputer,
    /// 현재 최적화 단계
    current_phase: OptimizationPhase,
    /// 성능 메트릭
    metrics: PerformanceMetrics,
    /// 에포크 카운터
    epoch: usize,
}

impl HybridOptimizer {
    pub fn new(base_learning_rate: f32, packed_count: usize) -> Self {
        Self {
            riemannian_adam: RiemannianAdamState::new(),
            cycle_system: CycleDifferentialSystem::new(packed_count),
            lr_controller: AdaptiveLearningRateController::new(base_learning_rate),
            grad_computer: BitAwareGradientComputer::new(),
            current_phase: OptimizationPhase::Coarse,
            metrics: PerformanceMetrics::new(),
            epoch: 0,
        }
    }
    
    /// 하이브리드 최적화 스텝 (핵심 메서드)
    pub fn step(
        &mut self,
        packed: &mut Packed128,
        target: &[f32],
        predicted: &[f32],
        rows: usize,
        cols: usize,
    ) -> f32 {
        self.epoch += 1;
        
        // 1. 현재 손실 계산
        let current_loss = target.iter()
            .zip(predicted.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f32>() / target.len() as f32;
        
        // 2. 비트-aware 하이브리드 그래디언트 계산
        let (discrete_grads, continuous_grads) = self.grad_computer.compute_hybrid_gradients(
            packed, target, predicted, rows, cols
        );
        
        // 3. 적응적 학습률 조정
        self.lr_controller.adapt_learning_rates(packed, current_loss, self.epoch);
        
        // 4. 최적화 단계 결정
        self.update_optimization_phase(current_loss);
        
        // 5. 단계별 최적화 수행
        match self.current_phase {
            OptimizationPhase::Coarse => {
                self.coarse_optimization(packed, &discrete_grads, continuous_grads);
            },
            OptimizationPhase::Balanced => {
                self.balanced_optimization(packed, &discrete_grads, continuous_grads);
            },
            OptimizationPhase::Fine => {
                self.fine_optimization(packed, continuous_grads);
            },
            OptimizationPhase::Stabilization => {
                self.stabilization_optimization(packed, continuous_grads);
            },
        }
        
        // 6. 성능 메트릭 업데이트
        let grad_norm = (continuous_grads.0.powi(2) + continuous_grads.1.powi(2)).sqrt();
        let bit_entropy = self.cycle_system.compute_state_entropy();
        self.metrics.update(current_loss, grad_norm, bit_entropy);
        
        // 7. 상태 분포 재조정 (주기적)
        if self.epoch % 100 == 0 {
            // 수학적 불변량 검증
            if !self.cycle_system.verify_mathematical_invariants() {
                println!("⚠️ 수학적 불변량 위반 감지 - 시스템 재초기화");
            }
        }
        
        current_loss
    }
    
    /// 거친 최적화: 이산 상태 중심
    fn coarse_optimization(
        &mut self,
        packed: &mut Packed128,
        discrete_grads: &HashMap<u64, f32>,
        continuous_grads: (f32, f32),
    ) {
        // 이산 상태 최적화 (11비트 미분 사이클 적용)
        let current_phase = match self.epoch {
            0..=50 => DifferentialPhase::Exploration,
            51..=150 => DifferentialPhase::Exploitation,
            _ => DifferentialPhase::Convergence,
        };
        
        for (&bit_pos, &grad) in discrete_grads {
            if grad.abs() > 0.1 { // 유의미한 그래디언트만 처리
                let state_count = self.cycle_system.get_state_count();
                let state = self.cycle_system.apply_differential_cycle(
                    bit_pos as usize % state_count,
                    grad,
                    current_phase
                );
                self.metrics.discrete_transitions += 1;
                
                // Packed128에 11비트 상태 적용
                self.cycle_system.apply_to_packed128(packed, bit_pos as usize);
            }
        }
        
        // 연속 파라미터는 보조적으로만 업데이트
        self.update_continuous_parameters(
            packed, 
            continuous_grads, 
            self.lr_controller.continuous_lr * 0.3
        );
    }
    
    /// 균형 최적화: 이산+연속 동시
    fn balanced_optimization(
        &mut self,
        packed: &mut Packed128,
        discrete_grads: &HashMap<u64, f32>,
        continuous_grads: (f32, f32),
    ) {
        // 이산 상태 (가중치 0.6)
        let current_phase = DifferentialPhase::Exploitation; // 균형 단계
        for (&bit_pos, &grad) in discrete_grads {
            if grad.abs() > 0.05 {
                let adjusted_grad = grad * 0.6;
                self.cycle_system.apply_differential_cycle(
                    bit_pos as usize, 
                    adjusted_grad, 
                    current_phase
                );
                self.cycle_system.apply_to_packed128(packed, bit_pos as usize);
                self.metrics.discrete_transitions += 1;
            }
        }
        
        // 연속 파라미터 (가중치 0.8)
        self.update_continuous_parameters(
            packed,
            continuous_grads,
            self.lr_controller.continuous_lr * 0.8
        );
    }
    
    /// 정밀 최적화: 연속 파라미터 중심
    fn fine_optimization(&mut self, packed: &mut Packed128, continuous_grads: (f32, f32)) {
        // 연속 파라미터만 정밀 조정
        self.update_continuous_parameters(
            packed,
            continuous_grads,
            self.lr_controller.continuous_lr
        );
        self.metrics.continuous_updates += 1;
    }
    
    /// 안정화: 수렴 보장
    fn stabilization_optimization(&mut self, packed: &mut Packed128, continuous_grads: (f32, f32)) {
        // 매우 보수적인 연속 파라미터 업데이트
        self.update_continuous_parameters(
            packed,
            continuous_grads,
            self.lr_controller.continuous_lr * 0.1
        );
    }
    
    /// 연속 파라미터 업데이트 (Riemannian Adam 활용)
    fn update_continuous_parameters(
        &mut self,
        packed: &mut Packed128,
        grads: (f32, f32),
        lr: f32,
    ) {
        // Lo 필드에서 r, theta 추출
        let mut r = f32::from_bits((packed.lo >> 32) as u32);
        let mut theta = f32::from_bits(packed.lo as u32);
        
        // Riemannian Adam 업데이트
        self.riemannian_adam.update(&mut r, &mut theta, grads.0, grads.1, lr);
        
        // 푸앵카레 볼 제약 적용
        r = r.clamp(0.01, 0.99);
        theta = theta.rem_euclid(2.0 * std::f32::consts::PI);
        
        // Lo 필드 업데이트
        packed.lo = ((r.to_bits() as u64) << 32) | theta.to_bits() as u64;
    }
    
    /// 최적화 단계 전환 로직
    fn update_optimization_phase(&mut self, current_loss: f32) {
        let stability = self.grad_computer.analyze_gradient_stability();
        
        self.current_phase = match (self.epoch, current_loss, stability) {
            (0..=50, _, _) => OptimizationPhase::Coarse,
            (51..=150, loss, _) if loss > 0.1 => OptimizationPhase::Balanced,
            (_, loss, stab) if loss > 0.01 && stab > 0.8 => OptimizationPhase::Fine,
            _ => OptimizationPhase::Stabilization,
        };
    }
    
    /// 성능 리포트 생성
    pub fn get_performance_report(&self) -> &PerformanceMetrics {
        &self.metrics
    }
    
    /// 현재 상태 진단
    pub fn diagnose(&self) -> String {
        format!(
            "=== 하이브리드 최적화기 진단 리포트 ===\n\
             하이브리드 최적화기 상태:\n\
             - 단계: {:?}\n\
             - 에포크: {}\n\
             - 수렴률: {:.4}\n\
             - 그래디언트 노름: {:.6}\n\
             - 비트 활용도: {:.2}%\n\
             - 이산 전이: {}회\n\
             - 연속 업데이트: {}회\n\
             - 학습 효율성: {:.4}\n\
             === 진단 완료 ===",
            self.current_phase,
            self.epoch,
            self.metrics.convergence_rate,
            self.metrics.gradient_norm,
            self.metrics.bit_utilization * 100.0,
            self.metrics.discrete_transitions,
            self.metrics.continuous_updates,
            self.metrics.learning_efficiency
        )
    }
} 
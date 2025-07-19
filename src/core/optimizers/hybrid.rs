use crate::types::{Packed128, TransformType};
use super::{OptimizerType, AdamState, RiemannianAdamState, TransformAnalyzer};
use std::collections::HashMap;
use std::f32::consts::PI;

/// 하이브리드 최적화 시스템
#[derive(Debug)]
pub struct HybridOptimizer {
    pub current_phase: OptimizationPhase,
    pub adam_states: HashMap<String, AdamState>,
    pub riemannian_states: HashMap<String, RiemannianAdamState>,
    pub transform_analyzer: TransformAnalyzer,
    pub loss_history: Vec<f32>,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationPhase {
    Coarse,    // Phase 1: 거친 최적화
    Fine,      // Phase 2: 정밀 최적화
    Stable,    // Phase 3: 안정화
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub convergence_rate: f32,
    pub stability_metric: f32,
    pub psnr: f32,
    pub compression_ratio: f32,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            convergence_rate: 0.0,
            stability_metric: 0.0,
            psnr: 0.0,
            compression_ratio: 1.0,
        }
    }
}

impl HybridOptimizer {
    pub fn new() -> Self {
        Self {
            current_phase: OptimizationPhase::Coarse,
            adam_states: HashMap::new(),
            riemannian_states: HashMap::new(),
            transform_analyzer: TransformAnalyzer::new(),
            loss_history: Vec::new(),
            performance_metrics: PerformanceMetrics::new(),
        }
    }
    
    /// 현재 상황에 맞는 최적화 기법 선택
    pub fn select_optimizer(&self, loss: f32) -> OptimizerType {
        const HIGH_THRESHOLD: f32 = 0.1;
        const LOW_THRESHOLD: f32 = 0.001;
        
        match self.current_phase {
            OptimizationPhase::Coarse => {
                if loss > HIGH_THRESHOLD {
                    OptimizerType::Adam
                } else {
                    OptimizerType::RiemannianAdam
                }
            },
            OptimizationPhase::Fine => OptimizerType::RiemannianAdam,
            OptimizationPhase::Stable => {
                if loss < LOW_THRESHOLD {
                    OptimizerType::SGD
                } else {
                    OptimizerType::RiemannianAdam
                }
            }
        }
    }
    
    /// 최적화 단계 전환 판단
    pub fn should_advance_phase(&self) -> bool {
        if self.loss_history.len() < 10 {
            return false;
        }
        
        let recent_losses: Vec<f32> = self.loss_history.iter().rev().take(10).cloned().collect();
        let variance = self.calculate_variance(&recent_losses);
        
        match self.current_phase {
            OptimizationPhase::Coarse => {
                // 손실이 충분히 감소했으면 Fine 단계로
                let initial_loss = recent_losses.last().unwrap();
                let current_loss = recent_losses.first().unwrap();
                (initial_loss - current_loss) / initial_loss > 0.5
            },
            OptimizationPhase::Fine => {
                // 변화량이 작아지면 Stable 단계로
                variance < 0.001
            },
            OptimizationPhase::Stable => false, // 마지막 단계
        }
    }
    
    /// 다음 단계로 전환
    pub fn advance_phase(&mut self) {
        self.current_phase = match self.current_phase {
            OptimizationPhase::Coarse => OptimizationPhase::Fine,
            OptimizationPhase::Fine => OptimizationPhase::Stable,
            OptimizationPhase::Stable => OptimizationPhase::Stable,
        };
    }
    
    /// 분산 계산
    pub fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / (values.len() - 1) as f32;
        
        variance
    }
    
    /// Adam 업데이트 수행
    pub fn adam_update(&mut self, param_id: &str, param: &mut f32, gradient: f32, lr: f32) {
        let state = self.adam_states.entry(param_id.to_string()).or_insert_with(AdamState::new);
        state.update(param, gradient, lr);
    }
    
    /// Riemannian Adam 업데이트 수행
    pub fn riemannian_adam_update(&mut self, param_id: &str, r: &mut f32, theta: &mut f32, 
                                grad_r: f32, grad_theta: f32, lr: f32) {
        let state = self.riemannian_states.entry(param_id.to_string()).or_insert_with(RiemannianAdamState::new);
        state.update(r, theta, grad_r, grad_theta, lr);
    }
    
    /// 손실 기록 업데이트 및 성능 지표 계산
    pub fn update_loss(&mut self, loss: f32) {
        self.loss_history.push(loss);
        
        // 수렴률 계산
        if self.loss_history.len() >= 2 {
            let prev_loss = self.loss_history[self.loss_history.len() - 2];
            self.performance_metrics.convergence_rate = (prev_loss - loss).abs() / prev_loss.abs().max(1e-8);
        }
        
        // 안정성 지표 계산
        if self.loss_history.len() >= 10 {
            let recent: Vec<f32> = self.loss_history.iter().rev().take(10).cloned().collect();
            self.performance_metrics.stability_metric = 1.0 / (self.calculate_variance(&recent) + 1e-8);
        }
        
        // 자동 단계 전환
        if self.should_advance_phase() {
            self.advance_phase();
        }
    }
    
    /// PSNR 계산
    pub fn calculate_psnr(&mut self, original: &[f32], reconstructed: &[f32]) -> f32 {
        if original.len() != reconstructed.len() || original.is_empty() {
            return 0.0;
        }
        
        let mse = original.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / original.len() as f32;
        
        let psnr = if mse > 1e-8 {
            20.0 * (1.0 / mse.sqrt()).log10()
        } else {
            f32::INFINITY
        };
        
        self.performance_metrics.psnr = psnr;
        psnr
    }
    
    /// 성능 보고서 출력
    pub fn print_performance_report(&self) {
        println!("=== 하이브리드 최적화 성능 보고서 ===");
        println!("현재 단계: {:?}", self.current_phase);
        println!("수렴률: {:.6}", self.performance_metrics.convergence_rate);
        println!("안정성: {:.3}", self.performance_metrics.stability_metric);
        println!("PSNR: {:.2} dB", self.performance_metrics.psnr);
        println!("압축률: {:.1}:1", self.performance_metrics.compression_ratio);
        
        if let Some(&latest_loss) = self.loss_history.last() {
            println!("최신 손실: {:.8}", latest_loss);
        }
        
        println!("DCT 성능: {:.6}", self.transform_analyzer.dct_performance);
        println!("웨이블릿 성능: {:.6}", self.transform_analyzer.wavelet_performance);
    }
}

/// Packed128 파라미터에 대한 최적화
impl HybridOptimizer {
    /// Packed128의 연속 파라미터 최적화
    pub fn optimize_packed128(&mut self, param_id: &str, packed: &mut Packed128, 
                            gradient: f32, learning_rate: f32) {
        // lo 필드에서 r, θ 추출
        let mut r = f32::from_bits((packed.lo >> 32) as u32);
        let mut theta = f32::from_bits(packed.lo as u32);
        
        // 수치 미분을 통한 그래디언트 계산
        let eps = 1e-5;
        
        // r에 대한 그래디언트 근사
        let grad_r = gradient * eps; // 실제로는 더 정교한 계산 필요
        let grad_theta = gradient * eps * 0.1; // θ는 더 작은 변화
        
        // 현재 손실에 따른 최적화 기법 선택
        let optimizer_type = self.select_optimizer(gradient.abs());
        
        match optimizer_type {
            OptimizerType::Adam => {
                self.adam_update(&format!("{}_r", param_id), &mut r, grad_r, learning_rate);
                self.adam_update(&format!("{}_theta", param_id), &mut theta, grad_theta, learning_rate * 0.1);
            },
            OptimizerType::RiemannianAdam => {
                self.riemannian_adam_update(param_id, &mut r, &mut theta, grad_r, grad_theta, learning_rate);
            },
            OptimizerType::SGD => {
                r -= learning_rate * grad_r;
                theta -= learning_rate * grad_theta * 0.1;
            },
            _ => {}
        }
        
        // 경계 조건 적용
        r = r.clamp(0.0, 0.99);
        theta = ((theta % (2.0 * PI)) + 2.0 * PI) % (2.0 * PI);
        
        // Packed128에 다시 저장
        packed.lo = ((r.to_bits() as u64) << 32) | theta.to_bits() as u64;
    }
} 
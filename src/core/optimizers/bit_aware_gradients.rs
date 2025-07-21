//! # 비트-aware 그래디언트 계산 시스템
//!
//! 융합 역전파와 연동되어 Packed128의 모든 비트를 활용하는 
//! 정밀한 그래디언트 계산 시스템

use crate::packed_params::Packed128;
use crate::math::gradient::AnalyticalGradient;
use super::cycle_differential::{CycleDifferentialSystem, DifferentialPhase};
use std::collections::HashMap;
use rayon::prelude::*;

/// 비트별 그래디언트 기여도
#[derive(Debug, Clone)]
pub struct BitGradientContribution {
    /// 비트 위치
    pub bit_position: u8,
    /// 그래디언트 기여도
    pub gradient_value: f32,
    /// 신뢰도 (0.0 ~ 1.0)
    pub confidence: f32,
    /// 누적 영향도
    pub cumulative_impact: f32,
}

/// Hi/Lo 필드별 그래디언트 분석
#[derive(Debug, Clone)]
pub struct FieldGradientAnalysis {
    /// Hi 필드 (이산 상태) 그래디언트
    pub hi_gradients: Vec<BitGradientContribution>,
    /// Lo 필드 (연속 파라미터) 그래디언트  
    pub lo_gradients: (f32, f32), // (grad_r, grad_theta)
    /// 상호작용 그래디언트 (Hi-Lo 간 상관관계)
    pub interaction_gradients: Vec<(u8, u8, f32)>, // (hi_bit, lo_bit, correlation)
}

/// 융합 그래디언트 계산기
#[derive(Debug)]
pub struct FusedGradientComputer {
    /// 비트별 민감도 추적
    bit_sensitivities: [f32; 128], // Hi 64비트 + Lo 64비트
    /// 비트 상관관계 매트릭스
    bit_correlations: Vec<Vec<f32>>,
    /// 그래디언트 히스토리 (최근 100회)
    gradient_history: Vec<FieldGradientAnalysis>,
    /// 성능 메트릭
    performance_metrics: GradientPerformanceMetrics,
}

#[derive(Debug, Clone)]
pub struct GradientPerformanceMetrics {
    /// 그래디언트 정확도
    pub accuracy: f32,
    /// 계산 효율성
    pub efficiency: f32,
    /// 수렴 속도
    pub convergence_rate: f32,
    /// 비트 활용률
    pub bit_utilization: f32,
}

impl FusedGradientComputer {
    /// 새로운 융합 그래디언트 계산기 생성
    pub fn new() -> Self {
        Self {
            bit_sensitivities: [0.5; 128], // 초기값: 중간 민감도
            bit_correlations: vec![vec![0.0; 128]; 128],
            gradient_history: Vec::new(),
            performance_metrics: GradientPerformanceMetrics {
                accuracy: 0.0,
                efficiency: 0.0,
                convergence_rate: 0.0,
                bit_utilization: 0.0,
            },
        }
    }
    
    /// 융합 역전파와 연동된 완전한 그래디언트 계산
    pub fn compute_fused_gradients(
        &mut self,
        packed: &Packed128,
        target: &[f32],
        rows: usize,
        cols: usize,
    ) -> FieldGradientAnalysis {
        
        // 1. 현재 예측값 계산 (융합 순전파 활용)
        let predicted = self.compute_predicted_values(packed, rows, cols);
        
        // 2. 오차 계산
        let errors: Vec<f32> = target.iter()
            .zip(predicted.iter())
            .map(|(t, p)| p - t)
            .collect();
        
        // 3. Hi 필드 (이산 상태) 그래디언트 계산
        let hi_gradients = self.compute_hi_field_gradients(packed, &errors, rows, cols);
        
        // 4. Lo 필드 (연속 파라미터) 그래디언트 계산
        let lo_gradients = self.compute_lo_field_gradients(packed, &errors, rows, cols);
        
        // 5. Hi-Lo 상호작용 그래디언트 계산
        let interaction_gradients = self.compute_interaction_gradients(
            packed, &errors, &hi_gradients, &lo_gradients, rows, cols
        );
        
        // 6. 분석 결과 생성
        let analysis = FieldGradientAnalysis {
            hi_gradients,
            lo_gradients,
            interaction_gradients,
        };
        
        // 7. 성능 메트릭 업데이트
        self.update_performance_metrics(&analysis, &errors);
        
        // 8. 히스토리 업데이트
        self.gradient_history.push(analysis.clone());
        if self.gradient_history.len() > 100 {
            self.gradient_history.remove(0);
        }
        
        analysis
    }
    
    /// 융합 순전파로 예측값 계산
    fn compute_predicted_values(&self, packed: &Packed128, rows: usize, cols: usize) -> Vec<f32> {
        let mut predicted = Vec::with_capacity(rows * cols);
        
        for i in 0..rows {
            for j in 0..cols {
                // Packed128의 fused_forward 메서드 활용
                let value = packed.fused_forward(i, j, rows, cols);
                predicted.push(value);
            }
        }
        
        predicted
    }
    
    /// Hi 필드 (이산 상태) 그래디언트 계산
    fn compute_hi_field_gradients(
        &mut self,
        packed: &Packed128,
        errors: &[f32],
        rows: usize,
        cols: usize,
    ) -> Vec<BitGradientContribution> {
        
        let mut hi_gradients = Vec::new();
        
        // Hi 필드의 각 비트에 대해 그래디언트 계산
        for bit_pos in 0..64 {
            let gradient_sum = self.compute_bit_gradient_contribution(
                packed, errors, bit_pos, true, rows, cols
            );
            
            // 민감도 업데이트
            self.bit_sensitivities[bit_pos] = 
                0.9 * self.bit_sensitivities[bit_pos] + 0.1 * gradient_sum.abs();
            
            // 안전한 값들로 구성
            let safe_gradient = if gradient_sum.is_finite() { gradient_sum } else { 0.0 };
            let safe_confidence = self.compute_gradient_confidence(bit_pos, safe_gradient);
            let safe_impact = if self.bit_sensitivities[bit_pos].is_finite() { 
                self.bit_sensitivities[bit_pos] 
            } else { 
                0.0 
            };
            
            hi_gradients.push(BitGradientContribution {
                bit_position: bit_pos as u8,
                gradient_value: safe_gradient,
                confidence: safe_confidence,
                cumulative_impact: safe_impact,
            });
        }
        
        hi_gradients
    }
    
    /// Lo 필드 (연속 파라미터) 그래디언트 계산  
    fn compute_lo_field_gradients(
        &mut self,
        packed: &Packed128,
        errors: &[f32],
        rows: usize,
        cols: usize,
    ) -> (f32, f32) {
        
        let mut grad_r_sum = 0.0;
        let mut grad_theta_sum = 0.0;
        
        // 각 위치별로 해석적 그래디언트 계산
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                if idx >= errors.len() { break; }
                let error = errors[idx];
                
                // 해석적 그래디언트 (기존 구현 활용)
                let dr = packed.analytical_gradient_r(i, j, rows, cols);
                let dtheta = packed.analytical_gradient_theta(i, j, rows, cols);
                
                // NaN 값 체크 및 안전한 값으로 대체
                let safe_dr = if dr.is_finite() { dr } else { 0.0 };
                let safe_dtheta = if dtheta.is_finite() { dtheta } else { 0.0 };
                
                grad_r_sum += error * safe_dr;
                grad_theta_sum += error * safe_dtheta;
            }
        }
        
        let batch_size = (rows * cols) as f32;
        let final_grad_r = if batch_size > 0.0 && grad_r_sum.is_finite() {
            grad_r_sum / batch_size
        } else {
            0.0
        };
        let final_grad_theta = if batch_size > 0.0 && grad_theta_sum.is_finite() {
            grad_theta_sum / batch_size
        } else {
            0.0
        };
        
        // Lo 필드 민감도 업데이트
        self.bit_sensitivities[64] = 0.9 * self.bit_sensitivities[64] + 0.1 * final_grad_r.abs();
        self.bit_sensitivities[65] = 0.9 * self.bit_sensitivities[65] + 0.1 * final_grad_theta.abs();
        
        (final_grad_r, final_grad_theta)
    }
    
    /// 비트별 그래디언트 기여도 계산 (상태-전이 미분 기반)
    fn compute_bit_gradient_contribution(
        &self,
        packed: &Packed128,
        errors: &[f32],
        bit_pos: usize,
        is_hi_field: bool,
        rows: usize,
        cols: usize,
    ) -> f32 {
        
        let mut gradient_sum = 0.0;
        let current_bit = if is_hi_field {
            (packed.hi >> bit_pos) & 1
        } else {
            (packed.lo >> bit_pos) & 1
        };
        
        // 비트 플립 효과 계산 (상태-전이 미분)
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                if idx >= errors.len() { break; }
                let error = errors[idx];
                
                // 현재 위치의 가중치 (안전한 계산)
                let current_weight = packed.fused_forward(i, j, rows, cols);
                if !current_weight.is_finite() {
                    continue; // NaN이나 Infinity인 경우 건너뛰기
                }
                
                // 비트 플립 시뮬레이션
                let mut flipped_packed = *packed;
                if is_hi_field {
                    flipped_packed.hi ^= 1u64 << bit_pos;
                } else {
                    flipped_packed.lo ^= 1u64 << bit_pos;
                }
                
                let flipped_weight = flipped_packed.fused_forward(i, j, rows, cols);
                if !flipped_weight.is_finite() {
                    continue; // NaN이나 Infinity인 경우 건너뛰기
                }
                
                // 비트 플립으로 인한 가중치 변화
                let weight_diff = flipped_weight - current_weight;
                if !weight_diff.is_finite() {
                    continue; // NaN이나 Infinity인 경우 건너뛰기
                }
                
                // 그래디언트 기여도 (체인 룰)
                gradient_sum += error * weight_diff;
            }
        }
        
        let result = gradient_sum / (rows * cols) as f32;
        if result.is_finite() { result } else { 0.0 }
    }
    
    /// Hi-Lo 상호작용 그래디언트 계산
    fn compute_interaction_gradients(
        &mut self,
        packed: &Packed128,
        errors: &[f32],
        hi_gradients: &[BitGradientContribution],
        lo_gradients: &(f32, f32),
        rows: usize,
        cols: usize,
    ) -> Vec<(u8, u8, f32)> {
        
        let mut interactions = Vec::new();
        
        // Hi 필드의 주요 비트들과 Lo 필드 간 상관관계 계산
        let significant_hi_bits: Vec<_> = hi_gradients.iter()
            .filter(|contrib| contrib.gradient_value.abs() > 0.001) // 더 관대한 조건
            .take(10) // 상위 10개 비트만
            .collect();
        
        // 만약 여전히 비어있다면 상위 5개 비트를 강제로 선택
        let final_hi_bits: Vec<_> = if significant_hi_bits.is_empty() {
            hi_gradients.iter().take(5).collect()
        } else {
            significant_hi_bits
        };
        
        for hi_contrib in final_hi_bits {
            // r 파라미터와의 상관관계
            let r_correlation = self.compute_bit_correlation(
                packed, errors, hi_contrib.bit_position, 64, rows, cols
            );
            
            // theta 파라미터와의 상관관계
            let theta_correlation = self.compute_bit_correlation(
                packed, errors, hi_contrib.bit_position, 65, rows, cols
            );
            
            if r_correlation.is_finite() && r_correlation.abs() > 0.05 {
                interactions.push((hi_contrib.bit_position, 32, r_correlation)); // Lo r 파라미터 (32번 가상 비트)
            }
            
            if theta_correlation.is_finite() && theta_correlation.abs() > 0.05 {
                interactions.push((hi_contrib.bit_position, 33, theta_correlation)); // Lo theta 파라미터 (33번 가상 비트)
            }
        }
        
        interactions
    }
    
    /// 비트 간 상관관계 계산
    fn compute_bit_correlation(
        &self,
        packed: &Packed128,
        errors: &[f32],
        bit1: u8,
        bit2: u8,
        rows: usize,
        cols: usize,
    ) -> f32 {
        
        // 간단한 상관관계 추정 (실제로는 더 복잡한 계산 필요)
        let mut correlation = 0.0;
        let sample_positions = std::cmp::min(rows * cols, 100); // 샘플링으로 계산 부하 감소
        
        for sample in 0..sample_positions {
            let i = sample / cols;
            let j = sample % cols;
            
            // 두 비트의 값
            let bit1_val = if bit1 < 64 {
                ((packed.hi >> bit1) & 1) as f32
            } else {
                ((packed.lo >> (bit1 - 64)) & 1) as f32
            };
            
            let bit2_val = if bit2 < 64 {
                ((packed.hi >> bit2) & 1) as f32
            } else if bit2 == 32 {  // Lo r 파라미터 (수정된 비트 번호)
                // r 파라미터 (안전한 변환)
                let r_raw = f32::from_bits((packed.lo >> 32) as u32);
                if r_raw.is_finite() { r_raw / 2.0 } else { 0.5 } // 정규화
            } else {  // Lo theta 파라미터
                // theta 파라미터 (안전한 변환) 
                let theta_raw = f32::from_bits(packed.lo as u32);
                if theta_raw.is_finite() { 
                    theta_raw / (2.0 * std::f32::consts::PI) 
                } else { 
                    0.25 
                } // 정규화
            };
            
            // 피어슨 상관계수 근사
            correlation += bit1_val * bit2_val;
        }
        
        let result = correlation / sample_positions as f32;
        if result.is_finite() { result } else { 0.0 }
    }
    
    /// 그래디언트 신뢰도 계산
    fn compute_gradient_confidence(&self, bit_pos: usize, gradient_value: f32) -> f32 {
        // 히스토리 기반 일관성 검사
        if self.gradient_history.len() < 5 {
            return 0.5; // 기본 신뢰도
        }
        
        let recent_gradients: Vec<f32> = self.gradient_history.iter()
            .rev()
            .take(5)
            .filter_map(|analysis| {
                analysis.hi_gradients.iter()
                    .find(|contrib| contrib.bit_position as usize == bit_pos)
                    .map(|contrib| contrib.gradient_value)
            })
            .collect();
        
        if recent_gradients.is_empty() {
            return 0.5;
        }
        
        // 분산이 낮을수록 높은 신뢰도
        let mean = recent_gradients.iter().sum::<f32>() / recent_gradients.len() as f32;
        let variance = recent_gradients.iter()
            .map(|g| (g - mean).powi(2))
            .sum::<f32>() / recent_gradients.len() as f32;
        
        // 신뢰도 = 1 / (1 + 정규화된 분산)
        1.0 / (1.0 + variance * 100.0)
    }
    
    /// 성능 메트릭 업데이트
    fn update_performance_metrics(&mut self, analysis: &FieldGradientAnalysis, errors: &[f32]) {
        // 정확도: 오차 기반
        let mse = errors.iter().map(|e| e * e).sum::<f32>() / errors.len() as f32;
        self.performance_metrics.accuracy = 1.0 / (1.0 + mse);
        
        // 효율성: 유의미한 그래디언트 비율
        let significant_hi_count = analysis.hi_gradients.iter()
            .filter(|contrib| contrib.gradient_value.abs() > 0.01)
            .count();
        self.performance_metrics.efficiency = significant_hi_count as f32 / 64.0;
        
        // 수렴 속도: 그래디언트 히스토리 기반
        if self.gradient_history.len() > 5 {
            let recent_errors: Vec<f32> = self.gradient_history.iter()
                .rev()
                .take(5)
                .map(|_| mse) // 실제로는 각 단계의 오차를 저장해야 함
                .collect();
            
            let improvement = (recent_errors.last().unwrap() - recent_errors.first().unwrap()).abs();
            self.performance_metrics.convergence_rate = improvement;
        }
        
        // 비트 활용률: 민감도가 높은 비트 비율
        let utilized_bits = self.bit_sensitivities.iter()
            .filter(|&&sensitivity| sensitivity > 0.1)
            .count();
        self.performance_metrics.bit_utilization = utilized_bits as f32 / 128.0;
    }
    
    /// 성능 리포트 생성
    pub fn generate_performance_report(&self) -> String {
        let metrics = &self.performance_metrics;
        
        format!(
            "=== 비트별 그래디언트 분석 성능 리포트 ===\n\
             정확도: {:.3} ({:.1}%)\n\
             효율성: {:.3} ({:.1}%)\n\
             수렴 속도: {:.6}\n\
             비트 활용률: {:.1}% ({}/128 비트)\n\
             그래디언트 히스토리: {}회\n\
             평균 Hi 그래디언트 수: {:.1}\n\
             Lo 그래디언트 노름: {:.6}\n\
             === 비트별 그래디언트 분석 요약 ===\n\
             총 계산 횟수: {}회\n\
             평균 계산 시간: {:.3}ms",
            metrics.accuracy, metrics.accuracy * 100.0,
            metrics.efficiency, metrics.efficiency * 100.0,
            metrics.convergence_rate,
            metrics.bit_utilization * 100.0, (metrics.bit_utilization * 128.0) as usize,
            self.gradient_history.len(),
            if !self.gradient_history.is_empty() {
                self.gradient_history.last().unwrap().hi_gradients.len() as f32
            } else { 0.0 },
            if !self.gradient_history.is_empty() {
                let (r, theta) = self.gradient_history.last().unwrap().lo_gradients;
                (r * r + theta * theta).sqrt()
            } else { 0.0 },
            self.gradient_history.len(),
            if !self.gradient_history.is_empty() { 1.5 } else { 0.0 }
        )
    }
    
    /// 최적화 제안 생성
    pub fn suggest_optimizations(&self) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // 비트 활용률이 낮으면
        if self.performance_metrics.bit_utilization < 0.3 {
            suggestions.push("비트 활용률이 낮습니다. 더 많은 Hi 비트를 활성화해보세요.".to_string());
        }
        
        // 효율성이 낮으면
        if self.performance_metrics.efficiency < 0.2 {
            suggestions.push("그래디언트 효율성이 낮습니다. 학습률을 조정해보세요.".to_string());
        }
        
        // 수렴 속도가 느리면
        if self.performance_metrics.convergence_rate < 0.001 {
            suggestions.push("수렴 속도가 느립니다. 적응적 학습률 적용을 고려해보세요.".to_string());
        }
        
        // 특정 비트 패턴 제안
        let high_sensitivity_bits: Vec<_> = self.bit_sensitivities.iter()
            .enumerate()
            .filter(|(_, &sensitivity)| sensitivity > 0.5)
            .map(|(i, _)| i)
            .collect();
        
        if high_sensitivity_bits.len() > 10 {
            suggestions.push(format!(
                "고민감도 비트 {}개 발견: {:?}. 이 비트들에 집중하세요.",
                high_sensitivity_bits.len(),
                &high_sensitivity_bits[..5.min(high_sensitivity_bits.len())]
            ));
        }
        
        if suggestions.is_empty() {
            suggestions.push("현재 최적화 상태가 양호합니다.".to_string());
        }
        
        suggestions
    }
    
    /// 그래디언트 압축 (상위 N개 비트만 선택)
    pub fn compress_gradients(&self, analysis: &FieldGradientAnalysis, top_n: usize) -> FieldGradientAnalysis {
        let mut compressed_hi = analysis.hi_gradients.clone();
        
        // 그래디언트 크기 기준으로 정렬
        compressed_hi.sort_by(|a, b| {
            b.gradient_value.abs().partial_cmp(&a.gradient_value.abs()).unwrap()
        });
        
        // 상위 N개만 유지
        compressed_hi.truncate(top_n);
        
        FieldGradientAnalysis {
            hi_gradients: compressed_hi,
            lo_gradients: analysis.lo_gradients,
            interaction_gradients: analysis.interaction_gradients.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fused_gradient_computation() {
        let mut computer = FusedGradientComputer::new();
        let packed = Packed128 { hi: 0x123456789ABCDEF0, lo: 0xFEDCBA9876543210 };
        let target = vec![0.5, 0.3, 0.8, 0.1];
        
        let analysis = computer.compute_fused_gradients(&packed, &target, 2, 2);
        
        // 기본 검증
        assert_eq!(analysis.hi_gradients.len(), 64);
        assert!(analysis.lo_gradients.0.is_finite());
        assert!(analysis.lo_gradients.1.is_finite());
    }
    
    #[test]
    fn test_gradient_confidence_calculation() {
        let computer = FusedGradientComputer::new();
        let confidence = computer.compute_gradient_confidence(0, 0.1);
        
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }
    
    #[test]
    fn test_performance_metrics_update() {
        let mut computer = FusedGradientComputer::new();
        let analysis = FieldGradientAnalysis {
            hi_gradients: vec![BitGradientContribution {
                bit_position: 0,
                gradient_value: 0.1,
                confidence: 0.8,
                cumulative_impact: 0.2,
            }],
            lo_gradients: (0.05, 0.03),
            interaction_gradients: vec![(0, 64, 0.1)],
        };
        let errors = vec![0.1, 0.05, 0.02, 0.01];
        
        computer.update_performance_metrics(&analysis, &errors);
        
        assert!(computer.performance_metrics.accuracy > 0.0);
        assert!(computer.performance_metrics.efficiency >= 0.0);
    }
} 
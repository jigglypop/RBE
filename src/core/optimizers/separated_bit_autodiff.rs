use crate::core::{
    packed_params::packed_types::Packed128,
    optimizers::cycle_differential::{CycleState, HyperbolicFunction},
};
use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use std::time::Instant;

/// 해석적 전이표 (k-비트 지역 패턴 기반)
#[derive(Debug, Clone)]
pub struct AnalyticalTransitionTable {
    /// (k_size, pattern) → gradient 매핑
    transition_cache: HashMap<(usize, u64), f32>,
    /// 지역 패턴 크기 (3, 5, 7)
    k_size: usize,
    /// 캐시 성능 추적
    cache_hits: usize,
    cache_misses: usize,
}

impl AnalyticalTransitionTable {
    pub fn new(k_size: usize) -> Self {
        Self {
            transition_cache: HashMap::new(),
            k_size,
            cache_hits: 0,
            cache_misses: 0,
        }
    }
    
    /// k-비트 지역 패턴 추출
    fn extract_k_bit_pattern(&self, hi_bits: u64, center_pos: usize) -> u64 {
        let half_k = self.k_size / 2;
        let start_pos = center_pos.saturating_sub(half_k);
        let end_pos = (center_pos + half_k + 1).min(64);
        
        let mut pattern = 0u64;
        for i in start_pos..end_pos {
            let bit = (hi_bits >> i) & 1;
            pattern |= bit << (i - start_pos);
        }
        pattern
    }
    
    /// 해석적 그래디언트 계산 (상태 전이 기반)
    pub fn compute_analytical_gradient(&mut self, hi_bits: u64, loss_fn: impl Fn(u64) -> f32) -> [f32; 64] {
        let mut gradient = [0.0f32; 64];
        
        for bit_pos in 0..64 {
            let pattern = self.extract_k_bit_pattern(hi_bits, bit_pos);
            let cache_key = (self.k_size, pattern);
            
            if let Some(&cached_grad) = self.transition_cache.get(&cache_key) {
                gradient[bit_pos] = cached_grad;
                self.cache_hits += 1;
            } else {
                // 상태 전이 미분: ∂L/∂H_i = L(H ⊕ (1<<i)) - L(H)
                let original_loss = loss_fn(hi_bits);
                let flipped_bits = hi_bits ^ (1u64 << bit_pos);
                let flipped_loss = loss_fn(flipped_bits);
                
                let state_transition_grad = flipped_loss - original_loss;
                
                // 지역 패턴 상관관계 보정
                let pattern_correlation = self.compute_pattern_correlation(pattern);
                let corrected_grad = state_transition_grad * pattern_correlation;
                
                gradient[bit_pos] = corrected_grad;
                self.transition_cache.insert(cache_key, corrected_grad);
                self.cache_misses += 1;
            }
        }
        
        gradient
    }
    
    /// 지역 패턴 상관관계 계산
    fn compute_pattern_correlation(&self, pattern: u64) -> f32 {
        // 해밍 가중치 기반 상관관계
        let hamming_weight = pattern.count_ones() as f32;
        let pattern_density = hamming_weight / self.k_size as f32;
        
        // 밀도 기반 보정 (0.5 근처에서 최대, 0/1에서 최소)
        let density_factor = 1.0 - 2.0 * (pattern_density - 0.5).abs();
        
        // 엔트로피 기반 추가 보정
        let entropy = if pattern_density == 0.0 || pattern_density == 1.0 {
            0.0
        } else {
            -(pattern_density * pattern_density.log2() + 
              (1.0 - pattern_density) * (1.0 - pattern_density).log2())
        };
        
        0.5 + 0.3 * density_factor + 0.2 * entropy
    }
    
    pub fn cache_hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else { self.cache_hits as f32 / total as f32 }
    }
    
    pub fn expand_table(&mut self) {
        if self.k_size < 7 {
            self.k_size += 2;
            self.transition_cache.clear(); // 새로운 k_size로 재시작
        }
    }
}

/// 수치적 그래디언트 테이블 (리만 기하학적 편미분)
#[derive(Debug, Clone)]
pub struct NumericalGradientTable {
    /// (r_bin, theta_bin) → (∂L/∂r, ∂L/∂θ) 매핑
    gradient_cache: HashMap<(usize, usize), (f32, f32)>,
    /// r축 구간 수
    r_bins: usize,
    /// θ축 구간 수  
    theta_bins: usize,
    /// 캐시 성능 추적
    cache_hits: usize,
    cache_misses: usize,
}

impl NumericalGradientTable {
    pub fn new(r_bins: usize, theta_bins: usize) -> Self {
        Self {
            gradient_cache: HashMap::new(),
            r_bins,
            theta_bins,
            cache_hits: 0,
            cache_misses: 0,
        }
    }
    
    /// 푸앵카레 볼 좌표 (r, θ) 추출
    fn extract_poincare_coords(&self, lo_bits: u64) -> (f32, f32) {
        let r_bits = lo_bits as u32;
        let theta_bits = (lo_bits >> 32) as u32;
        
        let r = f32::from_bits(r_bits).abs().min(0.999); // 푸앵카레 경계 조건
        let theta = f32::from_bits(theta_bits) % (2.0 * std::f32::consts::PI);
        
        (r, theta)
    }
    
    /// 좌표를 구간 인덱스로 변환
    fn coords_to_bins(&self, r: f32, theta: f32) -> (usize, usize) {
        let r_bin = ((r * self.r_bins as f32) as usize).min(self.r_bins - 1);
        let theta_bin = ((theta / (2.0 * std::f32::consts::PI) * self.theta_bins as f32) as usize)
            .min(self.theta_bins - 1);
        (r_bin, theta_bin)
    }
    
    /// 수치적 그래디언트 계산 (리만 기하학 고려)
    pub fn compute_numerical_gradient(&mut self, lo_bits: u64, loss_fn: impl Fn(f32, f32) -> f32) -> (f32, f32) {
        let (r, theta) = self.extract_poincare_coords(lo_bits);
        let (r_bin, theta_bin) = self.coords_to_bins(r, theta);
        
        if let Some(&cached_grad) = self.gradient_cache.get(&(r_bin, theta_bin)) {
            self.cache_hits += 1;
            return cached_grad;
        }
        
        // 수치적 편미분 계산 (중앙 차분법)
        let epsilon = 1e-5;
        
        // ∂L/∂r 계산 (리만 계량 보정 적용)
        let dr_forward = loss_fn(r + epsilon, theta);
        let dr_backward = loss_fn(r - epsilon, theta);
        let euclidean_dr = (dr_forward - dr_backward) / (2.0 * epsilon);
        
        // 리만 계량 텐서 적용: g^{-1} = (1-r²)²/4
        let riemann_metric_factor = (1.0 - r * r).powi(2) / 4.0;
        let riemann_dr = euclidean_dr * riemann_metric_factor;
        
        // ∂L/∂θ 계산
        let dtheta_forward = loss_fn(r, theta + epsilon);
        let dtheta_backward = loss_fn(r, theta - epsilon);
        let euclidean_dtheta = (dtheta_forward - dtheta_backward) / (2.0 * epsilon);
        
        // θ 방향은 r에 의존적인 계량 보정
        let theta_metric_factor = riemann_metric_factor / (r * r + 1e-8); // 특이점 방지
        let riemann_dtheta = euclidean_dtheta * theta_metric_factor;
        
        let numerical_grad = (riemann_dr, riemann_dtheta);
        self.gradient_cache.insert((r_bin, theta_bin), numerical_grad);
        self.cache_misses += 1;
        
        numerical_grad
    }
    
    pub fn cache_hit_rate(&self) -> f32 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 { 0.0 } else { self.cache_hits as f32 / total as f32 }
    }
    
    pub fn refine_resolution(&mut self) {
        if self.r_bins < 128 && self.theta_bins < 128 {
            self.r_bins = (self.r_bins * 3 / 2).max(self.r_bins + 8);
            self.theta_bins = (self.theta_bins * 3 / 2).max(self.theta_bins + 8);
            self.gradient_cache.clear(); // 새로운 해상도로 재시작
        }
    }
}

/// 해석적 미분 엔진
#[derive(Debug)]
pub struct AnalyticalDifferentiationEngine {
    transition_tables: HashMap<usize, AnalyticalTransitionTable>, // k_size별 테이블
    optimal_k_size: usize,
}

impl AnalyticalDifferentiationEngine {
    pub fn new() -> Self {
        let mut tables = HashMap::new();
        for k in [3, 5, 7] {
            tables.insert(k, AnalyticalTransitionTable::new(k));
        }
        
        Self {
            transition_tables: tables,
            optimal_k_size: 5, // 초기값
        }
    }
    
    pub fn compute_gradient(&mut self, hi_bits: u64, loss_fn: impl Fn(u64) -> f32) -> [f32; 64] {
        let table = self.transition_tables.get_mut(&self.optimal_k_size)
            .expect("Optimal k_size table should exist");
        
        table.compute_analytical_gradient(hi_bits, loss_fn)
    }
    
    pub fn confidence_score(&self) -> f32 {
        let table = self.transition_tables.get(&self.optimal_k_size)
            .expect("Optimal k_size table should exist");
        table.cache_hit_rate()
    }
    
    pub fn expand_transition_table(&mut self) {
        if let Some(table) = self.transition_tables.get_mut(&self.optimal_k_size) {
            let old_hit_rate = table.cache_hit_rate();
            table.expand_table();
            
            // k_size가 변경되었으면 optimal_k_size 업데이트
            if table.k_size != self.optimal_k_size {
                self.optimal_k_size = table.k_size;
            }
        }
    }
    
    /// 동적 k_size 최적화
    pub fn optimize_k_size(&mut self, hi_bits: u64, loss_fn: impl Fn(u64) -> f32) {
        let mut best_k = self.optimal_k_size;
        let mut best_performance = 0.0f32;
        
        for (&k_size, table) in &self.transition_tables {
            let hit_rate = table.cache_hit_rate();
            let cache_size = table.transition_cache.len() as f32;
            
            // 성능 점수: 적중률 * 효율성
            let performance_score = hit_rate * (1.0 / (1.0 + cache_size / 1000.0));
            
            if performance_score > best_performance {
                best_performance = performance_score;
                best_k = k_size;
            }
        }
        
        self.optimal_k_size = best_k;
    }
}

/// 수치적 미분 엔진  
#[derive(Debug)]
pub struct NumericalDifferentiationEngine {
    gradient_table: NumericalGradientTable,
}

impl NumericalDifferentiationEngine {
    pub fn new() -> Self {
        Self {
            gradient_table: NumericalGradientTable::new(32, 32), // 초기 해상도
        }
    }
    
    pub fn compute_gradient(&mut self, lo_bits: u64, loss_fn: impl Fn(f32, f32) -> f32) -> (f32, f32) {
        self.gradient_table.compute_numerical_gradient(lo_bits, loss_fn)
    }
    
    pub fn precision_estimate(&self) -> f32 {
        self.gradient_table.cache_hit_rate()
    }
    
    pub fn refine_grid_resolution(&mut self) {
        self.gradient_table.refine_resolution();
    }
    
    pub fn cache_hit_rate(&self) -> f32 {
        self.gradient_table.cache_hit_rate()
    }
}

/// 분리형 그래디언트 구조
#[derive(Debug, Clone)]
pub struct SeparatedBitGradient {
    /// 해석적 그래디언트 (64차원)
    pub analytical_grad: [f32; 64],
    /// 수치적 그래디언트 (2차원: r, theta)  
    pub numerical_grad: (f32, f32),
    /// 그래디언트 메타데이터
    pub analytical_confidence: f32,
    pub numerical_precision: f32,
    /// 통합 그래디언트 크기
    pub magnitude: f32,
}

impl SeparatedBitGradient {
    pub fn new(analytical_grad: [f32; 64], numerical_grad: (f32, f32), 
               analytical_confidence: f32, numerical_precision: f32) -> Self {
        let analytical_magnitude = analytical_grad.iter().map(|&x| x.abs()).sum::<f32>();
        let numerical_magnitude = (numerical_grad.0.abs() + numerical_grad.1.abs());
        let magnitude = analytical_magnitude + numerical_magnitude;
        
        Self {
            analytical_grad,
            numerical_grad,
            analytical_confidence,
            numerical_precision,
            magnitude,
        }
    }
    
    /// 그래디언트 품질 점수 (0-1)
    pub fn quality_score(&self) -> f32 {
        let confidence_score = self.analytical_confidence * 0.6 + self.numerical_precision * 0.4;
        let magnitude_score = (1.0 / (1.0 + (-self.magnitude).exp())); // 시그모이드
        
        confidence_score * 0.7 + magnitude_score * 0.3
    }
}

/// 통합 분리형 비트 자동미분 시스템
#[derive(Debug)]
pub struct SeparatedBitAutoDiff {
    /// 해석적 미분 엔진
    analytical_engine: AnalyticalDifferentiationEngine,
    /// 수치적 미분 엔진
    numerical_engine: NumericalDifferentiationEngine,
    /// 성능 통계
    total_computations: usize,
    total_time_us: u128,
    average_quality_score: f32,
}

impl SeparatedBitAutoDiff {
    pub fn new() -> Self {
        Self {
            analytical_engine: AnalyticalDifferentiationEngine::new(),
            numerical_engine: NumericalDifferentiationEngine::new(),
            total_computations: 0,
            total_time_us: 0,
            average_quality_score: 0.0,
        }
    }
    
    /// 분리형 그래디언트 계산
    pub fn compute_separated_gradient(&mut self, packed: &Packed128, loss_fn: impl Fn(u64, f32, f32) -> f32 + Send + Sync) -> SeparatedBitGradient {
        let start_time = Instant::now();
        
        let (hi_bits, lo_bits) = (packed.hi, packed.lo);
        
        // 해석적 그래디언트 계산 (lo_coords 미리 추출)
        let (r, theta) = self.extract_lo_coords(lo_bits);
        let analytical_grad = {
            self.analytical_engine.compute_gradient(hi_bits, |h| {
                loss_fn(h, r, theta)
            })
        };
        
        // 수치적 그래디언트 계산
        let numerical_grad = {
            self.numerical_engine.compute_gradient(lo_bits, |r, theta| {
                loss_fn(hi_bits, r, theta)
            })
        };
        
        // 신뢰도 계산 (별도 스코프)
        let analytical_confidence = self.analytical_engine.confidence_score();
        let numerical_precision = self.numerical_engine.precision_estimate();
        
        let gradient = SeparatedBitGradient::new(
            analytical_grad, 
            numerical_grad, 
            analytical_confidence, 
            numerical_precision
        );
        
        // 성능 통계 업데이트
        let elapsed = start_time.elapsed();
        self.total_time_us += elapsed.as_micros();
        self.total_computations += 1;
        self.average_quality_score = (self.average_quality_score * (self.total_computations - 1) as f32 + 
                                     gradient.quality_score()) / self.total_computations as f32;
        
        gradient
    }
    
    /// 배치 그래디언트 계산 (완전 병렬)
    pub fn compute_batch_gradients(&mut self, packed_batch: &[Packed128], 
                                  loss_fn: impl Fn(u64, f32, f32) -> f32 + Sync) -> Vec<SeparatedBitGradient> {
        let start_time = Instant::now();
        
        let results: Vec<_> = packed_batch.par_iter().map(|packed| {
            let (hi_bits, lo_bits) = (packed.hi, packed.lo);
            
            let (analytical_grad, numerical_grad) = rayon::join(
                || {
                    // 로컬 해석적 계산 (thread-safe)
                    let mut local_analytical = AnalyticalDifferentiationEngine::new();
                    local_analytical.compute_gradient(hi_bits, |h| {
                        let (r, theta) = self.extract_lo_coords(lo_bits);
                        loss_fn(h, r, theta)
                    })
                },
                || {
                    // 로컬 수치적 계산 (thread-safe)
                    let mut local_numerical = NumericalDifferentiationEngine::new();
                    local_numerical.compute_gradient(lo_bits, |r, theta| {
                        loss_fn(hi_bits, r, theta)
                    })
                }
            );
            
            SeparatedBitGradient::new(analytical_grad, numerical_grad, 0.9, 0.9) // 임시 신뢰도
        }).collect();
        
        let elapsed = start_time.elapsed();
        self.total_time_us += elapsed.as_micros();
        self.total_computations += packed_batch.len();
        
        results
    }
    
    /// 적응적 최적화
    pub fn adaptive_optimization(&mut self) {
        let analytical_hit_rate = self.analytical_engine.confidence_score();
        let numerical_hit_rate = self.numerical_engine.cache_hit_rate();
        
        // 적중률이 낮으면 캐시 확장
        if analytical_hit_rate < 0.8 {
            self.analytical_engine.expand_transition_table();
        }
        
        if numerical_hit_rate < 0.7 {
            self.numerical_engine.refine_grid_resolution();
        }
    }
    
    pub fn extract_lo_coords(&self, lo_bits: u64) -> (f32, f32) {
        let r_bits = lo_bits as u32;
        let theta_bits = (lo_bits >> 32) as u32;
        
        let r = f32::from_bits(r_bits).abs().min(0.999);
        let theta = f32::from_bits(theta_bits) % (2.0 * std::f32::consts::PI);
        
        (r, theta)
    }
    
    /// 해석적 엔진 캐시 적중률
    pub fn analytical_cache_hit_rate(&self) -> f32 {
        self.analytical_engine.confidence_score()
    }
    
    /// 수치적 엔진 캐시 적중률  
    pub fn numerical_cache_hit_rate(&self) -> f32 {
        self.numerical_engine.cache_hit_rate()
    }
    
    /// 성능 리포트
    pub fn performance_report(&self) -> String {
        let avg_time_us = if self.total_computations > 0 {
            self.total_time_us as f64 / self.total_computations as f64
        } else {
            0.0
        };
        
        format!(
            "분리형 비트 자동미분 성능 리포트:\n\
            - 총 계산 횟수: {}\n\
            - 총 시간: {:.2}ms\n\
            - 평균 계산 시간: {:.2}μs\n\
            - 평균 품질 점수: {:.3}\n\
            - 해석적 신뢰도: {:.3}\n\
            - 수치적 정밀도: {:.3}",
            self.total_computations,
            self.total_time_us as f64 / 1000.0,
            avg_time_us,
            self.average_quality_score,
            self.analytical_engine.confidence_score(),
            self.numerical_engine.cache_hit_rate()
        )
    }
} 
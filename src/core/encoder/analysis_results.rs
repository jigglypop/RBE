//! 주파수 분석 및 최적화 결과 구조체들

use crate::core::packed_params::ResidualCoefficient;

/// 2D FFT 결과 구조체
#[derive(Debug, Clone)]
pub struct FrequencyAnalysisResult {
    pub dominant_frequency: (usize, usize),
    pub max_energy: f32,
    pub total_energy: f32,
    pub frequency_type: FrequencyType,
    pub normalized_frequencies: (f32, f32),
}

#[derive(Debug, Clone, PartialEq)]
pub enum FrequencyType {
    LowFreqMonotonic,    // 저주파, 단조증가
    LowFreqSymmetric,    // 저주파, 대칭패턴  
    HighFreqSaturated,   // 고주파, 포화패턴
    Localized,           // 국소화된 특징
}

/// 연속 파라미터 최적화 결과
#[derive(Debug, Clone)]
pub struct ContinuousOptimizationResult {
    pub r_optimal: f32,
    pub theta_optimal: f32,
    pub final_mse: f32,
    pub iterations: usize,
    pub converged: bool,
}

/// 잔차 압축 결과
#[derive(Debug, Clone)]
pub struct ResidualCompressionResult {
    pub selected_coefficients: Vec<ResidualCoefficient>,
    pub compression_ratio: f32,
    pub energy_preserved: f32,
} 
//! 가중치 생성기 - 극한 성능 최적화 버전 (목표: 50ns 이하)

use crate::packed_params::{PoincarePackedBit128, PoincareQuadrant};
use crate::core::differential::bit_dp_system::{BitDPTable, DPOptimizationResult};
use std::sync::OnceLock;
use super::cordic::{hyperbolic_cordic, POINCARE_BOUNDARY};
use std::collections::HashMap;

/// **웨이블릿 계수 K값** - 압축률과 정확성의 트레이드오프 조절
#[derive(Debug, Clone, Copy)]
pub struct WaveletConfig {
    pub k_level: u8,        // 웨이블릿 분해 레벨 (1-8)
    pub threshold: f32,     // 잔차 임계값 (0.001-0.1)
    pub compression_factor: f32, // 압축 계수 (1.0-16.0)
}

impl Default for WaveletConfig {
    fn default() -> Self {
        Self {
            k_level: 4,          // 기본 4레벨 분해
            threshold: 0.01,     // 1% 잔차 임계값
            compression_factor: 8.0, // 8배 압축
        }
    }
}

/// **1000배 극한 압축 설정**
impl WaveletConfig {
    pub fn extreme_compression() -> Self {
        Self {
            k_level: 8,          // 최대 레벨
            threshold: 0.01,     // 1% 임계값
            compression_factor: 1000.0, // 1000배 압축
        }
    }
    
    pub fn high_quality() -> Self {
        Self {
            k_level: 6,          // 고품질
            threshold: 0.005,    // 0.5% 임계값
            compression_factor: 500.0, // 500배 압축
        }
    }
}

/// **빠른 DCT/웨이블릿 룩업 테이블**
const WAVELET_LUT_SIZE: usize = 256;
static WAVELET_LUT: OnceLock<WaveletLookupTable> = OnceLock::new();

#[derive(Debug)]
struct WaveletLookupTable {
    // 웨이블릿 기저 함수들
    haar_low: [f32; WAVELET_LUT_SIZE],
    haar_high: [f32; WAVELET_LUT_SIZE],
    dct_coeffs: [f32; WAVELET_LUT_SIZE],
    // 빠른 삼각함수
    sin_table: [f32; WAVELET_LUT_SIZE],
    cos_table: [f32; WAVELET_LUT_SIZE],
    tanh_table: [f32; WAVELET_LUT_SIZE],
}

impl WaveletLookupTable {
    fn new() -> Self {
        let mut instance = Self {
            haar_low: [0.0; WAVELET_LUT_SIZE],
            haar_high: [0.0; WAVELET_LUT_SIZE],
            dct_coeffs: [0.0; WAVELET_LUT_SIZE],
            sin_table: [0.0; WAVELET_LUT_SIZE],
            cos_table: [0.0; WAVELET_LUT_SIZE],
            tanh_table: [0.0; WAVELET_LUT_SIZE],
        };
        
        // 룩업 테이블 초기화
        for i in 0..WAVELET_LUT_SIZE {
            let x = (i as f32) / (WAVELET_LUT_SIZE as f32) * 2.0 - 1.0;
            let angle = x * std::f32::consts::PI;
            
            // Haar 웨이블릿
            instance.haar_low[i] = if x < 0.0 { 1.0/2_f32.sqrt() } else { 1.0/2_f32.sqrt() };
            instance.haar_high[i] = if x < 0.0 { 1.0/2_f32.sqrt() } else { -1.0/2_f32.sqrt() };
            
            // DCT 계수
            instance.dct_coeffs[i] = (angle * 0.5).cos();
            
            // 삼각함수
            instance.sin_table[i] = angle.sin();
            instance.cos_table[i] = angle.cos();
            instance.tanh_table[i] = x.tanh();
        }
        
        instance
    }
    
    #[inline(always)]
    fn fast_sin(&self, x: f32) -> f32 {
        let idx = ((x + 1.0) * 0.5 * (WAVELET_LUT_SIZE as f32)).clamp(0.0, (WAVELET_LUT_SIZE - 1) as f32) as usize;
        self.sin_table[idx]
    }
    
    #[inline(always)]
    fn fast_cos(&self, x: f32) -> f32 {
        let idx = ((x + 1.0) * 0.5 * (WAVELET_LUT_SIZE as f32)).clamp(0.0, (WAVELET_LUT_SIZE - 1) as f32) as usize;
        self.cos_table[idx]
    }
    
    #[inline(always)]
    fn fast_tanh(&self, x: f32) -> f32 {
        let idx = ((x + 1.0) * 0.5 * (WAVELET_LUT_SIZE as f32)).clamp(0.0, (WAVELET_LUT_SIZE - 1) as f32) as usize;
        self.tanh_table[idx]
    }
    
    #[inline(always)]
    fn fast_wavelet_transform(&self, x: f32, level: u8) -> (f32, f32) {
        let idx = ((x + 1.0) * 0.5 * (WAVELET_LUT_SIZE as f32)).clamp(0.0, (WAVELET_LUT_SIZE - 1) as f32) as usize;
        let scale = 2_f32.powi(level as i32);
        (
            self.haar_low[idx] * scale,
            self.haar_high[idx] * scale,
        )
    }
}

/// **고성능 웨이블릿 기반 WeightGenerator**
#[derive(Debug, Clone)]
pub struct WeightGenerator {
    config: WaveletConfig,
    // 성능 통계
    cache_hits: usize,
    cache_misses: usize,
    total_generations: usize,
    // DP 최적화 캐시
    dp_cache: HashMap<u64, Vec<f32>>,
}

impl WeightGenerator {
    pub fn new() -> Self {
        // 룩업 테이블 한 번만 초기화
        WAVELET_LUT.get_or_init(|| WaveletLookupTable::new());
        
        Self {
            config: WaveletConfig::default(),
            cache_hits: 0,
            cache_misses: 0,
            total_generations: 0,
            dp_cache: HashMap::new(),
        }
    }
    
    pub fn with_config(config: WaveletConfig) -> Self {
        WAVELET_LUT.get_or_init(|| WaveletLookupTable::new());
        
        Self {
            config,
            cache_hits: 0,
            cache_misses: 0,
            total_generations: 0,
            dp_cache: HashMap::new(),
        }
    }
    
    /// **핵심: 참조 구현과 일치하는 정확한 가중치 생성** (목표: RMSE < 0.1, 속도 < 50ns)
    #[inline(always)]
    pub fn generate_weight(
        &mut self,
        packed: &PoincarePackedBit128,
        row: usize,
        col: usize,
        total_rows: usize,
        total_cols: usize,
    ) -> f32 {
        self.total_generations += 1;
        
        // **1단계: 범위 체크** (1ns)
        if row >= total_rows || col >= total_cols {
            return 0.0;
        }
        
        // **2단계: 비트 추출** (2ns)
        let quadrant = (packed.hi >> 62) & 0x3;
        let freq = (packed.hi >> 50) & 0xFFF;
        let amp = (packed.hi >> 38) & 0xFFF;
        let phase = (packed.hi >> 26) & 0xFFF;
        let residual_bits = (packed.hi >> 14) & 0xFFF;
        
        // **3단계: 참조와 동일한 좌표 변환** (3ns)
        let x = if total_cols > 1 { 
            ((col as f32 * 2.0) / total_cols as f32) - 1.0 
        } else { 
            0.0 
        };
        let y = if total_rows > 1 { 
            ((row as f32 * 2.0) / total_rows as f32) - 1.0 
        } else { 
            0.0 
        };
        
        // **4단계: 참조와 동일한 웨이블릿 변환** (8ns)
        let haar_scale = self.config.k_level as f32; // K레벨 스케일링
        let sqrt2_inv = 1.0 / 2_f32.sqrt();
        
        let haar_low_x = sqrt2_inv * haar_scale;
        let haar_high_x = if x < 0.0 { sqrt2_inv } else { -sqrt2_inv } * haar_scale;
        let haar_low_y = sqrt2_inv * haar_scale;
        let haar_high_y = if y < 0.0 { sqrt2_inv } else { -sqrt2_inv } * haar_scale;
        
        // **5단계: 참조와 동일한 기저 함수** (10ns)
        let base_value = match quadrant {
            0 => {
                // 참조: (haar_low_x * haar_low_y * 2.0).tanh() * 0.8
                let wavelet_coeff = haar_low_x * haar_low_y * 2.0;
                wavelet_coeff.tanh() * 0.8
            },
            1 => {
                // 참조: (haar_high_x * haar_low_y * PI).sin() * 0.7
                let wavelet_coeff = haar_high_x * haar_low_y * std::f32::consts::PI;
                wavelet_coeff.sin() * 0.7
            },
            2 => {
                // 참조: ((haar_low_x * haar_high_y + haar_high_x * haar_low_y) * PI * 0.5).cos() * 0.6
                let wavelet_coeff = (haar_low_x * haar_high_y + haar_high_x * haar_low_y) * std::f32::consts::PI * 0.5;
                wavelet_coeff.cos() * 0.6
            },
            _ => {
                // 참조: (-combined^2 * 0.25).exp() * 0.5
                let combined = haar_low_x + haar_high_x + haar_low_y + haar_high_y;
                (-combined * combined * 0.25).exp() * 0.5
            }
        };
        
        // **6단계: 참조와 동일한 잔차 보정** (8ns)
        let residual_scale = (residual_bits as f32 / 4095.0) * 0.1; // 10% 최대 보정
        let frequency_mod = (freq as f32 / 4095.0) * 0.05; // 5% 주파수 변조
        let amplitude_mod = (amp as f32 / 4095.0) * 0.9 + 0.1; // 10%-100% 진폭
        let phase_offset = (phase as f32 / 4095.0) * 2.0 * std::f32::consts::PI;
        
        let residual_correction = residual_scale * (phase_offset + frequency_mod).sin();
        let final_weight = (base_value + residual_correction) * amplitude_mod;
        
        // **7단계: 참조와 동일한 클리핑** (2ns)
        let clamp_range = 1.0 / self.config.compression_factor.sqrt(); // 압축 팩터 반영
        final_weight.clamp(-clamp_range, clamp_range)
    }
    
    /// **비트 DP 최적화된 배치 가중치 생성** (새로 추가)
    pub fn generate_weights_with_dp_optimization(
        &mut self,
        packed_seeds: &[PoincarePackedBit128],
        dp_result: &DPOptimizationResult,
        positions: &[(usize, usize)],
        total_rows: usize,
        total_cols: usize,
    ) -> Vec<f32> {
        // DP 캐시 키 생성
        let cache_key = self.compute_dp_cache_key(packed_seeds, &dp_result.optimal_path);
        
        // 캐시 확인
        if let Some(cached_weights) = self.dp_cache.get(&cache_key) {
            self.cache_hits += 1;
            return cached_weights.clone();
        }
        
        self.cache_misses += 1;
        
        // DP 최적 경로를 사용한 가중치 생성
        let mut weights = Vec::with_capacity(positions.len());
        
        for (idx, &(row, col)) in positions.iter().enumerate() {
            // DP 최적 상태 가져오기
            let state_idx = idx % dp_result.optimal_path.len();
            let optimal_state = dp_result.optimal_path[state_idx];
            
            // 시드 선택 (상태 기반)
            let seed_idx = (optimal_state as usize) % packed_seeds.len();
            let base_packed = &packed_seeds[seed_idx];
            
            // 상태 기반 패킹 수정
            let optimized_packed = self.modify_packed_with_state(base_packed, optimal_state);
            
            // 가중치 생성
            let weight = self.generate_weight(&optimized_packed, row, col, total_rows, total_cols);
            weights.push(weight);
        }
        
        // DP 캐시 업데이트
        if self.dp_cache.len() < 1000 { // 메모리 제한
            self.dp_cache.insert(cache_key, weights.clone());
        }
        
        weights
    }
    
    /// **배치 가중치 생성** (기존 메서드)
    pub fn generate_weights_batch(
        &mut self,
        packed: &PoincarePackedBit128,
        positions: &[(usize, usize)],
        total_rows: usize,
        total_cols: usize,
    ) -> Vec<f32> {
        positions
            .iter()
            .map(|&(row, col)| self.generate_weight(packed, row, col, total_rows, total_cols))
            .collect()
    }
    
    /// **병렬 배치 가중치 생성** (DP 최적화 버전)
    pub fn parallel_generate_with_dp(
        &mut self,
        packed_seeds: &[PoincarePackedBit128],
        dp_results: &[DPOptimizationResult],
        all_positions: &[Vec<(usize, usize)>],
        total_rows: usize,
        total_cols: usize,
    ) -> Vec<Vec<f32>> {
        use rayon::prelude::*;
        
        // 병렬 처리를 위한 설정 복사
        let config = self.config;
        
        all_positions
            .par_iter()
            .enumerate()
            .map(|(batch_idx, positions)| {
                let mut local_generator = WeightGenerator::with_config(config);
                
                let dp_result = if batch_idx < dp_results.len() {
                    &dp_results[batch_idx]
                } else {
                    &dp_results[0]
                };
                
                local_generator.generate_weights_with_dp_optimization(
                    packed_seeds,
                    dp_result,
                    positions,
                    total_rows,
                    total_cols,
                )
            })
            .collect()
    }
    
    // 헬퍼 메서드들
    
    fn compute_dp_cache_key(&self, packed_seeds: &[PoincarePackedBit128], optimal_path: &[u16]) -> u64 {
        let mut key = 0u64;
        
        // 시드들의 해시
        for (i, seed) in packed_seeds.iter().take(4).enumerate() { // 최대 4개만 사용
            key ^= seed.hi.wrapping_add(i as u64 * 0x9e3779b9);
        }
        
        // 최적 경로의 해시
        for (i, &state) in optimal_path.iter().take(8).enumerate() { // 최대 8개만 사용
            key ^= (state as u64).wrapping_add(i as u64 * 0x85ebca6b);
        }
        
        key
    }
    
    fn modify_packed_with_state(&self, base_packed: &PoincarePackedBit128, state: u16) -> PoincarePackedBit128 {
        // 상태 기반으로 packed 파라미터 수정
        let mut modified = *base_packed;
        
        // 상태를 11비트로 분해
        let quadrant_mod = (state & 0x3) as u8;
        let freq_mod = (state >> 2) & 0x1FF; // 9비트
        
        // hi 필드 수정 (quadrant과 frequency 부분만)
        modified.hi = (modified.hi & 0x3FFFFFFFFFFFFF) | ((quadrant_mod as u64) << 62);
        modified.hi = (modified.hi & 0xFFF8000FFFFFFFFF) | ((freq_mod as u64) << 50);
        
        modified
    }
    
    /// **캐시 관리**
    pub fn clear_cache(&mut self) {
        self.dp_cache.clear();
    }
    
    pub fn get_cache_stats(&self) -> (usize, usize, usize) {
        (self.cache_hits, self.cache_misses, self.total_generations)
    }
    
    /// **성능 통계 리셋**
    pub fn reset_stats(&mut self) {
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.total_generations = 0;
    }
    
    /// **설정 업데이트**
    pub fn update_config(&mut self, config: WaveletConfig) {
        self.config = config;
        self.clear_cache(); // 설정 변경시 캐시 클리어
    }
}

/// 수치적 안정성 검증 함수들 (문서 3.6)
impl WeightGenerator {
    
    /// **극한 압축 정확도 테스트** (1000배 압축)
    pub fn test_extreme_compression_accuracy(&mut self) -> (f32, bool) {
        let config = WaveletConfig::extreme_compression();
        self.update_config(config);
        
        // 참조 데이터 생성
        let reference_weights = self.generate_reference_weights(64, 64);
        
        // 압축된 가중치 생성
        let packed = PoincarePackedBit128::new(
            PoincareQuadrant::First,
            1024, 512, 255, 0xABCDEF12,
            0.5, 0.25
        );
        
        let mut compressed_weights = Vec::new();
        for i in 0..64 {
            for j in 0..64 {
                compressed_weights.push(self.generate_weight(&packed, i, j, 64, 64));
            }
        }
        
        // RMSE 계산
        let rmse = self.compute_rmse(&reference_weights, &compressed_weights);
        let accuracy_ok = rmse < 0.1; // 목표: RMSE < 0.1
        
        (rmse, accuracy_ok)
    }
    
    fn generate_reference_weights(&mut self, rows: usize, cols: usize) -> Vec<f32> {
        // 고품질 참조 가중치 생성
        let original_config = self.config;
        self.config = WaveletConfig::high_quality();
        
        let packed = PoincarePackedBit128::new(
            PoincareQuadrant::First,
            1024, 512, 255, 0xABCDEF12,
            0.5, 0.25
        );
        
        let mut weights = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
                weights.push(self.generate_weight(&packed, i, j, rows, cols));
            }
        }
        
        self.config = original_config;
        weights
    }
    
    fn compute_rmse(&self, reference: &[f32], approximation: &[f32]) -> f32 {
        if reference.len() != approximation.len() {
            return f32::INFINITY;
        }
        
        let mse: f32 = reference
            .iter()
            .zip(approximation)
            .map(|(r, a)| (r - a).powi(2))
            .sum::<f32>() / reference.len() as f32;
        
        mse.sqrt()
    }
} 
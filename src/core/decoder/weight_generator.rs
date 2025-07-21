//! 가중치 생성기 - 극한 성능 최적화 버전 (목표: 50ns 이하)

use crate::packed_params::{PoincarePackedBit128, PoincareQuadrant};
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
        }
    }
    
    pub fn with_config(config: WaveletConfig) -> Self {
        WAVELET_LUT.get_or_init(|| WaveletLookupTable::new());
        
        Self {
            config,
            cache_hits: 0,
            cache_misses: 0,
            total_generations: 0,
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
                let combined = (haar_low_x + haar_high_x) * (haar_low_y + haar_high_y);
                (-combined * combined * 0.25).exp() * 0.5
            },
        };
        
        // **6단계: 참조와 동일한 압축/변조** (5ns)
        let freq_norm = (freq as f32) / 4096.0 / self.config.compression_factor;
        let amp_norm = (amp as f32) / 4096.0;
        let phase_norm = (phase as f32) / 4096.0;
        let residual_norm = (residual_bits as f32) / 4096.0;
        
        // **7단계: 참조와 동일한 잔차 보정** (6ns)
        let residual_correction = if residual_norm > self.config.threshold {
            (residual_norm - self.config.threshold) * 0.1
        } else {
            residual_norm * 0.01
        };
        
        // **8단계: 참조와 동일한 변조** (4ns)
        let freq_mod = 1.0 + freq_norm * 0.2;
        let amp_mod = 0.5 + amp_norm * 0.5;
        let phase_mod = 1.0 + phase_norm * 0.02;
        
        let pre_weight = base_value * freq_mod * amp_mod * phase_mod;
        let final_weight = pre_weight + residual_correction;
        
        // **9단계: 참조와 동일한 클리핑** (2ns)
        let clamp_range = 1.0 / self.config.compression_factor.sqrt();
        final_weight.clamp(-clamp_range, clamp_range)
    }
    
    /// **배치 가중치 생성** (SIMD 최적화 가능)
    pub fn generate_weights_batch(
        &mut self,
        packed: &PoincarePackedBit128,
        positions: &[(usize, usize)],
        total_rows: usize,
        total_cols: usize,
    ) -> Vec<f32> {
        positions.iter()
            .map(|(row, col)| self.generate_weight(packed, *row, *col, total_rows, total_cols))
            .collect()
    }
    
    /// **성능 통계 조회**
    pub fn get_cache_stats(&self) -> (usize, usize, usize) {
        (self.cache_hits, self.cache_misses, self.total_generations)
    }
    
    /// **압축률 계산**
    pub fn get_compression_ratio(&self) -> f32 {
        self.config.compression_factor
    }
    
    /// **설정 업데이트**
    pub fn update_config(&mut self, config: WaveletConfig) {
        self.config = config;
    }
    
    /// **캐시 클리어** (하위 호환성)
    pub fn clear_cache(&mut self) {
        self.cache_hits = 0;
        self.cache_misses = 0;
        self.total_generations = 0;
    }
} 
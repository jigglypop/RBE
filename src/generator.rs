//! 행렬 생성 관련 유틸리티
//! 
//! 이 모듈은 다양한 패턴의 행렬을 생성하는 헬퍼 함수들을 제공합니다.

use crate::types::{Packed128, PoincareMatrix};
use std::f32::consts::PI;

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
    
    /// 특정 패턴을 학습한 행렬 생성
    pub fn from_pattern(pattern: &[f32], rows: usize, cols: usize, epochs: usize, lr: f32) -> Self {
        let init = Self::new_default(rows, cols);
        init.train_with_adam128(pattern, rows, cols, epochs, lr)
    }
} 
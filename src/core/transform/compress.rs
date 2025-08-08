//! f32 가중치 → Packed128 압축기

use crate::core::tensors::{Packed128, DecodedParams};
use super::TransformStats;
use std::time::Instant;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// 가중치 압축기 (즉시 압축)
pub struct WeightCompressor {
    pub target_shape: (usize, usize),
    /// 열 단위 블록 크기 (hidden_size). 0이면 전체 행렬 한 번에 압축.
    pub block_cols: usize,
}

impl WeightCompressor {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { target_shape: (rows, cols), block_cols: 0 }
    }

    /// 블록 크기를 설정한 새 컴프레서 생성
    pub fn with_block_cols(mut self, block_cols: usize) -> Self {
        self.block_cols = block_cols;
        self
    }

    /// f32 배열을 Packed128로 즉시 압축
    pub fn compress_weights(&self, weights: &[f32]) -> anyhow::Result<(Packed128, TransformStats)> {
        let (rows, cols_total) = self.target_shape;
        let block = if self.block_cols == 0 { cols_total } else { self.block_cols.min(cols_total) };

        if cols_total % block != 0 {
            return Err(anyhow::anyhow!("cols {} not divisible by block size {}", cols_total, block));
        }

        let blocks = cols_total / block;
        let mut seeds: Vec<Packed128> = Vec::with_capacity(blocks);
        let mut total_rmse = 0.0;
        let mut total_orig_mb = 0.0;
        let mut total_comp_mb = 0.0;
        let start = Instant::now();

        for b in 0..blocks {
            let offset = b * block;
            let mut block_weights = Vec::with_capacity(rows * block);
            for r in 0..rows {
                let row_start = r * cols_total + offset;
                block_weights.extend_from_slice(&weights[row_start..row_start + block]);
            }

            let mut seed = self.create_optimal_seed(&block_weights, rows, block);
            // 전역 기본값 설정: αg=1.0, δ=0.0, κ=0.0 (문서 규범)
            seed.set_alpha_g(1.0);
            seed.set_delta(0.0);
            seed.set_kappa(0.0);
            
            let rmse = self.calculate_rmse_fast(&seed, &block_weights, rows, block);

            seeds.push(seed);
            total_rmse += rmse;
            total_orig_mb += (rows * block * 4) as f64 / (1024.0 * 1024.0);
            total_comp_mb += std::mem::size_of::<Packed128>() as f64 / (1024.0 * 1024.0);
        }

        let duration = start.elapsed();

        // 평균 RMSE
        let avg_rmse = total_rmse / blocks as f64;

        let stats = TransformStats {
            original_size_mb: total_orig_mb,
            compressed_size_mb: total_comp_mb,
            compression_ratio: total_orig_mb / total_comp_mb,
            rmse: avg_rmse,
            transform_ms: duration.as_secs_f64() * 1000.0,
            restore_ms: 0.0,
        };

        // 여러 블록이면 시드 배열을 해시로 결합 (간단: XOR)
        let final_seed = if seeds.len() == 1 {
            seeds[0]
        } else {
            let mut acc = seeds[0];
            for s in seeds.iter().skip(1) {
                acc.hi ^= s.hi;
                acc.lo ^= s.lo;
            }
            acc
        };

        Ok((final_seed, stats))
    }
    
    /// 최적 시드 즉시 생성 (가중치 기반 휴리스틱)
    fn create_optimal_seed(&self, weights: &[f32], rows: usize, cols: usize) -> Packed128 {
        // 1. 빠른 파워 Iteration으로 1번 singular vector 추정
        let mut rng = StdRng::from_entropy();
        let mut v: Vec<f32> = (0..cols.min(256)).map(|_| rand::Rng::gen::<f32>(&mut rng)).collect();
        let norm = (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>()).sqrt();
        v.iter_mut().for_each(|x| *x /= norm as f32);

        for _ in 0..3 { // 3회만
            // w = A * v  (샘플 몇 행만)
            let mut w = vec![0.0f32; rows.min(256)];
            for r in 0..rows.min(256) {
                let row_start = r * cols;
                let mut sum = 0.0f32;
                for c in 0..v.len() {
                    sum += weights[row_start + c] * v[c];
                }
                w[r] = sum;
            }
            // v = Aᵀ * w
            let mut v_new = vec![0.0f32; v.len()];
            for c in 0..v.len() {
                let mut sum = 0.0f32;
                for r in 0..w.len() {
                    sum += weights[r * cols + c] * w[r];
                }
                v_new[c] = sum;
            }
            let norm_new = (v_new.iter().map(|x| (*x as f64).powi(2)).sum::<f64>()).sqrt();
            v = v_new.into_iter().map(|x| x / norm_new as f32).collect();
        }

        // 2. r,theta 를 singular vector 통계로 매핑
        let mean_v = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
        let std_v = (v.iter().map(|&x| (x as f64 - mean_v).powi(2)).sum::<f64>() / v.len() as f64).sqrt();

        let r = (std_v * 3.0).min(0.999) as f32; // 분산 기반 확대
        let theta = ((mean_v * std::f64::consts::PI).sin() + 1.0) * std::f64::consts::PI;

        let params = DecodedParams { r_fp32: r, theta_fp32: theta as f32 };
        Packed128::from_continuous(&params)
    }
    
    /// 빠른 RMSE 계산
    fn calculate_rmse_fast(&self, seed: &Packed128, target_weights: &[f32], rows: usize, cols: usize) -> f64 {
        let mut total_error = 0.0f64;
        
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                let predicted = seed.fused_forward(i, j, rows, cols);
                let target = target_weights[idx];
                let error = (predicted - target) as f64;
                total_error += error * error;
            }
        }
        
        (total_error / (rows * cols) as f64).sqrt()
    }
} 
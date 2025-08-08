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
    /// 시드 로컬 정밀화 반복 횟수 (0이면 사용하지 않음)
    pub refine_iters: usize,
    /// 블록당 시드 개수 (1이면 단일 시드)
    pub seeds_per_block: usize,
}

impl WeightCompressor {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { target_shape: (rows, cols), block_cols: 0, refine_iters: 0, seeds_per_block: 1 }
    }

    /// 블록 크기를 설정한 새 컴프레서 생성
    pub fn with_block_cols(mut self, block_cols: usize) -> Self {
        self.block_cols = block_cols;
        self
    }

    /// 로컬 정밀화 반복 횟수 설정
    pub fn with_refine_iters(mut self, refine_iters: usize) -> Self {
        self.refine_iters = refine_iters;
        self
    }

    /// 블록당 시드 개수 설정
    pub fn with_seeds_per_block(mut self, seeds_per_block: usize) -> Self {
        self.seeds_per_block = seeds_per_block.max(1);
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

            if self.refine_iters > 0 {
                self.refine_seed(&mut seed, &block_weights, rows, block, self.refine_iters);
            }
            
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

    /// f32 배열을 블록 단위로 압축하여 각 블록의 시드를 반환
    /// - block_cols가 0이면 전체 열을 하나의 블록으로 처리
    /// - 반환: (블록 시드 벡터, 집계 통계)
    pub fn compress_weights_blocks(&self, weights: &[f32]) -> anyhow::Result<(Vec<Packed128>, TransformStats)> {
        let (rows, cols_total) = self.target_shape;
        let block = if self.block_cols == 0 { cols_total } else { self.block_cols.min(cols_total) };

        if cols_total % block != 0 {
            return Err(anyhow::anyhow!("cols {} not divisible by block size {}", cols_total, block));
        }

        let blocks = cols_total / block;
        let k = self.seeds_per_block.max(1);
        let mut seeds: Vec<Packed128> = Vec::with_capacity(blocks * k);
        let mut total_rmse = 0.0f64;
        let mut total_orig_mb = 0.0f64;
        let mut total_comp_mb = 0.0f64;
        let start = Instant::now();

        for b in 0..blocks {
            let offset = b * block;
            let mut block_weights = Vec::with_capacity(rows * block);
            for r in 0..rows {
                let row_start = r * cols_total + offset;
                block_weights.extend_from_slice(&weights[row_start..row_start + block]);
            }

            // Matching-pursuit 스타일 다중 시드 적합
            let mut residual = block_weights.clone();
            let mut block_seeds: Vec<Packed128> = Vec::with_capacity(k);
            for _s in 0..k {
                let mut seed = self.create_optimal_seed(&residual, rows, block);
                seed.set_alpha_g(1.0);
                seed.set_delta(0.0);
                seed.set_kappa(0.0);
                if self.refine_iters > 0 {
                    self.refine_seed(&mut seed, &residual, rows, block, self.refine_iters);
                }
                // residual -= seed_prediction
                for i in 0..rows {
                    for j in 0..block {
                        let idx = i * block + j;
                        let pred = seed.fused_forward(i, j, rows, block);
                        residual[idx] -= pred;
                    }
                }
                block_seeds.push(seed);
            }
            // 블록 최종 RMSE (원본 vs 합성합)
            let mut se_sum_sq = 0.0f64;
            for i in 0..rows {
                for j in 0..block {
                    let idx = i * block + j;
                    let mut pred_sum = 0.0f32;
                    for s in 0..k { pred_sum += block_seeds[s].fused_forward(i, j, rows, block); }
                    let e = (block_weights[idx] - pred_sum) as f64;
                    se_sum_sq += e * e;
                }
            }
            let rmse_block = (se_sum_sq / (rows * block) as f64).sqrt();
            total_rmse += rmse_block;

            // 수집
            seeds.extend(block_seeds.into_iter());
            total_orig_mb += (rows * block * 4) as f64 / (1024.0 * 1024.0);
            total_comp_mb += (k as f64) * (std::mem::size_of::<Packed128>() as f64) / (1024.0 * 1024.0);
        }

        let duration = start.elapsed();
        let avg_rmse = total_rmse / blocks as f64;

        let stats = TransformStats {
            original_size_mb: total_orig_mb,
            compressed_size_mb: total_comp_mb,
            compression_ratio: if total_comp_mb > 0.0 { total_orig_mb / total_comp_mb } else { 0.0 },
            rmse: avg_rmse,
            transform_ms: duration.as_secs_f64() * 1000.0,
            restore_ms: 0.0,
        };

        Ok((seeds, stats))
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
        if rows == 0 { return 0.0; }
        // 실제 타깃 버퍼의 stride (열 수). refine 경로에서 cols(예: 2)와 다를 수 있어 안전하게 산정
        let cols_target = (target_weights.len() / rows).max(1);
        for i in 0..rows {
            for j in 0..cols {
                // 타깃 인덱스는 실제 stride 사용, 열 오버런 방지를 위해 wrap
                let idx = i * cols_target + (j % cols_target);
                let predicted = seed.fused_forward(i, j, rows, cols);
                let target = target_weights[idx];
                let error = (predicted - target) as f64;
                total_error += error * error;
            }
        }
        (total_error / (rows * cols) as f64).sqrt()
    }
} 

impl WeightCompressor {
    /// 시드 로컬 정밀화 (수치미분 기반, KISS)
    fn refine_seed(&self, seed: &mut Packed128, weights: &[f32], rows: usize, cols: usize, iters: usize) {
        // 안전한 최소 cols=2 처리 (cols=1의 경우 정규화 분모 0 방지)
        let cols_eff = if cols < 2 { 2 } else { cols };
        let mut current_rmse = self.calculate_rmse_fast(seed, weights, rows, cols_eff);
        let mut lr_r = 0.05f32;
        let mut lr_th = 0.05f32;
        for t in 0..iters {
            let params = seed.decode();
            let r0 = params.r_fp32.clamp(0.0, 0.9999);
            let th0 = params.theta_fp32;
            // 적응 스텝
            let decay = 0.95f32.powi((t as i32).max(0));
            let lr_r_t = lr_r * decay;
            let lr_th_t = lr_th * decay;
            let eps_r = (1.0 - r0).abs().max(1e-4) * 1e-2;
            let eps_th = 1e-3f32;

            // dr 수치미분
            let mut s_plus = *seed; {
                let mut p = s_plus.decode(); p.r_fp32 = (r0 + eps_r).clamp(0.0, 0.9999); s_plus.update_from_continuous(&p);
            }
            let mut s_minus = *seed; {
                let mut p = s_minus.decode(); p.r_fp32 = (r0 - eps_r).clamp(0.0, 0.9999); s_minus.update_from_continuous(&p);
            }
            let rmse_plus = self.calculate_rmse_fast(&s_plus, weights, rows, cols_eff) as f64;
            let rmse_minus = self.calculate_rmse_fast(&s_minus, weights, rows, cols_eff) as f64;
            let grad_r = ((rmse_plus - rmse_minus) as f32) / (2.0 * eps_r);

            // dtheta 수치미분
            let mut s_plus = *seed; {
                let mut p = s_plus.decode(); p.theta_fp32 = th0 + eps_th; s_plus.update_from_continuous(&p);
            }
            let mut s_minus = *seed; {
                let mut p = s_minus.decode(); p.theta_fp32 = th0 - eps_th; s_minus.update_from_continuous(&p);
            }
            let rmse_plus = self.calculate_rmse_fast(&s_plus, weights, rows, cols_eff) as f64;
            let rmse_minus = self.calculate_rmse_fast(&s_minus, weights, rows, cols_eff) as f64;
            let grad_th = ((rmse_plus - rmse_minus) as f32) / (2.0 * eps_th);

            // 업데이트
            let mut p = seed.decode();
            p.r_fp32 = (r0 - lr_r_t * grad_r).clamp(0.0, 0.9999);
            p.theta_fp32 = th0 - lr_th_t * grad_th;
            seed.update_from_continuous(&p);

            // 개선 없는 경우 조기 종료
            let new_rmse = self.calculate_rmse_fast(seed, weights, rows, cols_eff);
            if (current_rmse - new_rmse).abs() < 1e-7 { break; }
            current_rmse = new_rmse;
        }
    }
}
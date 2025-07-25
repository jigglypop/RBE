//! f32 가중치 → Packed128 압축기

use crate::core::tensors::{Packed128, DecodedParams};
use super::TransformStats;
use std::time::Instant;

/// 가중치 압축기
pub struct WeightCompressor {
    pub target_shape: (usize, usize),
    pub optimization_iterations: usize,
}

impl WeightCompressor {
    pub fn new(rows: usize, cols: usize) -> Self {
        // 크기에 따른 적응적 최적화 (더 공격적)
        let iterations = match rows * cols {
            0..=1000 => 20,       // 작은 행렬: 초고속
            1001..=50000 => 30,   // 중간 행렬: 빠른 압축  
            50001..=500000 => 40, // 큰 행렬: 균형
            _ => 60,              // 초대형: 최소한의 정확도
        };
        
        Self {
            target_shape: (rows, cols),
            optimization_iterations: iterations,
        }
    }
    
    /// f32 배열을 Packed128로 압축
    pub fn compress_weights(&self, weights: &[f32]) -> Result<(Packed128, TransformStats), Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        let (rows, cols) = self.target_shape;
        if weights.len() != rows * cols {
            return Err(format!("크기 불일치: {} vs {}x{}", weights.len(), rows, cols).into());
        }
        
        println!("압축 시작: {}x{} 행렬 ({} 파라미터)", rows, cols, weights.len());
        
        // 1. 최적화 기반 시드 찾기
        let mut best_seed = self.find_optimal_seed(weights, rows, cols)?;
        
        // 2. 미세 조정
        best_seed = self.fine_tune_seed(best_seed, weights, rows, cols)?;
        
        let compress_time = start_time.elapsed().as_millis() as f64;
        
        // 3. 복원하여 정확도 측정
        let restored = self.restore_from_seed(&best_seed, rows, cols);
        let rmse = self.calculate_rmse(weights, &restored);
        
        let original_size = weights.len() * 4; // f32 = 4 bytes
        let compressed_size = std::mem::size_of::<Packed128>();
        let compression_ratio = original_size as f64 / compressed_size as f64;
        
        let stats = TransformStats {
            original_size_mb: original_size as f64 / 1024.0 / 1024.0,
            compressed_size_mb: compressed_size as f64 / 1024.0 / 1024.0,
            compression_ratio,
            rmse,
            transform_ms: compress_time,
            restore_ms: 0.0, // 별도 측정
        };
        
        println!("압축 완료: {:.1}:1 압축률, RMSE {:.6}", compression_ratio, rmse);
        
        Ok((best_seed, stats))
    }
    
    /// 최적 시드 탐색 (유전 알고리즘 스타일)
    fn find_optimal_seed(&self, target: &[f32], rows: usize, cols: usize) -> Result<Packed128, Box<dyn std::error::Error>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // 적응적 집단 크기 (더 작게)
        let population_size = match rows * cols {
            0..=1000 => 10,       // 작은 행렬: 최소 집단
            1001..=50000 => 15,   // 중간 행렬: 작은 집단
            50001..=500000 => 20, // 큰 행렬: 중간 집단
            _ => 25,              // 초대형: 큰 집단
        };
        
        let mut population: Vec<Packed128> = (0..population_size)
            .map(|_| Packed128::random(&mut rng))
            .collect();
        
        let mut best_fitness = f64::INFINITY;
        let mut best_seed = population[0];
        
        for generation in 0..self.optimization_iterations {
            // 적합도 평가 (샘플링 기반 빠른 평가)
            let mut fitness_scores = Vec::new();
            let sample_ratio = match rows * cols {
                0..=1000 => 0.2,      // 20% 샘플링
                1001..=50000 => 0.05, // 5% 샘플링  
                _ => 0.02,            // 2% 샘플링
            };
            
            for seed in &population {
                let rmse = if generation < self.optimization_iterations / 3 {
                    // 초기 세대: 빠른 샘플 평가
                    self.quick_evaluate_seed(seed, target, rows, cols, sample_ratio)
                } else {
                    // 후기 세대: 정확한 전체 평가
                    let restored = self.restore_from_seed(seed, rows, cols);
                    self.calculate_rmse(target, &restored)
                };
                
                fitness_scores.push(rmse as f64);
                
                if rmse < best_fitness {
                    best_fitness = rmse;
                    best_seed = *seed;
                }
            }
            
            if generation % 20 == 0 {
                println!("세대 {}: 최고 RMSE {:.6}", generation, best_fitness);
            }
            
            // 조기 종료 조건 (크기별 적응적)
            let target_rmse = match rows * cols {
                0..=1000 => 0.05,      // 작은 행렬: 관대한 기준
                1001..=50000 => 0.08,  // 중간 행렬: 더 관대
                _ => 0.12,             // 큰 행렬: 매우 관대
            };
            
            if best_fitness < target_rmse {
                println!("목표 정확도 달성! RMSE: {:.6}", best_fitness);
                break;
            }
            
            // 새 세대 생성
            population = self.evolve_population(&population, &fitness_scores, &mut rng);
        }
        
        println!("최적화 완료: 최종 RMSE {:.6}", best_fitness);
        Ok(best_seed)
    }
    
    /// 집단 진화
    fn evolve_population(&self, population: &[Packed128], fitness: &[f64], rng: &mut impl rand::Rng) -> Vec<Packed128> {
        let mut new_pop = Vec::new();
        
        // 엘리트 보존 (상위 10%)
        let elite_count = population.len() / 10;
        let mut indexed_fitness: Vec<(usize, f64)> = fitness.iter()
            .enumerate()
            .map(|(i, &f)| (i, f))
            .collect();
        indexed_fitness.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        for i in 0..elite_count {
            new_pop.push(population[indexed_fitness[i].0]);
        }
        
        // 돌연변이 생성
        while new_pop.len() < population.len() {
            let parent_idx = indexed_fitness[rng.gen_range(0..elite_count * 2)].0;
            let mut child = population[parent_idx];
            
            // 비트 돌연변이
            if rng.gen::<f32>() < 0.3 {
                child.hi ^= 1u64 << rng.gen_range(0..64);
            }
            if rng.gen::<f32>() < 0.3 {
                child.lo ^= 1u64 << rng.gen_range(0..64);
            }
            
            new_pop.push(child);
        }
        
        new_pop
    }
    
    /// 미세 조정 (그래디언트 기반)
    fn fine_tune_seed(&self, mut seed: Packed128, target: &[f32], rows: usize, cols: usize) -> Result<Packed128, Box<dyn std::error::Error>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let mut best_rmse = {
            let restored = self.restore_from_seed(&seed, rows, cols);
            self.calculate_rmse(target, &restored)
        };
        
        let learning_rate = 0.01;
        
        for iter in 0..50 {
            // 연속 파라미터 미세 조정
            let decoded = seed.decode();
            let mut new_r = decoded.r_fp32;
            let mut new_theta = decoded.theta_fp32;
            
            // 수치 그래디언트 근사
            let epsilon = 0.001f64;
            
            // r 그래디언트
            let r_plus = DecodedParams { r_fp32: new_r + epsilon as f32, theta_fp32: new_theta };
            let r_minus = DecodedParams { r_fp32: new_r - epsilon as f32, theta_fp32: new_theta };
            
            let seed_plus = Packed128::from_continuous(&r_plus);
            let seed_minus = Packed128::from_continuous(&r_minus);
            
            let restored_plus = self.restore_from_seed(&seed_plus, rows, cols);
            let restored_minus = self.restore_from_seed(&seed_minus, rows, cols);
            
            let rmse_plus = self.calculate_rmse(target, &restored_plus);
            let rmse_minus = self.calculate_rmse(target, &restored_minus);
            
            let grad_r = (rmse_plus - rmse_minus) / (2.0 * epsilon);
            
            // theta 그래디언트
            let theta_plus = DecodedParams { r_fp32: new_r, theta_fp32: new_theta + epsilon as f32 };
            let theta_minus = DecodedParams { r_fp32: new_r, theta_fp32: new_theta - epsilon as f32 };
            
            let seed_theta_plus = Packed128::from_continuous(&theta_plus);
            let seed_theta_minus = Packed128::from_continuous(&theta_minus);
            
            let restored_theta_plus = self.restore_from_seed(&seed_theta_plus, rows, cols);
            let restored_theta_minus = self.restore_from_seed(&seed_theta_minus, rows, cols);
            
            let rmse_theta_plus = self.calculate_rmse(target, &restored_theta_plus);
            let rmse_theta_minus = self.calculate_rmse(target, &restored_theta_minus);
            
            let grad_theta = (rmse_theta_plus - rmse_theta_minus) / (2.0 * epsilon);
            
            // 그래디언트 업데이트
            new_r -= (learning_rate * grad_r) as f32;
            new_theta -= (learning_rate * grad_theta) as f32;
            
            // 경계 조건
            new_r = new_r.clamp(0.0, 0.99);
            new_theta = new_theta.rem_euclid(2.0 * std::f32::consts::PI);
            
            let new_params = DecodedParams { r_fp32: new_r, theta_fp32: new_theta };
            let candidate_seed = Packed128::from_continuous(&new_params);
            
            let restored = self.restore_from_seed(&candidate_seed, rows, cols);
            let new_rmse = self.calculate_rmse(target, &restored);
            
            if new_rmse < best_rmse {
                best_rmse = new_rmse;
                seed = candidate_seed;
                
                if iter % 10 == 0 {
                    println!("미세조정 {}: RMSE {:.6}", iter, best_rmse);
                }
            }
        }
        
        Ok(seed)
    }
    
    /// 시드로부터 가중치 복원
    fn restore_from_seed(&self, seed: &Packed128, rows: usize, cols: usize) -> Vec<f32> {
        let mut restored = Vec::with_capacity(rows * cols);
        
        for i in 0..rows {
            for j in 0..cols {
                let weight = seed.fused_forward(i, j, rows, cols);
                restored.push(weight);
            }
        }
        
        restored
    }
    
    /// 빠른 샘플 기반 평가
    fn quick_evaluate_seed(&self, seed: &Packed128, target: &[f32], rows: usize, cols: usize, sample_ratio: f64) -> f64 {
        let total_elements = rows * cols;
        let sample_count = ((total_elements as f64 * sample_ratio) as usize).max(10);
        
        let mut total_error = 0.0f64;
        let step = total_elements / sample_count;
        
        for i in (0..total_elements).step_by(step).take(sample_count) {
            let row = i / cols;
            let col = i % cols;
            let predicted = seed.fused_forward(row, col, rows, cols);
            let actual = target[i];
            let diff = (predicted as f64) - (actual as f64);
            total_error += diff * diff;
        }
        
        (total_error / sample_count as f64).sqrt()
    }
    
    /// RMSE 계산
    fn calculate_rmse(&self, original: &[f32], restored: &[f32]) -> f64 {
        if original.len() != restored.len() {
            return f64::INFINITY;
        }
        
        let mse: f64 = original.iter()
            .zip(restored.iter())
            .map(|(a, b)| {
                let diff = (*a as f64) - (*b as f64);
                diff * diff
            })
            .sum::<f64>() / original.len() as f64;
        
        mse.sqrt()
    }
} 
use crate::types::{PoincareMatrix, Packed64};
use crate::math::{analyze_global_pattern, suggest_basis_functions,compute_sampled_rmse, local_search_exhaustive, compute_full_rmse};
use rand::{Rng, seq::SliceRandom};

impl PoincareMatrix {
    /// FP32 행렬을 64비트 시드로 압축 (문서의 방식)
    pub fn compress(matrix: &[f32], rows: usize, cols: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // 행렬에서 가장 큰 값의 위치를 찾아 패턴 추정
        let mut max_val = 0.0;
        let mut max_i = 0;
        let mut max_j = 0;
        
        for i in 0..rows {
            for j in 0..cols {
                let val = matrix[i * cols + j].abs();
                if val > max_val {
                    max_val = val;
                    max_i = i;
                    max_j = j;
                }
            }
        }
        
        // 최대값 위치에서 극좌표 추정
        let x = 2.0 * (max_j as f32) / ((cols - 1) as f32) - 1.0;
        let y = 2.0 * (max_i as f32) / ((rows - 1) as f32) - 1.0;
        let r = (x * x + y * y).sqrt().min(0.9);
        let theta = y.atan2(x).rem_euclid(2.0 * std::f32::consts::PI);
        
        // 랜덤 탐색으로 최적화
        let mut best_seed = Packed64::new(r, theta, 0, 0, false, 0, 0, 0);
        let mut min_error = f32::INFINITY;
        
        for _ in 0..10000 {
            let r_test = rng.gen_range(0.1..0.95);
            let theta_test = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
            let basis_id = rng.gen_range(0..4);
            let d_theta = rng.gen_range(0..4);
            let d_r = rng.gen::<bool>();
            let rot_code = rng.gen_range(0..10);
            let log2_c = rng.gen_range(-3..4);
            
            let test_seed = Packed64::new(
                r_test, theta_test, basis_id, d_theta, d_r, rot_code, log2_c, 0
            );
            
            let mut error = 0.0;
            for i in 0..rows.min(10) {  // 빠른 평가를 위해 일부만
                for j in 0..cols.min(10) {
                    let original = matrix[i * cols + j];
                    let reconstructed = test_seed.compute_weight(i, j, rows, cols);
                    error += (original - reconstructed).powi(2);
                }
            }
            
            if error < min_error {
                min_error = error;
                best_seed = test_seed;
            }
        }
        
        PoincareMatrix {
            seed: best_seed,
            rows,
            cols,
        }
    }
    
    /// 시드로부터 행렬 복원
    pub fn decompress(&self) -> Vec<f32> {
        let mut matrix = vec![0.0; self.rows * self.cols];
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                matrix[i * self.cols + j] = self.seed.compute_weight(i, j, self.rows, self.cols);
            }
        }
        
        matrix
    }

    pub fn deep_compress(matrix: &[f32], rows: usize, cols: usize) -> Self {
        println!("Deep compression starting... (this may take a while)");
        // Stage 1: Global pattern analysis
        let pattern_features = analyze_global_pattern(matrix);
        let candidate_bases = suggest_basis_functions(&pattern_features);
        // Stage 2: Generate candidates with GA
        let mut rng = rand::thread_rng();
        let mut population: Vec<(Packed64, f32)> = (0..100).map(|_| {
            let r = rng.gen_range(0.1..0.95);
            let theta = rng.gen_range(0.0..2.0*std::f32::consts::PI);
            let basis_id = *candidate_bases.choose(&mut rng).unwrap_or(&0);
            let d_theta = rng.gen_range(0..4);
            let d_r = rng.gen::<bool>();
            let rot_code = rng.gen_range(0..16);
            let log2_c = rng.gen_range(-4..=3);
            let seed = Packed64::new(r, theta, basis_id, d_theta, d_r, rot_code, log2_c, 0);
            let rmse = compute_sampled_rmse(matrix, seed, rows, cols);
            (seed, rmse)
        }).collect();
        for _ in 0..10 {  // Generations
            population.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
            let elite = population[0..10].to_vec();
            let mut new_pop = elite.clone();
            while new_pop.len() < 100 {
                let parent1 = &elite[rng.gen_range(0..elite.len())].0;
                let parent2 = &elite[rng.gen_range(0..elite.len())].0;
                let child = crossover(parent1, parent2, &mut rng);
                let mutated = mutate(child, &mut rng);
                let rmse = compute_sampled_rmse(matrix, mutated, rows, cols);
                new_pop.push((mutated, rmse));
            }
            population = new_pop;
        }
        population.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap());
        let top_10 = &population[0..10];
        // Stage 3: Fine optimization
        let mut best_seed = top_10[0].0;
        let mut best_rmse = f32::INFINITY;
        for (seed, _) in top_10 {
            let optimized = local_search_exhaustive(*seed, matrix, rows, cols);
            let full_rmse = compute_full_rmse(matrix, optimized, rows, cols);
            if full_rmse < best_rmse {
                best_rmse = full_rmse;
                best_seed = optimized;
                println!("New best: RMSE = {:.6}", best_rmse);
            }
        }
        PoincareMatrix { seed: best_seed, rows, cols }
    }
} 

fn crossover(p1: &Packed64, p2: &Packed64, rng: &mut impl Rng) -> Packed64 {
    let params1 = p1.decode();
    let params2 = p2.decode();
    Packed64::new(
        if rng.gen() {params1.r} else {params2.r},
        if rng.gen() {params1.theta} else {params2.theta},
        if rng.gen() {params1.basis_id} else {params2.basis_id},
        if rng.gen() {params1.d_theta} else {params2.d_theta},
        if rng.gen() {params1.d_r} else {params2.d_r},
        if rng.gen() {params1.rot_code} else {params2.rot_code},
        if rng.gen() {params1.log2_c} else {params2.log2_c},
        0
    )
}

fn mutate(seed: Packed64, rng: &mut impl Rng) -> Packed64 {
    let mut params = seed.decode();
    if rng.gen_ratio(1,5) { params.r = (params.r + rng.gen_range(-0.05..0.05)).clamp(0.0,0.999); }
    if rng.gen_ratio(1,5) { params.theta = (params.theta + rng.gen_range(-0.1..0.1)).rem_euclid(2.0*std::f32::consts::PI); }
    Packed64::new(params.r, params.theta, params.basis_id, params.d_theta, params.d_r, params.rot_code, params.log2_c, params.reserved)
} 
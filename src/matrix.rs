use crate::types::{PoincareMatrix, Packed64};
use crate::math::compute_full_rmse;
use rayon::prelude::*;
use rand::Rng;

impl PoincareMatrix {
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

    /// 유전 알고리즘을 사용한 압축
    pub fn compress_with_genetic_algorithm(
        matrix: &[f32],
        rows: usize,
        cols: usize,
        population_size: usize,
        generations: usize,
        mutation_rate: f32,
    ) -> Self {
        let mut rng = rand::thread_rng();

        // 1. 초기 집단 생성
        let mut population: Vec<Packed64> = (0..population_size)
            .map(|_| {
                let r: f32 = rng.gen();
                let theta: f32 = rng.gen_range(0.0..2.0 * std::f32::consts::PI);
                let basis_id: u8 = rng.gen_range(0..13);
                let freq_x: u8 = rng.gen_range(1..32);
                let freq_y: u8 = rng.gen_range(1..32);
                let amplitude: f32 = rng.gen_range(0.25..4.0);
                let offset: f32 = rng.gen_range(-2.0..2.0);
                let pattern_mix: u8 = rng.gen_range(0..16);
                let decay_rate: f32 = rng.gen_range(0.0..4.0);
                let d_theta: u8 = rng.gen_range(0..4);
                let d_r: bool = rng.gen();
                let log2_c: i8 = rng.gen_range(-2..=1);
                Packed64::new(r, theta, basis_id, freq_x, freq_y, amplitude, offset, pattern_mix, decay_rate, d_theta, d_r, log2_c)
            })
            .collect();

        let mut best_overall_seed = population[0];
        let mut best_overall_rmse = f32::INFINITY;
        let mut stagnation_counter = 0;
        let mut current_mutation_rate = mutation_rate;

        // 2. 세대 반복
        for gen in 0..generations {
            let mut fitness_scores: Vec<(f32, Packed64)> = population
                .par_iter()
                .map(|seed| (compute_full_rmse(matrix, *seed, rows, cols), *seed))
                .collect();
            
            fitness_scores.par_sort_by(|a, b| a.0.total_cmp(&b.0));

            if fitness_scores.is_empty() {
                // 집단이 비어있으면 다음 세대로 넘어감
                population = Vec::new(); // Or handle appropriately
                continue;
            }

            if fitness_scores[0].0 < best_overall_rmse {
                best_overall_rmse = fitness_scores[0].0;
                best_overall_seed = fitness_scores[0].1;
                println!("Gen {}: New best RMSE = {:.6} (mutation_rate: {})", gen, best_overall_rmse, current_mutation_rate);
                stagnation_counter = 0; // 개선되었으므로 카운터 리셋
                current_mutation_rate = mutation_rate; // 돌연변이율 초기화
            } else {
                stagnation_counter += 1;
            }

            // 10세대 동안 개선이 없으면 돌연변이율을 5배 높여 탐색 공간 확장
            if stagnation_counter > 10 {
                current_mutation_rate = mutation_rate * 5.0;
                stagnation_counter = 0; // 다시 탐색 시작
            }
            
            let sorted_population: Vec<Packed64> = fitness_scores
                .into_iter()
                .map(|(_, seed)| seed)
                .collect();
            
            let mut next_generation = Vec::with_capacity(population_size);
            let elite_count = (population_size as f32 * 0.1).ceil() as usize;
            next_generation.extend_from_slice(&sorted_population[..elite_count]);
            
            while next_generation.len() < population_size {
                let parent1 = selection(&sorted_population, &mut rng);
                let parent2 = selection(&sorted_population, &mut rng);
                let (child1, child2) = crossover(parent1, parent2, &mut rng);
                next_generation.push(child1);
                if next_generation.len() < population_size {
                    next_generation.push(child2);
                }
            }

            for individual in next_generation.iter_mut().skip(elite_count) {
                mutate(individual, current_mutation_rate, &mut rng);
            }

            population = next_generation;
        }

        PoincareMatrix {
            seed: best_overall_seed,
            rows,
            cols,
        }
    }
}

// 토너먼트 선택 (적합도가 높은, 즉 인덱스가 낮은 개체를 선택)
fn selection<'a, R: Rng>(population: &'a [Packed64], rng: &mut R) -> &'a Packed64 {
    let mut best_index = rng.gen_range(0..population.len());
    for _ in 1..5 { // Tournament size = 5
        let current_index = rng.gen_range(0..population.len());
        if current_index < best_index {
            best_index = current_index;
        }
    }
    &population[best_index]
}

// 2점 교차
fn crossover<R: Rng>(parent1: &Packed64, parent2: &Packed64, rng: &mut R) -> (Packed64, Packed64) {
    let p1_bits = parent1.0;
    let p2_bits = parent2.0;

    let mut cross_points = [rng.gen_range(1..63), rng.gen_range(1..63)];
    cross_points.sort_unstable();
    let [cp1, cp2] = cross_points;

    let mask = (1u64 << (cp2 - cp1)) - 1;
    let mask = mask << cp1;

    let child1_bits = (p1_bits & !mask) | (p2_bits & mask);
    let child2_bits = (p2_bits & !mask) | (p1_bits & mask);

    (Packed64(child1_bits), Packed64(child2_bits))
}

// 비트 플립 및 실수 값 노이즈 기반 돌연변이
fn mutate<R: Rng>(individual: &mut Packed64, mutation_rate: f32, rng: &mut R) {
    // 1. 기존의 비트 플립 돌연변이
    for i in 0..64 {
        if rng.gen::<f32>() < mutation_rate {
            individual.0 ^= 1 << i;
        }
    }

    // 2. 실수 공간에서의 미세 조정 돌연변이 (더 높은 확률로 적용)
    if rng.gen::<f32>() < 0.1 { // 10% 확률로 미세 조정
        let mut params = individual.decode();
        let choice = rng.gen_range(0..5);
        match choice {
            0 => params.r = (params.r + rng.gen_range(-0.05..0.05)).clamp(0.0, 0.999),
            1 => params.theta = (params.theta + rng.gen_range(-0.1..0.1)).rem_euclid(2.0 * std::f32::consts::PI),
            2 => params.amplitude = (params.amplitude + rng.gen_range(-0.2..0.2)).clamp(0.25, 4.0),
            3 => params.offset = (params.offset + rng.gen_range(-0.2..0.2)).clamp(-2.0, 2.0),
            4 => params.decay_rate = (params.decay_rate + rng.gen_range(-0.2..0.2)).clamp(0.0, 4.0),
            _ => {},
        }
        *individual = Packed64::from_params(&params);
    }
} 
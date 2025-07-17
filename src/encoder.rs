//! 행렬 압축(인코딩) 관련 기능
//! 
//! 이 모듈은 행렬을 128비트로 압축하는 기능을 제공합니다.

use crate::types::{Packed64, Packed128, PoincareMatrix};
use crate::math::compute_full_rmse;
use rand::Rng;

impl PoincareMatrix {
    /// 행렬을 128비트로 압축
    /// 
    /// # Arguments
    /// * `matrix` - 압축할 행렬 데이터
    /// * `rows` - 행 수
    /// * `cols` - 열 수
    /// 
    /// # Returns
    /// 압축된 PoincareMatrix
    pub fn compress(matrix: &[f32], rows: usize, cols: usize) -> Self {
        // 간단한 brute-force 탐색
        let mut best_seed = 0u64;
        let mut best_rmse = f32::INFINITY;
        
        let mut rng = rand::thread_rng();
        
        // 1000번 랜덤 시도
        for _ in 0..1000 {
            let seed = rng.gen::<u64>();
            let rmse = compute_full_rmse(matrix, &Packed64 { rotations: seed }, rows, cols);
            
            if rmse < best_rmse {
                best_rmse = rmse;
                best_seed = seed;
            }
        }
        
        println!("[Compress] Best Seed: 0x{:X}, RMSE: {}", best_seed, best_rmse);
        
        PoincareMatrix {
            seed: Packed128 { hi: best_seed, lo: 0 },
            rows,
            cols,
        }
    }
    
    /// 더 정교한 압축 알고리즘 (미래 확장용)
    /// 
    /// 현재는 기본 compress와 동일하지만, 
    /// 추후 유전 알고리즘이나 gradient descent 등을 적용할 수 있습니다.
    pub fn compress_advanced(matrix: &[f32], rows: usize, cols: usize) -> Self {
        // TODO: 유전 알고리즘 구현
        // TODO: Gradient-based 최적화
        // TODO: Multi-scale 접근
        Self::compress(matrix, rows, cols)
    }
    
    /// 역 CORDIC 알고리즘을 사용한 정교한 압축
    pub fn compress_cordic(matrix: &[f32], rows: usize, cols: usize) -> Self {
        let key_points = extract_key_points(matrix, rows, cols);
        let mut best_seed = Packed64::new(0);
        let mut best_rmse = f32::INFINITY;

        for point in key_points {
            let candidate_seed = find_seed_for_point(point, rows, cols);
            let rmse = compute_full_rmse(matrix, &candidate_seed, rows, cols);

            if rmse < best_rmse {
                best_rmse = rmse;
                best_seed = candidate_seed;
            }
        }

        println!("[Compress CORDIC] Best Seed: 0x{:X}, RMSE: {:.6}", best_seed.rotations, best_rmse);
        PoincareMatrix { seed: Packed128 { hi: best_seed.rotations, lo: 0 }, rows, cols }
    }
    
    /// 그리드 기반 압축 - 큰 행렬을 작은 블록으로 나누어 압축
    /// 
    /// # Arguments
    /// * `matrix` - 압축할 행렬 데이터
    /// * `rows` - 전체 행 수
    /// * `cols` - 전체 열 수
    /// * `block_size` - 각 블록의 크기 (예: 32면 32x32 블록)
    /// 
    /// # Returns
    /// 그리드 압축된 결과
    pub fn compress_grid(matrix: &[f32], rows: usize, cols: usize, block_size: usize) -> GridCompressedMatrix {
        let grid_rows = (rows + block_size - 1) / block_size;
        let grid_cols = (cols + block_size - 1) / block_size;
        let mut blocks = Vec::with_capacity(grid_rows * grid_cols);
        
        for grid_i in 0..grid_rows {
            for grid_j in 0..grid_cols {
                // 블록의 시작과 끝 계산
                let start_i = grid_i * block_size;
                let start_j = grid_j * block_size;
                let end_i = (start_i + block_size).min(rows);
                let end_j = (start_j + block_size).min(cols);
                
                let block_rows = end_i - start_i;
                let block_cols = end_j - start_j;
                
                // 블록 데이터 추출
                let mut block_data = Vec::with_capacity(block_rows * block_cols);
                for i in start_i..end_i {
                    for j in start_j..end_j {
                        block_data.push(matrix[i * cols + j]);
                    }
                }
                
                // 블록 압축
                let compressed_block = Self::compress(&block_data, block_rows, block_cols);
                blocks.push(compressed_block);
            }
        }
        
        GridCompressedMatrix {
            blocks,
            grid_rows,
            grid_cols,
            block_size,
            total_rows: rows,
            total_cols: cols,
        }
    }
}

/// CORDIC 기반으로 최적의 회전 시퀀스를 찾습니다
fn find_seed_for_point(point: (usize, usize, f32), rows: usize, cols: usize) -> Packed64 {
    let (i, j, target_value) = point;
    let mut rotations = 0u64;

    // 초기 벡터 설정
    let mut x = (j as f32 / (cols.saturating_sub(1)) as f32) * 2.0 - 1.0;
    let mut y = (i as f32 / (rows.saturating_sub(1)) as f32) * 2.0 - 1.0;

    // 최종 목표 벡터를 (target_value, 0)으로 가정
    let target_angle = 0.0f32.atan2(target_value);

    for k in 0..64 {
        let power_of_2 = (2.0f32).powi(-(k as i32));
        
        // 현재 벡터의 각도
        let current_angle = y.atan2(x);
        
        // 목표 각도까지 남은 차이
        let angle_diff = target_angle - current_angle;

        // CORDIC 회전 각도 (arctan(2^-k))
        let cordic_angle = power_of_2.atan();

        // 각도 차이를 줄이는 방향으로 회전
        let sigma = -angle_diff.signum();

        if sigma > 0.0 {
             rotations |= 1 << k;
        }

        let x_new = x - sigma * y * power_of_2;
        let y_new = y + sigma * x * power_of_2;
        x = x_new;
        y = y_new;

        if k % 4 == 0 {
            let r = (x * x + y * y).sqrt();
            if r > 1e-9 {
                let tanh_r = r.tanh();
                x *= tanh_r;
                y *= tanh_r;
            }
        }
    }

    Packed64::new(rotations)
}

/// 행렬에서 분석할 주요 특징점을 추출합니다
fn extract_key_points(matrix: &[f32], rows: usize, cols: usize) -> Vec<(usize, usize, f32)> {
    let mut points = Vec::with_capacity(5);
    points.push((0, 0, matrix[0]));
    points.push((0, cols - 1, matrix[cols - 1]));
    points.push((rows - 1, 0, matrix[(rows - 1) * cols]));
    points.push((rows - 1, cols - 1, matrix[rows * cols - 1]));
    points.push((rows / 2, cols / 2, matrix[rows / 2 * cols + cols / 2]));
    points
} 

/// 그리드로 압축된 행렬
pub struct GridCompressedMatrix {
    pub blocks: Vec<PoincareMatrix>,
    pub grid_rows: usize,
    pub grid_cols: usize,
    pub block_size: usize,
    pub total_rows: usize,
    pub total_cols: usize,
}

impl GridCompressedMatrix {
    /// 압축률 계산
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.total_rows * self.total_cols * 4; // f32 = 4 bytes
        let compressed_size = self.blocks.len() * 16; // 각 블록 = 128 bits = 16 bytes
        original_size as f32 / compressed_size as f32
    }
    
    /// 메타데이터 포함 압축률
    pub fn effective_compression_ratio(&self) -> f32 {
        let original_size = self.total_rows * self.total_cols * 4;
        // 블록 데이터 + 메타데이터 (grid info)
        let compressed_size = self.blocks.len() * 16 + 24; // 24 bytes for metadata
        original_size as f32 / compressed_size as f32
    }
} 
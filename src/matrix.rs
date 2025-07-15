use crate::math::calculate_rmse;
use crate::types::{Packed64, PoincareMatrix};

impl PoincareMatrix {
    /// 주어진 시드로부터 행렬을 복원(생성)합니다.
    pub fn decompress(&self) -> Vec<f32> {
        let mut matrix = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                matrix[i * self.cols + j] = self.seed.compute_weight(i, j, self.rows, self.cols);
            }
        }
        matrix
    }

    /// 역 CORDIC 알고리즘을 사용하여 행렬을 압축하고 최적의 시드를 찾습니다.
    pub fn compress(matrix: &[f32], rows: usize, cols: usize) -> Self {
        let key_points = extract_key_points(matrix, rows, cols);
        let mut best_seed = Packed64::new(0);
        let mut best_rmse = f32::INFINITY;

        for point in key_points {
            let candidate_seed = find_seed_for_point(point, rows, cols);
            let rmse = calculate_rmse(matrix, &candidate_seed, rows, cols);

            if rmse < best_rmse {
                best_rmse = rmse;
                best_seed = candidate_seed;
            }
        }

        println!("[Compress] Best Seed: 0x{:X}, RMSE: {:.6}", best_seed.rotations, best_rmse);
        PoincareMatrix { seed: best_seed, rows, cols }
    }
}

/// CORDIC.md의 아이디어를 기반으로, 각도 차이를 최소화하는 그리디 방식으로 최적의 회전 시퀀스를 찾습니다.
fn find_seed_for_point(point: (usize, usize, f32), rows: usize, cols: usize) -> Packed64 {
    let (i, j, target_value) = point;
    let mut rotations = 0u64;

    // 초기 벡터 설정
    let mut x = (j as f32 / (cols.saturating_sub(1)) as f32) * 2.0 - 1.0;
    let mut y = (i as f32 / (rows.saturating_sub(1)) as f32) * 2.0 - 1.0;

    // 최종 목표 벡터를 (target_value, 0)으로 가정합니다.
    // 이는 최종 x값이 target_value가 되기를 바라는 휴리스틱입니다.
    let target_angle = 0.0f32.atan2(target_value);

    for k in 0..64 {
        let power_of_2 = (2.0f32).powi(-(k as i32));
        
        // 현재 벡터의 각도
        let current_angle = y.atan2(x);
        
        // 목표 각도까지 남은 차이
        let angle_diff = target_angle - current_angle;

        // CORDIC 회전 각도 (arctan(2^-k))
        let cordic_angle = power_of_2.atan();

        // 각도 차이를 줄이는 방향으로 회전합니다.
        let sigma = if angle_diff.abs() < cordic_angle {
            // 이미 목표 각도에 가깝다면 더 회전하지 않습니다.
            // 하지만 이럴 경우 남은 비트가 모두 0이 되므로,
            // angle_diff의 부호에 따라 약간의 조정을 계속합니다.
            -angle_diff.signum()
        } else {
             // 차이가 충분히 크면, 차이를 줄이는 방향으로 회전
            -angle_diff.signum()
        };

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

/// 행렬에서 분석할 주요 특징점을 추출합니다.
fn extract_key_points(matrix: &[f32], rows: usize, cols: usize) -> Vec<(usize, usize, f32)> {
    let mut points = Vec::with_capacity(5);
    points.push((0, 0, matrix[0]));
    points.push((0, cols - 1, matrix[cols - 1]));
    points.push((rows - 1, 0, matrix[(rows - 1) * cols]));
    points.push((rows - 1, cols - 1, matrix[rows * cols - 1]));
    points.push((rows / 2, cols / 2, matrix[rows / 2 * cols + cols / 2]));
    points
} 
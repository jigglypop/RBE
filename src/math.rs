use crate::types::Packed64;
use rand::Rng;

/// RMSE (Root Mean Square Error)를 계산합니다.
///
/// # Arguments
/// * `matrix` - 원본 행렬 데이터
/// * `seed` - 재구성에 사용할 Packed64 시드
/// * `rows` - 행렬의 행 수
/// * `cols` - 행렬의 열 수
pub fn calculate_rmse(matrix: &[f32], seed: &Packed64, rows: usize, cols: usize) -> f32 {
    let mut error = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let original_value = matrix[i * cols + j];
            let reconstructed_value = seed.compute_weight(i, j, rows, cols);
            error += (original_value - reconstructed_value).powi(2);
        }
    }
    (error / (rows * cols) as f32).sqrt()
}

/// 유전 알고리즘의 변이(mutation) 연산을 수행합니다.
///
/// 주어진 시드의 각 비트를 `mutation_rate` 확률로 뒤집습니다.
///
/// # Arguments
/// * `seed` - 변이를 적용할 원본 Packed64 시드
/// * `mutation_rate` - 각 비트가 변이될 확률 (0.0 ~ 1.0)
///
/// # Returns
/// * 변이된 새로운 Packed64 시드
pub fn mutate_seed(seed: Packed64, mutation_rate: f32) -> Packed64 {
    let mut rng = rand::thread_rng();
    let mut new_rotations = seed.rotations;
    for i in 0..64 {
        if rng.gen_bool(mutation_rate as f64) {
            new_rotations ^= 1 << i;
        }
    }
    Packed64::new(new_rotations)
}

// CORDIC 구현에 필요할 수 있는 수학 함수들은 유지합니다.

pub fn bessel_j0(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 
            + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 
            + y * (59272.64853 + y * (267.8532712 + y))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164; // PI/4
        let ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 
            + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 
            + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
        (2.0 / (std::f32::consts::PI * ax)).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2)
    }
}

// CORDIC 특수 함수 헬퍼 (CORDIC.md 기반)
// 이 함수들은 현재 compute_weight에서 직접 사용되지는 않지만,
// 향후 CORDIC 알고리즘 확장을 위해 남겨둡니다.
pub fn apply_bessel_cordic(result: (f32, f32), special: u16) -> f32 {
    let r = (result.0 * result.0 + result.1 * result.1).sqrt();
    bessel_j0(r * (special as f32 / 63.0))
}

pub fn apply_elliptic_cordic(result: (f32, f32), special: u16) -> f32 {
    let angle = result.1.atan2(result.0);
    (angle * (special as f32 / 63.0 - 1.0)).tanh()
}

pub fn apply_theta_cordic(result: (f32, f32), special: u16) -> f32 {
    let theta = result.1.atan2(result.0);
    theta * (special as f32 / 63.0)
} 

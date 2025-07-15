use std::f32::consts::PI;
use crate::types::Packed64;
use noise::{NoiseFn, Perlin};

/// 회전 각도 계산
pub fn get_rotation_angle(rot_code: u8) -> f32 {
    match rot_code {
        0 => 0.0,
        1 => PI / 8.0,
        2 => PI / 6.0,
        3 => PI / 4.0,
        4 => PI / 3.0,
        5 => PI / 2.0,
        6 => 2.0 * PI / 3.0,
        7 => 3.0 * PI / 4.0,
        8 => 5.0 * PI / 6.0,
        9 => 7.0 * PI / 8.0,
        _ => 0.0,
    }
}

/// 각도 미분 적용
pub fn apply_angular_derivative(theta: f32, d_theta: u8, basis_id: u8) -> f32 {
    let is_sin_based = (basis_id & 0x1) == 0;
    
    match (is_sin_based, d_theta % 4) {
        (true, 0) => theta.sin(),
        (true, 1) => theta.cos(),
        (true, 2) => -theta.sin(),
        (true, 3) => -theta.cos(),
        (false, 0) => theta.cos(),
        (false, 1) => -theta.sin(),
        (false, 2) => -theta.cos(),
        (false, 3) => theta.sin(),
        _ => unreachable!(),
    }
}

/// 반지름 미분 적용
pub fn apply_radial_derivative(r: f32, d_r: bool, basis_id: u8) -> f32 {
    let is_sinh_based = (basis_id & 0x2) == 0;
    
    match (is_sinh_based, d_r) {
        (true, false) => r.sinh(),
        (true, true) => r.cosh(),
        (false, false) => r.cosh(),
        (false, true) => r.sinh(),
    }
}

// 베셀 함수들
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
        let xx = ax - 0.785398164;
        let ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 
            + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 
            + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
        (2.0 / (PI * ax)).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2)
    }
}

pub fn bessel_i0(x: f32) -> f32 {
    if x.abs() < 3.75 {
        let t = (x / 3.75).powi(2);
        let mut result = 1.0;
        let mut term = 1.0;
        
        for k in 1..=10 {
            term *= t / (k * k) as f32;
            result += term;
        }
        result
    } else {
        let t = 3.75 / x.abs();
        let mut result = 0.39894228;
        result += 0.01328592 * t;
        result += 0.00225319 * t * t;
        result -= 0.00157565 * t.powi(3);
        result * x.exp() / x.sqrt()
    }
}

pub fn bessel_k0(x: f32) -> f32 {
    if x < 2.0 {
        let i0 = bessel_i0(x);
        -x.ln() * i0 + 0.5772156649
    } else {
        let mut result = 1.2533141;
        result -= 0.07832358 * (2.0 / x);
        result += 0.02189568 * (2.0 / x).powi(2);
        result * (-x).exp() / x.sqrt()
    }
}

pub fn bessel_y0(x: f32) -> f32 {
    let j0 = bessel_j0(x);
    2.0 / PI * (x.ln() * j0 + 0.07832358)
}

pub fn sech(x: f32) -> f32 {
    2.0 / (x.exp() + (-x).exp())
}

pub fn triangle_wave(x: f32) -> f32 {
    let phase = x / PI;
    let t = phase - phase.floor();
    if t < 0.5 {
        4.0 * t - 1.0
    } else {
        3.0 - 4.0 * t
    }
}

pub fn morlet_wavelet(r: f32, theta: f32, freq: f32) -> f32 {
    let sigma = 1.0 / freq.sqrt();
    let gaussian = (-0.5 * (r / sigma).powi(2)).exp();
    let oscillation = (freq * theta).cos();
    gaussian * oscillation
} 

use rustfft::{FftPlanner, num_complex::Complex};

// Global pattern analysis with FFT and stats
pub fn analyze_global_pattern(matrix: &[f32]) -> Vec<f32> {
    let mut spectrum: Vec<Complex<f32>> = matrix.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
    let fft = FftPlanner::new().plan_fft_forward(matrix.len());
    fft.process(&mut spectrum);
    
    let mean = matrix.iter().sum::<f32>() / matrix.len() as f32;
    let variance = matrix.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / matrix.len() as f32;
    
    let mut max_mag = 0.0;
    let mut peak_freq = 0.0;
    for &val in &spectrum {
        let mag = val.norm();
        if mag > max_mag {
            max_mag = mag;
            peak_freq = mag / spectrum.len() as f32;
        }
    }
    vec![mean, variance, peak_freq, max_mag]
}

// Suggest bases based on features
pub fn suggest_basis_functions(features: &Vec<f32>) -> Vec<u8> {
    let mut suggestions = Vec::new();
    if features[2] > 0.1 { // High frequency -> periodic
        suggestions.extend(0..=3);
    }
    if features[1] > 0.5 { // High variance -> bessel
        suggestions.extend(4..=7);
    }
    suggestions.extend(8..=11); // Always include special
    suggestions
}

pub fn optimize_for_periodic(matrix: &[f32], _basis_id: u8, _rows: usize, cols: usize) -> (f32, f32) {
    let mut spectrum: Vec<Complex<f32>> = matrix.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();
    let fft = FftPlanner::new().plan_fft_forward(matrix.len());
    fft.process(&mut spectrum);
    let mut max_mag = 0.0;
    let mut peak_idx = 0;
    for (i, &val) in spectrum.iter().enumerate() {
        let mag = val.norm();
        if mag > max_mag {
            max_mag = mag;
            peak_idx = i;
        }
    }
    let kx = (peak_idx % cols) as f32;
    let ky = (peak_idx / cols) as f32;
    let r = (kx * kx + ky * ky).sqrt() / matrix.len() as f32;
    let theta = ky.atan2(kx);
    (r.min(0.99), theta)
}

// Optimize for Bessel: find r, theta that match radial decay
pub fn optimize_for_bessel(matrix: &[f32], basis_id: u8, rows: usize, cols: usize) -> (f32, f32) {
    let mut best_r = 0.5;
    let mut best_theta = 0.0;
    let mut min_rmse = f32::INFINITY;
    
    for r_trial in (1..99).step_by(5) {
        let r = r_trial as f32 / 100.0;
        for theta_trial in (0..360).step_by(10) {
            let theta = theta_trial as f32 * PI / 180.0;
            // 새로운 시그니처: 기본값으로 설정
            let seed = Packed64::new(r, theta, basis_id, 1, 1, 1.0, 0.0, 0, 0.0, 0, false, 0);
            let rmse = compute_sampled_rmse(matrix, seed, rows, cols);
            if rmse < min_rmse {
                min_rmse = rmse;
                best_r = r;
                best_theta = theta;
            }
        }
    }
    (best_r, best_theta)
}

// Similar for special functions
pub fn optimize_for_special(matrix: &[f32], basis_id: u8, rows: usize, cols: usize) -> (f32, f32) {
    let mut best_r = 0.5;
    let mut best_theta = 0.0;
    let mut min_rmse = f32::INFINITY;
    
    for r_trial in (1..99).step_by(5) {
        let r = r_trial as f32 / 100.0;
        for theta_trial in (0..360).step_by(10) {
            let theta = theta_trial as f32 * PI / 180.0;
            // 새로운 시그니처: 기본값으로 설정
            let seed = Packed64::new(r, theta, basis_id, 1, 1, 1.0, 0.0, 0, 0.0, 0, false, 0);
            let rmse = compute_sampled_rmse(matrix, seed, rows, cols);
            if rmse < min_rmse {
                min_rmse = rmse;
                best_r = r;
                best_theta = theta;
            }
        }
    }
    (best_r, best_theta)
}

pub fn compute_sampled_rmse(matrix: &[f32], seed: Packed64, rows: usize, cols: usize) -> f32 {
    let mut error = 0.0;
    let samples = 100;
    for _ in 0..samples {
        let i = rand::random::<usize>() % rows;
        let j = rand::random::<usize>() % cols;
        let original = matrix[i * cols + j];
        let reconstructed = seed.compute_weight(i, j, rows, cols);
        error += (original - reconstructed).powi(2);
    }
    (error / samples as f32).sqrt()
}

pub fn compute_full_rmse(matrix: &[f32], seed: Packed64, rows: usize, cols: usize) -> f32 {
    let mut error = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let original = matrix[i * cols + j];
            let reconstructed = seed.compute_weight(i, j, rows, cols);
            error += (original - reconstructed).powi(2);
        }
    }
    (error / (rows * cols) as f32).sqrt()
}

// Local exhaustive search around seed
pub fn local_search_exhaustive(seed: Packed64, matrix: &[f32], rows: usize, cols: usize) -> Packed64 {
    let params = seed.decode();
    let mut best_seed = seed;
    let mut best_rmse = compute_full_rmse(matrix, seed, rows, cols);
    
    // Fine-tune r and theta
    for dr in -5..=5 {
        let r_new = (params.r + dr as f32 * 0.01).clamp(0.01, 0.99);
        for dtheta in -5..=5 {
            let theta_new = (params.theta + dtheta as f32 * 0.1) % (2.0 * PI);
            let new_seed = Packed64::new(
                r_new, theta_new, params.basis_id, 
                params.freq_x, params.freq_y,
                params.amplitude, params.offset,
                params.pattern_mix, params.decay_rate,
                params.d_theta, params.d_r, params.log2_c
            );
            let rmse = compute_full_rmse(matrix, new_seed, rows, cols);
            if rmse < best_rmse {
                best_rmse = rmse;
                best_seed = new_seed;
            }
        }
    }
    
    // Try nearby discrete params
    for d_basis in -1..=1 {
        let basis_new = ((params.basis_id as i8 + d_basis).clamp(0, 11)) as u8;
        for d_dtheta in -1..=1 {
            let dtheta_new = ((params.d_theta as i8 + d_dtheta).clamp(0, 3)) as u8;
            let new_seed = Packed64::new(
                params.r, params.theta, basis_new,
                params.freq_x, params.freq_y,
                params.amplitude, params.offset,
                params.pattern_mix, params.decay_rate,
                dtheta_new, params.d_r, params.log2_c
            );
            let rmse = compute_full_rmse(matrix, new_seed, rows, cols);
            if rmse < best_rmse {
                best_rmse = rmse;
                best_seed = new_seed;
            }
        }
    }
    
    best_seed
} 

/// Reserved 비트 활용: scale 파라미터 (0-3)
pub fn get_scale_factor(scale_code: u8) -> f32 {
    match scale_code & 0x3 {
        0 => 1.0,
        1 => 1.5,
        2 => 2.0,
        3 => 0.5,
        _ => 1.0,
    }
}

/// Reserved 비트 활용: frequency modulation 파라미터 (0-3)
pub fn get_freq_modulation(freq_code: u8) -> f32 {
    match freq_code & 0x3 {
        0 => 1.0,
        1 => 1.25,
        2 => 1.5,
        3 => 0.75,
        _ => 1.0,
    }
}

/// Reserved 비트 활용: phase shift 파라미터 (0-3)
pub fn get_phase_shift(phase_code: u8) -> f32 {
    match phase_code & 0x3 {
        0 => 0.0,
        1 => PI / 4.0,
        2 => PI / 2.0,
        3 => 3.0 * PI / 4.0,
        _ => 0.0,
    }
} 

/// 2D 가우시안 함수
pub fn gaussian_2d(x: f32, y: f32, sigma: f32) -> f32 {
    let r_squared = x * x + y * y;
    (-r_squared / (2.0 * sigma * sigma)).exp()
}

/// 체커보드 패턴
pub fn checkerboard(x: f32, y: f32) -> f32 {
    let x_checker = (x * PI).sin().signum();
    let y_checker = (y * PI).sin().signum();
    x_checker * y_checker
}

/// 리플(물결) 패턴
pub fn ripple(r: f32, theta: f32) -> f32 {
    (r * 5.0).sin() * theta.cos()
}

/// 나선형 패턴
pub fn spiral(r: f32, theta: f32) -> f32 {
    ((r * 10.0 + theta).rem_euclid(2.0 * PI) / PI - 1.0).tanh()
}

/// Gabor 2D 필터
pub fn gabor_2d(x: f32, y: f32, freq_x: f32, freq_y: f32) -> f32 {
    let gaussian = gaussian_2d(x, y, 0.3);
    let sinusoid = (2.0 * PI * (x * freq_x + y * freq_y)).cos();
    gaussian * sinusoid
} 

/// 2D 펄린 노이즈
pub fn perlin_2d(x: f32, y: f32, freq: f32) -> f32 {
    let perlin = Perlin::new(1); // 시드 1로 고정하여 재현성 보장
    perlin.get([x as f64 * freq as f64, y as f64 * freq as f64]) as f32
} 
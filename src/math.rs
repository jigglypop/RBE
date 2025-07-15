use std::f32::consts::PI;

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
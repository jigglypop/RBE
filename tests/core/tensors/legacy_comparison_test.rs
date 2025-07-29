//! Legacy 수학 함수와 Enhanced128 구현 간의 1:1 비교 테스트

use rbe_llm::core::{
    tensors::Enhanced128,
    optimizers::adam::RBESeed,
};
use rand::SeedableRng;

// --- Legacy 수학 함수 구현 (src/legacy/src/rbe/math.rs 에서 복사) ---

mod legacy_math {
    use std::f32::consts::PI;

    pub fn bessel_j0(x: f32) -> f32 {
        let ax = x.abs();
        if ax < 8.0 {
            let y = x * x;
            let ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
            let ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))));
            (ans1 / ans2) as f32
        } else {
            let z = 8.0 / ax;
            let y = z * z;
            let xx = ax - 0.785398164;
            let ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
            let ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
            ((2.0 / (PI * ax)).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2)) as f32
        }
    }

    pub fn bessel_i0(x: f32) -> f32 {
        if x.abs() < 3.75 {
            let y = x / 3.75;
            let y2 = y * y;
            (1.0 + y2 * (3.5156229 + y2 * (3.0899424 + y2 * (1.2067492 + y2 * (0.2659732 + y2 * (0.0360768 + y2 * 0.0045813)))))) as f32
        } else {
            let ax = x.abs();
            let inv_ax = 1.0 / ax;
            (ax.exp() / (2.0 * PI * ax).sqrt()) * (0.39894228 + inv_ax * (0.01328592 + inv_ax * (0.00225319 + inv_ax * (-0.00157565 + inv_ax * (0.00916281 + inv_ax * (-0.02057706 + inv_ax * (0.02635537 + inv_ax * (-0.01647633 + inv_ax * 0.00392377)))))))) as f32
        }
    }

    pub fn bessel_k0(x: f32) -> f32 {
        if x <= 2.0 {
            let y = x * x / 4.0;
            (-x.ln() * bessel_i0(x)) + (-0.57721566 + y * (0.42278420 + y * (0.23069756 + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740)))))) as f32
        } else {
            let inv_x = 1.0 / x;
            (x.exp() * (PI / (2.0 * x)).sqrt()) * (1.25331414 + inv_x * (-0.07832358 + inv_x * (0.02189568 + inv_x * (-0.01062446 + inv_x * (0.00587872 + inv_x * (-0.00251540 + inv_x * 0.00053208)))))) as f32
        }
    }
    
    pub fn bessel_y0(x: f32) -> f32 {
        if x < 8.0 {
            let y = x * x;
            let ans1 = -2957821389.0 + y * (7062834065.0 + y * (-512359803.6 + y * (10879881.29 + y * (-86324.90036 + y * 228.4622733))));
            let ans2 = 40076544269.0 + y * (745249964.8 + y * (7189466.438 + y * (47447.26470 + y * (226.1030244 + y))));
            (bessel_j0(x) * (2.0 / PI) * x.ln()) + (ans1 / ans2) as f32
        } else {
            let z = 8.0 / x;
            let y = z * z;
            let xx = x - 0.785398164;
            let ans1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
            let ans2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934945152e-7)));
            ((2.0 / (PI * x)).sqrt() * (xx.sin() * ans1 + z * xx.cos() * ans2)) as f32
        }
    }

    pub fn sech(x: f32) -> f32 {
        1.0 / x.cosh()
    }
    
    pub fn triangle_wave(x: f32) -> f32 {
        4.0 / PI * (x.sin() - (3.0*x).sin()/9.0 + (5.0*x).sin()/25.0)
    }

    pub fn morlet_wavelet(r: f32, theta: f32, freq: f32) -> f32 {
        let sigma = 1.0 / freq.sqrt();
        let gaussian = (-0.5 * (r / sigma).powi(2)).exp();
        let oscillation = (freq * theta).cos();
        gaussian * oscillation
    }
}

// --- 테스트 스위트 ---

fn run_comparison_test(basis_id: u8, legacy_fn: fn(f32,f32) -> f32, enhanced_fn: fn(&Enhanced128, f32, f32) -> f32) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(basis_id as u64);
    let test_samples = 1000;
    let mut max_abs_error = 0.0;
    
    for _ in 0..test_samples {
        let seed = Enhanced128::random(&mut rng);
        let params = seed.decode();
        
        let r = params.r_fp32;
        let theta = params.theta_fp32;
        
        // Legacy 값 계산 (단순화된 기저 함수만)
        let legacy_value = legacy_fn(r, theta);

        // Enhanced128 값 계산
        let enhanced_value = enhanced_fn(&seed, r, theta);
        
        let error = (legacy_value - enhanced_value).abs();
        if error > max_abs_error {
            max_abs_error = error;
        }
    }
    
    println!("  기저함수 #{}: 최대 절대 오차 = {:.9}", basis_id, max_abs_error);
    assert!(max_abs_error < 1e-5, "기저함수 #{} 오차가 너무 큽니다!", basis_id);
}

#[test]
fn legacy와_enhanced128_기저함수_정밀비교_테스트() {
    println!("\nLegacy vs Enhanced128 기저함수 정밀 비교");
    
    // 테스트할 기저 함수 목록
    let tests: &[(u8, fn(f32,f32)->f32, fn(&Enhanced128, f32,f32)->f32)] = &[
        (4, |r, _| legacy_math::bessel_j0(r * 10.0), |s, _, _| s.basis_function_value(4)),
        (5, |r, _| legacy_math::bessel_i0(r * 10.0), |s, _, _| s.basis_function_value(5)),
        (6, |r, _| legacy_math::bessel_k0(r * 10.0), |s, _, _| s.basis_function_value(6)),
        (7, |r, _| legacy_math::bessel_y0(r * 10.0), |s, _, _| s.basis_function_value(7)),
        (9, |r, theta| legacy_math::sech(r) * legacy_math::triangle_wave(theta), |s, _, _| s.basis_function_value(9)),
        (10, |r, theta| (-r).exp() * theta.sin(), |s, _, _| s.basis_function_value(10)),
        (11, |r, theta| legacy_math::morlet_wavelet(r, theta, 5.0), |s, _, _| s.basis_function_value(11)),
    ];
    
    for &(id, legacy_fn, enhanced_fn) in tests {
        run_comparison_test(id, legacy_fn, enhanced_fn);
    }
} 
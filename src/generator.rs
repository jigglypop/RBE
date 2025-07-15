use crate::types::Packed64;
use crate::math::*;

impl Packed64 {
    /// 가중치 계산 - 향상된 패턴 생성
    pub fn compute_weight(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32 {
        let params = self.decode();
        
        // 곡률 계산
        let c = 2.0f32.powi(params.log2_c as i32);
        
        // 좌표를 [-1, 1] 범위로 정규화
        let x = 2.0 * (j as f32) / ((cols - 1) as f32) - 1.0;
        let y = 2.0 * (i as f32) / ((rows - 1) as f32) - 1.0;
        
        // 로컬 극좌표
        let r_local = (x * x + y * y).sqrt().min(0.999999);
        let theta_local = y.atan2(x);
        
        // X/Y 주파수를 적용한 좌표 변환
        let x_freq = x * params.freq_x as f32;
        let y_freq = y * params.freq_y as f32;
        
        // 주요 각도 계산 (주파수 적용)
        let theta_final = params.theta + theta_local * params.freq_x as f32;
        
        // 미분 순환성 적용
        let angular_value = apply_angular_derivative(theta_final, params.d_theta, params.basis_id);
        let radial_value = apply_radial_derivative(c * params.r, params.d_r, params.basis_id);
        
        // 첫 번째 패턴: 기존 기저 함수
        let pattern1 = match params.basis_id {
            0..=3 => angular_value * radial_value,
            4 => bessel_j0(r_local * 10.0 * params.freq_x as f32),
            5 => bessel_i0(r_local * 10.0 * params.freq_y as f32),
            6 => bessel_k0(r_local * 5.0),
            7 => bessel_y0(r_local * 5.0),
            8 => (c * r_local).tanh() * theta_final.cos().signum(),
            9 => sech(c * r_local) * triangle_wave(theta_final),
            10 => (-c * r_local * params.decay_rate).exp() * theta_final.sin(),
            11 => morlet_wavelet(r_local, theta_final, params.freq_x as f32),
            12 => perlin_2d(x_freq, y_freq, 5.0), // 펄린 노이즈 추가
            _ => 0.0,
        };
        
        // 두 번째 패턴: pattern_mix에 따른 추가 패턴
        let pattern2 = match params.pattern_mix {
            0 => 0.0,  // 믹싱 없음
            1 => (x_freq * std::f32::consts::PI).sin() * (y_freq * std::f32::consts::PI).cos(),
            2 => (x_freq + y_freq).sin(),
            3 => ((x_freq * x_freq + y_freq * y_freq).sqrt() * 2.0 * std::f32::consts::PI).sin(),
            4 => (x_freq * y_freq).tanh(),
            5 => gaussian_2d(x, y, 0.5) * (theta_local * params.freq_x as f32).cos(),
            6 => checkerboard(x_freq, y_freq),
            7 => ripple(r_local * params.freq_x as f32, theta_local),
            8 => spiral(r_local, theta_local * params.freq_y as f32),
            _ => gabor_2d(x, y, params.freq_x as f32 * 0.5, params.freq_y as f32 * 0.5),
        };
        
        // 패턴 믹싱 비율 계산
        let mix_ratio = (params.pattern_mix as f32) / 15.0;
        let mixed_pattern = pattern1 * (1.0 - mix_ratio) + pattern2 * mix_ratio;
        
        // 감쇠 적용 (중심에서 멀어질수록)
        let decay_factor = (-r_local * params.decay_rate).exp();
        
        // 야코비안 계산
        let jacobian = (1.0 - c * params.r * params.r).powi(-2).sqrt();
        
        // 최종 값: 진폭, 오프셋, 감쇠 적용
        (mixed_pattern * jacobian * decay_factor * params.amplitude) + params.offset
    }
} 
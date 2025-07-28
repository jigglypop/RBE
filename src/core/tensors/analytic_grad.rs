//! Analytic Gradient 모듈 - Enhanced128용 정확한 해석적 미분
//!
//! 12개 기저함수의 폐쇄형 미분을 Q16.16 고정소수점 LUT로 제공
//! 수치미분 대신 해석적 미분으로 정확도 혁신

use std::f32::consts::PI;

/// Q16.16 고정소수점 연산 유틸리티
pub struct FixedPointMath;

impl FixedPointMath {
    pub const SCALE: i32 = 65536; // 2^16
    pub const HALF: i32 = 32768;  // 0.5 in Q16.16

    #[inline(always)]
    pub fn from_f32(f: f32) -> i32 {
        (f * Self::SCALE as f32).round() as i32
    }

    #[inline(always)]
    pub fn to_f32(fixed: i32) -> f32 {
        fixed as f32 / Self::SCALE as f32
    }

    #[inline(always)]
    pub fn mul(a: i32, b: i32) -> i32 {
        ((a as i64 * b as i64) >> 16) as i32
    }

    #[inline(always)]
    pub fn div(a: i32, b: i32) -> i32 {
        if b == 0 { return 0; }
        ((a as i64) << 16) as i32 / b
    }
}

/// 해석적 그래디언트 LUT 관리자
pub struct AnalyticGradient {
    /// r 방향 그래디언트 LUT [basis_id][r_idx][theta_idx] (64x64, 힙 할당, i32)
    grad_r_lut: Box<[[[i32; 64]; 64]; 12]>,
    /// theta 방향 그래디언트 LUT [basis_id][r_idx][theta_idx] (64x64, 힙 할당, i32)
    grad_theta_lut: Box<[[[i32; 64]; 64]; 12]>,
    /// LUT 초기화 완료 플래그
    initialized: bool,
}

impl AnalyticGradient {
    /// 새로운 AnalyticGradient 인스턴스 생성
    pub fn new() -> Self {
        let mut instance = Self {
            grad_r_lut: Box::new([[[0i32; 64]; 64]; 12]),
            grad_theta_lut: Box::new([[[0i32; 64]; 64]; 12]),
            initialized: false,
        };
        instance.initialize_luts();
        instance
    }

    /// 모든 기저함수의 LUT 초기화
    fn initialize_luts(&mut self) {
        println!("🔧 Analytic Gradient LUT 초기화 시작...");
        
        for basis_id in 0..12 {
            self.build_basis_lut(basis_id);
        }
        
        self.initialized = true;
        println!("✅ Analytic Gradient LUT 초기화 완료");
    }

    /// 특정 기저함수의 LUT 구축
    fn build_basis_lut(&mut self, basis_id: usize) {
        for r_idx in 0..64 {
            for theta_idx in 0..64 {
                // 정규화된 좌표 계산 (63으로 나누어 0.999999 최대값 유지)
                let r = (r_idx as f32) / 63.0 * 0.999999;
                let theta = (theta_idx as f32) / 63.0 * 2.0 * PI;

                // 해석적 그래디언트 계산
                let (grad_r, grad_theta) = self.compute_analytic_gradient(basis_id as u8, r, theta);
                
                // 그래디언트 값 소프트 클리핑 (과도한 값 방지)
                let grad_r_clipped = self.soft_clip(grad_r);
                let grad_theta_clipped = self.soft_clip(grad_theta);

                // Q16.16 고정소수점 변환
                let grad_r_fixed = (grad_r_clipped * 65536.0).round() as i32;
                let grad_theta_fixed = (grad_theta_clipped * 65536.0).round() as i32;

                self.grad_r_lut[basis_id][r_idx][theta_idx] = grad_r_fixed;
                self.grad_theta_lut[basis_id][r_idx][theta_idx] = grad_theta_fixed;
            }
        }
    }

    /// 그래디언트 소프트 클리핑 함수
    fn soft_clip(&self, val: f32) -> f32 {
        const THRESHOLD: f32 = 100.0;
        if val.abs() > THRESHOLD {
            val.signum() * (THRESHOLD + (val.abs() - THRESHOLD + 1.0).ln())
        } else {
            val
        }
    }

    /// 기저함수별 해석적 그래디언트 계산
    fn compute_analytic_gradient(&self, basis_id: u8, r: f32, theta: f32) -> (f32, f32) {
        match basis_id {
            0 => self.grad_basis_0(r, theta),   // sin(θ) × sinh(r)
            1 => self.grad_basis_1(r, theta),   // cos(θ) × sinh(r)
            2 => self.grad_basis_2(r, theta),   // sin(θ) × cosh(r)
            3 => self.grad_basis_3(r, theta),   // cos(θ) × cosh(r)
            4 => self.grad_basis_4(r, theta),   // Bessel J₀
            5 => self.grad_basis_5(r, theta),   // Modified Bessel I₀
            6 => self.grad_basis_6(r, theta),   // Modified Bessel K₀
            7 => self.grad_basis_7(r, theta),   // Bessel Y₀ (Neumann)
            8 => self.grad_basis_8(r, theta),   // tanh × signum
            9 => self.grad_basis_9(r, theta),   // sech × triangle
            10 => self.grad_basis_10(r, theta), // exp × sin
            11 => self.grad_basis_11(r, theta, 5.0), // Morlet wavelet (기본 omega=5.0)
            _ => (0.0, 0.0),
        }
    }

    /// 기저 0: sin(θ) × sinh(r) 미분
    fn grad_basis_0(&self, r: f32, theta: f32) -> (f32, f32) {
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let sinh_r = r.sinh();
        let cosh_r = r.cosh();
        
        let grad_r = sin_theta * cosh_r;     // ∂/∂r[sin(θ) × sinh(r)]
        let grad_theta = cos_theta * sinh_r; // ∂/∂θ[sin(θ) × sinh(r)]
        
        (grad_r, grad_theta)
    }

    /// 기저 1: cos(θ) × sinh(r) 미분
    fn grad_basis_1(&self, r: f32, theta: f32) -> (f32, f32) {
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let sinh_r = r.sinh();
        let cosh_r = r.cosh();
        
        let grad_r = cos_theta * cosh_r;      // ∂/∂r[cos(θ) × sinh(r)]
        let grad_theta = -sin_theta * sinh_r; // ∂/∂θ[cos(θ) × sinh(r)]
        
        (grad_r, grad_theta)
    }

    /// 기저 2: sin(θ) × cosh(r) 미분
    fn grad_basis_2(&self, r: f32, theta: f32) -> (f32, f32) {
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let sinh_r = r.sinh();
        let cosh_r = r.cosh();
        
        let grad_r = sin_theta * sinh_r;     // ∂/∂r[sin(θ) × cosh(r)]
        let grad_theta = cos_theta * cosh_r; // ∂/∂θ[sin(θ) × cosh(r)]
        
        (grad_r, grad_theta)
    }

    /// 기저 3: cos(θ) × cosh(r) 미분
    fn grad_basis_3(&self, r: f32, theta: f32) -> (f32, f32) {
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        let sinh_r = r.sinh();
        let cosh_r = r.cosh();
        
        let grad_r = cos_theta * sinh_r;      // ∂/∂r[cos(θ) × cosh(r)]
        let grad_theta = -sin_theta * cosh_r; // ∂/∂θ[cos(θ) × cosh(r)]
        
        (grad_r, grad_theta)
    }

    /// 기저 4: Bessel J₀ 미분 (J₀'(x) = -J₁(x))
    fn grad_basis_4(&self, r: f32, _theta: f32) -> (f32, f32) {
        let x = r * 10.0; // 스케일링
        let j1_val = self.bessel_j1_approx(x);
        let grad_r = -j1_val * 10.0; // 체인 룰
        let grad_theta = 0.0; // θ에 독립적
        
        (grad_r, grad_theta)
    }

    /// 기저 5: Modified Bessel I₀ 미분 (I₀'(x) = I₁(x))
    fn grad_basis_5(&self, r: f32, _theta: f32) -> (f32, f32) {
        let x = r * 10.0;
        let i1_val = self.bessel_i1_approx(x);
        let grad_r = i1_val * 10.0;
        let grad_theta = 0.0;
        
        (grad_r, grad_theta)
    }

    /// 기저 6: Modified Bessel K₀ 미분 (K₀'(x) = -K₁(x))
    fn grad_basis_6(&self, r: f32, _theta: f32) -> (f32, f32) {
        let x = r * 10.0;
        let k1_val = self.bessel_k1_approx(x);
        let grad_r = -k1_val * 10.0;
        let grad_theta = 0.0;
        
        (grad_r, grad_theta)
    }

    /// 기저 7: Bessel Y₀ 미분 (Y₀'(x) = -Y₁(x))
    fn grad_basis_7(&self, r: f32, _theta: f32) -> (f32, f32) {
        let x = r * 10.0;
        let y1_val = self.bessel_y1_approx(x);
        let grad_r = -y1_val * 10.0;
        let grad_theta = 0.0;
        
        (grad_r, grad_theta)
    }

    /// 기저 8: tanh(cr) × signum(cos(θ)) 미분
    fn grad_basis_8(&self, r: f32, theta: f32) -> (f32, f32) {
        let c = 2.0; // 기본 곡률
        let cr = c * r;
        let sech2_cr = 1.0 / cr.cosh().powi(2); // sech²(cr)
        let cos_theta = theta.cos();
        
        let signum_cos = if cos_theta > 0.0 { 1.0 } else if cos_theta < 0.0 { -1.0 } else { 0.0 };
        
        let grad_r = c * sech2_cr * signum_cos; // ∂/∂r[tanh(cr)] × signum
        
        // signum 미분은 디랙 델타이므로 실용적으로 0으로 근사
        let grad_theta = 0.0;
        
        (grad_r, grad_theta)
    }

    /// 기저 9: sech(cr) × triangle_wave(θ) 미분
    fn grad_basis_9(&self, r: f32, theta: f32) -> (f32, f32) {
        let c = 2.0;
        let cr = c * r;
        let sech_cr = 1.0 / cr.cosh();
        let tanh_cr = cr.tanh();
        
        let triangle_val = self.triangle_wave(theta);
        let triangle_grad = self.triangle_wave_derivative(theta);
        
        let grad_r = -c * sech_cr * tanh_cr * triangle_val; // ∂/∂r[sech(cr)]
        let grad_theta = sech_cr * triangle_grad;           // ∂/∂θ[triangle(θ)]
        
        (grad_r, grad_theta)
    }

    /// 기저 10: exp(-cr) × sin(θ) 미분
    fn grad_basis_10(&self, r: f32, theta: f32) -> (f32, f32) {
        let c = 2.0;
        let cr = c * r;
        let exp_neg_cr = (-cr).exp();
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();
        
        let grad_r = -c * exp_neg_cr * sin_theta;  // ∂/∂r[exp(-cr)]
        let grad_theta = exp_neg_cr * cos_theta;   // ∂/∂θ[sin(θ)]
        
        (grad_r, grad_theta)
    }

    /// 기저 11: Morlet Wavelet 미분
    fn grad_basis_11(&self, r: f32, theta: f32, omega: f32) -> (f32, f32) {
        let sigma = 1.0;
        let t = r * theta.cos(); // 투영된 시간축
        let gaussian = (-(t * t) / (2.0 * sigma * sigma)).exp();
        let cosine = (omega * t).cos();
        let sine = (omega * t).sin();
        
        // ∂/∂r = ∂/∂t × ∂t/∂r
        let dt_dr = theta.cos();
        let dgaussian_dt = gaussian * (-t / (sigma * sigma));
        let dcosine_dt = -omega * sine;
        
        let grad_r = dt_dr * (dgaussian_dt * cosine + gaussian * dcosine_dt);
        
        // ∂/∂θ = ∂/∂t × ∂t/∂θ
        let dt_dtheta = -r * theta.sin();
        let grad_theta = dt_dtheta * (dgaussian_dt * cosine + gaussian * dcosine_dt);
        
        (grad_r, grad_theta)
    }

    /// LUT에서 그래디언트 조회 (이중선형 보간)
    pub fn lookup_gradient(&self, basis_id: u8, r: f32, theta: f32) -> (f32, f32) {
        if !self.initialized || basis_id >= 12 {
            return (0.0, 0.0);
        }

        // 정규화된 인덱스 계산 (63으로 나누어 64x64 LUT에 맞춤)
        let r_norm = (r.clamp(0.0, 0.999999) * 63.0).min(62.0);
        let theta_norm = (theta.rem_euclid(2.0 * PI) / (2.0 * PI) * 63.0).min(62.0);

        let r_idx = r_norm as usize;
        let theta_idx = theta_norm as usize;
        
        // 이중선형 보간을 위한 가중치
        let r_frac = r_norm - r_idx as f32;
        let theta_frac = theta_norm - theta_idx as f32;

        // 4개 코너 점의 그래디언트 값
        let gr_00 = self.grad_r_lut[basis_id as usize][r_idx][theta_idx] as f32 / 65536.0;
        let gr_01 = self.grad_r_lut[basis_id as usize][r_idx][theta_idx + 1] as f32 / 65536.0;
        let gr_10 = self.grad_r_lut[basis_id as usize][r_idx + 1][theta_idx] as f32 / 65536.0;
        let gr_11 = self.grad_r_lut[basis_id as usize][r_idx + 1][theta_idx + 1] as f32 / 65536.0;

        let gt_00 = self.grad_theta_lut[basis_id as usize][r_idx][theta_idx] as f32 / 65536.0;
        let gt_01 = self.grad_theta_lut[basis_id as usize][r_idx][theta_idx + 1] as f32 / 65536.0;
        let gt_10 = self.grad_theta_lut[basis_id as usize][r_idx + 1][theta_idx] as f32 / 65536.0;
        let gt_11 = self.grad_theta_lut[basis_id as usize][r_idx + 1][theta_idx + 1] as f32 / 65536.0;

        // 이중선형 보간
        let grad_r = (1.0 - r_frac) * (1.0 - theta_frac) * gr_00
                   + (1.0 - r_frac) * theta_frac * gr_01
                   + r_frac * (1.0 - theta_frac) * gr_10
                   + r_frac * theta_frac * gr_11;

        let grad_theta = (1.0 - r_frac) * (1.0 - theta_frac) * gt_00
                       + (1.0 - r_frac) * theta_frac * gt_01
                       + r_frac * (1.0 - theta_frac) * gt_10
                       + r_frac * theta_frac * gt_11;

        (grad_r, grad_theta)
    }

    // ========== 보조 함수들 ==========

    /// Bessel J₁ 근사 함수
    fn bessel_j1_approx(&self, x: f32) -> f32 {
        let ax = x.abs();
        if ax < 8.0 {
            let y = x * x;
            let ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
            let ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
            ans1 / ans2
        } else {
            let z = 8.0 / ax;
            let y = z * z;
            let xx = ax - 2.356194491;
            let ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
            let ans2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            sign * (2.0 / (PI * ax)).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2)
        }
    }

    /// Modified Bessel I₁ 근사 함수
    fn bessel_i1_approx(&self, x: f32) -> f32 {
        let ax = x.abs();
        if ax < 3.75 {
            let y = (x / 3.75) * (x / 3.75);
            x * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934 + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))))
        } else {
            let z = 3.75 / ax;
            let ans = 0.2282967 + z * (-0.2895312e-1 + z * (0.1787654e-1 + z * (-0.420059e-2)));
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            sign * (ax.exp() / ax.sqrt()) * (0.39894228 + z * ans)
        }
    }

    /// Modified Bessel K₁ 근사 함수
    fn bessel_k1_approx(&self, x: f32) -> f32 {
        if x <= 2.0 {
            let y = x * x / 4.0;
            (x.ln() * self.bessel_i1_approx(x)) + (1.0 / x) * (1.0 + y * (0.15443144 + y * (-0.67278579 + y * (-0.18156897 + y * (-0.01919402 + y * (-0.00110404 + y * (-0.00004686)))))))
        } else {
            let z = 2.0 / x;
            ((-x).exp() / x.sqrt()) * (1.25331414 + z * (0.23498619 + z * (-0.03655620 + z * (0.01504268 + z * (-0.00780353 + z * (0.00325614 + z * (-0.00068245)))))))
        }
    }

    /// Bessel Y₁ 근사 함수
    fn bessel_y1_approx(&self, x: f32) -> f32 {
        if x < 8.0 {
            let y = x * x;
            let j1 = self.bessel_j1_approx(x);
            j1 * (2.0 / PI) * x.ln() - (1.0 / x) * (1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6)))))
        } else {
            let z = 8.0 / x;
            let y = z * z;
            let xx = x - 2.356194491;
            let ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
            let ans2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
            (2.0 / (PI * x)).sqrt() * (xx.sin() * ans1 + z * xx.cos() * ans2)
        }
    }

    /// Triangle wave 함수
    fn triangle_wave(&self, x: f32) -> f32 {
        let normalized = x / PI;
        let t = normalized - normalized.floor();
        if t < 0.5 { 2.0 * t } else { 2.0 * (1.0 - t) }
    }

    /// Triangle wave 미분
    fn triangle_wave_derivative(&self, x: f32) -> f32 {
        let normalized = (x / PI) % 1.0;
        if normalized < 0.5 { 2.0 / PI } else { -2.0 / PI }
    }
}

impl Default for AnalyticGradient {
    fn default() -> Self {
        Self::new()
    }
}

/// 전역 Analytic Gradient 인스턴스 (lazy static)
static mut ANALYTIC_GRADIENT: Option<AnalyticGradient> = None;
static INIT: std::sync::Once = std::sync::Once::new();

/// 전역 Analytic Gradient 인스턴스 획득
pub fn get_analytic_gradient() -> &'static AnalyticGradient {
    unsafe {
        INIT.call_once(|| {
            ANALYTIC_GRADIENT = Some(AnalyticGradient::new());
        });
        ANALYTIC_GRADIENT.as_ref().unwrap()
    }
} 
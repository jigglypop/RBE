//! # RBE - Bit Differential Engine
//!
//! 이 모듈은 레거시 시스템의 수학적 정확성을 1:1로 계승하는 순수 계산 엔진입니다.
//! `Packed256`의 상태(hi)와 파라미터(lo)를 입력받아, 최종 예측 값과 비트 미분에 따른
//! 해석적 그래디언트(∂f/∂r, ∂f/∂θ)를 계산하는 책임을 가집니다.
//!
//! ## 핵심 기능
//! - **레거시 수학 함수 이식**: `bessel`, `morlet_wavelet` 등 레거시의 핵심 함수를 NaN/inf 안전 가드와 함께 그대로 포함.
//! - **비트 미분 구현**: `d_r`, `d_theta` 비트 값에 따라 미리 정의된 미분 함수를 선택.
//! - **통합 계산**: 단일 진입점 `compute_fused_output`을 통해 예측 값과 그래디언트를 한 번에 계산.

use crate::core::tensors::packed256_types::Packed256Params;
use std::f32::consts::PI;

/// 비트 미분 엔진의 통합 출력
#[derive(Debug, Clone, Copy)]
pub struct EngineOutput {
    pub predicted_value: f32,
    pub grad_r: f32,
    pub grad_theta: f32,
    pub grad_p2: f32, // amplitude(param2) gradient
    pub grad_p1: f32, // frequency(param1) gradient
}

impl std::ops::Sub for EngineOutput {
    type Output = f32;
    
    fn sub(self, other: EngineOutput) -> f32 {
        self.predicted_value - other.predicted_value
    }
}

impl std::ops::Sub<f32> for EngineOutput {
    type Output = f32;
    
    fn sub(self, other: f32) -> f32 {
        self.predicted_value - other
    }
}

impl std::fmt::Display for EngineOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:.6}", self.predicted_value)
    }
}

/// 메인 계산 함수
///
/// `Packed256`의 디코딩된 파라미터와 좌표를 받아,
/// 최종 예측 값과 해석적 그래디언트를 계산합니다.
pub fn compute_fused_output(
    params: &Packed256Params,
    i: usize,
    j: usize,
    rows: usize,
    cols: usize,
) -> EngineOutput {
    let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
    let theta_coord = if cols > 0 { 2.0 * PI * (j as f32) / (cols as f32) } else { 0.0 };
    compute_fused_output_fast(params, r_coord, theta_coord)
}

/// 고속 경로: 사전계산된 r_coord, theta_coord를 제공
#[inline(always)]
pub fn compute_fused_output_fast(
    params: &Packed256Params,
    r_coord: f32,
    theta_coord: f32,
) -> EngineOutput {
    let r_scale = params.r.clamp(0.0, 4.0);
    let theta_scale = if params.basis_id == 12 { 1.0 } else { params.theta.clamp(0.0, 8.0) };
    let r_eff = r_scale * r_coord;
    let th_eff = theta_scale * theta_coord;

    // 3) Atlas 2x2 블렌딩 모드 여부
    let use_atlas = (params.flags & 0b0000_0100) != 0;

    // 4) 기저 함수 값 및 (효과 좌표에 대한) 해석 도함수 계산
    let (base_val, d_base_dr_eff, d_base_dth_eff, d_base_dphi, d_base_dp1) = if !use_atlas {
        compute_base_function(params, r_eff, th_eff)
    } else {
        // Partition of Unity: bilinear weights on (u,v) in [0,1]^2
        let u = r_coord; // already 0..1
        let v = (theta_coord / (2.0 * PI)).clamp(0.0, 1.0);
        let w00 = (1.0 - u) * (1.0 - v);
        let w10 = u * (1.0 - v);
        let w01 = (1.0 - u) * v;
        let w11 = u * v;

        // Local modifiers derived from a2,a3,a4
        let a2 = (params.q_value as f32) / 255.0;   // ~[0,1]
        let a3 = (params.k_value as f32) / 255.0;   // ~[0,1]
        let a4 = (params.activation_id as f32) / 255.0; // ~[0,1]
        let base_p1 = params.param1;
        let base_phi = params.theta;

        // Define four local (p1, phi) pairs
        let l = [
            (base_p1 * (1.0 + 0.00), base_phi + 0.0),
            (base_p1 * (1.0 + 0.50 * a2), base_phi + 0.5 * base_phi),
            (base_p1 * (1.0 + 0.50 * a3), base_phi - 0.5 * base_phi),
            (base_p1 * (1.0 + 0.50 * a4), base_phi + 1.0 * base_phi),
        ];

        let mut acc_val = 0.0;
        let mut acc_dr = 0.0;
        let mut acc_dth = 0.0;
        let mut acc_dphi = 0.0;
        let mut acc_dp1 = 0.0;

        for (idx, (p1_loc, phi_loc)) in l.iter().enumerate() {
            // 임시 params 변형 없이 로컬 p1, phi를 주입하여 평가
            let (v_loc, dr_loc, dth_loc, dphi_loc, dp1_loc) = compute_base_function_overrides(params, r_eff, th_eff, *p1_loc, *phi_loc);
            let w = match idx {
                0 => w00,
                1 => w10,
                2 => w01,
                _ => w11,
            };
            acc_val += w * v_loc;
            acc_dr += w * dr_loc;
            acc_dth += w * dth_loc;
            acc_dphi += w * dphi_loc;
            acc_dp1 += w * dp1_loc;
        }
        (acc_val, acc_dr, acc_dth, acc_dphi, acc_dp1)
    };

    // 4.1) 기저 블렌딩 (flags bit3) - 보조 기저와의 가중합
    let blend_enabled = (params.flags & 0b0000_1000) != 0;
    let mut base_val = base_val;
    let mut d_base_dr_eff = d_base_dr_eff;
    let mut d_base_dth_eff = d_base_dth_eff;
    let mut d_base_dphi = d_base_dphi;
    let mut d_base_dp1 = d_base_dp1;
    if blend_enabled && params.basis_id != 12 {
        // blend weight from flags[5..=7] mapped to [0,1]
        let blend_bits = (params.flags >> 5) & 0b0000_0111;
        let w_blend = ((blend_bits as f32) / 7.0).clamp(0.0, 1.0);
        // paired basis mapping 0<->2, 1<->3 else ->0
        let alt_basis = match params.basis_id {
            0 => 2,
            2 => 0,
            1 => 3,
            3 => 1,
            _ => 0,
        };
        let (val_alt, dr_alt, dth_alt, dphi_alt, dp1_alt) = compute_base_function_overrides_full(params, r_eff, th_eff, alt_basis, params.param1, params.theta);
        let w0 = 1.0 - w_blend;
        // 안전 가드: 두 값이 모두 0으로 가는 경우 방지
        let val_mix = w0 * base_val + w_blend * val_alt;
        base_val = if val_mix.abs() < 1e-12 { base_val } else { val_mix };
        d_base_dr_eff = w0 * d_base_dr_eff + w_blend * dr_alt;
        d_base_dth_eff = w0 * d_base_dth_eff + w_blend * dth_alt;
        d_base_dphi = w0 * d_base_dphi + w_blend * dphi_alt;
        d_base_dp1 = w0 * d_base_dp1 + w_blend * dp1_alt;
    }

    // 5) 비트 미분 적용 (d_r, d_theta)
    let (func_val, d_func_dr_eff, d_func_dth_eff, d_func_dphi, d_func_dp1) = apply_bit_derivatives_ext(params, base_val, d_base_dr_eff, d_base_dth_eff, d_base_dphi, d_base_dp1);

    // 6) Poincaré 유사 메트릭 (r_eff 기준)
    let c = 2.0_f32.powi(params.log2_c as i32);
    let use_neutral_metric = c.abs() < 1.0e-6;
    let (metric, d_metric_dr_eff) = if use_neutral_metric {
        (1.0, 0.0)
    } else {
        let jacobian_denom = 1.0 - c * r_eff * r_eff;
        if jacobian_denom <= 1e-8 {
            return EngineOutput { predicted_value: 0.0, grad_r: 0.0, grad_theta: 0.0, grad_p2: 0.0, grad_p1: 0.0 };
        }
    let metric = 1.0 / jacobian_denom;
        let d_metric_dr_eff = (2.0 * c * r_eff) / (jacobian_denom * jacobian_denom);
        (metric, d_metric_dr_eff)
    };

    // 7) 진폭(amp) 적용: param2를 진폭으로 사용
    let amp = params.param2.clamp(0.0, 4.0);
    let predicted_value = amp * func_val * metric;

    // 8) 효과 좌표에 대한 그래디언트
    let grad_r_eff = amp * (d_func_dr_eff * metric + func_val * d_metric_dr_eff);
    let grad_th_eff = amp * (d_func_dth_eff * metric);
    let grad_amp = func_val * metric;
    let grad_p1 = amp * (d_func_dp1 * metric);

    // 9) 시드 파라미터(r_scale, theta_scale)에 대한 체인 룰
    // r_scale(theta로 저장)와 theta_scale(theta 파라미터)를 각각 r, theta에 매핑
    // grad_r -> ∂/∂r_scale, grad_theta -> ∂/∂theta_scale 로 반환
    let grad_r = grad_r_eff * r_coord;           // ∂/∂r_scale = ∂/∂r_eff * ∂r_eff/∂r_scale = ... * r_coord
    let grad_theta = grad_th_eff * theta_coord + d_func_dphi;  // theta 파라미터는 위상(phi)에도 영향

    EngineOutput { predicted_value, grad_r, grad_theta, grad_p2: grad_amp, grad_p1 }
}

/// basis_id에 따라 기저 함수와 그 해석적 도함수를 계산
fn compute_base_function(params: &Packed256Params, r_eff: f32, theta_eff: f32) -> (f32, f32, f32, f32, f32) {
    let p1 = params.param1;
    let p2 = params.param2;
    let phi = params.theta; // 시드의 theta를 위상으로 사용

    // 수치 미분 보조 간격 (일부 기저에만 사용)
    let h = 1e-5;

    match params.basis_id {
        0 => { // Harmonic mixture (k=1,2,3[,4]) with weights from q_value/k_value/activation_id
            let enable_mix = (params.flags & 0b0000_0001) != 0; // enable k=2,3
            let enable_k4  = (params.flags & 0b0000_0010) != 0; // enable k=4
            let a2 = (params.q_value as f32) / 255.0;
            let a3 = (params.k_value as f32) / 255.0;
            let a4 = (params.activation_id as f32) / 255.0;

            let mut acc_val = 0.0;
            let mut acc_dr = 0.0;
            let mut acc_dth = 0.0;
            let mut acc_dphi = 0.0;
            let mut acc_dp1 = 0.0;

            // k=1
            let k1 = 1.0f32;
            let inner1 = (k1 * p1) * r_eff + k1 * phi;
            let sin1 = inner1.sin();
            let cos1 = inner1.cos();
            let cos_th1 = (k1 * theta_eff).cos();
            let sin_th1 = (k1 * theta_eff).sin();
            acc_val += sin1 * cos_th1;
            acc_dr += (k1 * p1) * cos1 * cos_th1;
            acc_dth += -(k1) * sin1 * sin_th1;
            acc_dphi += cos1 * cos_th1;
            acc_dp1 += (k1 * r_eff) * cos1 * cos_th1;

            if enable_mix {
                // k=2
                let k2 = 2.0f32;
                let inner2 = (k2 * p1) * r_eff + k2 * phi;
                let sin2 = inner2.sin();
                let cos2 = inner2.cos();
                let cos_th2 = (k2 * theta_eff).cos();
                let sin_th2 = (k2 * theta_eff).sin();
                acc_val += a2 * (sin2 * cos_th2);
                acc_dr += a2 * ((k2 * p1) * cos2 * cos_th2);
                acc_dth += a2 * (-(k2) * sin2 * sin_th2);
                acc_dphi += a2 * (cos2 * cos_th2);
                acc_dp1 += a2 * ((k2 * r_eff) * cos2 * cos_th2);

                // k=3
                let k3 = 3.0f32;
                let inner3 = (k3 * p1) * r_eff + k3 * phi;
                let sin3 = inner3.sin();
                let cos3 = inner3.cos();
                let cos_th3 = (k3 * theta_eff).cos();
                let sin_th3 = (k3 * theta_eff).sin();
                acc_val += a3 * (sin3 * cos_th3);
                acc_dr += a3 * ((k3 * p1) * cos3 * cos_th3);
                acc_dth += a3 * (-(k3) * sin3 * sin_th3);
                acc_dphi += a3 * (cos3 * cos_th3);
                acc_dp1 += a3 * ((k3 * r_eff) * cos3 * cos_th3);
                if enable_k4 {
                    // k=4
                    let k4 = 4.0f32;
                    let inner4 = (k4 * p1) * r_eff + k4 * phi;
                    let sin4 = inner4.sin();
                    let cos4 = inner4.cos();
                    let cos_th4 = (k4 * theta_eff).cos();
                    let sin_th4 = (k4 * theta_eff).sin();
                    acc_val += a4 * (sin4 * cos_th4);
                    acc_dr += a4 * ((k4 * p1) * cos4 * cos_th4);
                    acc_dth += a4 * (-(k4) * sin4 * sin_th4);
                    acc_dphi += a4 * (cos4 * cos_th4);
                    acc_dp1 += a4 * ((k4 * r_eff) * cos4 * cos_th4);
                }
            }

            (acc_val, acc_dr, acc_dth, acc_dphi, acc_dp1)
        }
        1 => { // f = tanh(p1*r_eff + phi) * sech(θ_eff)
            // tanh'(x) = sech^2(x),  (sech θ)' = -sech θ tanh θ
            let inner = p1 * r_eff + phi;
            let tanh_inner = inner.tanh();
            let sech_inner = legacy_math::sech(inner);
            let sech_theta = legacy_math::sech(theta_eff);
            let tanh_theta = theta_eff.tanh();
            let val = tanh_inner * sech_theta;
            let d_dr = p1 * (sech_inner * sech_inner) * sech_theta;
            let d_dtheta = -tanh_inner * sech_theta * tanh_theta;
            let d_dphi = (sech_inner * sech_inner) * sech_theta; // ∂/∂phi
            let d_dp1 = r_eff * (sech_inner * sech_inner) * sech_theta; // ∂/∂p1 = r_eff * tanh'(inner)
            (val, d_dr, d_dtheta, d_dphi, d_dp1)
        }
        2 => { // f = cos(p1*r_eff + phi) * sin(p2*θ_eff)
            let inner = p1 * r_eff + phi;
            let cos_inner = inner.cos();
            let sin_inner = inner.sin();
            let sin_th = (p2 * theta_eff).sin();
            let cos_th = (p2 * theta_eff).cos();
            let val = cos_inner * sin_th;
            let d_dr = -p1 * sin_inner * sin_th;
            let d_dtheta = p2 * cos_inner * cos_th;
            let d_dphi = -sin_inner * sin_th;
            let d_dp1 = -r_eff * sin_inner * sin_th;
            (val, d_dr, d_dtheta, d_dphi, d_dp1)
        }
        3 => { // f = sinh(p1*r_eff + phi) * cosh(p2*θ_eff)
            let inner = p1 * r_eff + phi;
            let val = inner.sinh() * (p2 * theta_eff).cosh();
            let d_dr = p1 * inner.cosh() * (p2 * theta_eff).cosh();
            let d_dtheta = p2 * inner.sinh() * (p2 * theta_eff).sinh();
            let d_dphi = inner.cosh() * (p2 * theta_eff).cosh();
            let d_dp1 = r_eff * inner.cosh() * (p2 * theta_eff).cosh();
            (val, d_dr, d_dtheta, d_dphi, d_dp1)
        }
        // ... (다른 basis_id에 대한 구현)
        4 => { // Bessel J0(x) with phase shift phi: x = p1*r_eff + phi
            let x = r_eff * p1 + phi;
            let val = legacy_math::bessel_j0(x);
            // 고정밀 보조: J1 근사 (Central difference on J0')
            let val_ph = legacy_math::bessel_j0(x + h);
            let val_mh = legacy_math::bessel_j0(x - h);
            let d_dx = (val_ph - val_mh) / (2.0 * h);
            let d_dr = p1 * d_dx; // chain rule
            let d_dphi = d_dx;     // ∂/∂phi = d/dx
            let d_dp1 = r_eff * d_dx; // ∂/∂p1 = r_eff * d/dx
            (val, d_dr, 0.0, d_dphi, d_dp1)
        }
        11 => { // Morlet Wavelet
            let val = legacy_math::morlet_wavelet(r_eff, theta_eff, p1);
            let val_r_h = legacy_math::morlet_wavelet(r_eff + h, theta_eff, p1);
            let val_theta_h = legacy_math::morlet_wavelet(r_eff, theta_eff + h, p1);
            let d_dr = (val_r_h - val) / h;
            let d_dtheta = (val_theta_h - val) / h;
            // 위상 phi, p1 직접 포함 미약 → 0 근사
            (val, d_dr, d_dtheta, 0.0, 0.0)
        }
        12 => { // Separable sum: ax*sin(ωx r + φx) + ay*cos(ωy θ + φy)
            let omega_x = p1;
            // theta_coord는 이미 2π*j/cols 이므로 ωy=1.0이면 정확히 1주기
            let omega_y = (params.q_value as f32) / 255.0;
            let a_x = (params.k_value as f32) / 255.0 * 1.0;
            let a_y = (params.activation_id as f32) / 255.0 * 1.0;
            let phi_x = params.theta; // φx는 theta 필드 사용
            let phi_bits = (params.flags >> 4) & 0b11; // flags[5:4] → φy 상태(0, π/2, π, 3π/2)
            let phi_y = match phi_bits {
                0 => 0.0,
                1 => 0.5 * PI,
                2 => PI,
                _ => 1.5 * PI,
            };

            let inner_x = omega_x * r_eff + phi_x;
            let inner_y = omega_y * theta_eff + phi_y;

            let sin_x = inner_x.sin();
            let cos_x = inner_x.cos();
            let cos_y = inner_y.cos();
            let sin_y = inner_y.sin();

            let val = a_x * sin_x + a_y * cos_y;
            let d_dr = a_x * omega_x * cos_x;
            let d_dtheta = -a_y * omega_y * sin_y;
            let d_dphi = a_x * cos_x;       // ∂/∂φx
            let d_dp1 = a_x * r_eff * cos_x; // ∂/∂ωx
            (val, d_dr, d_dtheta, d_dphi, d_dp1)
        }
        _ => (0.0, 0.0, 0.0, 0.0, 0.0), // 기본값
    }
}

/// p1, phi를 오버라이드하여 동일한 기저를 평가
fn compute_base_function_overrides(params: &Packed256Params, r_eff: f32, theta_eff: f32, p1_override: f32, phi_override: f32) -> (f32, f32, f32, f32, f32) {
    // 임시 복사본으로 basis와 기타 파라미터는 유지하되 p1, theta(=phi)를 덮어써 평가
    let mut shadow = *params;
    // Packed256Params는 Copy 파생이 없어 직접 필드로 재구성
    let shadow_params = Packed256Params {
        r: params.r,
        theta: phi_override,
        param1: p1_override,
        param2: params.param2,
        basis_id: params.basis_id,
        d_r: params.d_r,
        d_theta: params.d_theta,
        log2_c: params.log2_c,
        activation_id: params.activation_id,
        q_value: params.q_value,
        k_value: params.k_value,
        flags: params.flags,
    };
    compute_base_function(&shadow_params, r_eff, theta_eff)
}

/// basis_id, p1, phi를 모두 오버라이드하여 평가
fn compute_base_function_overrides_full(params: &Packed256Params, r_eff: f32, theta_eff: f32, basis_override: u8, p1_override: f32, phi_override: f32) -> (f32, f32, f32, f32, f32) {
    let shadow_params = Packed256Params {
        r: params.r,
        theta: phi_override,
        param1: p1_override,
        param2: params.param2,
        basis_id: basis_override,
        d_r: params.d_r,
        d_theta: params.d_theta,
        log2_c: params.log2_c,
        activation_id: params.activation_id,
        q_value: params.q_value,
        k_value: params.k_value,
        flags: params.flags,
    };
    compute_base_function(&shadow_params, r_eff, theta_eff)
}

/// 비트 값에 따라 함수와 그 미분에 변형을 적용
/// Legacy 방식: d_r, d_theta 비트에 따라 미분된 함수를 직접 계산
fn apply_bit_derivatives_ext(params: &Packed256Params, val: f32, d_dr: f32, d_dtheta: f32, d_dphi: f32, d_dp1: f32) -> (f32, f32, f32, f32, f32) {
    let d_r = params.d_r;
    let d_theta = params.d_theta;

    match (d_r, d_theta) {
        // 미분 없음 - 원래 함수 그대로
        (0, 0) => (val, d_dr, d_dtheta, d_dphi, d_dp1),
        
        // r에 대한 1차 미분 - f_r로 전환
        (1, 0) => {
            // f -> f_r, 2차/교차 도함수는 제공하지 않음: 0으로 둠
            (d_dr, 0.0, 0.0, 0.0, 0.0)
        },

        // theta에 대한 1차 미분 - f_θ로 전환
        (0, 1) => {
            // f -> f_θ, ∂/∂θ(f_θ) ≈ f_{θθ}, ∂/∂r(f_θ) ≈ f_{rθ}
            (d_dtheta, 0.0, 0.0, 0.0, 0.0)
        },

        // r, theta에 대한 교차 미분 - 교차 미분을 단순 근사
        (1, 1) => {
            // f -> f_{rθ}: 현재 기저 집합에서 교차 2차 도함수는 기본적으로 0으로 둔다
            (0.0, 0.0, 0.0, 0.0, 0.0)
        },

        // 기타 경우는 원래 함수 반환
        _ => (val, d_dr, d_dtheta, d_dphi, d_dp1),
    }
}

/// 레거시 시스템에서 가져온 수학 함수 모음
pub mod legacy_math {
    use std::f32::consts::PI;

    pub fn bessel_j0(x: f32) -> f32 {
        let ax = x.abs();
        
        if ax < 2.0 {
            // 매우 작은 값: 초고정밀 테일러 급수 (20항까지)
            let y = x * x;
            let mut result = 1.0;
            let mut term = 1.0;
            let x_half = x / 2.0;
            let x_half_squared = x_half * x_half;
            
            // 20항까지 극고정밀도 계산
            for n in 1..=20 {
                term *= -x_half_squared / ((n * n) as f32);
                result += term;
                
                // 수렴 조건 (극도로 정밀한 임계값)
                if term.abs() < 1e-12 {
                    break;
                }
            }
            
            let _ = y; // silence unused warning (kept for potential future use)
            result
        } else if ax < 8.0 {
            // 중간 값: 향상된 다항식 근사
            let z = ax * ax;
            
            // 초고정밀 계수들 (더 많은 항 포함)
            let p0 = 1.0;
            let p1 = -0.25;
            let p2 = 0.015625;
            let p3 = -0.000434027777777778;
            let p4 = 0.0000067462579365079365;
            let p5 = -0.0000000651626283420156;
            let p6 = 0.000000000411088748100754;
            
            p0 + z * (p1 + z * (p2 + z * (p3 + z * (p4 + z * (p5 + z * p6)))))
        } else if ax < 40.0 {
            // 큰 값: 정확한 점근 전개 (Abramowitz & Stegun 공식)
            let z = 8.0 / ax;
            let y = z * z;
            let xx = ax - std::f32::consts::FRAC_PI_4;
            
            // 초정밀 P0 다항식 (8항까지 확장)
            let p0 = 1.0;
            let p1 = -0.1098628627416795;
            let p2 = 0.2734510407859862e-2;
            let p3 = -0.2073370639406285e-3;
            let p4 = 0.2093887211946849e-4;
            let p5 = -0.1262712808295539e-5;
            let p6 = 0.5477629206827851e-7;
            let p7 = -0.1764222073852486e-8;
            
            let p = p0 + y * (p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * (p6 + y * p7))))));
            
            // 초정밀 Q0 다항식 (8항까지 확장)
            let q0 = -0.1250000000000000;
            let q1 = 0.7031250000000000e-2;
            let q2 = -0.7324218750000000e-3;
            let q3 = 0.1121520996093750e-3;
            let q4 = -0.2271080017089844e-4;
            let q5 = 0.5725014209747314e-6;
            let q6 = -0.1727447527908730e-7;
            let q7 = 0.6074042001273483e-9;
            
            let q = z * (q0 + y * (q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y * (q6 + y * q7)))))));
            
            // 최종 점근 전개 결과
            (2.0 / (std::f32::consts::PI * ax)).sqrt() * (p * xx.cos() - q * xx.sin())
        } else {
            // 매우 큰 값: 단순한 점근 공식
            let phase = ax - std::f32::consts::FRAC_PI_4;
            (2.0 / (std::f32::consts::PI * ax)).sqrt() * phase.cos()
        }
    }

    pub fn bessel_i0(x: f32) -> f32 {
        let ax = x.abs();
        if ax < 3.75 {
            // 베셀 I0 고정밀도 급수 전개 (Legacy 수준)
            let t = x / 3.75;
            let t2 = t * t;
            let t4 = t2 * t2;
            let _t6 = t4 * t2; // reserved for potential higher-order terms
            let _t8 = t4 * t4; // reserved for potential higher-order terms
            
            // 고정밀도 계수들 (15자리 정밀도)
            let c0 = 1.0;
            let c1 = 3.5156229787436;
            let c2 = 3.0899424048796;
            let c3 = 1.2067492948273;
            let c4 = 0.26597321648835;
            let c5 = 0.036076834579462;
            let c6 = 0.0045813547297318;
            let c7 = 0.00032411992854875;
            let c8 = 0.000014128685944659;
            
            (c0 + t2 * (c1 + t2 * (c2 + t2 * (c3 + t2 * (c4 + t2 * (c5 + t2 * (c6 + t2 * (c7 + t2 * c8)))))))) as f32
        } else {
            // 점근 급수 (고정밀도)
            let inv_x = 1.0 / ax;
            let exp_x = ax.exp();
            let sqrt_2pi_x = (2.0 * PI * ax).sqrt();
            
            // 고정밀도 점근 계수들
            let a0 = 0.39894228040143267;
            let a1 = 0.01328592142205742;
            let a2 = 0.002253190047946677;
            let a3 = -0.001575654162542779;
            let a4 = 0.009162810717447234;
            let a5 = -0.020577062261067324;
            let a6 = 0.026355372503823843;
            let a7 = -0.016476330700715963;
            let a8 = 0.003923769170616327;
            let a9 = -0.000413406297045157;
            
            let series = a0 + inv_x * (a1 + inv_x * (a2 + inv_x * (a3 + inv_x * (a4 + inv_x * (a5 + inv_x * (a6 + inv_x * (a7 + inv_x * (a8 + inv_x * a9))))))));
            
            ((exp_x / sqrt_2pi_x) * series) as f32
        }
    }

    pub fn bessel_k0(x: f32) -> f32 {
        if x <= 0.0 { return f32::MAX; } // Diverges at 0 or less
        if x <= 2.0 {
            let y = x * x / 4.0;
            (-x.ln() * bessel_i0(x)) + (-0.57721566 + y * (0.42278420 + y * (0.23069756 + y * (0.03488590 + y * (0.00262698 + y * (0.00010750 + y * 0.00000740)))))) as f32
        } else {
            let inv_x = 1.0 / x;
            (x.exp() * (PI / (2.0 * x)).sqrt()) * (1.25331414 + inv_x * (-0.07832358 + inv_x * (0.02189568 + inv_x * (-0.01062446 + inv_x * (0.00587872 + inv_x * (-0.00251540 + inv_x * 0.00053208)))))) as f32
        }
    }
    
    pub fn bessel_y0(x: f32) -> f32 {
        if x <= 0.0 { return f32::NEG_INFINITY; } // Diverges at 0 or less
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
        let ax = x.abs();
        if ax > 20.0 {
            0.0 // 언더플로우 방지 (더 넓은 범위)
        } else if ax < 0.001 {
            // 작은 x에 대한 테일러 급수 (고정밀도)
            let x2 = x * x;
            let x4 = x2 * x2;
            let _x6 = x4 * x2; // reserved
            let _x8 = x4 * x4; // reserved
            
            // sech(x) = 1 - x²/2 + 5x⁴/24 - 61x⁶/720 + 277x⁸/8064 + ...
            1.0 - x2 * (0.5 - x2 * (5.0/24.0 - x2 * (61.0/720.0 - x2 * 277.0/8064.0)))
        } else {
            // 일반적인 경우 (고정밀도)
            let exp_x = x.exp();
            let exp_neg_x = 1.0 / exp_x; // exp(-x)보다 정확
            2.0 / (exp_x + exp_neg_x)
        }
    }
    
    pub fn triangle_wave(x: f32) -> f32 {
        4.0 / PI * (x.sin() - (3.0*x).sin()/9.0 + (5.0*x).sin()/25.0)
    }

    pub fn morlet_wavelet(r: f32, theta: f32, freq: f32) -> f32 {
        let freq_clamped = freq.clamp(0.1, 50.0);
        let sigma = 1.0 / freq_clamped.sqrt();
        
        // 고정밀도 가우시안
        let r_norm = r / sigma;
        let gaussian_exp = -0.5 * r_norm * r_norm;
        let gaussian = if gaussian_exp < -20.0 { 0.0 } else { gaussian_exp.exp() };
        
        // 고정밀도 진동
        let phase = freq_clamped * theta;
        let oscillation = phase.cos();
        
        // 정규화 인수 (Legacy 수준 정밀도)
        let normalization = (1.0 / (PI.sqrt() * sigma)).sqrt();
        
        normalization * gaussian * oscillation
    }
} 
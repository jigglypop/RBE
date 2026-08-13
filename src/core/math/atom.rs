//! 쌍곡 평면파 원자 (논문 정의 13.1, 정리 13.2; 하네스 CP-5)
//!
//!   W(z) = A * e^{rho * B_b(z)} * cos(lambda * B_b(z) + phi),  rho = 1/2
//!
//! rho = 1/2 은 선택이 아니라 라플라시안 고유값의 실수성이 강제하는 값이다 (부록 C.1).
//!
//! 미분 (정리 13.2): b-측지 방향 도함수는 파라미터 공간의 아핀 사상
//!   log A += (1/2) log(rho^2 + lambda^2),   phi += delta,  delta = atan2(lambda, rho)
//! 이며, delta 를 위상 격자로 1회 양자화해 저장하면 이후 모든 차수의 미분이
//! 정수 덧셈 하나다 (부록 A.3). n계 도함수의 위상 오차는 최초 양자화 1회분뿐.

use super::busemann::busemann_polar;
use super::phase_state::{PhaseState, PHASE_MASK};

/// 스펙트럼 상수 rho = 1/2 (부록 C.1 이 강제)
pub const RHO: f64 = 0.5;

#[derive(Clone, Copy, Debug)]
pub struct Atom {
    /// 경계점 방향
    pub theta_b: f64,
    /// 스펙트럼 파라미터 (>= 0)
    pub lambda: f64,
    /// 위상 레지스터 (20비트)
    pub phi_q: u32,
    /// 로그 진폭 log2(A)
    pub log2_amp: f64,
}

impl Atom {
    /// 미분 위상 이동 delta = atan2(lambda, rho) 의 격자 양자화 (1회 저장용)
    pub fn q_delta(&self) -> u32 {
        PhaseState::quantize_phase(self.lambda.atan2(RHO))
    }

    /// 미분당 로그 진폭 증가량: (1/2) log2(rho^2 + lambda^2)
    pub fn log2_amp_step(&self) -> f64 {
        0.5 * (RHO * RHO + self.lambda * self.lambda).log2()
    }

    /// 주어진 Busemann 좌표값 b 에서의 평가 (측지 좌표계 — 미분 검증용 코어)
    pub fn eval_at_b(&self, b: f64) -> f64 {
        let amp = self.log2_amp.exp2();
        amp * (RHO * b).exp() * (self.lambda * b + PhaseState::phase_value(self.phi_q)).cos()
    }

    /// 원판 좌표 (r, theta) 에서의 평가 (프로덕션 경로: 상쇄 없는 극좌표 Busemann)
    pub fn eval(&self, r: f64, theta: f64) -> f64 {
        self.eval_at_b(busemann_polar(r, theta, self.theta_b))
    }

    /// n계 미분을 상태 연산만으로: 위상 += n * q_delta (단일 덧셈), 로그 진폭 += n * step
    /// 반환된 원자를 eval 하면 n계 도함수 값이 나온다 (정리 13.2 + 부록 A.3)
    pub fn differentiated_n(&self, n: u32) -> Atom {
        Atom {
            theta_b: self.theta_b,
            lambda: self.lambda,
            phi_q: (self.phi_q.wrapping_add(self.q_delta().wrapping_mul(n))) & PHASE_MASK,
            log2_amp: self.log2_amp + n as f64 * self.log2_amp_step(),
        }
    }

    /// 1계 미분
    pub fn differentiated(&self) -> Atom {
        self.differentiated_n(1)
    }
}

/// 로그 진폭 16비트 양자화: a = round(log2A * 2048) + 32768  [논문 4.1절 레이아웃]
pub fn quantize_amp(log2_amp: f64) -> u16 {
    let code = (log2_amp * 2048.0).round() + 32768.0;
    code.clamp(0.0, 65535.0) as u16
}

pub fn dequantize_amp(code: u16) -> f64 {
    (code as f64 - 32768.0) / 2048.0
}

/// tanh 트랙 도함수 지름길 (논문 3.4절): W = A tanh(u) 일 때
/// dW/du = A (1 - tanh^2 u) = A - W^2 / A  — 순전파 값에서 FMA 2회로 도함수를 얻는다.
pub fn tanh_derivative_shortcut(w: f64, amp: f64) -> f64 {
    amp - w * w / amp
}

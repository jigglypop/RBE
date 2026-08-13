//! 검증 하네스 기반 모듈 (docs/test/verification_harness.md CP-1)
//!
//! 규칙: 테스트의 모든 허용오차는 이 모듈의 `bounds` 함수에서만 나온다.
//! 테스트 코드에 부동소수점 리터럴 허용오차를 직접 쓰는 것은 하네스 위반이다.
//! 각 bounds 함수는 논문(PAPER_POINCARE_RBE.md) 부록의 유도 번호를 주석으로 참조한다.

/// 닫힌형 허용오차 계산기. 값 변경은 논문 유도 변경 + 사용자 승인을 요구한다.
pub mod bounds {
    use std::f64::consts::{LN_2, PI};

    /// f32 단위 반올림 u = 2^-24 (정확값)
    pub const U32: f64 = 1.0 / 16_777_216.0;
    /// f64 단위 반올림 u = 2^-53 (정확값)
    pub const U64: f64 = 1.0 / 9_007_199_254_740_992.0;

    /// 사분파 LUT 선형 보간 오차 상계: (1/8) * (pi / 2^(k+1))^2  [부록 E.3]
    pub fn lut_interp(k: u32) -> f64 {
        let h = PI / 2f64.powi(k as i32 + 1);
        h * h / 8.0
    }

    /// (1-r)(1+r) 형태의 1-r^2 계산 상대오차 상계: 2u + u^2  [부록 E.2, E.1]
    pub fn sterbenz_product() -> f64 {
        2.0 * U32 + U32 * U32
    }

    /// b비트 위상 양자화의 반스텝 오차: pi / 2^b  [논문 7.3절]
    pub fn phase_quant(b: u32) -> f64 {
        PI / 2f64.powi(b as i32)
    }

    /// 로그 진폭 양자화 상대오차 반스텝: ln2 / 2^(frac_bits+1)  [논문 7.3절]
    pub fn amp_quant(frac_bits: u32) -> f64 {
        LN_2 / 2f64.powi(frac_bits as i32 + 1)
    }

    /// f32 내적(길이 n)의 전진오차 계수 gamma_n = n*u/(1-n*u)  [Higham, 논문 15.4절 검증용]
    /// 주의: 이 상계는 Sum|x_i y_i| 에 상대적이다 (|Sum x_i y_i| 가 아님).
    pub fn dot_product(n: usize) -> f64 {
        let nu = n as f64 * U32;
        assert!(nu < 1.0, "dot_product 상계는 n < 1/u 에서만 유효");
        nu / (1.0 - nu)
    }

    /// f64 중심차분 전체 오차 상계: h^2 * M3 / 6 + 2u * M0 / h
    /// (절단 오차 + 반올림 오차; M0 = 함수값 상계, M3 = 3계 도함수 상계)
    pub fn central_diff(h: f64, m0: f64, m3: f64) -> f64 {
        h * h * m3 / 6.0 + 2.0 * U64 * m0 / h
    }

    /// 위 상계를 최소화하는 스텝: h* = (3u*M0/M3)^(1/3)
    pub fn central_diff_h_opt(m0: f64, m3: f64) -> f64 {
        (3.0 * U64 * m0 / m3).cbrt()
    }

    /// RMSE 추정치의 상대 신뢰구간 (카이제곱 CLT, 5-시그마 정책): 5 / sqrt(2n)
    pub fn rmse_ci_rel(n: usize) -> f64 {
        5.0 / (2.0 * n as f64).sqrt()
    }

    /// 샤논 하한 가드: sigma * 2^-bpw * (1 - CI)  [부록 D.1 + 유한크기 완화]
    /// 인코더 실측 RMSE 가 이 값 미만이면 측정/구현 버그로 판정한다.
    pub fn shannon_floor(sigma: f64, bpw: f64, n: usize) -> f64 {
        sigma * 2f64.powf(-bpw) * (1.0 - rmse_ci_rel(n))
    }

    /// f64 초등함수 합성(연산 k회)의 반올림 전파 상계(상대): 2k * u
    /// (각 연산 1 ulp + 전파 여유 2배; 오라클 자기검증 전용.
    ///  주의: 감산 상쇄가 있는 식에는 부적합 — 상쇄 증폭 항을 포함한 전용 상계를 유도할 것)
    pub fn f64_chain(ops: u32) -> f64 {
        2.0 * ops as f64 * U64
    }

    /// f32 출력 LUT 평가의 합성 상계: 보간 오차 + 출력 반올림 반 ulp  [부록 E.3 + 7.3절]
    /// |sin| <= 1 이므로 f32 반올림 절대오차 <= 2^-25 = U32/2.
    pub fn lut_eval_f32(k: u32) -> f64 {
        lut_interp(k) + U32 / 2.0
    }

    /// 오라클 부제만-라디얼 자기검증(부록 F.3) 상계.
    ///
    /// 유도: 오라클은 정의 직역이므로 소박한 1-|z|^2, |z-b|^2 감산을 포함하고,
    /// r -> 1 에서 상쇄 증폭이 일어난다 (논문 7.2절이 경고하는 바로 그 현상).
    /// 1차 오차 전파:
    ///   num = 1-|z|^2 의 상대오차 <= u * (2 + 6 * r^2/(1-r^2))   [|z|^2 반올림의 상쇄 증폭]
    ///   den = |z-b|^2 의 상대오차 <= u * (4 + 4 * r/(1-r))       [z-b 감산의 상쇄 증폭]
    ///   B = ln(num) - ln(den) 의 절대오차 <= 두 상대오차의 합
    /// 상수를 보수적으로 8/8 로 잡고 |B| = 2 artanh r 로 정규화한다.
    pub fn busemann_radial_oracle(r: f64) -> f64 {
        let one_minus_r2 = (1.0 - r) * (1.0 + r);
        let amp = 8.0 + 8.0 * (r * r / one_minus_r2 + r / (1.0 - r));
        let b = 2.0 * r.atanh();
        amp * U64 / b.abs().max(1.0)
    }
}

/// f64 직역 참조 구현. 최적화 금지, 프로덕션 경로 호출 금지 (자기 참조 검증 방지).
pub mod oracle {
    /// Busemann 좌표 B_b(z) = log[(1-|z|^2) / |z-b|^2], b = (cos t, sin t)  [논문 13.2절 정의 직역]
    pub fn busemann(z: (f64, f64), theta_b: f64) -> f64 {
        let (x, y) = z;
        let (bx, by) = (theta_b.cos(), theta_b.sin());
        let num = 1.0 - (x * x + y * y);
        let den = (x - bx) * (x - bx) + (y - by) * (y - by);
        (num / den).ln()
    }

    /// 중심차분 (f64)
    pub fn central_diff(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }
}

/// 판정 헬퍼: measured <= bound 가 아니면 (measured, bound, ratio) 를 출력하며 실패.
pub fn check(name: &str, measured: f64, bound: f64) {
    assert!(
        measured <= bound,
        "[하네스 위반] {}: measured={:e} bound={:e} ratio={:.3}",
        name,
        measured,
        bound,
        measured / bound
    );
}

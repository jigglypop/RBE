//! CP-5: 평면파 원자 미분 검증 (하네스 3-4절)
//! (a) 위상이동 미분 경로 == 해석 공식 (양자화 1회 오차 상계)
//! (b) 미분값 == f64 중심차분 (스텝 자동 산출, 상계는 정리 13.2 자체가 주는 도함수 상계)
//! (c) n계 상태 누적의 정확성, tanh 지름길, 위상+진폭 양자화 바닥

use crate::core::math::atom::{dequantize_amp, quantize_amp, tanh_derivative_shortcut, Atom, RHO};
use crate::core::math::phase_state::{PhaseState, PHASE_MASK};
use crate::core::math::verification::{bounds, check, oracle};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

fn random_atom(rng: &mut StdRng) -> Atom {
    Atom {
        theta_b: rng.gen_range(0.0..2.0 * PI),
        lambda: rng.gen_range(0.0..50.0f64),
        phi_q: rng.r#gen::<u32>() & PHASE_MASK,
        log2_amp: rng.gen_range(-2.0..2.0f64),
    }
}

#[test]
fn 위상이동미분_해석식_동치() {
    // 정리 13.2: 비트 경로(양자화된 q_delta)와 해석 공식(정확한 delta)의 차이는
    // delta 양자화 반스텝뿐: |차이| <= A' * phase_quant(20), A' = A sqrt(rho^2+lambda^2) e^{rho B}
    let mut rng = StdRng::seed_from_u64(0x5242_4551);
    for _ in 0..20_000 {
        let atom = random_atom(&mut rng);
        let b = rng.gen_range(-3.0..3.0f64);

        // (a) 비트 경로: 상태 미분 후 같은 평가기 재사용
        let d_bit = atom.differentiated().eval_at_b(b);

        // (b) 해석 공식: A sqrt(rho^2+lambda^2) e^{rho b} cos(lambda b + phi + delta)
        let amp = atom.log2_amp.exp2();
        let scale = (RHO * RHO + atom.lambda * atom.lambda).sqrt();
        let delta = atom.lambda.atan2(RHO);
        let d_exact = amp
            * scale
            * (RHO * b).exp()
            * (atom.lambda * b + PhaseState::phase_value(atom.phi_q) + delta).cos();

        let envelope = amp * scale * (RHO * b).exp();
        check(
            "위상이동미분 해석식 동치",
            (d_bit - d_exact).abs(),
            envelope * (bounds::phase_quant(20) + bounds::f64_chain(16)),
        );
    }
}

#[test]
fn 미분_중심차분_대조() {
    // 해석 도함수(정확한 delta)를 f64 중심차분과 대조.
    // m0, m3 는 정리 13.2 가 닫힌형으로 준다:
    //   |f^(k)| <= A (rho^2+lambda^2)^{k/2} e^{rho b} 이므로
    //   m0 = 1.1 A e^{rho b}, m3 = 1.1 A (rho^2+lambda^2)^{3/2} e^{rho b}
    let mut rng = StdRng::seed_from_u64(0x5242_4552);
    for _ in 0..10_000 {
        let atom = random_atom(&mut rng);
        let b = rng.gen_range(-3.0..3.0f64);

        let amp = atom.log2_amp.exp2();
        let scale2 = RHO * RHO + atom.lambda * atom.lambda;
        let envelope = amp * (RHO * b).exp();
        // 함수값 평가 오차는 순수 반올림 u*m0 이 아니라 cos 인자 조건수가 지배한다:
        //   arg = lambda*b + phi 가 크면 인자 자체의 반올림 u*|arg| 가 |sin|<=1 을 타고
        //   절대오차 u * envelope * |arg| 로 전파된다. m0 에 이 조건수를 포함한다 (유도 보정).
        let arg_mag = atom.lambda * (b.abs() + 1.0) + 2.0 * std::f64::consts::PI;
        let m0 = envelope * (4.0 + arg_mag);
        let m3 = 1.1 * envelope * scale2.powf(1.5);
        let h = bounds::central_diff_h_opt(m0, m3);

        let numeric = oracle::central_diff(|t| atom.eval_at_b(t), b, h);
        let delta = atom.lambda.atan2(RHO);
        let exact = amp
            * scale2.sqrt()
            * (RHO * b).exp()
            * (atom.lambda * b + PhaseState::phase_value(atom.phi_q) + delta).cos();

        check(
            "미분 중심차분 대조",
            (numeric - exact).abs(),
            2.0 * bounds::central_diff(h, m0, m3),
        );
    }
}

#[test]
fn n계미분_상태누적_동일성() {
    // 부록 A.3: n계 = 단일 덧셈. 위상 레지스터는 반복 적용과 비트 동일,
    // 로그 진폭은 n * step 닫힌형과 f64 합산 순서 차이 이내.
    let mut rng = StdRng::seed_from_u64(0x5242_4553);
    for _ in 0..5_000 {
        let atom = random_atom(&mut rng);
        let n = rng.gen_range(0u32..16);

        let single = atom.differentiated_n(n);
        let mut repeated = atom;
        for _ in 0..n {
            repeated = repeated.differentiated();
        }
        assert_eq!(single.phi_q, repeated.phi_q, "위상 누적 불일치 n={}", n);
        check(
            "로그진폭 n계 누적",
            (single.log2_amp - repeated.log2_amp).abs(),
            bounds::f64_chain(2 * n.max(1)) * single.log2_amp.abs().max(1.0),
        );
    }
}

#[test]
fn tanh지름길_동치() {
    // 논문 3.4절: A - W^2/A == A (1 - tanh^2 u)
    let mut rng = StdRng::seed_from_u64(0x5242_4554);
    for _ in 0..50_000 {
        let u = rng.gen_range(-5.0..5.0f64);
        let amp = rng.gen_range(0.1..10.0f64);
        let w = amp * u.tanh();
        let shortcut = tanh_derivative_shortcut(w, amp);
        let direct = amp * (1.0 - u.tanh() * u.tanh());
        check(
            "tanh 지름길",
            (shortcut - direct).abs(),
            bounds::f64_chain(8) * amp,
        );
    }
}

#[test]
fn 위상_진폭_양자화_바닥() {
    // 논문 7.3절 (위상·진폭 성분): 위상 20비트 + 로그 진폭 16비트 양자화의 재현 오차는
    //   |dW| <= A e^{rho B} * (phase_quant(20) + amp_quant(11) * (1 + amp_quant(11)))
    // (위상: |dW/dphi| <= A e^{rho B}; 진폭: 상대 오차 반스텝 ln2/2^12)
    let mut rng = StdRng::seed_from_u64(0x5242_4555);
    for _ in 0..20_000 {
        let phi = rng.gen_range(0.0..2.0 * PI);
        let log2_amp = rng.gen_range(-2.0..2.0f64);
        let lambda = rng.gen_range(0.0..50.0f64);
        let b = rng.gen_range(-3.0..3.0f64);

        // 연속 원자 (오라클)
        let exact = log2_amp.exp2() * (RHO * b).exp() * (lambda * b + phi).cos();

        // 양자화된 원자 (프로덕션): 위상 20비트, 진폭 16비트 (lambda, b 는 이 테스트에서 정확)
        let atom = Atom {
            theta_b: 0.0,
            lambda,
            phi_q: PhaseState::quantize_phase(phi),
            log2_amp: dequantize_amp(quantize_amp(log2_amp)),
        };
        let quantized = atom.eval_at_b(b);

        let envelope = log2_amp.exp2() * (RHO * b).exp();
        let amp_rel = bounds::amp_quant(11);
        check(
            "위상+진폭 양자화 바닥",
            (quantized - exact).abs(),
            envelope
                * (bounds::phase_quant(20) + amp_rel * (1.0 + amp_rel) + bounds::f64_chain(16)),
        );
    }
}

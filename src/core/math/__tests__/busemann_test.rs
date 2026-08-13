//! CP-4: Busemann 좌표 L1/L2 검증 (하네스 3-4절)
//! 모든 허용오차는 bounds 함수 + 케이스별 증폭 인자의 유도식으로만 구성한다.

use crate::core::math::busemann::{busemann_polar, busemann_xy, poisson_polar, Mobius};
use crate::core::math::verification::{bounds, check, oracle};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

#[test]
fn 부제만_대수항등식() {
    // 부록 B.2 핵심 항등식: 2<z, z-b> + (1-|z|^2) == |z-b|^2
    // 모든 항이 O(1) 이므로 절대오차 상계 = f64_chain(8) * 최대항크기(4)
    let mut rng = StdRng::seed_from_u64(0x5242_4541);
    let bound = bounds::f64_chain(8) * 4.0;
    for _ in 0..100_000 {
        let r = rng.gen_range(0.0..0.999f64);
        let t = rng.gen_range(0.0..2.0 * PI);
        let tb = rng.gen_range(0.0..2.0 * PI);
        let (x, y) = (r * t.cos(), r * t.sin());
        let (bx, by) = (tb.cos(), tb.sin());
        let (dx, dy) = (x - bx, y - by);
        let lhs = 2.0 * (x * dx + y * dy) + (1.0 - (x * x + y * y));
        let rhs = dx * dx + dy * dy;
        check("부제만 대수항등식", (lhs - rhs).abs(), bound);
    }
}

#[test]
fn 극좌표식_직교좌표식_일치() {
    // 상쇄 없는 극좌표 형(프로덕션)과 직역 직교좌표 형의 일치.
    // 케이스별 상계: 직교좌표 형의 상쇄 증폭 (1/(1-r^2), 1/den) 포함.
    let mut rng = StdRng::seed_from_u64(0x5242_4542);
    for _ in 0..100_000 {
        let r = rng.gen_range(0.0..0.9f64);
        let t = rng.gen_range(0.0..2.0 * PI);
        let tb = rng.gen_range(0.0..2.0 * PI);
        let b_polar = busemann_polar(r, t, tb);
        let b_xy = busemann_xy(r * t.cos(), r * t.sin(), tb.cos(), tb.sin());
        let s = (0.5 * (t - tb)).sin();
        let den = (1.0 - r) * (1.0 - r) + 4.0 * r * s * s;
        let amp = 1.0 + 1.0 / ((1.0 - r) * (1.0 + r)) + 1.0 / den;
        check(
            "극좌표-직교좌표 일치",
            (b_polar - b_xy).abs(),
            bounds::f64_chain(10) * amp,
        );
    }
}

#[test]
fn 부제만_단위기울기_수치검증() {
    // 부록 B.2: ||grad B||_hyp = 1. 중심차분(f64)으로 유클리드 기울기를 재고
    // 등각 인자 lambda = 2/(1-r^2) 로 나눈다. 상계는 중심차분 유도식.
    // 3계 도함수 상계: log 합성의 도함수 스케일 <= 4 * (2/(1-r_max))^3, r_max = r + 2h.
    let mut rng = StdRng::seed_from_u64(0x5242_4543);
    for _ in 0..2_000 {
        let r = rng.gen_range(0.0..0.85f64);
        let t = rng.gen_range(0.0..2.0 * PI);
        let tb = rng.gen_range(0.0..2.0 * PI);
        let (x0, y0) = (r * t.cos(), r * t.sin());
        let (bx, by) = (tb.cos(), tb.sin());

        let m0 = busemann_xy(x0, y0, bx, by).abs() + 1.0;
        let m3 = 4.0 * (2.0 / (1.0 - 0.87f64)).powi(3);
        let h = bounds::central_diff_h_opt(m0, m3);
        let gx = oracle::central_diff(|x| busemann_xy(x, y0, bx, by), x0, h);
        let gy = oracle::central_diff(|y| busemann_xy(x0, y, bx, by), y0, h);

        let lambda = 2.0 / ((1.0 - r) * (1.0 + r));
        let hyp_norm = (gx * gx + gy * gy).sqrt() / lambda;
        check(
            "부제만 단위기울기",
            (hyp_norm - 1.0).abs(),
            2.0 * bounds::central_diff(h, m0, m3),
        );
    }
}

#[test]
fn 호로사이클_등위선() {
    // 부록 B.3: P(z,b) = c 의 등위선은 중심 (c/(c+1))*b, 반지름 1/(c+1) 인 원 (b 에 내접).
    // 그 원 위의 점에서 P == c 를 확인한다. b 근방(den -> 0)은 증폭이 커지므로 t 를 제한.
    for c in [0.5f64, 1.0, 2.0, 5.0] {
        let center = c / (c + 1.0);
        let radius = 1.0 / (c + 1.0);
        for i in 0..1_000 {
            let t = 0.15 + (2.0 * PI - 0.3) * i as f64 / 1_000.0;
            let x = center + radius * t.cos();
            let y = radius * t.sin();
            let r = (x * x + y * y).sqrt();
            let theta = y.atan2(x);
            let p = poisson_polar(r, theta); // theta_b = 0
            let s = (0.5 * theta).sin();
            let den = (1.0 - r) * (1.0 - r) + 4.0 * r * s * s;
            let amp = 1.0 + 1.0 / den + 1.0 / ((1.0 - r) * (1.0 + r));
            check(
                "호로사이클 등위선",
                (p - c).abs() / c,
                bounds::f64_chain(16) * amp,
            );
        }
    }
}

#[test]
fn 뫼비우스_공변성() {
    // 부록 C.3: P(gz, gb) * |g'(b)| == P(z, b).
    // 케이스별 상계: 변환 후 좌표의 상쇄 증폭 인자를 실제 값에서 합성.
    let mut rng = StdRng::seed_from_u64(0x5242_4544);
    for _ in 0..20_000 {
        let r = rng.gen_range(0.0..0.9f64);
        let t = rng.gen_range(0.0..2.0 * PI);
        let tb = rng.gen_range(0.0..2.0 * PI);
        let ar = rng.gen_range(0.0..0.9f64);
        let at = rng.gen_range(0.0..2.0 * PI);
        let g = Mobius::new(ar * at.cos(), ar * at.sin(), rng.gen_range(0.0..2.0 * PI));

        let (x, y) = (r * t.cos(), r * t.sin());
        let (bx, by) = (tb.cos(), tb.sin());
        let p_orig = poisson_polar(r, t - tb);

        let (gx, gy) = g.apply(x, y);
        let (gbx, gby) = g.apply_boundary(bx, by);
        let p_img = {
            let num = 1.0 - (gx * gx + gy * gy);
            let (dx, dy) = (gx - gbx, gy - gby);
            num / (dx * dx + dy * dy)
        };
        let lhs = p_img * g.deriv_boundary_abs(bx, by);

        let gr2 = gx * gx + gy * gy;
        let gden = (gx - gbx) * (gx - gbx) + (gy - gby) * (gy - gby);
        let amp = 1.0 + 1.0 / (1.0 - gr2) + 1.0 / gden + 1.0 / ((1.0 - r) * (1.0 + r));
        check(
            "뫼비우스 공변성",
            (lhs - p_orig).abs() / p_orig,
            bounds::f64_chain(32) * amp,
        );
    }
}

#[test]
fn 라디얼_일치_경계까지_프로덕션() {
    // 부록 F.3: B_b(r*b) = 2 artanh r. 프로덕션 극좌표 형은 상쇄가 없으므로
    // r = 1 - 2^-30 까지 균일한 상계로 성립해야 한다 (CP-1 오라클 소박형과 대조되는 지점).
    for j in 1..=30 {
        let r = 1.0 - 0.5f64.powi(j);
        for tb in [0.0, 1.7, -2.4] {
            let b_val = busemann_polar(r, tb, tb);
            let exact = 2.0 * r.atanh();
            check(
                "라디얼 일치 (프로덕션, 경계 포함)",
                (b_val - exact).abs() / exact.abs().max(1.0),
                bounds::f64_chain(6),
            );
        }
    }
}

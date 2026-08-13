//! Busemann 좌표와 뫼비우스 작용 (논문 13.2절, 부록 B·C; 하네스 CP-4)
//!
//! 프로덕션 경로는 극좌표 형을 사용한다:
//!   P(z,b) = num / den,  num = (1-r)(1+r)          [Sterbenz 형, 부록 E]
//!   den = |z-b|^2 = (1-r)^2 + 4 r sin^2(dtheta/2)  [모든 항 비음 — 상쇄 없음]
//! 직교좌표 형(busemann_xy)은 r -> 1 에서 1-|z|^2 감산 상쇄가 있으므로
//! 일반 용도/검증 전용이며, 원자 평가는 극좌표 형만 쓴다 (하네스 9.3).

/// 푸아송 핵 P(z,b), z = (r, theta), b = theta_b 방향 경계점, dtheta = theta - theta_b
pub fn poisson_polar(r: f64, dtheta: f64) -> f64 {
    let num = (1.0 - r) * (1.0 + r);
    let s = (0.5 * dtheta).sin();
    let den = (1.0 - r) * (1.0 - r) + 4.0 * r * s * s;
    num / den
}

/// Busemann 좌표 B_b(z) = ln P(z,b) (극좌표, 상쇄 없는 프로덕션 형)
pub fn busemann_polar(r: f64, theta: f64, theta_b: f64) -> f64 {
    poisson_polar(r, theta - theta_b).ln()
}

/// Busemann 좌표 (직교좌표 형 — r ~ 1 에서 상쇄 주의, 검증/일반 용도)
pub fn busemann_xy(x: f64, y: f64, cos_b: f64, sin_b: f64) -> f64 {
    let num = 1.0 - (x * x + y * y);
    let dx = x - cos_b;
    let dy = y - sin_b;
    (num / (dx * dx + dy * dy)).ln()
}

/// 원판 자기동형 g(z) = e^{i rot} (z - a) / (1 - conj(a) z),  |a| < 1
#[derive(Clone, Copy, Debug)]
pub struct Mobius {
    pub ax: f64,
    pub ay: f64,
    pub rot: f64,
}

fn c_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn c_div(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

impl Mobius {
    pub fn new(ax: f64, ay: f64, rot: f64) -> Self {
        debug_assert!(ax * ax + ay * ay < 1.0, "뫼비우스 중심은 원판 내부여야 함");
        Self { ax, ay, rot }
    }

    /// g(z) — 원판 내부 점에 적용
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let numer = (x - self.ax, y - self.ay);
        // 1 - conj(a) z = 1 - (ax - i ay)(x + i y)
        let denom = (
            1.0 - (self.ax * x + self.ay * y),
            -(self.ax * y - self.ay * x),
        );
        let w = c_div(numer, denom);
        c_mul(((self.rot).cos(), (self.rot).sin()), w)
    }

    /// 경계점 b = (cos t, sin t) 의 상. 수치 안정성을 위해 단위 원으로 재정규화한다
    /// (경계는 경계로 사상되는 것이 정확한 수학적 사실이므로 정규화는 반올림 보정일 뿐).
    pub fn apply_boundary(&self, cos_b: f64, sin_b: f64) -> (f64, f64) {
        let (gx, gy) = self.apply(cos_b, sin_b);
        let n = (gx * gx + gy * gy).sqrt();
        (gx / n, gy / n)
    }

    /// |g'(b)| = (1 - |a|^2) / |1 - conj(a) b|^2  [부록 C.3]
    pub fn deriv_boundary_abs(&self, cos_b: f64, sin_b: f64) -> f64 {
        let a2 = self.ax * self.ax + self.ay * self.ay;
        let dr = 1.0 - (self.ax * cos_b + self.ay * sin_b);
        let di = -(self.ax * sin_b - self.ay * cos_b);
        (1.0 - a2) / (dr * dr + di * di)
    }
}

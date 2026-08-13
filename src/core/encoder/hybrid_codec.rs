//! 하이브리드 코덱 (논문 17장 R2 운용점; 하네스 CP-7)
//!
//! 구성: W ~= K(원자 구조부) + Q_b(W - K)(백색화 잔차의 b비트 양자화)
//! - 잔차 양자화기는 가우시안 Lloyd-Max 를 런타임에 수렴시켜 얻는다
//!   (레벨을 하드코딩하지 않고 해석적 중심(무게중심) 공식으로 반복 — 상수 이식 오류 방지).
//! - c(원자 에너지 포획률) 추정기: c = 1 - ||W-K||^2 / ||W||^2  (논문 17.1절)
//!
//! 이론 왜곡 (단위 분산): D = 1 - sum_i p_i * l_i^2  (중심 조건 하 해석적)

use libm::erf;
use std::f64::consts::PI;

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// 표준정규 구간 [a, b] 의 확률질량과 무게중심
fn cell_mass_centroid(a: f64, b: f64) -> (f64, f64) {
    let mass = norm_cdf(b) - norm_cdf(a);
    let centroid = if mass > 0.0 {
        (norm_pdf(a) - norm_pdf(b)) / mass
    } else {
        0.5 * (a + b)
    };
    (mass, centroid)
}

/// 가우시안 Lloyd-Max 스칼라 양자화기 (단위 분산 기준)
pub struct LloydMaxQuantizer {
    pub levels: Vec<f64>,
    pub thresholds: Vec<f64>,
    /// 해석적 상대 왜곡 D = E[(x - q(x))^2] / sigma^2 = 1 - sum p_i l_i^2
    pub distortion_rel: f64,
}

impl LloydMaxQuantizer {
    pub fn new_gaussian(bits: u32) -> Self {
        let n = 1usize << bits;
        // 초기 레벨: [-3, 3] 균등
        let mut levels: Vec<f64> = (0..n)
            .map(|i| -3.0 + 6.0 * (i as f64 + 0.5) / n as f64)
            .collect();
        let mut thresholds = vec![0.0; n - 1];
        // Lloyd 반복: 임계 = 이웃 레벨 중점, 레벨 = 셀 무게중심
        for _ in 0..500 {
            for i in 0..n - 1 {
                thresholds[i] = 0.5 * (levels[i] + levels[i + 1]);
            }
            for i in 0..n {
                let a = if i == 0 {
                    f64::NEG_INFINITY
                } else {
                    thresholds[i - 1]
                };
                let b = if i == n - 1 {
                    f64::INFINITY
                } else {
                    thresholds[i]
                };
                let (_, c) = cell_mass_centroid(a, b);
                levels[i] = c;
            }
        }
        // 해석적 왜곡
        let mut d = 1.0;
        for i in 0..n {
            let a = if i == 0 {
                f64::NEG_INFINITY
            } else {
                thresholds[i - 1]
            };
            let b = if i == n - 1 {
                f64::INFINITY
            } else {
                thresholds[i]
            };
            let (p, _) = cell_mass_centroid(a, b);
            d -= p * levels[i] * levels[i];
        }
        Self {
            levels,
            thresholds,
            distortion_rel: d,
        }
    }

    pub fn quantize(&self, x: f64) -> usize {
        // 임계 선형 탐색 (레벨 수 <= 16)
        self.thresholds.iter().take_while(|&&t| x > t).count()
    }

    pub fn index_to_value(&self, idx: usize) -> f64 {
        self.levels[idx]
    }

    /// 첨도 민감도 D' = (1/24) sum_i int_cell (x-l_i)^2 He4(x) phi(x) dx  [부록 D.3]
    /// 대칭 양자화기에서 왜도 항은 정확히 상쇄되므로 4차 항이 최초 보정이다.
    pub fn kurtosis_sensitivity(&self) -> f64 {
        let n = self.levels.len();
        let mut d4 = 0.0;
        for i in 0..n {
            let a = if i == 0 {
                f64::NEG_INFINITY
            } else {
                self.thresholds[i - 1]
            };
            let b = if i == n - 1 {
                f64::INFINITY
            } else {
                self.thresholds[i]
            };
            let l = self.levels[i];
            let im = truncated_moments(a, b);
            // (x-l)^2 He4(x) = x^6 - 2l x^5 + (l^2-6)x^4 + 12l x^3 + (3-6l^2)x^2 - 6l x + 3l^2
            let co = [
                3.0 * l * l,
                -6.0 * l,
                3.0 - 6.0 * l * l,
                12.0 * l,
                l * l - 6.0,
                -2.0 * l,
                1.0,
            ];
            d4 += co.iter().enumerate().map(|(m, c)| c * im[m]).sum::<f64>();
        }
        d4 / 24.0
    }

    /// 첨도 1차 보정 상대 왜곡 (부록 D.3): D(kappa) = D_phi + kappa * D'.
    /// kappa = 0 이면 가우시안 해석값으로 환원. 유효 범위 |kappa| << 1.
    pub fn distortion_rel_kurtosis(&self, kappa: f64) -> f64 {
        self.distortion_rel + kappa * self.kurtosis_sensitivity()
    }
}

/// 표준정규 절단 모멘트 I_m = int_a^b x^m phi dx, m = 0..=6 의 부분적분 점화식 [부록 D.3]
/// I_0 = Phi(b)-Phi(a), I_1 = phi(a)-phi(b), I_m = (m-1) I_{m-2} + a^{m-1}phi(a) - b^{m-1}phi(b)
fn truncated_moments(a: f64, b: f64) -> [f64; 7] {
    let tail = |x: f64, m: u32| {
        if x.is_finite() {
            x.powi(m as i32) * norm_pdf(x)
        } else {
            0.0
        }
    };
    let mut im = [0.0; 7];
    im[0] = norm_cdf(b) - norm_cdf(a);
    im[1] = tail(a, 0) - tail(b, 0);
    for m in 2..7 {
        im[m] = (m - 1) as f64 * im[m - 2] + tail(a, m as u32 - 1) - tail(b, m as u32 - 1);
    }
    im
}

/// 잔차 부호화: 표본 표준편차로 정규화 후 Lloyd-Max 인덱스
pub fn encode_residual(residual: &[f64], q: &LloydMaxQuantizer) -> (Vec<u8>, f64) {
    let n = residual.len() as f64;
    let sigma = (residual.iter().map(|e| e * e).sum::<f64>() / n).sqrt();
    let scale = if sigma > 0.0 { sigma } else { 1.0 };
    let idx = residual
        .iter()
        .map(|&e| q.quantize(e / scale) as u8)
        .collect();
    (idx, sigma)
}

pub fn decode_residual(indices: &[u8], sigma: f64, q: &LloydMaxQuantizer) -> Vec<f64> {
    indices
        .iter()
        .map(|&i| sigma * q.index_to_value(i as usize))
        .collect()
}

/// 원자 에너지 포획률 c = 1 - ||W - K||^2 / ||W||^2  (논문 17.1절)
pub fn energy_capture(w: &[f64], k: &[f64]) -> f64 {
    assert_eq!(w.len(), k.len());
    let total: f64 = w.iter().map(|x| x * x).sum();
    if total == 0.0 {
        return 0.0;
    }
    let residual: f64 = w.iter().zip(k).map(|(x, y)| (x - y) * (x - y)).sum();
    1.0 - residual / total
}

/// 하이브리드 왕복: K + Q_b(W - K) 재현과 RMSE
pub fn hybrid_roundtrip(w: &[f64], k: &[f64], q: &LloydMaxQuantizer) -> (Vec<f64>, f64) {
    let residual: Vec<f64> = w.iter().zip(k).map(|(x, y)| x - y).collect();
    let (idx, sigma) = encode_residual(&residual, q);
    let deq = decode_residual(&idx, sigma, q);
    let recon: Vec<f64> = k.iter().zip(&deq).map(|(a, b)| a + b).collect();
    let mse = w
        .iter()
        .zip(&recon)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / w.len() as f64;
    (recon, mse.sqrt())
}

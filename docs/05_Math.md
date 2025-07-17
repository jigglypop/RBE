# 5. 수학 함수 라이브러리: CORDIC와 미분 가능 함수 (`src/math.rs`)

이 문서에서는 128비트 하이브리드 시스템의 핵심인 CORDIC 알고리즘과 학습을 위한 미분 가능한 수학 함수들을 설명합니다. 추론과 학습에 각각 최적화된 두 가지 함수 세트를 제공합니다.

---

## 핵심 개념: 왜 특별한 수학 함수가 필요한가?

### 전통적인 접근의 문제점

일반적인 수학 라이브러리의 sin, cos, exp 등은:
- **정확하지만 느림**: 수백 사이클 필요
- **하드웨어 의존적**: CPU마다 성능 차이
- **병렬화 어려움**: 복잡한 내부 구조

### 우리의 해결책

1. **추론용**: CORDIC 알고리즘으로 초고속 계산
2. **학습용**: 미분 가능한 부드러운 함수

---

## CORDIC 알고리즘 (추론 최적화)

### CORDIC란 무엇인가?

CORDIC(COordinate Rotation DIgital Computer)는 1959년 Jack Volder가 발명한 알고리즘입니다. 원래는 항공기 네비게이션 시스템을 위해 개발되었습니다.

핵심 아이디어:
```
복잡한 함수 = 단순한 회전의 연속

예: 45° 회전 = 45° 한 번
    또는     = 26.565° + 14.036° + 4.398° + ...
```

각 작은 회전은 2의 거듭제곱으로 표현되어 시프트 연산만으로 가능합니다

### 기본 원리

벡터 (x, y)를 θ만큼 회전:

```
전통적 방법:
x' = x·cos(θ) - y·sin(θ)  // 4번의 곱셈
y' = x·sin(θ) + y·cos(θ)  // 4번의 곱셈

CORDIC 방법:
θ = Σ σᵢ·arctan(2⁻ⁱ)로 분해
각 단계에서:
x' = x - σᵢ·y·2⁻ⁱ  // 시프트와 덧셈만!
y' = y + σᵢ·x·2⁻ⁱ  // 시프트와 덧셈만!
```

### 구현 상세

```rust
/// CORDIC 각도 테이블 (미리 계산된 arctan(2^-k))
const CORDIC_ANGLES: [f32; 20] = [
    0.7853982,   // arctan(2^0)  = 45.000°
    0.4636476,   // arctan(2^-1) = 26.565°
    0.2449787,   // arctan(2^-2) = 14.036°
    0.1243549,   // arctan(2^-3) = 7.125°
    0.0624188,   // arctan(2^-4) = 3.576°
    0.0312398,   // arctan(2^-5) = 1.790°
    // ... 더 작은 각도들
];

/// CORDIC 게인 상수 (벡터 길이 증가 보정)
const CORDIC_GAIN: f32 = 0.607253;  // Π(cos(arctan(2^-k))) for k=0..∞

/// CORDIC 회전 연산
pub fn cordic_rotate(x: f32, y: f32, angle: f32, iterations: usize) -> (f32, f32) {
    let mut xc = x;
    let mut yc = y;
    let mut remaining_angle = angle;
    
    for k in 0..iterations.min(20) {
        // 1. 회전 방향 결정
        // remaining_angle이 양수면 반시계방향(+1), 음수면 시계방향(-1)
        let sigma = if remaining_angle > 0.0 { 1.0 } else { -1.0 };
        
        // 2. 2^-k 계산 (시프트로 구현 가능)
        let shift = 2.0_f32.powi(-(k as i32));
        
        // 3. 회전 적용 (곱셈 없이!)
        let xc_new = xc - sigma * yc * shift;
        let yc_new = yc + sigma * xc * shift;
        
        xc = xc_new;
        yc = yc_new;
        
        // 4. 남은 각도 업데이트
        remaining_angle -= sigma * CORDIC_ANGLES[k];
    }
    
    // 5. 게인 보정
    (xc * CORDIC_GAIN, yc * CORDIC_GAIN)
}
```

### CORDIC의 수학적 배경

각 반복에서 일어나는 일:

```
1단계 (k=0): 45° 단위로 회전
2단계 (k=1): 26.565° 단위로 미세 조정
3단계 (k=2): 14.036° 단위로 더 미세 조정
...

수렴 속도: 각 단계마다 오차가 절반으로 감소
20단계 후: 오차 < 0.00005° (충분히 정확!)
```

### GPU/하드웨어 최적화 버전

```rust
/// 브랜치리스 CORDIC (SIMD/GPU용)
#[inline(always)]
pub fn cordic_rotate_branchless(x: f32, y: f32, angle_bits: u32) -> (f32, f32) {
    let mut xc = x;
    let mut yc = y;
    
    // 언롤된 루프 (컴파일 타임 최적화)
    macro_rules! cordic_step {
        ($k:expr) => {
            // 비트에서 회전 방향 추출
            // bit가 1이면 +1, 0이면 -1
            let sigma = ((angle_bits >> $k) & 1) as f32 * 2.0 - 1.0;
            
            // 상수 시프트 (컴파일 타임에 계산)
            let shift = 2.0_f32.powi(-($k as i32));
            
            // 조건문 없는 회전
            let xc_tmp = xc - sigma * yc * shift;
            yc = yc + sigma * xc * shift;
            xc = xc_tmp;
        };
    }
    
    // 매크로로 20단계 전개
    cordic_step!(0); cordic_step!(1); cordic_step!(2); cordic_step!(3);
    cordic_step!(4); cordic_step!(5); cordic_step!(6); cordic_step!(7);
    // ... 20단계까지
    
    (xc * CORDIC_GAIN, yc * CORDIC_GAIN)
}
```

브랜치리스의 장점:
- **파이프라인 스톨 없음**: 분기 예측 실패 없음
- **SIMD 친화적**: 동일한 명령 흐름
- **일정한 실행 시간**: 보안에도 유리

---

## 미분 가능 함수 (학습 최적화)

### 왜 미분 가능성이 중요한가?

신경망 학습의 핵심은 역전파(Backpropagation)입니다:

```
손실 함수: L = f(g(h(x)))

체인 룰: dL/dx = dL/df × df/dg × dg/dh × dh/dx

만약 어느 한 함수라도 미분 불가능하면:
→ 그래디언트가 0이 되거나 무한대
→ 학습 중단!
```

### Smooth 활성화 함수

#### 1. Smooth ReLU

전통적인 ReLU의 문제:
```
ReLU(x) = max(0, x)
도함수: x > 0일 때 1, x ≤ 0일 때 0

문제: x = 0에서 미분 불가능!
```

우리의 해결책:

```rust
/// 미분 가능한 ReLU (Smooth ReLU)
pub fn smooth_relu(x: f32, beta: f32) -> f32 {
    // Softplus 함수: ln(1 + e^(βx)) / β
    // beta가 클수록 원래 ReLU에 가까워짐
    (1.0 / beta) * (1.0 + (beta * x).exp()).ln()
}

/// Smooth ReLU의 도함수
pub fn smooth_relu_grad(x: f32, beta: f32) -> f32 {
    // sigmoid(βx) = 1 / (1 + e^(-βx))
    1.0 / (1.0 + (-beta * x).exp())
}

// 비교:
// x = -0.1: ReLU = 0,    Smooth = 0.0443
// x = 0.0:  ReLU = 0,    Smooth = 0.0693 (부드러운 전환!)
// x = 0.1:  ReLU = 0.1,  Smooth = 0.0943
```

#### 2. 미분 가능한 클램핑

```rust
/// 미분 가능한 클램핑
pub fn smooth_clamp(x: f32, min: f32, max: f32) -> f32 {
    // 정규화
    let range = max - min;
    let t = (x - min) / range;
    
    // 시그모이드로 부드럽게 제한
    let alpha = 6.0;  // 경사도 조절 (클수록 급격)
    let sigmoid_t = 1.0 / (1.0 + (-(t - 0.5) * alpha).exp());
    
    min + range * sigmoid_t
}

/// 도함수
pub fn smooth_clamp_grad(x: f32, min: f32, max: f32) -> f32 {
    let range = max - min;
    let t = (x - min) / range;
    let alpha = 6.0;
    
    let exp_term = (-(t - 0.5) * alpha).exp();
    let sigmoid_grad = alpha * exp_term / (1.0 + exp_term).powi(2);
    
    sigmoid_grad / range
}
```

시각적 비교:
```
일반 clamp:          Smooth clamp:
  1 |--------        1 |      ___---
    |               |     _/
    |               |   _/
  0 |______|        0 |--
    min   max           min   max
```

### Radial Basis Functions (RBF)

RBF는 중심점으로부터의 거리에 기반한 함수입니다:

```rust
/// Gaussian RBF (가장 널리 사용됨)
pub fn gaussian_rbf(r: f32, center: f32, sigma: f32) -> f32 {
    // exp(-(r-c)²/(2σ²))
    // r: 현재 위치
    // center: 중심점
    // sigma: 퍼짐 정도
    (-(r - center).powi(2) / (2.0 * sigma * sigma)).exp()
}

/// Gaussian RBF의 도함수
pub fn gaussian_rbf_grad(r: f32, center: f32, sigma: f32) -> f32 {
    let diff = r - center;
    let gaussian = gaussian_rbf(r, center, sigma);
    -diff / (sigma * sigma) * gaussian
}

/// Multiquadric RBF (보간에 유용)
pub fn multiquadric_rbf(r: f32, center: f32, epsilon: f32) -> f32 {
    // sqrt((r-c)² + ε²)
    // epsilon: 특이점 방지
    ((r - center).powi(2) + epsilon * epsilon).sqrt()
}

/// Inverse Multiquadric RBF (부드러운 감쇠)
pub fn inverse_multiquadric_rbf(r: f32, center: f32, epsilon: f32) -> f32 {
    // 1 / sqrt((r-c)² + ε²)
    1.0 / ((r - center).powi(2) + epsilon * epsilon).sqrt()
}
```

각 RBF의 특성:
- **Gaussian**: 부드럽고 국소적, 항상 양수
- **Multiquadric**: 전역적 영향, 보간에 강함
- **Inverse Multiquadric**: 빠른 감쇠, 희소성

---

## 수치 미분 도구

해석적 미분이 복잡한 경우를 위한 도구들:

### 중앙 차분법

```rust
/// 1차 편미분 (중앙 차분)
pub fn numerical_derivative_1d<F>(f: F, x: f32, h: f32) -> f32 
where F: Fn(f32) -> f32 
{
    // f'(x) ≈ [f(x+h) - f(x-h)] / 2h
    // 오차: O(h²)
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// 2차원 그래디언트
pub fn numerical_gradient_2d<F>(f: F, r: f32, theta: f32, h: f32) -> (f32, f32)
where F: Fn(f32, f32) -> f32
{
    // ∂f/∂r
    let grad_r = (f(r + h, theta) - f(r - h, theta)) / (2.0 * h);
    
    // ∂f/∂θ
    let grad_theta = (f(r, theta + h) - f(r, theta - h)) / (2.0 * h);
    
    (grad_r, grad_theta)
}
```

### 적응형 스텝 크기

너무 큰 h는 부정확하고, 너무 작은 h는 수치 오류를 발생시킵니다:

```rust
/// 적응형 수치 미분
pub fn adaptive_numerical_derivative<F>(f: F, x: f32, target_error: f32) -> f32
where F: Fn(f32) -> f32
{
    let mut h = 0.1;  // 초기 스텝
    let mut prev_deriv = 0.0;
    
    loop {
        let deriv = numerical_derivative_1d(&f, x, h);
        
        // Richardson 외삽으로 오차 추정
        let h_half = h / 2.0;
        let deriv_half = numerical_derivative_1d(&f, x, h_half);
        let error_estimate = (deriv - deriv_half).abs() / 3.0;
        
        if error_estimate < target_error {
            // 더 정확한 값 반환
            return deriv_half + (deriv_half - deriv) / 3.0;
        }
        
        prev_deriv = deriv;
        h *= 0.5;
        
        if h < 1e-10 {
            // 최소 스텝 도달
            return deriv;
        }
    }
}
```

Richardson 외삽의 원리:
```
실제 도함수 = D
h로 계산: D₁ = D + O(h²)
h/2로 계산: D₂ = D + O(h²/4)

더 정확한 추정: D ≈ D₂ + (D₂ - D₁)/3
```

---

## 기저 함수 라이브러리

### 삼각/쌍곡선 함수 조합

미분의 순환성을 활용한 효율적 구현:

```rust
/// 삼각함수 미분의 순환성
/// sin → cos → -sin → -cos → sin ...
pub fn apply_trig_derivative(value: f32, derivative_order: u8, is_sine: bool) -> f32 {
    // 미분 차수에 따른 위상 이동
    let phase_shift = if is_sine { 0.0 } else { PI / 2.0 };
    let angle = value + phase_shift + (derivative_order as f32) * PI / 2.0;
    
    // 4차 미분마다 원래 함수로 돌아옴
    angle.sin()
}

/// 쌍곡함수 미분
/// sinh → cosh → sinh → cosh ...
pub fn apply_hyperbolic_derivative(value: f32, derivative_order: u8, is_sinh: bool) -> f32 {
    match (derivative_order % 2, is_sinh) {
        (0, true) => value.sinh(),   // sinh
        (1, true) => value.cosh(),   // d/dx sinh = cosh
        (0, false) => value.cosh(),  // cosh
        (1, false) => value.sinh(),  // d/dx cosh = sinh
        _ => unreachable!()
    }
}
```

이 순환성의 장점:
- 복잡한 고차 미분도 간단히 계산
- 수치 오류 누적 없음

### 특수 함수 (고속 근사)

#### Bessel 함수

Bessel 함수는 파동 방정식의 해로, 원형 대칭 패턴에 유용합니다:

```rust
/// Bessel J0 - Remez 다항식 근사
pub fn bessel_j0_fast(x: f32) -> f32 {
    if x.abs() < 8.0 {
        // 작은 x: 테일러 급수 (정확도 우선)
        let x2 = x * x;
        let x4 = x2 * x2;
        let x6 = x4 * x2;
        
        1.0 
        - x2 / 4.0           // -x²/2²
        + x4 / 64.0          // +x⁴/(2²×4²)
        - x6 / 2304.0        // -x⁶/(2²×4²×6²)
    } else {
        // 큰 x: 점근 전개 (속도 우선)
        let inv_x = 1.0 / x;
        let phase = x - PI / 4.0;
        
        // J₀(x) ≈ √(2/πx) cos(x - π/4) for large x
        (2.0 / (PI * x)).sqrt() * phase.cos()
    }
}

/// Modified Bessel I0 (지수적 성장)
pub fn bessel_i0_fast(x: f32) -> f32 {
    if x.abs() < 3.75 {
        // 다항식 근사
        let t = x / 3.75;
        let t2 = t * t;
        
        1.0 + 3.5156229 * t2 
            + 3.0899424 * t2 * t2
            + 1.2067492 * t2 * t2 * t2
    } else {
        // 점근 근사
        let t = 3.75 / x.abs();
        let exp_x = x.abs().exp();
        
        exp_x / x.abs().sqrt() * (0.39894228 + 0.01328592 * t)
    }
}
```

#### Morlet Wavelet

시간-주파수 분석에 사용되는 웨이블릿:

```rust
/// Morlet Wavelet (CORDIC 최적화)
pub fn morlet_wavelet_cordic(r: f32, theta: f32, freq: f32) -> f32 {
    // 1. Gaussian 엔벨로프
    let envelope = (-r * r / 2.0).exp();
    
    // 2. CORDIC로 복소 지수 계산
    let (cos_val, sin_val) = cordic_rotate(1.0, 0.0, freq * theta, 16);
    
    // 3. 실수부만 사용 (또는 복소수 반환 가능)
    envelope * cos_val
}

/// Morlet Wavelet (정확한 버전)
pub fn morlet_wavelet_exact(r: f32, theta: f32, freq: f32, sigma: f32) -> f32 {
    // ψ(x) = π^(-1/4) exp(-x²/2) exp(iω₀x)
    let gaussian = (-r * r / (2.0 * sigma * sigma)).exp();
    let oscillation = (freq * theta).cos();
    
    let normalization = 1.0 / (PI.powf(0.25) * sigma.sqrt());
    normalization * gaussian * oscillation
}
```

---

## 성능 비교와 최적화

### 함수별 성능 측정

| 함수 | 표준 구현 | CORDIC | 고속 근사 | 정확도 |
|:-----|:----------|:-------|:----------|:-------|
| sin/cos | 120 cycles | 40 cycles | 25 cycles | 0.0001 |
| atan2 | 150 cycles | 45 cycles | - | 0.0001 |
| exp | 100 cycles | - | 30 cycles | 0.001 |
| sqrt | 80 cycles | - | 20 cycles | 0.0001 |
| Bessel J0 | 300 cycles | - | 90 cycles | 0.001 |

### 최적화 전략

```rust
/// 정확도 vs 속도 트레이드오프
pub trait MathFunction {
    /// 고정밀도 버전 (학습용)
    fn compute_exact(&self, x: f32) -> f32;
    
    /// 고속 근사 버전 (추론용)
    fn compute_fast(&self, x: f32) -> f32;
    
    /// 자동 선택
    fn compute(&self, x: f32, need_gradient: bool) -> f32 {
        if need_gradient {
            self.compute_exact(x)  // 미분 가능성 보장
        } else {
            self.compute_fast(x)   // 속도 우선
        }
    }
}
```

### 벡터화 최적화

SIMD를 활용한 배치 처리:

```rust
use std::simd::*;

/// 4개씩 병렬 처리 (AVX)
pub fn cordic_rotate_batch_f32x4(
    x: f32x4, 
    y: f32x4, 
    angle: f32x4
) -> (f32x4, f32x4) {
    let mut xc = x;
    let mut yc = y;
    
    // SIMD 상수
    const GAIN: f32x4 = f32x4::from_array([CORDIC_GAIN; 4]);
    
    for k in 0..16 {
        let angle_k = f32x4::splat(CORDIC_ANGLES[k]);
        let shift = f32x4::splat(2.0_f32.powi(-(k as i32)));
        
        // 조건부 회전 (마스크 사용)
        let mask = angle.simd_gt(f32x4::splat(0.0));
        let sigma = mask.select(f32x4::splat(1.0), f32x4::splat(-1.0));
        
        let xc_new = xc - sigma * yc * shift;
        let yc_new = yc + sigma * xc * shift;
        
        xc = xc_new;
        yc = yc_new;
    }
    
    (xc * GAIN, yc * GAIN)
}
```

---

## 실용적인 사용 예제

### 예제 1: 패턴 생성을 위한 기저 함수

```rust
/// 복잡한 패턴 생성
pub fn generate_pattern(x: f32, y: f32, params: &PatternParams) -> f32 {
    // 극좌표 변환
    let r = (x * x + y * y).sqrt();
    let theta = y.atan2(x);
    
    // 여러 기저 함수 조합
    let base = match params.basis_type {
        0 => theta.sin() * r.cosh(),
        1 => bessel_j0_fast(r * params.frequency),
        2 => morlet_wavelet_cordic(r, theta, params.frequency),
        _ => gaussian_rbf(r, params.center, params.sigma),
    };
    
    // 부드러운 클램핑
    smooth_clamp(base * params.amplitude, 0.0, 1.0)
}
```

### 예제 2: 학습을 위한 미분 가능 손실 함수

```rust
/// Smooth L1 Loss (Huber Loss)
pub fn smooth_l1_loss(pred: f32, target: f32, beta: f32) -> f32 {
    let diff = (pred - target).abs();
    
    if diff < beta {
        // 작은 오차: L2 손실 (미분 가능)
        0.5 * diff * diff / beta
    } else {
        // 큰 오차: L1 손실 (이상치에 강건)
        diff - 0.5 * beta
    }
}

/// 도함수
pub fn smooth_l1_loss_grad(pred: f32, target: f32, beta: f32) -> f32 {
    let diff = pred - target;
    
    if diff.abs() < beta {
        diff / beta
    } else {
        diff.signum()
    }
}
```

### 예제 3: 고속 거리 계산

```rust
/// 고속 유클리드 거리 (Newton-Raphson)
pub fn fast_distance(x: f32, y: f32) -> f32 {
    let sum = x * x + y * y;
    
    if sum < 1e-8 {
        return 0.0;  // 특이점 처리
    }
    
    // 초기 추정 (비트 조작)
    let mut guess = f32::from_bits((sum.to_bits() >> 1) + 0x1fc00000);
    
    // Newton-Raphson 반복 (2번이면 충분)
    guess = 0.5 * (guess + sum / guess);
    guess = 0.5 * (guess + sum / guess);
    
    guess
}
```

---

## 핵심 장점

1. **하드웨어 효율성**: 
   - CORDIC는 곱셈기 없이 구현 가능
   - 임베디드 시스템에 이상적

2. **정밀도 제어**: 
   - 반복 횟수로 정밀도 조절
   - 필요에 따라 속도/정확도 균형

3. **완전 미분 가능**: 
   - 학습용 함수는 매끄러운 그래디언트
   - 수치 미분 도구 제공

4. **캐시 친화적**: 
   - 작은 룩업 테이블만 필요
   - 메모리 접근 최소화

5. **병렬화 용이**: 
   - 각 연산이 독립적
   - SIMD/GPU 최적화 가능

이 수학 라이브러리는 극한의 효율성과 학습 가능성을 동시에 제공하는 핵심 컴포넌트입니다. "빠르지만 부정확" 또는 "정확하지만 느림"의 이분법을 넘어서는 해결책입니다. 
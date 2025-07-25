# Core Types API Documentation

## Packed64 (메인 비트 도메인 텐서)

### 개요
`Packed64`는 RBE 시스템의 핵심 데이터 구조로, 64비트 내에 푸앵카레볼 좌표계의 두 파라미터 `r`(반지름)과 `theta`(각도)를 Q32.32 고정소수점 형식으로 저장합니다.

### 구조
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Packed64 {
    pub data: u64,  // [63:32] r (Q32) | [31:0] theta (Q32)
}
```

### 필드 설명
- `data`: 64비트 데이터 필드
  - 상위 32비트 `[63:32]`: r 파라미터 (Q32 고정소수점)
  - 하위 32비트 `[31:0]`: theta 파라미터 (Q32 고정소수점)

### 핵심 메서드

#### 생성자
```rust
pub fn new(data: u64) -> Self
pub fn from_continuous(p: &DecodedParams) -> Self
pub fn random(rng: &mut impl Rng) -> Self
```

#### 인코딩/디코딩
```rust
pub fn decode(&self) -> DecodedParams
pub fn update_from_continuous(&mut self, params: &DecodedParams)
```

#### 순전파
```rust
pub fn fused_forward(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32
pub fn fused_forward_poincare(&self, i: usize, j: usize, rows: usize, cols: usize) -> f32
```

#### 그래디언트 계산
```rust
pub fn compute_gradients(
    &self, 
    i: usize, 
    j: usize, 
    rows: usize, 
    cols: usize,
    target: f32,
    use_l1: bool,
) -> (f32, f32, f32)  // (grad_r, grad_theta, predicted)

pub fn compute_riemannian_gradients(
    &self,
    i: usize,
    j: usize,
    rows: usize,
    cols: usize,
    target: f32,
    use_l1: bool,
) -> (f32, f32)  // (grad_r, grad_theta)
```

#### 파라미터 업데이트
```rust
pub fn update_with_riemannian_grad(&mut self, update_r: f32, update_theta: f32, lr: f32)
```

## DecodedParams (연속 파라미터)

### 구조
```rust
#[derive(Debug, Clone, Default)]
pub struct DecodedParams {
    pub r_fp32: f32,     // 반지름 [0, 1)
    pub theta_fp32: f32, // 각도 [0, 2π)
}
```

### 설명
- `r_fp32`: 푸앵카레볼 내 점의 반지름 (0에서 1 미만)
- `theta_fp32`: 각도 좌표 (0에서 2π 라디안)

## 수학적 원리

### 푸앵카레볼 기하학
푸앵카레볼은 무한한 쌍곡공간을 단위구 내부에 매핑하는 모델입니다:
- 중심 (r=0): 원점
- 경계 (r→1): 무한대
- 메트릭: ds² = 4/(1-r²)² (dr² + r²dθ²)

### 좌표 변환
```
푸앵카레볼 좌표 → 쌍곡 거리: d = 2 * atanh(r)
쌍곡 거리 → 함수값: f(d) = tanh(d)
각도 성분: sin(θ)
```

### 최종 출력 공식
```
output = tanh(2 * atanh(r)) * sin(θ) * spatial_modulation
```

## 압축 메커니즘

### 압축률
- 기존: N×M 행렬 = N×M×4 바이트 (f32)
- RBE: 단일 64비트 시드 = 8 바이트
- 압축률: (N×M×4) / 8 = N×M/2 : 1

### 복원 방법
단일 시드로부터 전체 행렬의 각 요소를 `fused_forward(i, j)`로 실시간 계산

## BitTensor (비트 도메인 텐서)

### 구조
```rust
#[derive(Debug, Clone)]
pub struct BitTensor {
    pub data: Vec<Packed64>,              // 시드 배열
    pub shape: Vec<usize>,                // 텐서 형상
    pub bit_gradients: BitGradientTracker, // 비트별 그래디언트
}
```

## BitGradientTracker (그래디언트 추적)

### 구조
```rust
#[derive(Debug, Clone)]
pub struct BitGradientTracker {
    bit_grads: Vec<[u8; 64]>,                        // 64비트별 그래디언트
    bit_interactions: HashMap<(u8, u8), u8>,        // 비트간 상호작용
    state_transition_grads: Vec<StateTransitionGrad>, // 상태 전이
}
```

## 별칭 타입들

### 호환성 유지
```rust
pub type Packed128 = Packed64;  // 기존 코드 호환성
```

## 성능 특성

### 메모리 효율성
- 구조체 크기: 8바이트
- 캐시 라인 효율성: 64바이트당 8개 구조체
- 메모리 정렬: 8바이트

### 연산 성능
- 목표 성능: ~735 ns/op
- 압축률: 평균 150:1
- 정확도: RMSE 0.01 이하

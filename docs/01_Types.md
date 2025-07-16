# 1. 핵심 데이터 타입 (`src/types.rs`)

이 문서에서는 푸앵카레 레이어 라이브러리의 핵심을 이루는 데이터 구조체들을 설명합니다. 특히 128비트 하이브리드 구조로 극한 압축과 학습 가능성을 동시에 달성하는 혁신적인 타입 시스템을 소개합니다.

---

## 🎯 핵심 혁신: 128비트 하이브리드 아키텍처

### `Packed128` - 압축과 학습의 완벽한 조화

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Packed128 {
    pub hi: u64,   // Seed0: 추론용 비트필드 (CORDIC + 양자화 파라미터)
    pub lo: u64,   // Seed1: 학습용 연속 파라미터 (FP32 × 2)
}
```

#### **구조 상세**

**Seed0 (`hi`: 64비트) - 추론 최적화**
```
[63:44] r_quantized    (20 bits) - Q0.20 고정소수점 반지름
[43:20] θ_quantized    (24 bits) - Q0.24 고정소수점 각도
[19:16] basis_id       (4 bits)  - 기저 함수 선택
[15:14] d_theta        (2 bits)  - 각도 미분 차수
[13]    d_r            (1 bit)   - 반지름 미분 여부
[12:9]  rot_code       (4 bits)  - 회전 변환 코드
[8:6]   log2_c         (3 bits)  - 곡률 계수 (부호 있는 3비트)
[5:0]   reserved       (6 bits)  - 향후 확장용
```

**Seed1 (`lo`: 64비트) - 학습 최적화**
```
[63:32] r_fp32         (32 bits) - IEEE 754 단정밀도 부동소수점
[31:0]  θ_fp32         (32 bits) - IEEE 754 단정밀도 부동소수점
```

#### **핵심 설계 철학**

1. **이중 표현 (Dual Representation)**
   - 추론: 양자화된 고정소수점 → 초고속 CORDIC 연산
   - 학습: 연속 부동소수점 → 정확한 그래디언트 계산

2. **메모리 효율성**
   - 32×32 행렬: 4KB → 16B (256:1 압축)
   - 추론 시 Seed0만 로드 (8B)

3. **학습 가능성**
   - 양자화로 인한 그래디언트 소실 문제 해결
   - 표준 Adam 옵티마이저 직접 사용 가능

---

### `Packed64` - 레거시 호환성

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Packed64(pub u64);
```

- **역할**: 이전 버전과의 호환성을 위한 64비트 시드
- **사용처**: `Packed128.hi`의 비트필드 연산
- **특징**: CORDIC 기반 초고속 디코딩 지원

---

### `DecodedParams` - 디코딩된 파라미터

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedParams {
    pub r: f32,              // 반지름 (0.0 ~ 1.0)
    pub theta: f32,          // 각도 (0 ~ 2π)
    pub basis_id: u8,        // 기저 함수 ID
    pub d_theta: u8,         // 각도 미분 차수 (0~3)
    pub d_r: bool,           // 반지름 미분 여부
    pub rot_code: u8,        // 회전 코드 (0~15)
    pub log2_c: i8,          // 곡률 계수 (-4 ~ 3)
    pub reserved: u8,        // 예비 필드
}
```

---

### `DecodedParams128` - 확장된 파라미터

```rust
#[derive(Debug, Clone)]
pub struct DecodedParams128 {
    pub base: DecodedParams,  // 기본 디코딩 파라미터
    pub r_fp32: f32,         // 연속 반지름 (학습용)
    pub theta_fp32: f32,     // 연속 각도 (학습용)
}
```

#### **주요 메서드**

```rust
impl Packed128 {
    /// 양방향 변환
    pub fn decode(&self) -> DecodedParams128 {
        let base = Packed64(self.hi).decode();
        let r_fp32 = f32::from_bits((self.lo >> 32) as u32);
        let theta_fp32 = f32::from_bits(self.lo as u32);
        DecodedParams128 { base, r_fp32, theta_fp32 }
    }
    
    /// 연속 파라미터로부터 생성
    pub fn from_continuous(p: &DecodedParams128) -> Self {
        let hi = Packed64::new(/* 양자화된 파라미터들 */);
        let lo = ((p.r_fp32.to_bits() as u64) << 32) | 
                 p.theta_fp32.to_bits() as u64;
        Packed128 { hi, lo }
    }
    
    /// 추론용 가중치 계산 (CORDIC)
    pub fn compute_weight(&self, i: usize, j: usize, 
                         rows: usize, cols: usize) -> f32 {
        Packed64(self.hi).compute_weight(i, j, rows, cols)
    }
    
    /// 학습용 가중치 계산 (연속 함수)
    pub fn compute_weight_continuous(&self, i: usize, j: usize,
                                   rows: usize, cols: usize) -> f32 {
        // Seed1의 연속 파라미터 사용
        let r = f32::from_bits((self.lo >> 32) as u32);
        let theta = f32::from_bits(self.lo as u32);
        
        // Radial gradient 함수 (미분 가능)
        radial_gradient(r, theta, i, j, rows, cols)
    }
}
```

---

### `BasisFunction` - 기저 함수 열거형

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BasisFunction {
    SinCosh = 0,    // sin(θ) × cosh(r)
    SinSinh = 1,    // sin(θ) × sinh(r)
    CosCosh = 2,    // cos(θ) × cosh(r)
    CosSinh = 3,    // cos(θ) × sinh(r)
    // ... 총 16가지 조합
}
```

---

### `PoincareMatrix` - 최상위 구조체

```rust
pub struct PoincareMatrix {
    pub seed: Packed128,  // 128비트 하이브리드 시드
    pub rows: usize,
    pub cols: usize,
}
```

#### **핵심 메서드**

```rust
impl PoincareMatrix {
    /// Adam 옵티마이저 기반 학습
    pub fn train_with_adam128(&self, target: &[f32], 
                             epochs: usize, lr: f32) -> Self {
        // Seed1의 연속 파라미터로 직접 학습
        // 양자화 없이 정확한 그래디언트 계산
    }
    
    /// 압축 (행렬 → 128비트)
    pub fn compress(matrix: &[f32], rows: usize, cols: usize) -> Self {
        // 1. 패턴 분석
        // 2. 초기 파라미터 추정
        // 3. Adam 기반 최적화
    }
    
    /// 복원 (128비트 → 행렬)
    pub fn decompress(&self) -> Vec<f32> {
        // CORDIC 기반 고속 디코딩
        // GPU 병렬화 가능
    }
}
```

---

## 🔑 타입 시스템의 장점

1. **극한 압축**: 4KB → 16B (256:1)
2. **학습 가능**: 표준 옵티마이저 사용
3. **고속 추론**: CORDIC 알고리즘
4. **호환성**: 64비트 모드 지원
5. **확장성**: 예비 비트 확보

이 타입 시스템은 "압축 vs 학습"의 트레이드오프를 해결하는 혁신적인 설계입니다. 
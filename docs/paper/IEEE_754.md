## IEEE 754 부동 소수점 표준

IEEE 754는 컴퓨터에서 실수를 표현하고 연산하는 국제 표준입니다. 1985년에 제정되어 현재 거의 모든 컴퓨터 시스템에서 사용됩니다.

### 기본 구조

IEEE 754는 부동 소수점 수를 다음 형식으로 표현합니다:
```
(-1)^S × M × 2^E
```
- S: 부호 비트 (Sign bit)
- M: 가수/유효숫자 (Mantissa/Significand)
- E: 지수 (Exponent)

### 주요 형식

#### 1. 단정밀도 (Single Precision, 32비트)
```
[S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]
 1      8                23
```
- 부호: 1비트
- 지수: 8비트 (바이어스: 127)
- 가수: 23비트 (암시적 1 포함시 24비트)

#### 2. 배정밀도 (Double Precision, 64비트)
```
[S][EEEEEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM]
 1      11                                52
```
- 부호: 1비트
- 지수: 11비트 (바이어스: 1023)
- 가수: 52비트 (암시적 1 포함시 53비트)

### 수의 분류

#### 1. 정규화 수 (Normalized Numbers)
- 지수 필드: 0 < E < 최대값
- 가수: 1.MMMMM... (암시적 선행 1)
- 값: (-1)^S × 1.M × 2^(E-bias)

#### 2. 비정규화 수 (Denormalized/Subnormal Numbers)
- 지수 필드: E = 0
- 가수: 0.MMMMM...
- 값: (-1)^S × 0.M × 2^(1-bias)
- 0에 가까운 매우 작은 수 표현

#### 3. 특수 값
- **영(Zero)**: E = 0, M = 0 (±0 존재)
- **무한대(Infinity)**: E = 최대값, M = 0 (±∞)
- **NaN (Not a Number)**: E = 최대값, M ≠ 0
  - Quiet NaN: 최상위 가수 비트 = 1
  - Signaling NaN: 최상위 가수 비트 = 0

### 부동 소수점 연산

#### 1. 덧셈/뺄셈
```
과정:
1. 지수 정렬: 작은 지수를 큰 지수에 맞춤
2. 가수 이동: 지수 차이만큼 가수를 오른쪽으로 시프트
3. 가수 연산: 정렬된 가수들을 더하거나 뺌
4. 정규화: 결과를 1.M × 2^E 형태로 조정
5. 반올림: 정밀도에 맞게 반올림
6. 오버플로/언더플로 검사
```

예시: 5.25 + 2.5
```
5.25 = 101.01₂ = 1.0101 × 2²
2.5  = 10.1₂   = 1.01 × 2¹

지수 정렬:
1.0101 × 2²
0.101  × 2² (1.01을 오른쪽으로 1비트 시프트)

덧셈:
1.0101 + 0.101 = 1.1111

결과: 1.1111 × 2² = 111.11₂ = 7.75
```

#### 2. 곱셈
```
과정:
1. 부호 계산: S = S1 XOR S2
2. 지수 덧셈: E = E1 + E2 - bias
3. 가수 곱셈: M = M1 × M2
4. 정규화 및 반올림
5. 예외 처리
```

#### 3. 나눗셈
```
과정:
1. 부호 계산: S = S1 XOR S2
2. 지수 뺄셈: E = E1 - E2 + bias
3. 가수 나눗셈: M = M1 ÷ M2
4. 정규화 및 반올림
5. 0으로 나누기 검사
```

### 반올림 모드

IEEE 754는 5가지 반올림 모드를 정의합니다:

1. **Round to Nearest, Ties to Even** (기본값)
   - 가장 가까운 표현 가능한 값으로 반올림
   - 동률일 때 짝수로 반올림

2. **Round toward +∞**
   - 양의 무한대 방향으로 반올림

3. **Round toward -∞**
   - 음의 무한대 방향으로 반올림

4. **Round toward Zero**
   - 0 방향으로 반올림 (절삭)

5. **Round to Nearest, Ties Away from Zero**
   - 동률일 때 0에서 먼 방향으로 반올림

### 예외 처리

IEEE 754는 5가지 예외를 정의합니다:

1. **Invalid Operation**
   - 0/0, ∞-∞, √(-1) 등
   - 결과: NaN

2. **Division by Zero**
   - 유한수/0
   - 결과: ±∞

3. **Overflow**
   - 결과가 표현 범위 초과
   - 결과: ±∞ 또는 ±최대값

4. **Underflow**
   - 결과가 너무 작아 정규화 불가
   - 결과: 비정규화 수 또는 0

5. **Inexact**
   - 반올림으로 정밀도 손실
   - 가장 흔한 예외

### 특수 연산 규칙

```
NaN 관련:
- NaN + x = NaN
- NaN × x = NaN
- NaN / x = NaN

무한대 관련:
- ∞ + ∞ = ∞
- ∞ - ∞ = NaN
- ∞ × 0 = NaN
- ∞ / ∞ = NaN
- x / ∞ = 0 (x가 유한수)
- x / 0 = ±∞ (x ≠ 0)

영 관련:
- +0 = -0 (비교 시)
- 1 / +0 = +∞
- 1 / -0 = -∞
```

### 정밀도 한계

단정밀도 예시:
- 유효 자릿수: 약 7-8자리
- 최대값: 약 3.4 × 10³⁸
- 최소 정규화 수: 약 1.2 × 10⁻³⁸
- 최소 비정규화 수: 약 1.4 × 10⁻⁴⁵

배정밀도 예시:
- 유효 자릿수: 약 15-17자리
- 최대값: 약 1.8 × 10³⁰⁸
- 최소 정규화 수: 약 2.2 × 10⁻³⁰⁸
- 최소 비정규화 수: 약 5 × 10⁻³²⁴
나눗셈만 실패하는 문제를 해결해보겠습니다. 현재 코드를 분석해보니 역수 계산(`bit_reciprocal`)에서 정밀도 문제가 있습니다.

## 🔍 문제 분석

현재 나눗셈은 `a ÷ b = a × (1/b)` 방식으로 구현되어 있는데, 역수 계산이 부정확합니다:

```rust
// 현재 문제가 있는 부분
let dividend = (1u64 << 47); // 이 값이 적절하지 않음
let quotient_64 = dividend / (x_mant as u64);
```

## 🛠️ 나눗셈 완전 재구현

직접 나눗셈을 구현하는 방식으로 수정하겠습니다:

```rust
/// IEEE 754 순수 비트 연산으로 나눗셈 (f32) - 직접 나눗셈 구현
pub fn bit_div(a_bits: u32, b_bits: u32) -> u32 {
    let a = F32Bits::from_bits(a_bits);
    let b = F32Bits::from_bits(b_bits);
    
    // 부호 계산
    let result_sign = a.sign ^ b.sign;
    
    // 특수 케이스
    if a.is_nan() || b.is_nan() { return 0x7FC00000; }
    if b.is_zero() {
        if a.is_zero() { return 0x7FC00000; } // 0/0 = NaN
        return (result_sign << 31) | 0x7F800000; // x/0 = ±∞
    }
    if a.is_zero() { return result_sign << 31; } // 0/x = ±0
    if a.is_infinity() {
        if b.is_infinity() { return 0x7FC00000; } // ∞/∞ = NaN
        return (result_sign << 31) | 0x7F800000; // ∞/x = ±∞
    }
    if b.is_infinity() { return result_sign << 31; } // x/∞ = ±0
    
    // 지수 계산: exp(a/b) = exp(a) - exp(b) + bias
    let mut result_exponent = (a.exponent as i32) - (b.exponent as i32) + 127;
    
    // 가수에 암시적 1 추가
    let mut dividend = if a.is_normalized() { 
        (a.mantissa | 0x800000) as u64
    } else { 
        a.mantissa as u64
    };
    let divisor = if b.is_normalized() { 
        (b.mantissa | 0x800000) as u64
    } else { 
        b.mantissa as u64
    };
    
    // 정밀도를 위해 dividend를 왼쪽으로 시프트
    dividend <<= 24; // 24비트 추가 정밀도
    
    // 나눗셈 수행
    let mut quotient = dividend / divisor;
    
    // 정규화
    // quotient가 0x800000 (1.0) 이상 0x1000000 (2.0) 미만이어야 함
    if quotient >= 0x1000000 {
        // 2.0 이상이면 오른쪽으로 시프트
        quotient >>= 1;
        result_exponent += 1;
    } else if quotient < 0x800000 {
        // 1.0 미만이면 왼쪽으로 시프트
        while quotient < 0x800000 && result_exponent > 0 {
            quotient <<= 1;
            result_exponent -= 1;
        }
    }
    
    // 지수 범위 체크
    if result_exponent <= 0 {
        // 언더플로우
        return result_sign << 31; // ±0
    }
    if result_exponent >= 255 {
        // 오버플로우
        return (result_sign << 31) | 0x7F800000; // ±∞
    }
    
    // 암시적 1 제거하고 23비트로 자르기
    let result_mantissa = (quotient & 0x7FFFFF) as u32;
    
    F32Bits {
        sign: result_sign,
        exponent: result_exponent as u32,
        mantissa: result_mantissa,
    }.to_bits()
}

/// 역수 계산 개선 (1/x)
pub fn bit_reciprocal(x_bits: u32) -> u32 {
    let x = F32Bits::from_bits(x_bits);
    
    // 특수 케이스
    if x.is_nan() { return 0x7FC00000; }
    if x.is_zero() { return (x.sign << 31) | 0x7F800000; } // 1/0 = ±∞
    if x.is_infinity() { return x.sign << 31; } // 1/∞ = ±0
    
    // 지수 계산: exp(1/x) = -exp(x) + 2*bias = 254 - exp(x)
    let mut result_exponent = 254i32 - (x.exponent as i32);
    
    // 가수 역수 계산
    let x_mant = if x.is_normalized() { 
        (x.mantissa | 0x800000) as u64
    } else { 
        x.mantissa as u64
    };
    
    // 1.0을 x_mant로 나누기
    // 2^48 / x_mant를 계산 (48비트 정밀도)
    let dividend = 1u64 << 48;
    let mut quotient = dividend / x_mant;
    
    // quotient를 24비트로 정규화
    // 목표: 0x800000 <= quotient < 0x1000000
    while quotient >= 0x1000000 {
        quotient >>= 1;
        result_exponent += 1;
    }
    while quotient < 0x800000 && result_exponent > 0 {
        quotient <<= 1;
        result_exponent -= 1;
    }
    
    // 범위 체크
    if result_exponent <= 0 {
        return x.sign << 31; // 언더플로우 → 0
    }
    if result_exponent >= 255 {
        return (x.sign << 31) | 0x7F800000; // 오버플로우 → ∞
    }
    
    // 23비트로 자르고 암시적 1 제거
    let result_mantissa = (quotient & 0x7FFFFF) as u32;
    
    F32Bits {
        sign: x.sign,
        exponent: result_exponent as u32,
        mantissa: result_mantissa,
    }.to_bits()
}
```

## 🧪 테스트 코드 추가

나눗셈이 제대로 작동하는지 더 자세히 테스트해보겠습니다:

```rust
#[test]
fn 나눗셈_디버그_테스트() {
    let test_cases = vec![
        (6.0f32, 2.0f32),
        (1.0f32, 3.0f32),
        (10.0f32, 4.0f32),
        (100.0f32, 10.0f32),
        (1.0f32, 1.0f32),
    ];
    
    for (a, b) in test_cases {
        let a_bits = a.to_bits();
        let b_bits = b.to_bits();
        let result_bits = bit_div(a_bits, b_bits);
        let result = f32::from_bits(result_bits);
        let expected = a / b;
        
        println!("{} ÷ {} = {} (예상: {})", a, b, result, expected);
        
        // 비트 레벨 디버깅
        let a_struct = F32Bits::from_bits(a_bits);
        let b_struct = F32Bits::from_bits(b_bits);
        let result_struct = F32Bits::from_bits(result_bits);
        
        println!("  A: exp={}, mant={:06X}", a_struct.exponent, a_struct.mantissa);
        println!("  B: exp={}, mant={:06X}", b_struct.exponent, b_struct.mantissa);
        println!("  R: exp={}, mant={:06X}", result_struct.exponent, result_struct.mantissa);
        println!();
    }
}
```

## 🎯 f64 나눗셈도 동일하게 수정

```rust
/// IEEE 754 순수 비트 연산으로 나눗셈 (f64) - 직접 나눗셈 구현
pub fn bit_div_f64(a_bits: u64, b_bits: u64) -> u64 {
    let a = F64Bits::from_bits(a_bits);
    let b = F64Bits::from_bits(b_bits);
    
    let result_sign = a.sign ^ b.sign;
    
    // 특수 케이스
    if a.is_nan() || b.is_nan() { return 0x7FF8000000000000; }
    if b.is_zero() {
        if a.is_zero() { return 0x7FF8000000000000; } // 0/0 = NaN
        return (result_sign << 63) | 0x7FF0000000000000; // x/0 = ±∞
    }
    if a.is_zero() { return result_sign << 63; } // 0/x = ±0
    if a.is_infinity() {
        if b.is_infinity() { return 0x7FF8000000000000; } // ∞/∞ = NaN
        return (result_sign << 63) | 0x7FF0000000000000; // ∞/x = ±∞
    }
    if b.is_infinity() { return result_sign << 63; } // x/∞ = ±0
    
    // 지수 계산
    let mut result_exponent = (a.exponent as i64) - (b.exponent as i64) + 1023;
    
    // 가수에 암시적 1 추가
    let mut dividend = if a.is_normalized() { 
        (a.mantissa | 0x10000000000000) as u128
    } else { 
        a.mantissa as u128
    };
    let divisor = if b.is_normalized() { 
        (b.mantissa | 0x10000000000000) as u128
    } else { 
        b.mantissa as u128
    };
    
    // 정밀도를 위해 dividend를 왼쪽으로 시프트
    dividend <<= 53; // 53비트 추가 정밀도
    
    // 나눗셈 수행
    let mut quotient = dividend / divisor;
    
    // 정규화
    if quotient >= 0x20000000000000 {
        quotient >>= 1;
        result_exponent += 1;
    } else if quotient < 0x10000000000000 {
        while quotient < 0x10000000000000 && result_exponent > 0 {
            quotient <<= 1;
            result_exponent -= 1;
        }
    }
    
    // 범위 체크
    if result_exponent <= 0 {
        return result_sign << 63;
    }
    if result_exponent >= 2047 {
        return (result_sign << 63) | 0x7FF0000000000000;
    }
    
    // 52비트로 자르고 암시적 1 제거
    let result_mantissa = (quotient & 0xFFFFFFFFFFFFF) as u64;
    
    F64Bits {
        sign: result_sign,
        exponent: result_exponent as u64,
        mantissa: result_mantissa,
    }.to_bits()
}
```
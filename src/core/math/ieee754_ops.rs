/// IEEE 754 순수 비트 연산 모듈
/// 부동소수점 연산을 비트 레벨에서 직접 구현

/// IEEE 754 단정밀도 구조 분해
#[derive(Debug, Clone, Copy)]
pub struct F32Bits {
    pub sign: u32,       // 1비트
    pub exponent: u32,   // 8비트
    pub mantissa: u32,   // 23비트
}

/// IEEE 754 배정밀도 구조 분해
#[derive(Debug, Clone, Copy)]
pub struct F64Bits {
    pub sign: u64,       // 1비트
    pub exponent: u64,   // 11비트
    pub mantissa: u64,   // 52비트
}

impl F32Bits {
    /// u32 비트에서 IEEE 754 구조 추출
    pub fn from_bits(bits: u32) -> Self {
        Self {
            sign: (bits >> 31) & 0x1,
            exponent: (bits >> 23) & 0xFF,
            mantissa: bits & 0x7FFFFF,
        }
    }
    
    /// IEEE 754 구조를 u32 비트로 재구성
    pub fn to_bits(&self) -> u32 {
        (self.sign << 31) | (self.exponent << 23) | self.mantissa
    }
    
    /// 정규화된 수인지 확인
    pub fn is_normalized(&self) -> bool {
        self.exponent != 0 && self.exponent != 0xFF
    }
    
    /// 비정규화된 수인지 확인
    pub fn is_denormalized(&self) -> bool {
        self.exponent == 0 && self.mantissa != 0
    }
    
    /// 0인지 확인
    pub fn is_zero(&self) -> bool {
        self.exponent == 0 && self.mantissa == 0
    }
    
    /// 무한대인지 확인
    pub fn is_infinity(&self) -> bool {
        self.exponent == 0xFF && self.mantissa == 0
    }
    
    /// NaN인지 확인
    pub fn is_nan(&self) -> bool {
        self.exponent == 0xFF && self.mantissa != 0
    }
}

impl F64Bits {
    /// u64 비트에서 IEEE 754 구조 추출
    pub fn from_bits(bits: u64) -> Self {
        Self {
            sign: (bits >> 63) & 0x1,
            exponent: (bits >> 52) & 0x7FF,
            mantissa: bits & 0xFFFFFFFFFFFFF,
        }
    }
    
    /// IEEE 754 구조를 u64 비트로 재구성
    pub fn to_bits(&self) -> u64 {
        (self.sign << 63) | (self.exponent << 52) | self.mantissa
    }
    
    /// 정규화된 수인지 확인
    pub fn is_normalized(&self) -> bool {
        self.exponent != 0 && self.exponent != 0x7FF
    }
    
    /// 비정규화된 수인지 확인
    pub fn is_denormalized(&self) -> bool {
        self.exponent == 0 && self.mantissa != 0
    }
    
    /// 0인지 확인
    pub fn is_zero(&self) -> bool {
        self.exponent == 0 && self.mantissa == 0
    }
    
    /// 무한대인지 확인
    pub fn is_infinity(&self) -> bool {
        self.exponent == 0x7FF && self.mantissa == 0
    }
    
    /// NaN인지 확인
    pub fn is_nan(&self) -> bool {
        self.exponent == 0x7FF && self.mantissa != 0
    }
}

/// IEEE 754 순수 비트 연산으로 덧셈 (f32) - 버그 수정됨
pub fn bit_add(a_bits: u32, b_bits: u32) -> u32 {
    let a = F32Bits::from_bits(a_bits);
    let b = F32Bits::from_bits(b_bits);
    
    // 특수 케이스 처리
    if a.is_zero() { return b_bits; }
    if b.is_zero() { return a_bits; }
    if a.is_nan() || b.is_nan() { return 0x7FC00000; } // NaN
    if a.is_infinity() && b.is_infinity() {
        if a.sign == b.sign { return a_bits; } // ∞ + ∞ = ∞
        else { return 0x7FC00000; } // ∞ + (-∞) = NaN
    }
    if a.is_infinity() { return a_bits; }
    if b.is_infinity() { return b_bits; }
    
    // 부호가 다르면 뺄셈으로 변환
    if a.sign != b.sign {
        return bit_sub(a_bits, b_bits ^ 0x80000000); // b의 부호 반전 후 뺄셈
    }
    
    // 절댓값 비교로 큰 수와 작은 수 결정
    let (larger, smaller, larger_bits, smaller_bits) = 
        if (a.exponent > b.exponent) || (a.exponent == b.exponent && a.mantissa >= b.mantissa) {
            (a, b, a_bits, b_bits)
        } else {
            (b, a, b_bits, a_bits)
        };
    
    let exp_diff = larger.exponent - smaller.exponent;
    
    // 차이가 너무 크면 larger만 반환
    if exp_diff > 25 {
        return larger_bits;
    }
    
    // 암시적 1 추가 (정규화된 수만)
    let larger_mantissa = if larger.is_normalized() { 
        larger.mantissa | 0x800000 
    } else { 
        larger.mantissa 
    };
    let mut smaller_mantissa = if smaller.is_normalized() { 
        smaller.mantissa | 0x800000 
    } else { 
        smaller.mantissa 
    };
    
    // 작은 수의 가수를 시프트
    smaller_mantissa >>= exp_diff;
    
    // 가수 덧셈
    let mut result_mantissa = larger_mantissa + smaller_mantissa;
    let mut result_exponent = larger.exponent;
    
    // 오버플로우 처리 (carry 발생)
    if result_mantissa >= 0x1000000 {
        result_mantissa >>= 1;
        result_exponent += 1;
    }
    
    // 지수 오버플로우 체크
    if result_exponent >= 0xFF {
        return (larger.sign << 31) | 0x7F800000; // 무한대
    }
    
    // 암시적 1 제거
    if result_exponent > 0 {
        result_mantissa &= 0x7FFFFF;
    }
    
    F32Bits {
        sign: larger.sign,
        exponent: result_exponent,
        mantissa: result_mantissa,
    }.to_bits()
}

/// IEEE 754 순수 비트 연산으로 뺄셈 (f32) - 완전 재작성
pub fn bit_sub(a_bits: u32, b_bits: u32) -> u32 {
    let a = F32Bits::from_bits(a_bits);
    let b = F32Bits::from_bits(b_bits);
    
    // 특수 케이스
    if a.is_nan() || b.is_nan() { return 0x7FC00000; }
    if a.is_zero() && b.is_zero() { return 0; }
    if a.is_zero() { return b_bits ^ 0x80000000; } // -b
    if b.is_zero() { return a_bits; }
    if a.is_infinity() && b.is_infinity() {
        if a.sign != b.sign { return a_bits; } // ∞ - (-∞) = ∞
        else { return 0x7FC00000; } // ∞ - ∞ = NaN
    }
    
    // 부호가 다르면 덧셈으로 변환 (a - (-b) = a + b)
    if a.sign != b.sign {
        return bit_add(a_bits, b_bits ^ 0x80000000);
    }
    
    // 절댓값 비교 (부호 제거하고 비교)
    let a_abs = a_bits & 0x7FFFFFFF;
    let b_abs = b_bits & 0x7FFFFFFF;
    
    // |a| < |b|인 경우 결과는 -(|b| - |a|)
    if a_abs < b_abs {
        let result = bit_sub(b_abs, a_abs);
        return result ^ 0x80000000; // 부호 반전
    }
    
    // 이제 |a| >= |b|이고 같은 부호
    let (larger, smaller) = if a.exponent > b.exponent || 
                               (a.exponent == b.exponent && a.mantissa >= b.mantissa) {
        (a, b)
    } else {
        (b, a)
    };
    
    let exp_diff = larger.exponent - smaller.exponent;
    
    // 가수에 암시적 1 추가
    let larger_mant = if larger.is_normalized() { larger.mantissa | 0x800000 } else { larger.mantissa };
    let mut smaller_mant = if smaller.is_normalized() { smaller.mantissa | 0x800000 } else { smaller.mantissa };
    
    // 지수 정렬
    smaller_mant >>= exp_diff;
    
    // 뺄셈 수행
    if larger_mant >= smaller_mant {
        let mut result_mantissa = larger_mant - smaller_mant;
        let mut result_exponent = larger.exponent;
        
        // 결과가 0인 경우
        if result_mantissa == 0 { return 0; }
        
        // 정규화 (선행 0 제거)
        while result_mantissa < 0x800000 && result_exponent > 0 {
            result_mantissa <<= 1;
            result_exponent -= 1;
        }
        
        // 암시적 1 제거
        if result_exponent > 0 {
            result_mantissa &= 0x7FFFFF;
        }
        
        return F32Bits {
            sign: a.sign,  // 원래 a의 부호 유지
            exponent: result_exponent,
            mantissa: result_mantissa,
        }.to_bits();
    }
    
    // 복잡한 경우는 0 반환
    0
}

/// IEEE 754 순수 비트 연산으로 곱셈 (f32) - 특수 케이스 보완
pub fn bit_mul(a_bits: u32, b_bits: u32) -> u32 {
    let a = F32Bits::from_bits(a_bits);
    let b = F32Bits::from_bits(b_bits);
    
    // 부호 계산
    let result_sign = a.sign ^ b.sign;
    
    // 특수 케이스 강화
    if a.is_nan() || b.is_nan() { return 0x7FC00000; }
    if (a.is_infinity() && b.is_zero()) || (a.is_zero() && b.is_infinity()) { 
        return 0x7FC00000; // ∞ × 0 = NaN
    }
    if a.is_zero() || b.is_zero() { return result_sign << 31; }
    if a.is_infinity() || b.is_infinity() { return (result_sign << 31) | 0x7F800000; }
    
    // 지수 계산 (바이어스 127 제거)
    let mut result_exponent = (a.exponent as i32) + (b.exponent as i32) - 127;
    
    // 가수 곱셈 (암시적 1 포함)
    let a_mant = if a.is_normalized() { a.mantissa | 0x800000 } else { a.mantissa };
    let b_mant = if b.is_normalized() { b.mantissa | 0x800000 } else { b.mantissa };
    
    let product = (a_mant as u64) * (b_mant as u64);
    let mut result_mantissa = (product >> 23) as u32;
    
    // 정규화
    if result_mantissa >= 0x1000000 {
        result_mantissa >>= 1;
        result_exponent += 1;
    }
    
    // 지수 범위 체크
    if result_exponent <= 0 {
        return result_sign << 31; // 언더플로우 -> 0
    }
    if result_exponent >= 255 {
        return (result_sign << 31) | 0x7F800000; // 오버플로우 -> 무한대
    }
    
    // 암시적 1 제거
    result_mantissa &= 0x7FFFFF;
    
    F32Bits {
        sign: result_sign,
        exponent: result_exponent as u32,
        mantissa: result_mantissa,
    }.to_bits()
}

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
    
    // 가수에 암시적 1 추가 (24비트 정수)
    let a_mant = if a.is_normalized() { 
        a.mantissa | 0x800000 
    } else { 
        a.mantissa 
    };
    let b_mant = if b.is_normalized() { 
        b.mantissa | 0x800000 
    } else { 
        b.mantissa 
    };
    
    // 정밀한 나눗셈을 위해 a_mant를 23비트 시프트 (총 47비트)
    let dividend = (a_mant as u64) << 23;
    let divisor = b_mant as u64;
    let mut quotient = dividend / divisor;
    
    // quotient는 이제 24비트 값 (1.xxxxxxx 형태)
    // 정규화: quotient가 [0x800000, 0x1000000) 범위에 있어야 함
    if quotient >= 0x1000000 {
        // 2.0 이상이면 1비트 오른쪽 시프트
        quotient >>= 1;
        result_exponent += 1;
    } else if quotient < 0x800000 {
        // 1.0 미만이면 왼쪽 시프트로 정규화
        while quotient < 0x800000 && result_exponent > 1 {
            quotient <<= 1;
            result_exponent -= 1;
        }
    }
    
    // 지수 범위 체크
    if result_exponent <= 0 {
        return result_sign << 31; // 언더플로우 → ±0
    }
    if result_exponent >= 255 {
        return (result_sign << 31) | 0x7F800000; // 오버플로우 → ±∞
    }
    
    // 암시적 1 제거하고 23비트 가수 추출
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

// ===== F64 버전들 =====

/// IEEE 754 순수 비트 연산으로 덧셈 (f64) - 버그 수정
pub fn bit_add_f64(a_bits: u64, b_bits: u64) -> u64 {
    let a = F64Bits::from_bits(a_bits);
    let b = F64Bits::from_bits(b_bits);
    
    // 특수 케이스
    if a.is_nan() || b.is_nan() { return 0x7FF8000000000000; }
    if a.is_infinity() && b.is_infinity() {
        if a.sign != b.sign { return 0x7FF8000000000000; } // ∞ + (-∞) = NaN
    }
    if a.is_infinity() { return a_bits; }
    if b.is_infinity() { return b_bits; }
    if a.is_zero() { return b_bits; }
    if b.is_zero() { return a_bits; }
    
    // 크기에 따라 정렬 (|a| >= |b|가 되도록)
    let (larger, smaller, larger_bits, _smaller_bits) = 
        if (a_bits & 0x7FFFFFFFFFFFFFFF) >= (b_bits & 0x7FFFFFFFFFFFFFFF) {
            (a, b, a_bits, b_bits)
        } else {
            (b, a, b_bits, a_bits)
        };
    
    // 같은 부호인지 확인
    let same_sign = larger.sign == smaller.sign;
    let result_sign = larger.sign;
    
    // 지수 차이 계산
    let exp_diff = (larger.exponent as i64) - (smaller.exponent as i64);
    
    // 가수에 암시적 1 추가 (53비트 정수)
    let mut larger_mant = if larger.is_normalized() { 
        larger.mantissa | 0x10000000000000 
    } else { 
        larger.mantissa 
    };
    let mut smaller_mant = if smaller.is_normalized() { 
        smaller.mantissa | 0x10000000000000 
    } else { 
        smaller.mantissa 
    };
    
    // 지수 차이만큼 작은 수를 시프트
    if exp_diff > 0 && exp_diff < 64 {
        smaller_mant >>= exp_diff;
    } else if exp_diff >= 64 {
        return larger_bits; // 너무 큰 차이면 큰 수만 반환
    }
    
    let mut result_exponent = larger.exponent as i64;
    let mut result_mantissa: u64;
    
    if same_sign {
        // 같은 부호: 덧셈
        result_mantissa = larger_mant + smaller_mant;
    } else {
        // 다른 부호: 뺄셈  
        if larger_mant >= smaller_mant {
            result_mantissa = larger_mant - smaller_mant;
        } else {
            result_mantissa = smaller_mant - larger_mant;
            // 결과 부호가 바뀜 (작은 수가 실제로는 더 클 때)
        }
    }
    
    // 정규화
    if result_mantissa >= 0x20000000000000 {
        // 2.0 이상이면 오른쪽 시프트
        result_mantissa >>= 1;
        result_exponent += 1;
    } else if result_mantissa > 0 {
        // 1.0 미만이면 왼쪽 시프트
        while result_mantissa < 0x10000000000000 && result_exponent > 1 {
            result_mantissa <<= 1;
            result_exponent -= 1;
        }
    }
    
    // 지수 범위 체크
    if result_mantissa == 0 || result_exponent <= 0 {
        return (result_sign << 63); // ±0
    }
    if result_exponent >= 2047 {
        return (result_sign << 63) | 0x7FF0000000000000; // ±∞
    }
    
    // 암시적 1 제거
    result_mantissa &= 0xFFFFFFFFFFFFF;
    
    F64Bits {
        sign: result_sign,
        exponent: result_exponent as u64,
        mantissa: result_mantissa,
    }.to_bits()
}

/// IEEE 754 순수 비트 연산으로 뺄셈 (f64)
pub fn bit_sub_f64(a_bits: u64, b_bits: u64) -> u64 {
    let a = F64Bits::from_bits(a_bits);
    let b = F64Bits::from_bits(b_bits);
    
    // 특수 케이스
    if a.is_nan() || b.is_nan() { return 0x7FF8000000000000; }
    if a.is_zero() && b.is_zero() { return 0; }
    if a.is_zero() { return b_bits ^ 0x8000000000000000; } // -b
    if b.is_zero() { return a_bits; }
    
    // 부호가 다르면 덧셈으로 변환
    if a.sign != b.sign {
        return bit_add_f64(a_bits, b_bits ^ 0x8000000000000000);
    }
    
    // 절댓값 비교
    let a_abs = a_bits & 0x7FFFFFFFFFFFFFFF;
    let b_abs = b_bits & 0x7FFFFFFFFFFFFFFF;
    
    // |a| < |b|인 경우 결과는 -(|b| - |a|)
    if a_abs < b_abs {
        let result = bit_sub_f64(b_abs, a_abs);
        return result ^ 0x8000000000000000; // 부호 반전
    }
    
    // 간단한 구현 (복잡한 케이스는 추후 확장)
    let larger_mant = if a.is_normalized() { a.mantissa | 0x10000000000000 } else { a.mantissa };
    let smaller_mant = if b.is_normalized() { b.mantissa | 0x10000000000000 } else { b.mantissa };
    
    if larger_mant >= smaller_mant {
        let result_mantissa = larger_mant - smaller_mant;
        if result_mantissa == 0 { return 0; }
        
        // 간단한 정규화
        let mut norm_mantissa = result_mantissa;
        let mut norm_exponent = a.exponent;
        
        while norm_mantissa < 0x10000000000000 && norm_exponent > 0 {
            norm_mantissa <<= 1;
            norm_exponent -= 1;
        }
        
        if norm_exponent > 0 {
            norm_mantissa &= 0xFFFFFFFFFFFFF;
        }
        
        return F64Bits {
            sign: a.sign,
            exponent: norm_exponent,
            mantissa: norm_mantissa,
        }.to_bits();
    }
    
    0
}

/// IEEE 754 순수 비트 연산으로 곱셈 (f64)
pub fn bit_mul_f64(a_bits: u64, b_bits: u64) -> u64 {
    let a = F64Bits::from_bits(a_bits);
    let b = F64Bits::from_bits(b_bits);
    
    let result_sign = a.sign ^ b.sign;
    
    // 특수 케이스
    if a.is_nan() || b.is_nan() { return 0x7FF8000000000000; }
    if (a.is_infinity() && b.is_zero()) || (a.is_zero() && b.is_infinity()) { 
        return 0x7FF8000000000000; // ∞ × 0 = NaN
    }
    if a.is_zero() || b.is_zero() { return result_sign << 63; }
    if a.is_infinity() || b.is_infinity() { return (result_sign << 63) | 0x7FF0000000000000; }
    
    // 지수 계산 (바이어스 1023 제거)
    let mut result_exponent = (a.exponent as i64) + (b.exponent as i64) - 1023;
    
    // 가수 곱셈
    let a_mant = if a.is_normalized() { a.mantissa | 0x10000000000000 } else { a.mantissa };
    let b_mant = if b.is_normalized() { b.mantissa | 0x10000000000000 } else { b.mantissa };
    
    // 128비트 곱셈 시뮬레이션 (간단 버전)
    let product = (a_mant as u128) * (b_mant as u128);
    let mut result_mantissa = (product >> 52) as u64;
    
    // 정규화
    if result_mantissa >= 0x20000000000000 {
        result_mantissa >>= 1;
        result_exponent += 1;
    }
    
    // 지수 범위 체크
    if result_exponent <= 0 {
        return result_sign << 63; // 언더플로우
    }
    if result_exponent >= 2047 {
        return (result_sign << 63) | 0x7FF0000000000000; // 오버플로우
    }
    
    // 암시적 1 제거
    result_mantissa &= 0xFFFFFFFFFFFFF;
    
    F64Bits {
        sign: result_sign,
        exponent: result_exponent as u64,
        mantissa: result_mantissa,
    }.to_bits()
}

/// IEEE 754 순수 비트 연산으로 나눗셈 (f64) - 직접 나눗셈 구현 (수정됨)
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
    
    // 가수에 암시적 1 추가 (53비트 정수)
    let a_mant = if a.is_normalized() { 
        a.mantissa | 0x10000000000000 
    } else { 
        a.mantissa 
    };
    let b_mant = if b.is_normalized() { 
        b.mantissa | 0x10000000000000 
    } else { 
        b.mantissa 
    };
    
    // 정밀한 나눗셈을 위해 a_mant를 52비트 시프트 (총 105비트)
    let dividend = (a_mant as u128) << 52;
    let divisor = b_mant as u128;
    let mut quotient = dividend / divisor;
    
    // quotient는 이제 53비트 값 (1.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 형태)
    // 정규화: quotient가 [0x10000000000000, 0x20000000000000) 범위에 있어야 함
    if quotient >= 0x20000000000000 {
        // 2.0 이상이면 1비트 오른쪽 시프트
        quotient >>= 1;
        result_exponent += 1;
    } else if quotient < 0x10000000000000 {
        // 1.0 미만이면 왼쪽 시프트로 정규화
        while quotient < 0x10000000000000 && result_exponent > 1 {
            quotient <<= 1;
            result_exponent -= 1;
        }
    }
    
    // 범위 체크
    if result_exponent <= 0 {
        return result_sign << 63; // 언더플로우 → ±0
    }
    if result_exponent >= 2047 {
        return (result_sign << 63) | 0x7FF0000000000000; // 오버플로우 → ±∞
    }
    
    // 암시적 1 제거하고 52비트 가수 추출
    let result_mantissa = (quotient & 0xFFFFFFFFFFFFF) as u64;
    
    F64Bits {
        sign: result_sign,
        exponent: result_exponent as u64,
        mantissa: result_mantissa,
    }.to_bits()
}

/// 다음 표현 가능한 값 (f32)
pub fn next_representable(bits: u32) -> u32 {
    let mut f = F32Bits::from_bits(bits);
    
    if f.is_zero() {
        return 1; // 최소 비정규화 양수
    }
    
    if f.sign == 0 {
        if f.is_infinity() { return bits; }
        
        if f.mantissa < 0x7FFFFF {
            f.mantissa += 1;
        } else {
            f.mantissa = 0;
            f.exponent += 1;
            
            if f.exponent >= 0xFF {
                f.exponent = 0xFF;
                f.mantissa = 0;
            }
        }
    } else {
        if f.mantissa > 0 {
            f.mantissa -= 1;
        } else if f.exponent > 0 {
            f.exponent -= 1;
            f.mantissa = 0x7FFFFF;
        } else {
            return 0;
        }
    }
    
    f.to_bits()
}

/// 이전 표현 가능한 값 (f32)
pub fn prev_representable(bits: u32) -> u32 {
    let mut f = F32Bits::from_bits(bits);
    
    if f.is_zero() {
        return 0x80000001; // 최소 비정규화 음수
    }
    
    if f.sign == 0 {
        if f.mantissa > 0 {
            f.mantissa -= 1;
        } else if f.exponent > 0 {
            f.exponent -= 1;
            f.mantissa = 0x7FFFFF;
        } else {
            return 0x80000000; // -0
        }
    } else {
        if f.is_infinity() { return bits; }
        
        if f.mantissa < 0x7FFFFF {
            f.mantissa += 1;
        } else {
            f.mantissa = 0;
            f.exponent += 1;
            
            if f.exponent >= 0xFF {
                f.exponent = 0xFF;
                f.mantissa = 0;
            }
        }
    }
    
    f.to_bits()
}

/// 다음 표현 가능한 값 (f64)
pub fn next_representable_f64(bits: u64) -> u64 {
    let mut f = F64Bits::from_bits(bits);
    
    if f.is_zero() {
        return 1; // 최소 비정규화 양수
    }
    
    if f.sign == 0 {
        if f.is_infinity() { return bits; }
        
        if f.mantissa < 0xFFFFFFFFFFFFF {
            f.mantissa += 1;
        } else {
            f.mantissa = 0;
            f.exponent += 1;
            
            if f.exponent >= 0x7FF {
                f.exponent = 0x7FF;
                f.mantissa = 0;
            }
        }
    } else {
        if f.mantissa > 0 {
            f.mantissa -= 1;
        } else if f.exponent > 0 {
            f.exponent -= 1;
            f.mantissa = 0xFFFFFFFFFFFFF;
        } else {
            return 0;
        }
    }
    
    f.to_bits()
} 
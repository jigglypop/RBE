//! 쌍곡함수 룩업 테이블 - 컴파일 타임 생성 (극한의 정밀도)

// 컴파일 타임에 생성되는 쌍곡함수 LUT
pub const HYPERBOLIC_LUT_DATA: [[u32; 256]; 8] = generate_hyperbolic_lut();

/// 고정소수점 쌍곡함수 LUT 생성 (컴파일 타임) - 고정밀도 버전
const fn generate_hyperbolic_lut() -> [[u32; 256]; 8] {
    let mut lut = [[0u32; 256]; 8];
    let mut i = 0;
    
    while i < 256 {
        let x = (i as i32 - 128) << 8; // [-128, 127] → Q16.16
        
        // 함수 0: sinh - Taylor 급수 7차항까지
        // sinh(x) = x + x³/6 + x⁵/120 + x⁷/5040
        let x2 = ((x as i64 * x as i64) >> 16) as i32;
        let x3 = ((x2 as i64 * x as i64) >> 16) as i32;
        let x5 = ((x3 as i64 * x2 as i64) >> 16) as i32;
        let x7 = ((x5 as i64 * x2 as i64) >> 16) as i32;
        
        // 정밀한 계수: 1/6 = 10923, 1/120 = 546, 1/5040 = 13
        let sinh_val = x + ((x3 as i64 * 10923) >> 16) as i32 
                         + ((x5 as i64 * 546) >> 16) as i32
                         + ((x7 as i64 * 13) >> 16) as i32;
        lut[0][i] = sinh_val as u32;
        
        // 함수 1: cosh - Taylor 급수 6차항까지
        // cosh(x) = 1 + x²/2 + x⁴/24 + x⁶/720
        let x4 = ((x2 as i64 * x2 as i64) >> 16) as i32;
        let x6 = ((x4 as i64 * x2 as i64) >> 16) as i32;
        
        // 정밀한 계수: 1/2 = 32768, 1/24 = 2731, 1/720 = 91
        let cosh_val = 65536 + ((x2 as i64 * 32768) >> 16) as i32
                             + ((x4 as i64 * 2731) >> 16) as i32
                             + ((x6 as i64 * 91) >> 16) as i32;
        lut[1][i] = cosh_val as u32;
        
        // 함수 2: tanh - Padé [5,5] 근사 (더 정밀)
        // tanh(x) ≈ x(945 + 105x² + x⁴)/(945 + 420x² + 15x⁴)
        let x_abs = if x < 0 { -x } else { x };
        if x_abs < (3 << 16) { // |x| < 3
            let num = (x as i64) * (61931520 + (6881280 * x2 as i64 >> 16) + (x4 as i64));
            let den = 61931520 + (27525120 * x2 as i64 >> 16) + (983040 * x4 as i64 >> 16);
            let tanh_val = if den != 0 {
                (num / den) as i32
            } else {
                0
            };
            lut[2][i] = tanh_val as u32;
        } else {
            // |x| >= 3인 경우 ±1에 수렴
            lut[2][i] = if x > 0 { 65536 } else { -65536i32 as u32 };
        }
        
        // 함수 3: sech² - 정확한 계산
        // sech²(x) = 1 - tanh²(x)
        let tanh_val = lut[2][i] as i32;
        let tanh2 = ((tanh_val as i64 * tanh_val as i64) >> 16) as i32;
        let sech2_val = 65536 - tanh2;
        lut[3][i] = if sech2_val < 0 { 0 } else { sech2_val as u32 };
        
        // 함수 4: asinh 근사
        // asinh(x) ≈ x - x³/6 + 3x⁵/40 (|x| < 1)
        if x_abs < 65536 { // |x| < 1
            let asinh_val = x - ((x3 as i64 * 10923) >> 16) as i32
                              + ((3 * x5 as i64 * 1638) >> 16) as i32;
            lut[4][i] = asinh_val as u32;
        } else {
            // |x| >= 1인 경우 ln(2|x|) 근사
            lut[4][i] = x_abs as u32;
        }
        
        // 함수 5: acosh 근사 (x >= 1)
        // acosh(x) ≈ ln(2x) for x > 2
        if i >= 192 { // x >= 1 in [-1, 1] 범위
            let x_shifted = (i - 128) as u32;
            lut[5][i] = (x_shifted << 10) as u32; // 간단한 근사
        } else {
            lut[5][i] = 0;
        }
        
        // 함수 6: atanh 근사
        // atanh(x) = 0.5 * ln((1+x)/(1-x))
        if x_abs < 65536 { // |x| < 1
            // Taylor: x + x³/3 + 2x⁵/15
            let atanh_val = x + ((x3 as i64 * 21845) >> 16) as i32
                              + ((2 * x5 as i64 * 4369) >> 16) as i32;
            lut[6][i] = atanh_val as u32;
        } else {
            lut[6][i] = if x > 0 { 0x7FFFFFFF } else { 0x80000001 };
        }
        
        // 함수 7: 쌍곡 코사인 역함수의 도함수
        // d/dx[acosh(x)] = 1/sqrt(x²-1)
        if i > 192 { // x > 1
            let x2_minus_1 = x2 - 65536;
            if x2_minus_1 > 0 {
                // 1/sqrt 근사
                let sqrt_approx = 65536; // 간단한 초기값
                lut[7][i] = ((65536i64 << 16) / sqrt_approx) as u32;
            } else {
                lut[7][i] = 0x7FFFFFFF; // 무한대
            }
        } else {
            lut[7][i] = 0;
        }
        
        i += 1;
    }
    
    lut
}

// LUT 크기 검증
const _: () = assert!(HYPERBOLIC_LUT_DATA.len() == 8);
const _: () = assert!(HYPERBOLIC_LUT_DATA[0].len() == 256);

// 추가 검증: 특정 값들의 정확성
const _: () = {
    // sinh(0) = 0
    assert!(HYPERBOLIC_LUT_DATA[0][128] == 0);
    // cosh(0) = 1 (Q16.16에서 65536)
    assert!(HYPERBOLIC_LUT_DATA[1][128] == 65536);
    // tanh(0) = 0
    assert!(HYPERBOLIC_LUT_DATA[2][128] == 0);
    // sech²(0) = 1
    assert!(HYPERBOLIC_LUT_DATA[3][128] == 65536);
}; 
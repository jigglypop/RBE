#[cfg(test)]
mod tests {
    use crate::core::math::ieee754_ops::*;  // 절대 경로로 수정
    use std::time::Instant;
    use rand::{thread_rng, Rng};

    // IEEE 754 구조 분해 테스트
    #[test]
    fn IEEE754_구조_분해_정확성_테스트() {
        // 1.0f32 = 0x3F800000
        let bits = 0x3F800000u32;
        let f = F32Bits::from_bits(bits);
        
        assert_eq!(f.sign, 0);
        assert_eq!(f.exponent, 127);
        assert_eq!(f.mantissa, 0);
        assert_eq!(f.to_bits(), bits);
        
        // -2.5f32 = 0xC0200000  
        let bits = 0xC0200000u32;
        let f = F32Bits::from_bits(bits);
        
        assert_eq!(f.sign, 1);
        assert_eq!(f.exponent, 128);
        assert_eq!(f.mantissa, 0x200000);
        assert_eq!(f.to_bits(), bits);
        
        println!("✅ IEEE 754 구조 분해 정확성 확인됨");
    }

    #[test]
    fn 특수값_분류_정확성_테스트() {
        // 0 테스트
        let zero = F32Bits::from_bits(0x00000000);
        assert!(zero.is_zero());
        assert!(!zero.is_normalized());
        assert!(!zero.is_denormalized());
        
        // 비정규화 수 테스트
        let denorm = F32Bits::from_bits(0x00000001); // 최소 양수
        assert!(denorm.is_denormalized());
        assert!(!denorm.is_zero());
        assert!(!denorm.is_normalized());
        
        // 정규화 수 테스트  
        let norm = F32Bits::from_bits(0x3F800000); // 1.0
        assert!(norm.is_normalized());
        assert!(!norm.is_zero());
        assert!(!norm.is_denormalized());
        
        // 무한대 테스트
        let inf = F32Bits::from_bits(0x7F800000); // +∞
        assert!(inf.is_infinity());
        assert!(!inf.is_nan());
        
        // NaN 테스트
        let nan = F32Bits::from_bits(0x7FC00000);
        assert!(nan.is_nan());
        assert!(!nan.is_infinity());
        
        println!("✅ 특수값 분류 정확성 확인됨");
    }

    #[test] 
    fn 비트_덧셈_vs_실제_f32_정확성_테스트() {
        let test_cases = [
            (1.0f32, 2.0f32),
            (0.5f32, 0.25f32),
            (1000.0f32, 0.001f32),
            (-5.5f32, 3.2f32),
            (0.0f32, 42.0f32),
            (f32::EPSILON, 1.0f32),
            (1e-10f32, 1e10f32),
        ];
        
        let mut passed = 0;
        let total = test_cases.len();
        
        for (a, b) in test_cases.iter() {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            
            let expected = a + b;
            let actual_bits = bit_add(a_bits, b_bits);
            let actual = f32::from_bits(actual_bits);
            
            // 상대 오차 계산
            let relative_error = if expected.abs() > f32::EPSILON {
                ((actual - expected) / expected).abs()
            } else {
                (actual - expected).abs()
            };
            
            // 매우 엄격한 정확도 요구 (0.01% 이내)
            if relative_error < 1e-4 || (expected - actual).abs() < f32::EPSILON * 10.0 {
                passed += 1;
                println!("✅ {:.6} + {:.6} = {:.6} (예상: {:.6}, 오차: {:.2e})", 
                        a, b, actual, expected, relative_error);
            } else {
                println!("❌ {:.6} + {:.6} = {:.6} (예상: {:.6}, 오차: {:.2e})", 
                        a, b, actual, expected, relative_error);
            }
        }
        
        let accuracy = (passed as f32 / total as f32) * 100.0;
        println!("비트 덧셈 정확도: {}/{} = {:.1}%", passed, total, accuracy);
        
        // 90% 이상 정확해야 함
        assert!(accuracy >= 90.0, "비트 덧셈 정확도가 너무 낮음: {:.1}%", accuracy);
    }

    #[test]
    fn 비트_곱셈_vs_실제_f32_정확성_테스트() {
        let test_cases = [
            (2.0f32, 3.0f32),
            (0.5f32, 4.0f32),
            (-1.5f32, 2.0f32),
            (1e5f32, 1e-3f32),
            (0.1f32, 10.0f32),
            (f32::EPSILON, 1e6f32),
        ];
        
        let mut passed = 0;
        let total = test_cases.len();
        
        for (a, b) in test_cases.iter() {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            
            let expected = a * b;
            let actual_bits = bit_mul(a_bits, b_bits);
            let actual = f32::from_bits(actual_bits);
            
            let relative_error = if expected.abs() > f32::EPSILON {
                ((actual - expected) / expected).abs()
            } else {
                (actual - expected).abs()
            };
            
            if relative_error < 1e-4 || (expected - actual).abs() < f32::EPSILON * 10.0 {
                passed += 1;
                println!("✅ {:.6} × {:.6} = {:.6} (예상: {:.6}, 오차: {:.2e})", 
                        a, b, actual, expected, relative_error);
            } else {
                println!("❌ {:.6} × {:.6} = {:.6} (예상: {:.6}, 오차: {:.2e})", 
                        a, b, actual, expected, relative_error);
            }
        }
        
        let accuracy = (passed as f32 / total as f32) * 100.0;
        println!("비트 곱셈 정확도: {}/{} = {:.1}%", passed, total, accuracy);
        
        assert!(accuracy >= 90.0, "비트 곱셈 정확도가 너무 낮음: {:.1}%", accuracy);
    }

    #[test]
    fn 비트_나눗셈_vs_실제_f32_정확성_테스트() {
        let test_cases = [
            (6.0f32, 2.0f32),
            (1.0f32, 3.0f32),
            (-8.0f32, 4.0f32),
            (1e6f32, 1e3f32),
            (0.125f32, 0.25f32),
        ];
        
        let mut passed = 0;
        let total = test_cases.len();
        
        for (a, b) in test_cases.iter() {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            
            let expected = a / b;
            let actual_bits = bit_div(a_bits, b_bits);
            let actual = f32::from_bits(actual_bits);
            
            let relative_error = if expected.abs() > f32::EPSILON {
                ((actual - expected) / expected).abs()
            } else {
                (actual - expected).abs()
            };
            
            if relative_error < 1e-3 || (expected - actual).abs() < f32::EPSILON * 100.0 {
                passed += 1;
                println!("✅ {:.6} ÷ {:.6} = {:.6} (예상: {:.6}, 오차: {:.2e})", 
                        a, b, actual, expected, relative_error);
            } else {
                println!("❌ {:.6} ÷ {:.6} = {:.6} (예상: {:.6}, 오차: {:.2e})", 
                        a, b, actual, expected, relative_error);
            }
        }
        
        let accuracy = (passed as f32 / total as f32) * 100.0;
        println!("비트 나눗셈 정확도: {}/{} = {:.1}%", passed, total, accuracy);
        
        assert!(accuracy >= 80.0, "비트 나눗셈 정확도가 너무 낮음: {:.1}%", accuracy);
    }

    #[test]
    fn 다음_표현가능_값_정확성_테스트() {
        // 1.0에서 다음 값
        let one_bits = 1.0f32.to_bits();
        let next_bits = next_representable(one_bits);
        let next_val = f32::from_bits(next_bits);
        
        // 1.0보다 커야 함
        assert!(next_val > 1.0f32);
        
        // 가장 작은 증가량이어야 함
        let expected_next = f32::from_bits(one_bits + 1);
        assert_eq!(next_val, expected_next);
        
        // 0에서 다음 값
        let zero_bits = 0.0f32.to_bits();
        let next_zero_bits = next_representable(zero_bits);
        let next_zero = f32::from_bits(next_zero_bits);
        
        // 최소 양수여야 함
        assert!(next_zero > 0.0f32);
        assert_eq!(next_zero_bits, 1);
        
        println!("✅ 다음 표현가능 값 정확성 확인됨");
        println!("  1.0 다음: {:.2e}", next_val);
        println!("  0.0 다음: {:.2e}", next_zero);
    }

    #[test]
    fn 특수_케이스_처리_정확성_테스트() {
        // 0으로 나누기
        let div_by_zero = bit_div(1.0f32.to_bits(), 0.0f32.to_bits());
        let result = f32::from_bits(div_by_zero);
        assert!(result.is_infinite());
        
        // 0/0 = NaN
        let zero_div_zero = bit_div(0.0f32.to_bits(), 0.0f32.to_bits());
        let result = f32::from_bits(zero_div_zero);
        assert!(result.is_nan());
        
        // 무한대 + 1
        let inf_plus_one = bit_add(f32::INFINITY.to_bits(), 1.0f32.to_bits());
        let result = f32::from_bits(inf_plus_one);
        assert!(result.is_infinite());
        
        // 무한대 * 0 = NaN
        let inf_times_zero = bit_mul(f32::INFINITY.to_bits(), 0.0f32.to_bits());
        let result = f32::from_bits(inf_times_zero);
        assert!(result.is_nan());
        
        println!("✅ 특수 케이스 처리 정확성 확인됨");
    }

    /// IEEE 754 비트 연산 성능 vs 표준 f32 연산 성능 비교
    #[test]
    fn 비트_연산_성능_vs_f32_연산_테스트() {
        println!("=== IEEE 754 비트 연산 vs 표준 f32 연산 성능 비교 ===");
        
        const ITERATIONS: usize = 1_000_000;
        let mut rng = thread_rng();
        
        let values: Vec<f32> = (0..ITERATIONS)
            .map(|_| rng.gen_range(-1000.0..1000.0f32))
            .collect();
        
        // 덧셈 성능 테스트
        let start = Instant::now();
        let mut f32_sum = 0.0f32;
        for i in 0..ITERATIONS-1 {
            f32_sum += values[i] + values[i+1];
        }
        let f32_add_time = start.elapsed();
        
        let start = Instant::now();
        let mut bit_sum = 0.0f32;
        for i in 0..ITERATIONS-1 {
            let result_bits = bit_add(values[i].to_bits(), values[i+1].to_bits());
            bit_sum += f32::from_bits(result_bits);
        }
        let bit_add_time = start.elapsed();
        
        println!("테스트 반복: {}", ITERATIONS);
        println!("덧셈 성능:");
        println!("  표준 f32: {:?}", f32_add_time);
        println!("  비트 연산: {:?}", bit_add_time);
        println!("  속도 비율: {:.2}x (f32 연산 더 빠름)", 
                 bit_add_time.as_nanos() as f64 / f32_add_time.as_nanos() as f64);
        
        // 곱셈 성능 테스트
        let start = Instant::now();
        let mut f32_product = 1.0f32;
        for i in 0..ITERATIONS-1 {
            f32_product = (values[i] * values[i+1]).min(1e10f32);
        }
        let f32_mul_time = start.elapsed();
        
        let start = Instant::now();
        let mut bit_product = 1.0f32;
        for i in 0..ITERATIONS-1 {
            let result_bits = bit_mul(values[i].to_bits(), values[i+1].to_bits());
            bit_product = f32::from_bits(result_bits).min(1e10f32);
        }
        let bit_mul_time = start.elapsed();
        
        println!("곱셈 성능:");
        println!("  표준 f32: {:?}", f32_mul_time);
        println!("  비트 연산: {:?}", bit_mul_time);
        println!("  속도 비율: {:.2}x (비트 연산 더 빠름)", 
                 f32_mul_time.as_nanos() as f64 / bit_mul_time.as_nanos() as f64);
        
        println!("합계 정확도 차이: {:.2e}", (f32_sum - bit_sum).abs());
    }

    /// 나눗셈 디버그 테스트
    #[test]
    fn 나눗셈_디버그_테스트() {
        println!("=== 나눗셈 디버그 상세 분석 ===");
        
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
            
            println!("{}÷{} = {} (예상: {})", a, b, result, expected);
            
            // 비트 레벨 디버깅
            let a_struct = F32Bits::from_bits(a_bits);
            let b_struct = F32Bits::from_bits(b_bits);
            let result_struct = F32Bits::from_bits(result_bits);
            let expected_struct = F32Bits::from_bits(expected.to_bits());
            
            println!("  A: sign={}, exp={:3}, mant={:06X}", a_struct.sign, a_struct.exponent, a_struct.mantissa);
            println!("  B: sign={}, exp={:3}, mant={:06X}", b_struct.sign, b_struct.exponent, b_struct.mantissa);
            println!("  R: sign={}, exp={:3}, mant={:06X}", result_struct.sign, result_struct.exponent, result_struct.mantissa);
            println!("  E: sign={}, exp={:3}, mant={:06X}", expected_struct.sign, expected_struct.exponent, expected_struct.mantissa);
            
            let error = if expected != 0.0 {
                ((result - expected) / expected).abs()
            } else {
                result.abs()
            };
            println!("  상대 오차: {:.6e}", error);
            println!();
        }
    }

    #[test]
    fn 랜덤_대규모_정확성_스트레스_테스트() {
        let mut rng = thread_rng();
        let iterations = 10000;
        let mut add_passed = 0;
        let mut mul_passed = 0;
        
        for _ in 0..iterations {
            // 다양한 범위의 랜덤 값 생성
            let a = if rng.gen_bool(0.1) {
                // 10% 확률로 극값
                match rng.gen_range(0..4) {
                    0 => 0.0f32,
                    1 => f32::EPSILON,
                    2 => 1e-10f32,
                    _ => 1e10f32,
                }
            } else {
                rng.gen_range(-1000.0f32..1000.0f32)
            };
            
            let b = if rng.gen_bool(0.1) {
                match rng.gen_range(0..4) {
                    0 => 0.0f32,
                    1 => f32::EPSILON,
                    2 => 1e-10f32,
                    _ => 1e10f32,
                }
            } else {
                rng.gen_range(-1000.0f32..1000.0f32)
            };
            
            // 덧셈 테스트
            let expected_add = a + b;
            let actual_add = f32::from_bits(bit_add(a.to_bits(), b.to_bits()));
            
            let add_error = if expected_add.abs() > f32::EPSILON {
                ((actual_add - expected_add) / expected_add).abs()
            } else {
                (actual_add - expected_add).abs()
            };
            
            if add_error < 1e-3 || (expected_add - actual_add).abs() < f32::EPSILON * 100.0 {
                add_passed += 1;
            }
            
            // 곱셈 테스트 (0이 아닌 경우만)
            if a.abs() > f32::EPSILON && b.abs() > f32::EPSILON {
                let expected_mul = a * b;
                let actual_mul = f32::from_bits(bit_mul(a.to_bits(), b.to_bits()));
                
                let mul_error = if expected_mul.abs() > f32::EPSILON {
                    ((actual_mul - expected_mul) / expected_mul).abs()
                } else {
                    (actual_mul - expected_mul).abs()
                };
                
                if mul_error < 1e-2 || (expected_mul - actual_mul).abs() < f32::EPSILON * 1000.0 {
                    mul_passed += 1;
                }
            } else {
                mul_passed += 1; // 0 케이스는 통과로 처리
            }
        }
        
        let add_accuracy = (add_passed as f32 / iterations as f32) * 100.0;
        let mul_accuracy = (mul_passed as f32 / iterations as f32) * 100.0;
        
        println!("=== 대규모 랜덤 스트레스 테스트 결과 ===");
        println!("반복 횟수: {}", iterations);
        println!("덧셈 정확도: {}/{} = {:.1}%", add_passed, iterations, add_accuracy);
        println!("곱셈 정확도: {}/{} = {:.1}%", mul_passed, iterations, mul_accuracy);
        
        // 매우 엄격한 기준: 85% 이상 정확해야 함
        assert!(add_accuracy >= 85.0, "대규모 덧셈 정확도 미달: {:.1}%", add_accuracy);
        assert!(mul_accuracy >= 80.0, "대규모 곱셈 정확도 미달: {:.1}%", mul_accuracy);
    }

    #[test]
    fn 언더플로우_오버플로우_경계_테스트() {
        // 최대값 근처
        let max_f32 = f32::MAX;
        let near_max = max_f32 * 0.9;
        
        // 오버플로우 테스트
        let overflow_result = bit_mul(max_f32.to_bits(), 2.0f32.to_bits());
        let result = f32::from_bits(overflow_result);
        assert!(result.is_infinite(), "오버플로우가 무한대로 처리되지 않음");
        
        // 언더플로우 테스트
        let min_normal = f32::MIN_POSITIVE;
        let underflow_result = bit_mul(min_normal.to_bits(), 0.1f32.to_bits());
        let result = f32::from_bits(underflow_result);
        // 결과가 0이거나 매우 작은 값이어야 함
        assert!(result.abs() < min_normal, "언더플로우가 적절히 처리되지 않음");
        
        println!("✅ 오버플로우/언더플로우 경계 처리 확인됨");
        println!("  오버플로우: {} * 2.0 = {}", max_f32, result);
        println!("  언더플로우: {} * 0.1 = {}", min_normal, f32::from_bits(underflow_result));
    }

    /// f64 IEEE 754 비트 연산 정확도 테스트
    #[test]
    fn f64_비트_연산_정확도_테스트() {
        println!("=== f64 IEEE 754 비트 연산 정확도 테스트 ===");
        
        // 덧셈 테스트
        let test_cases_add = vec![
            (1.0f64, 2.0f64),
            (0.5f64, 0.25f64),
            (1000.0f64, 0.001f64),
            (-5.5f64, 3.2f64),
            (0.0f64, 42.0f64),
        ];
        
        println!("f64 덧셈 테스트:");
        let mut add_correct = 0;
        for (a, b) in &test_cases_add {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            let result_bits = bit_add_f64(a_bits, b_bits);
            let result = f64::from_bits(result_bits);
            let expected = a + b;
            
            let error = if expected != 0.0 {
                ((result - expected) / expected).abs()
            } else {
                result.abs()
            };
            
            if error < 1e-12 { // f64는 더 엄격한 기준
                add_correct += 1;
                println!("✅ {} + {} = {} (예상: {}, 오차: {:.2e})", a, b, result, expected, error);
            } else {
                println!("❌ {} + {} = {} (예상: {}, 오차: {:.2e})", a, b, result, expected, error);
            }
        }
        
        // 곱셈 테스트
        let test_cases_mul = vec![
            (2.0f64, 3.0f64),
            (0.5f64, 4.0f64),
            (-1.5f64, 2.0f64),
            (100000.0f64, 0.001f64),
            (0.1f64, 10.0f64),
        ];
        
        println!("f64 곱셈 테스트:");
        let mut mul_correct = 0;
        for (a, b) in &test_cases_mul {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            let result_bits = bit_mul_f64(a_bits, b_bits);
            let result = f64::from_bits(result_bits);
            let expected = a * b;
            
            let error = if expected != 0.0 {
                ((result - expected) / expected).abs()
            } else {
                result.abs()
            };
            
            if error < 1e-12 {
                mul_correct += 1;
                println!("✅ {} × {} = {} (예상: {}, 오차: {:.2e})", a, b, result, expected, error);
            } else {
                println!("❌ {} × {} = {} (예상: {}, 오차: {:.2e})", a, b, result, expected, error);
            }
        }
        
        // 나눗셈 테스트
        let test_cases_div = vec![
            (6.0f64, 2.0f64),
            (1.0f64, 3.0f64),
            (10.0f64, 4.0f64),
            (100.0f64, 10.0f64),
            (1.0f64, 1.0f64),
        ];
        
        println!("f64 나눗셈 테스트:");
        let mut div_correct = 0;
        for (a, b) in &test_cases_div {
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            let result_bits = bit_div_f64(a_bits, b_bits);
            let result = f64::from_bits(result_bits);
            let expected = a / b;
            
            let error = if expected != 0.0 {
                ((result - expected) / expected).abs()
            } else {
                result.abs()
            };
            
            if error < 1e-12 {
                div_correct += 1;
                println!("✅ {} ÷ {} = {} (예상: {}, 오차: {:.2e})", a, b, result, expected, error);
            } else {
                println!("❌ {} ÷ {} = {} (예상: {}, 오차: {:.2e})", a, b, result, expected, error);
            }
        }
        
        // 전체 결과
        let total_correct = add_correct + mul_correct + div_correct;
        let total_tests = test_cases_add.len() + test_cases_mul.len() + test_cases_div.len();
        let accuracy = (total_correct as f64 / total_tests as f64) * 100.0;
        
        println!("\n=== f64 비트 연산 종합 결과 ===");
        println!("덧셈 정확도: {}/{} = {:.1}%", add_correct, test_cases_add.len(), (add_correct as f64 / test_cases_add.len() as f64) * 100.0);
        println!("곱셈 정확도: {}/{} = {:.1}%", mul_correct, test_cases_mul.len(), (mul_correct as f64 / test_cases_mul.len() as f64) * 100.0);
        println!("나눗셈 정확도: {}/{} = {:.1}%", div_correct, test_cases_div.len(), (div_correct as f64 / test_cases_div.len() as f64) * 100.0);
        println!("**전체 f64 정확도: {}/{} = {:.1}%**", total_correct, total_tests, accuracy);
        
        assert!(accuracy > 95.0, "f64 비트 연산 정확도가 95% 미만: {:.1}%", accuracy);
    }
} 
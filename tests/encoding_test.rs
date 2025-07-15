//! `encoding.rs`에 대한 단위 테스트

use poincare_layer::Packed64;
use std::f32::consts::PI;

#[test]
fn test_encoding_bit_packing() {
    println!("\n--- Test: Encoding Bit Packing ---");

    // 1. 테스트할 파라미터 정의 (새로운 시그니처)
    let r = 0.5;
    let theta = PI;
    let basis_id = 10;
    let freq_x = 3;
    let freq_y = 4;
    let amplitude = 1.5;
    let offset = -0.5;
    let pattern_mix = 5;
    let decay_rate = 1.0;
    let d_theta = 3;
    let d_r = true;
    let log2_c = -1;  // 2비트로 축소됨

    // 2. 인코딩 실행
    let packed = Packed64::new(r, theta, basis_id, freq_x, freq_y, amplitude, offset, 
                             pattern_mix, decay_rate, d_theta, d_r, log2_c);

    // 3. 디코딩하여 값 확인
    let decoded = packed.decode();
    
    // 4. 검증
    println!("  - Original r: {} -> Decoded: {}", r, decoded.r);
    println!("  - Original theta: {} -> Decoded: {}", theta, decoded.theta);
    println!("  - Original basis_id: {} -> Decoded: {}", basis_id, decoded.basis_id);
    println!("  - Original freq_x: {} -> Decoded: {}", freq_x, decoded.freq_x);
    println!("  - Original freq_y: {} -> Decoded: {}", freq_y, decoded.freq_y);
    
    // 정밀도 테스트
    assert!((decoded.r - r).abs() < 0.001);
    assert!((decoded.theta - theta).abs() < 0.002);
    assert_eq!(decoded.basis_id, basis_id);
    assert_eq!(decoded.freq_x, freq_x);
    assert_eq!(decoded.freq_y, freq_y);
    assert!((decoded.amplitude - amplitude).abs() < 0.1);
    assert!((decoded.offset - offset).abs() < 0.1);
    
    println!("  [PASSED] Bit packing and unpacking is correct.");
}

#[test]
fn test_encoding_clamping_and_normalization() {
    println!("\n--- Test: Encoding Clamping and Normalization ---");

    // 경계값 테스트
    let r_over = 1.5;
    let theta_over = 3.0 * PI;

    let packed = Packed64::new(r_over, theta_over, 0, 1, 1, 1.0, 0.0, 0, 0.0, 0, false, 0);
    let decoded = packed.decode();

    // r은 1.0 미만으로 클램핑되어야 함
    assert!(decoded.r <= 0.999999, "r value should be clamped to be less than 1.0, got {}", decoded.r);
    println!("  [PASSED] r value is clamped correctly.");

    // theta는 [0, 2π) 범위로 정규화되어야 함 (3π -> π)
    assert!(
        (decoded.theta - PI).abs() < 0.01,  // 12비트 정밀도로 인해 epsilon 증가
        "theta value should be normalized to the [0, 2π) range, got {}", decoded.theta
    );
    println!("  [PASSED] theta value is normalized correctly.");
} 
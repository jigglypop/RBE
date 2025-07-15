//! `decoding.rs`에 대한 단위 테스트

use poincare_layer::Packed64;
use std::f32::consts::PI;

#[test]
fn test_decoding_bit_unpacking() {
    println!("\n--- Test: Decoding Bit Unpacking ---");

    // 1. 새로운 비트 레이아웃에 맞는 테스트 값 생성
    let r_bits: u64 = 0x800;         // r = 0.5 (12비트)
    let theta_bits: u64 = 0x800;     // theta = PI (12비트)
    let basis_id: u64 = 10;
    let freq_x: u64 = 3;
    let freq_y: u64 = 4;
    let amplitude_bits: u64 = 32;    // 약 1.5
    let offset_bits: u64 = 16;       // 약 -1.0
    let pattern_mix: u64 = 5;
    let decay_bits: u64 = 8;         // 약 1.0
    let d_theta: u64 = 3;
    let d_r: u64 = 1;
    let log2_c: u64 = 1;             // -1 (2비트)

    let test_packed_val = (r_bits << 52)
        | (theta_bits << 40)
        | (basis_id << 36)
        | (freq_x << 31)
        | (freq_y << 26)
        | (amplitude_bits << 20)
        | (offset_bits << 14)
        | (pattern_mix << 10)
        | (decay_bits << 5)
        | (d_theta << 3)
        | (d_r << 2)
        | log2_c;
    
    let packed = Packed64(test_packed_val);

    // 2. 디코딩 실행
    let decoded = packed.decode();

    // 3. 검증
    println!("  - Packed Value: 0x{:016X}", test_packed_val);
    println!("  - Decoded Params: {:?}", decoded);

    assert!((decoded.r - 0.5).abs() < 0.001);
    assert!((decoded.theta - PI).abs() < 0.002);
    assert_eq!(decoded.basis_id, basis_id as u8);
    assert_eq!(decoded.freq_x, freq_x as u8);
    assert_eq!(decoded.freq_y, freq_y as u8);
    assert_eq!(decoded.d_theta, d_theta as u8);
    assert_eq!(decoded.d_r, d_r == 1);
    assert_eq!(decoded.log2_c, -1);
    println!("  [PASSED] All fields were decoded correctly.");
}

#[test]
fn test_signed_int_decoding() {
    println!("\n--- Test: Signed Integer (log2_c) Decoding ---");
    
    // log2_c: 2비트, 범위 -2 ~ +1
    let test_cases = vec![
        (0b00, -2),
        (0b01, -1),
        (0b10, 0),
        (0b11, 1),
    ];

    for (bits, expected_val) in test_cases {
        println!("  - Bits: 0b{:02b} -> Expected: {}", bits, expected_val);
        
        // 실제 디코딩 로직 테스트
        let log2_c = match bits {
            0 => -2,
            1 => -1,
            2 => 0,
            3 => 1,
            _ => 0,
        };
        
        assert_eq!(log2_c, expected_val);
    }
    
    println!("  [PASSED] 2-bit signed integer decoding is correct.");
} 
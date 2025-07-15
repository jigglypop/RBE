//! `decoding.rs`에 대한 단위 테스트

use poincare_layer::types::Packed64;

#[test]
/// CORDIC 모델에서 `decode` 메소드가 Packed64에 저장된
/// 64비트 회전 시퀀스를 정확하게 반환하는지 테스트합니다.
fn test_cordic_seed_decoding() {
    // 테스트용 원본 회전 시퀀스
    let original_rotations = 0xFEDCBA9876543210_u64;

    // `new`를 통해 인코딩 (저장)
    let packed = Packed64::new(original_rotations);

    // `decode`를 통해 디코딩 (검색)
    let decoded_rotations = packed.decode();

    // 디코딩된 값이 원본과 일치하는지 확인
    assert_eq!(
        decoded_rotations, original_rotations,
        "The decoded rotation sequence should match the original value."
    );

    println!("PASSED: test_cordic_seed_decoding");
} 
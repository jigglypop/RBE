//! `encoding.rs`에 대한 단위 테스트

use poincare_layer::types::Packed64;

#[test]
/// CORDIC 모델에서 Packed64 구조체가 64비트 회전 시퀀스를
/// 올바르게 저장하고 반환하는지 테스트합니다.
fn 코딕_시드_저장_및_검색_테스트() {
    // 임의의 u64 값을 회전 시퀀스로 사용
    let original_rotations = 0x1A2B3C4D5E6F7890_u64;

    // Packed64 생성
    let packed = Packed64::new(original_rotations);

    // decode 메소드를 통해 저장된 값 확인
    let decoded_rotations = packed.decode();

    // 원본 값과 디코딩된 값이 일치하는지 확인
    assert_eq!(
        original_rotations, decoded_rotations,
        "Decoded rotations should be identical to the original."
    );
    
    // 공개 필드를 직접 접근하여 확인
    assert_eq!(
        original_rotations, packed.rotations,
        "Public field `rotations` should be identical to the original."
    );

    println!("PASSED: 코딕_시드_저장_및_검색_테스트");
    
    // 압축률: 64비트로 32x32 행렬(4096 바이트) 저장 = 512:1
    println!("압축률: 32x32 행렬(4096 bytes) -> 64 bits = 512:1 압축");
} 
//! `math.rs`에 대한 단위 테스트

use poincare_layer::math::{compute_full_rmse, mutate_seed};
use poincare_layer::types::{Packed64, Packed128, PoincareMatrix};

#[test]
/// `calculate_rmse` 함수의 정확성을 테스트합니다.
fn test_calculate_rmse() {
    let rows = 4;
    let cols = 4;

    // 1. 초기 시드와 행렬 준비
    let original_seed = Packed64::new(0x123456789ABCDEF0);
    let poincare_matrix = PoincareMatrix { 
        seed: Packed128 { hi: original_seed.rotations, lo: 0 }, 
        rows: 8, 
        cols: 8 
    };
    let matrix = poincare_matrix.decompress();

    // 2. 변이 전 RMSE 계산
    let rmse_before = compute_full_rmse(&matrix, &original_seed, 8, 8);
    assert!(rmse_before < 1e-6, "RMSE before mutation should be close to zero.");

    // 3. 시드 변이
    let mutated_seed = mutate_seed(original_seed, 0.1);

    // 4. 변이 후 RMSE 계산
    let rmse_after = compute_full_rmse(&matrix, &mutated_seed, 8, 8);
    println!("\n--- Test: Seed Mutation ---");
    println!("  - RMSE before mutation: {}", rmse_before);
    println!("  - RMSE after mutation: {}", rmse_after);

    // 변이가 일어났다면 시드가 바뀌고, 결과적으로 RMSE가 0이 아니어야 합니다.
    assert!(rmse_after > 1e-6, "RMSE after mutation should be greater than 0.");
}

#[test]
/// `mutate_seed` 함수의 동작을 다양한 변이 확률에 대해 테스트합니다.
fn test_mutate_seed() {
    let original_seed_val = 0xAAAAAAAAAAAAAAAA; // 10101010...
    let original_seed = Packed64::new(original_seed_val);

    // 1. 변이 확률 0.0: 시드가 변경되지 않아야 함
    let mutated_seed_zero_rate = mutate_seed(original_seed, 0.0);
    assert_eq!(
        mutated_seed_zero_rate.rotations,
        original_seed_val,
        "Seed should not change with a mutation rate of 0.0"
    );
    println!("PASSED: Mutation rate of 0.0 causes no change.");

    // 2. 변이 확률 1.0: 모든 비트가 반전되어야 함
    let mutated_seed_full_rate = mutate_seed(original_seed, 1.0);
    assert_eq!(
        mutated_seed_full_rate.rotations,
        !original_seed_val,
        "Seed should be bitwise NOT with a mutation rate of 1.0"
    );
    println!("PASSED: Mutation rate of 1.0 inverts all bits.");

    // 3. 변이 확률 0.5: 시드가 변경되어야 함
    let mutated_seed_mid_rate = mutate_seed(original_seed, 0.5);
    assert_ne!(
        mutated_seed_mid_rate.rotations,
        original_seed_val,
        "Seed should change with a mutation rate of 0.5"
    );
    println!("PASSED: Mutation rate of 0.5 causes some change.");

    // 변경된 비트 수 확인 (통계적 검증)
    let flipped_bits = (original_seed_val ^ mutated_seed_mid_rate.rotations).count_ones();
    println!("  - Flipped bits at 50% rate: {} (expected ~32)", flipped_bits);
    assert!(flipped_bits > 10 && flipped_bits < 54, "Number of flipped bits is statistically unlikely for a 0.5 rate.");
} 
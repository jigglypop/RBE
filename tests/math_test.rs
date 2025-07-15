//! `math.rs`에 대한 단위 테스트

use poincare_layer::math::{calculate_rmse, mutate_seed};
use poincare_layer::types::{Packed64, PoincareMatrix};

#[test]
/// `calculate_rmse` 함수의 정확성을 테스트합니다.
fn test_calculate_rmse() {
    let rows = 4;
    let cols = 4;

    // 1. 완벽하게 재구성 가능한 경우 (RMSE ≈ 0)
    let original_seed = Packed64::new(0x1122334455667788);
    let matrix_generator = PoincareMatrix {
        seed: original_seed,
        rows,
        cols,
    };
    let original_matrix = matrix_generator.decompress();

    let rmse_perfect = calculate_rmse(&original_matrix, &original_seed, rows, cols);
    assert!(
        rmse_perfect < 1e-6,
        "RMSE for a perfectly reconstructed matrix should be close to 0, but was {}",
        rmse_perfect
    );
    println!("PASSED: RMSE is near zero for perfect reconstruction.");

    // 2. 다른 시드로 재구성하는 경우 (RMSE > 0)
    let different_seed = Packed64::new(0x8877665544332211);
    let rmse_different = calculate_rmse(&original_matrix, &different_seed, rows, cols);
    assert!(
        rmse_different > 1e-6,
        "RMSE for a differently reconstructed matrix should be greater than 0, but was {}",
        rmse_different
    );
    println!("PASSED: RMSE is non-zero for imperfect reconstruction.");
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
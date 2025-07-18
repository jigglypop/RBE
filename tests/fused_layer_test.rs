
use nalgebra::{DMatrix, DVector};
use poincare_layer::encoder::HybridEncoder;
use poincare_layer::types::{TransformType, HybridEncodedBlock};
use poincare_layer::EncodedLayer;

#[test]
fn test_fused_forward_correctness() {
    // 1. 4x4 행렬 생성 및 인코딩
    let rows = 4;
    let cols = 4;
    let original_matrix = DMatrix::from_fn(rows, cols, |r, c| {
        ((r + c * 10) as f32) / 100.0 // 간단한 패턴
    });
    let original_data: Vec<f32> = original_matrix.iter().cloned().collect();

    let mut encoder = HybridEncoder::new(2, TransformType::Dct); // K=2 DCT
    let encoded_block = encoder.encode_block(&original_data, rows, cols);

    // 2. EncodedLayer 생성
    let blocks_grid = vec![vec![encoded_block.clone()]];
    let layer = EncodedLayer::new(blocks_grid, rows, cols);

    // 3. 디코딩하여 예상 결과 계산
    let decoded_data = encoded_block.decode();
    let decoded_matrix = DMatrix::from_vec(rows, cols, decoded_data);
    
    // 4. 임의의 입력 벡터 생성
    let input_vector = DVector::from_vec(vec![0.1, -0.2, 0.3, 0.5]);

    // 5. 예상 결과 계산 (W * x)
    let expected_output: DVector<f64> = decoded_matrix.map(|e| e as f64) * input_vector.map(|e| e as f64);
    
    // 6. fused_forward로 실제 결과 계산
    let actual_output: DVector<f64> = layer.fused_forward(&input_vector.map(|e| e as f64));
    
    // 7. 결과 비교
    let tolerance = 1e-6;
    let difference = (&expected_output - &actual_output).norm();
    
    println!("Decoded Matrix:\n{}", decoded_matrix);
    println!("Input Vector:\n{}", input_vector);
    println!("Expected Output (Decoded * x):\n{}", expected_output);
    println!("Actual Output (fused_forward):\n{}", actual_output);
    println!("Difference norm: {}", difference);

    assert!(
        difference < tolerance,
        "The output of fused_forward should be very close to the output of the decoded matrix multiplication."
    );
}

#[test]
fn test_fused_backward_correctness() {
    // 1. Setup: Same as forward test
    let rows = 4;
    let cols = 4;
    let original_matrix = DMatrix::from_fn(rows, cols, |r, c| ((r + c * 10) as f32) / 100.0);
    let mut encoder = poincare_layer::encoder::HybridEncoder::new(2, TransformType::Dct);
    let encoded_block = encoder.encode_block(&original_matrix.iter().cloned().collect::<Vec<f32>>(), rows, cols);
    let layer = poincare_layer::EncodedLayer::new(vec![vec![encoded_block.clone()]], rows, cols);
    let decoded_matrix = DMatrix::from_vec(rows, cols, encoded_block.decode()).map(|e| e as f64);

    // 2. Arbitrary input and output gradient
    let input_vector = DVector::from_vec(vec![0.1, -0.2, 0.3, 0.5]);
    let d_loss_d_y = DVector::from_vec(vec![-0.1, 0.2, -0.3, 0.4]);

    // 3. Expected result: d_loss/d_x = W^T * d_loss/d_y
    let expected_d_loss_d_x = decoded_matrix.transpose() * &d_loss_d_y;

    // 4. Actual result from fused_backward
    let (actual_d_loss_d_x, _param_grads) = layer.fused_backward(&input_vector, &d_loss_d_y);

    // 5. Compare
    let tolerance = 1e-6;
    let difference = (&expected_d_loss_d_x - &actual_d_loss_d_x).norm();

    println!("Decoded Matrix Transposed:\n{}", decoded_matrix.transpose());
    println!("d_loss/d_y:\n{}", d_loss_d_y);
    println!("Expected d_loss/d_x:\n{}", expected_d_loss_d_x);
    println!("Actual d_loss/d_x:\n{}", actual_d_loss_d_x);
    println!("Difference norm: {}", difference);

    assert!(
        difference < tolerance,
        "d_loss/d_x from fused_backward should match the reference calculation."
    );
} 
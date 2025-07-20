use crate::nlp::linear::rbe_linear::*;
use crate::core::*;
use anyhow::Result;

/// 테스트용 유틸리티 함수들
fn generate_random_weights(rows: usize, cols: usize) -> Vec<f32> {
    use rand::prelude::*;
    let mut rng = thread_rng();
    (0..rows * cols).map(|_| rng.gen_range(-0.5..0.5)).collect()
}

fn generate_random_input(batch_size: usize, seq_len: usize, dim: usize) -> Vec<f32> {
    use rand::prelude::*;
    let mut rng = thread_rng();
    (0..batch_size * seq_len * dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn compute_reference_linear(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    input_dim: usize,
    output_dim: usize,
    batch_size: usize,
    seq_len: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch_size * seq_len * output_dim];
    
    for b in 0..batch_size {
        for s in 0..seq_len {
            let input_offset = (b * seq_len + s) * input_dim;
            let output_offset = (b * seq_len + s) * output_dim;
            
            // Matrix multiplication: output = input * weights^T
            for i in 0..output_dim {
                let mut sum = 0.0f32;
                for j in 0..input_dim {
                    sum += input[input_offset + j] * weights[i * input_dim + j];
                }
                output[output_offset + i] = sum;
                
                // Add bias
                if let Some(bias_vec) = bias {
                    output[output_offset + i] += bias_vec[i];
                }
            }
        }
    }
    
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rbe_linear_basic_functionality() -> Result<()> {
        println!("=== RBE Linear Layer 기본 기능 테스트 ===");
        
        let input_dim = 128;
        let output_dim = 64;
        let batch_size = 4;
        let seq_len = 32;
        let block_size = 32;
        let compression_ratio = 100;
        
        // 1. 랜덤 가중치 생성
        let weights = generate_random_weights(output_dim, input_dim);
        let bias = Some((0..output_dim).map(|_| 0.1f32).collect::<Vec<_>>());
        
        // 2. RBE Linear Layer 생성
        let rbe_linear = RBELinear::from_dense_weights(
            &weights,
            input_dim,
            output_dim,
            bias.clone(),
            block_size,
            compression_ratio,
        )?;
        
        println!("✓ RBE Linear Layer 생성 완료");
        println!("  - 입력 차원: {}", input_dim);
        println!("  - 출력 차원: {}", output_dim);
        println!("  - 블록 크기: {}", block_size);
        println!("  - 압축률: {}", compression_ratio);
        
        // 3. 테스트 입력 생성
        let input = generate_random_input(batch_size, seq_len, input_dim);
        
        // 4. RBE 순전파
        let rbe_output = rbe_linear.forward(&input, batch_size, seq_len)?;
        
        // 5. 참조 구현 (일반 linear layer)
        let reference_output = compute_reference_linear(
            &input, &weights, bias.as_deref(),
            input_dim, output_dim, batch_size, seq_len
        );
        
        // 6. 정확도 검증
        let relative_error = crate::nlp::compute_relative_error(&reference_output, &rbe_output);
        println!("✓ 순전파 완료");
        println!("  - 상대 오차: {:.6}", relative_error);
        
        // 정확도 임계값 확인 (압축으로 인한 오차 허용)
        assert!(relative_error < 0.1, "정확도가 너무 낮습니다: {}", relative_error);
        
        // 7. 메모리 사용량 확인
        let original_memory = weights.len() * 4 + bias.as_ref().map_or(0, |b| b.len() * 4);
        let compressed_memory = rbe_linear.memory_usage_bytes();
        let compression_ratio_actual = original_memory as f32 / compressed_memory as f32;
        
        println!("✓ 메모리 효율성");
        println!("  - 원본 메모리: {} bytes", original_memory);
        println!("  - 압축 메모리: {} bytes", compressed_memory);
        println!("  - 실제 압축률: {:.2}x", compression_ratio_actual);
        
        assert!(compression_ratio_actual > 2.0, "압축률이 너무 낮습니다: {:.2}x", compression_ratio_actual);
        
        Ok(())
    }
    
    #[test]
    fn test_rbe_linear_backward_pass() -> Result<()> {
        println!("=== RBE Linear Layer 역전파 테스트 ===");
        
        let input_dim = 64;
        let output_dim = 32;
        let batch_size = 2;
        let seq_len = 16;
        let block_size = 32;
        
        // 1. RBE Linear Layer 생성
        let weights = generate_random_weights(output_dim, input_dim);
        let rbe_linear = RBELinear::from_dense_weights(
            &weights, input_dim, output_dim, None, block_size, 50
        )?;
        
        // 2. 순전파
        let input = generate_random_input(batch_size, seq_len, input_dim);
        let output = rbe_linear.forward(&input, batch_size, seq_len)?;
        
        // 3. 역전파
        let grad_output = generate_random_input(batch_size, seq_len, output_dim);
        let (grad_input, gradients) = rbe_linear.backward(&grad_output, &input, batch_size, seq_len)?;
        
        // 4. 크기 검증
        assert_eq!(grad_input.len(), input.len());
        assert_eq!(gradients.grad_weights.len(), output_dim * input_dim);
        
        println!("✓ 역전파 크기 검증 통과");
        
        // 5. 수치적 gradient 검증 (작은 perturbation)
        let epsilon = 1e-4;
        let mut numerical_grad = vec![0.0f32; input.len()];
        
        for i in 0..input.len().min(10) { // 처음 10개만 테스트 (속도)
            let mut input_plus = input.clone();
            let mut input_minus = input.clone();
            input_plus[i] += epsilon;
            input_minus[i] -= epsilon;
            
            let output_plus = rbe_linear.forward(&input_plus, batch_size, seq_len)?;
            let output_minus = rbe_linear.forward(&input_minus, batch_size, seq_len)?;
            
            let mut grad_sum = 0.0f32;
            for j in 0..output.len() {
                let numerical_partial = (output_plus[j] - output_minus[j]) / (2.0 * epsilon);
                grad_sum += grad_output[j] * numerical_partial;
            }
            numerical_grad[i] = grad_sum;
        }
        
        // 수치적 gradient와 비교
        let mut grad_error = 0.0f32;
        for i in 0..10 {
            let error = (grad_input[i] - numerical_grad[i]).abs();
            grad_error = grad_error.max(error);
        }
        
        println!("✓ 수치적 gradient 검증");
        println!("  - 최대 오차: {:.6}", grad_error);
        
        assert!(grad_error < 1e-2, "Gradient 오차가 너무 큽니다: {}", grad_error);
        
        Ok(())
    }
    
    #[test]
    fn test_performance_comparison() -> Result<()> {
        println!("=== RBE vs 참조 구현 성능 비교 ===");
        
        let input_dim = 512;
        let output_dim = 256;
        let batch_size = 8;
        let seq_len = 64;
        let block_size = 64;
        
        // 1. 테스트 데이터
        let weights = generate_random_weights(output_dim, input_dim);
        let input = generate_random_input(batch_size, seq_len, input_dim);
        
        // 2. RBE Linear Layer
        let rbe_linear = RBELinear::from_dense_weights(
            &weights, input_dim, output_dim, None, block_size, 150
        )?;
        
        // 3. RBE 성능 측정
        let rbe_start = std::time::Instant::now();
        let _rbe_output = rbe_linear.forward(&input, batch_size, seq_len)?;
        let rbe_time = rbe_start.elapsed();
        
        // 4. 참조 구현 성능 측정
        let ref_start = std::time::Instant::now();
        let _ref_output = compute_reference_linear(
            &input, &weights, None, input_dim, output_dim, batch_size, seq_len
        );
        let ref_time = ref_start.elapsed();
        
        // 5. 메모리 사용량
        let original_memory = weights.len() * 4;
        let compressed_memory = rbe_linear.memory_usage_bytes();
        
        println!("✓ 성능 비교 결과");
        println!("  - RBE 시간: {:.3}ms", rbe_time.as_millis());
        println!("  - 참조 시간: {:.3}ms", ref_time.as_millis());
        println!("  - 속도 비율: {:.2}x", ref_time.as_nanos() as f32 / rbe_time.as_nanos() as f32);
        println!("  - 메모리 절약: {:.2}x", original_memory as f32 / compressed_memory as f32);
        
        Ok(())
    }
    
    #[test]
    fn test_different_block_sizes() -> Result<()> {
        println!("=== 다양한 블록 크기 테스트 ===");
        
        let input_dim = 256;
        let output_dim = 128;
        let batch_size = 4;
        let seq_len = 32;
        
        let weights = generate_random_weights(output_dim, input_dim);
        let input = generate_random_input(batch_size, seq_len, input_dim);
        let reference_output = compute_reference_linear(
            &input, &weights, None, input_dim, output_dim, batch_size, seq_len
        );
        
        let block_sizes = vec![16, 32, 64, 128];
        
        for &block_size in &block_sizes {
            // block_size가 차원보다 클 경우 스킵
            if block_size > input_dim.min(output_dim) {
                continue;
            }
            
            let rbe_linear = RBELinear::from_dense_weights(
                &weights, input_dim, output_dim, None, block_size, 100
            )?;
            
            let rbe_output = rbe_linear.forward(&input, batch_size, seq_len)?;
            let error = crate::nlp::compute_relative_error(&reference_output, &rbe_output);
            let memory_ratio = (weights.len() * 4) as f32 / rbe_linear.memory_usage_bytes() as f32;
            
            println!("  블록 크기 {}: 오차 {:.6}, 압축률 {:.2}x", 
                    block_size, error, memory_ratio);
            
            assert!(error < 0.2, "블록 크기 {}에서 오차가 너무 큽니다: {}", block_size, error);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_edge_cases() -> Result<()> {
        println!("=== 경계 조건 테스트 ===");
        
        // 1. 매우 작은 차원
        let small_linear = RBELinear::from_dense_weights(
            &vec![1.0, 2.0, 3.0, 4.0], 2, 2, None, 2, 10
        )?;
        
        let small_input = vec![1.0, 0.5, -1.0, 1.5]; // batch=2, seq=1, dim=2
        let small_output = small_linear.forward(&small_input, 2, 1)?;
        assert_eq!(small_output.len(), 4); // batch=2, seq=1, out_dim=2
        
        println!("✓ 작은 차원 테스트 통과");
        
        // 2. 단일 토큰
        let single_input = vec![1.0, 0.5]; // batch=1, seq=1, dim=2
        let single_output = small_linear.forward(&single_input, 1, 1)?;
        assert_eq!(single_output.len(), 2);
        
        println!("✓ 단일 토큰 테스트 통과");
        
        // 3. 긴 시퀀스
        let long_input = generate_random_input(1, 256, 64);
        let long_weights = generate_random_weights(32, 64);
        let long_linear = RBELinear::from_dense_weights(
            &long_weights, 64, 32, None, 32, 50
        )?;
        
        let long_output = long_linear.forward(&long_input, 1, 256)?;
        assert_eq!(long_output.len(), 256 * 32);
        
        println!("✓ 긴 시퀀스 테스트 통과");
        
        Ok(())
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_full_pipeline() -> Result<()> {
        println!("=== 전체 파이프라인 통합 테스트 ===");
        
        let input_dim = 384;
        let output_dim = 192;
        let batch_size = 8;
        let seq_len = 128;
        let block_size = 48;
        
        // 1. 초기 가중치
        let initial_weights = generate_random_weights(output_dim, input_dim);
        let bias = Some((0..output_dim).map(|i| (i as f32) * 0.01).collect::<Vec<_>>());
        
        // 2. RBE 압축
        let rbe_linear = RBELinear::from_dense_weights(
            &initial_weights, input_dim, output_dim, bias.clone(), block_size, 200
        )?;
        
        // 3. 여러 번의 순전파/역전파
        let mut total_error = 0.0f32;
        let iterations = 5;
        
        for i in 0..iterations {
            let input = generate_random_input(batch_size, seq_len, input_dim);
            let grad_output = generate_random_input(batch_size, seq_len, output_dim);
            
            // 순전파
            let output = rbe_linear.forward(&input, batch_size, seq_len)?;
            
            // 참조 결과와 비교
            let reference = compute_reference_linear(
                &input, &initial_weights, bias.as_deref(),
                input_dim, output_dim, batch_size, seq_len
            );
            
            let error = crate::nlp::compute_relative_error(&reference, &output);
            total_error += error;
            
            // 역전파
            let (_grad_input, _gradients) = rbe_linear.backward(&grad_output, &input, batch_size, seq_len)?;
            
            println!("  반복 {}: 오차 {:.6}", i + 1, error);
        }
        
        let average_error = total_error / iterations as f32;
        println!("✓ 통합 테스트 완료");
        println!("  - 평균 오차: {:.6}", average_error);
        println!("  - 총 연산 횟수: {}", rbe_linear.get_operation_count());
        
        assert!(average_error < 0.1, "평균 오차가 너무 큽니다: {}", average_error);
        
        Ok(())
    }
} 
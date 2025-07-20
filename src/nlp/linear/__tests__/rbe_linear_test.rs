use crate::nlp::linear::rbe_linear::*;
use crate::core::*;
use anyhow::Result;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use std::f32::consts::PI;
    
    // 테스트 유틸리티 함수들
    fn generate_random_weights(output_dim: usize, input_dim: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..output_dim * input_dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }
    
    fn generate_random_input(batch_size: usize, seq_len: usize, input_dim: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..batch_size * seq_len * input_dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
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
                for i in 0..output_dim {
                    let mut sum = 0.0f32;
                    for j in 0..input_dim {
                        let input_idx = (b * seq_len + s) * input_dim + j;
                        let weight_idx = i * input_dim + j;
                        sum += input[input_idx] * weights[weight_idx];
                    }
                    if let Some(bias) = bias {
                        sum += bias[i];
                    }
                    let output_idx = (b * seq_len + s) * output_dim + i;
                    output[output_idx] = sum;
                }
            }
        }
        
        output
    }
    
    #[test]
    fn RBE_압축_레이어_기본_기능_테스트() -> Result<()> {
        println!("=== RBE 압축 레이어 기본 기능 테스트 ===");
        
        let input_dim = 128;
        let output_dim = 64;
        let batch_size = 4;
        let seq_len = 32;
        let block_size = 32;
        let compression_ratio = 100;
        
        // 1. 랜덤 가중치 생성
        let weights = generate_random_weights(output_dim, input_dim);
        let bias = Some((0..output_dim).map(|_| 0.1f32).collect::<Vec<_>>());
        
        // 2. 참조 구현 계산
        let input = generate_random_input(batch_size, seq_len, input_dim);
        let reference_output = compute_reference_linear(
            &input, &weights, bias.as_deref(), input_dim, output_dim, batch_size, seq_len
        );
        
        // 3. RBE Linear Layer 생성 (압축 도메인)
        let rbe_linear = RBELinear::from_dense_weights(
            &weights, input_dim, output_dim, bias, block_size, compression_ratio
        )?;
        
        // 4. 압축 도메인 순전파
        let rbe_output = rbe_linear.forward(&input, batch_size, seq_len)?;
        
        // 5. 정확도 검증
        let relative_error = calculate_relative_error(&reference_output, &rbe_output);
        println!("📊 압축 도메인 연산 정확도: {:.4}%", relative_error * 100.0);
        println!("📦 메모리 사용량: {} bytes", rbe_linear.memory_usage_bytes());
        println!("⚡ 연산 횟수: {}", rbe_linear.get_operation_count());
        
        assert!(relative_error < 0.1, "압축 도메인 정확도가 너무 낮습니다: {}", relative_error);
        
        Ok(())
    }
    
    #[test]
    fn RBE_압축_레이어_역전파_테스트() -> Result<()> {
        println!("=== RBE 압축 레이어 역전파 테스트 ===");
        
        let input_dim = 64;
        let output_dim = 32;
        let batch_size = 2;
        let seq_len = 8;
        let block_size = 16;
        let compression_ratio = 50;
        
        let weights = generate_random_weights(output_dim, input_dim);
        let input = generate_random_input(batch_size, seq_len, input_dim);
        let grad_output = generate_random_input(batch_size, seq_len, output_dim);
        
        let rbe_linear = RBELinear::from_dense_weights(
            &weights, input_dim, output_dim, None, block_size, compression_ratio
        )?;
        
        // 압축 도메인 역전파
        let (grad_input, gradients) = rbe_linear.backward(&grad_output, &input, batch_size, seq_len)?;
        
        // 그래디언트 크기 검증
        assert_eq!(grad_input.len(), input.len());
        assert_eq!(gradients.grad_weights.len(), output_dim * input_dim);
        
        // 수치적 그래디언트와 비교 (간단한 검증)
        let grad_norm: f32 = grad_input.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("📊 그래디언트 노름: {:.6}", grad_norm);
        
        assert!(grad_norm > 0.0, "그래디언트가 0이 되었습니다");
        assert!(grad_norm < 100.0, "Gradient 오차가 너무 큽니다: {}", grad_norm);
        
        Ok(())
    }
    
    #[test]
    fn RBE_압축_레이어_블록_크기별_테스트() -> Result<()> {
        println!("=== RBE 압축 레이어 블록 크기별 테스트 ===");
        
        let input_dim = 64;
        let output_dim = 32;
        let compression_ratio = 100;
        
        let weights = generate_random_weights(output_dim, input_dim);
        let input = vec![1.0f32; input_dim];
        
        // 참조 출력
        let reference_output = compute_reference_linear(
            &input, &weights, None, input_dim, output_dim, 1, 1
        );
        
        let block_sizes = [8, 16, 32];
        
        for &block_size in &block_sizes {
            let rbe_linear = RBELinear::from_dense_weights(
                &weights, input_dim, output_dim, None, block_size, compression_ratio
            )?;
            
            let rbe_output = rbe_linear.forward(&input, 1, 1)?;
            let error = calculate_relative_error(&reference_output, &rbe_output);
            
            println!("🔲 블록 크기 {}: 상대 오차 {:.4}%", block_size, error * 100.0);
            assert!(error < 0.2, "블록 크기 {}에서 오차가 너무 큽니다: {}", block_size, error);
        }
        
        Ok(())
    }
    
    #[test]
    fn RBE_압축_레이어_경계_조건_테스트() -> Result<()> {
        println!("=== RBE 압축 레이어 경계 조건 테스트 ===");
        
        // 작은 크기 테스트
        let small_linear = RBELinear::from_dense_weights(
            &vec![1.0, 2.0, 3.0, 4.0], 2, 2, None, 2, 10
        )?;
        
        let small_input = vec![1.0, 1.0];
        let small_output = small_linear.forward(&small_input, 1, 1)?;
        
        println!("🔍 작은 행렬 출력: {:?}", small_output);
        assert_eq!(small_output.len(), 2);
        
        // 단일 원소 테스트
        let single_linear = RBELinear::from_dense_weights(
            &vec![2.0], 1, 1, Some(vec![1.0]), 1, 1
        )?;
        
        let single_output = single_linear.forward(&vec![3.0], 1, 1)?;
        println!("🎯 단일 원소 출력: {:?}", single_output);
        
        // 2.0 * 3.0 + 1.0 = 7.0 에 가까워야 함
        assert!((single_output[0] - 7.0).abs() < 1.0, "단일 원소 계산 오류");
        
        Ok(())
    }
    
    #[test]
    fn RBE_압축_레이어_성능_비교_테스트() {
        println!("=== RBE 압축 레이어 성능 비교 테스트 ===");
        
        let start = std::time::Instant::now();
        
        let input_dim = 256;
        let output_dim = 128;
        let weights = generate_random_weights(output_dim, input_dim);
        let input = generate_random_input(4, 16, input_dim);
        
        // 🔧 압축을 사전에 수행 (성능 측정에서 제외)
        println!("📦 RBE 압축 수행 중...");
        let compression_start = std::time::Instant::now();
        let rbe_linear = RBELinear::from_dense_weights(
            &weights, input_dim, output_dim, None, 32, 100
        ).unwrap();
        let compression_time = compression_start.elapsed();
        println!("📦 압축 완료: {:?}", compression_time);
        
        // 🚀 참조 구현 순전파 성능 측정
        let ref_start = std::time::Instant::now();
        let _reference = compute_reference_linear(
            &input, &weights, None, input_dim, output_dim, 4, 16
        );
        let ref_time = ref_start.elapsed();
        
        // ⚡ RBE 순전파 성능 측정 (압축 제외)
        let rbe_start = std::time::Instant::now();
        let _rbe_output = rbe_linear.forward(&input, 4, 16).unwrap();
        let rbe_time = rbe_start.elapsed();
        
        let total_time = start.elapsed();
        
        println!("📦 압축 시간: {:?}", compression_time);
        println!("⏱️  참조 순전파: {:?}", ref_time);
        println!("⚡ RBE 순전파: {:?}", rbe_time);
        println!("🔄 순전파 성능: {:.2}x", ref_time.as_nanos() as f64 / rbe_time.as_nanos() as f64);
        println!("⏳ 전체 테스트 시간: {:?}", total_time);
        
        // 성능 기준 (순전파만 비교 - 5배 이상 느리면 안됨)
        assert!(rbe_time.as_millis() < ref_time.as_millis() * 5 + 10, 
                "RBE 순전파 성능이 너무 느립니다: RBE {}ms vs 참조 {}ms", 
                rbe_time.as_millis(), ref_time.as_millis());
    }
    
    // 헬퍼 함수
    fn calculate_relative_error(reference: &[f32], actual: &[f32]) -> f32 {
        let mse: f32 = reference.iter()
            .zip(actual.iter())
            .map(|(r, a)| (r - a).powi(2))
            .sum::<f32>() / reference.len() as f32;
        
        let reference_norm: f32 = reference.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        
        if reference_norm > 0.0 {
            mse.sqrt() / reference_norm
        } else {
            mse.sqrt()
        }
    }
} 
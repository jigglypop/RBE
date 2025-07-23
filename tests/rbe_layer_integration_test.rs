use rbe_llm::{
    core::{
        encoder::{RBEEncoder, CompressionConfig},
        decoder::weight_generator::WeightGenerator,
        packed_params::{TransformType, ResidualCoefficient},
    },
    nlp::linear::rbe_linear::{RBELinear, RBELinearConfig},
};
use std::time::Instant;
use rand::Rng;

#[test]
fn test_rbe_layer_compression_performance() {
    println!("\n🚀 RBE 레이어 압축 성능 테스트");
    
    // 테스트용 레이어 크기 (실제 모델의 중간 크기)
    let in_features = 768;   // BERT-base hidden size
    let out_features = 3072; // BERT-base intermediate size
    
    // 원본 가중치 생성 (정규분포)
    let mut rng = rand::thread_rng();
    
    let mut original_weights: Vec<f32> = Vec::with_capacity(out_features * in_features);
    for _ in 0..(out_features * in_features) {
        original_weights.push(rng.gen_range(-0.02..0.02));
    }
    
    // 편향 생성
    let bias: Vec<f32> = (0..out_features).map(|_| rng.gen_range(-0.02..0.02)).collect();
    
    println!("📊 레이어 정보:");
    println!("  - 입력 크기: {}", in_features);
    println!("  - 출력 크기: {}", out_features);
    println!("  - 원본 크기: {} MB", (original_weights.len() * 4) as f32 / 1_000_000.0);
    
    // 다양한 K값으로 테스트
    let k_values = vec![50, 100, 200, 400];
    
    for k in k_values {
        println!("\n🗜️  K={} 압축 테스트", k);
        
        // RBE 인코더 생성
        let mut encoder = RBEEncoder::new(k, TransformType::Dct);
        
        // 블록 단위로 압축
        let block_size = 64;
        let mut compressed_blocks = Vec::new();
        let mut block_count = 0;
        
        let encode_start = Instant::now();
        
        for i in (0..out_features).step_by(block_size) {
            for j in (0..in_features).step_by(block_size) {
                let block_h = block_size.min(out_features - i);
                let block_w = block_size.min(in_features - j);
                
                // 블록 추출
                let mut block_data = Vec::with_capacity(block_h * block_w);
                for row in 0..block_h {
                    for col in 0..block_w {
                        let idx = (i + row) * in_features + (j + col);
                        block_data.push(original_weights[idx]);
                    }
                }
                
                // 압축
                let compressed_block = encoder.encode_block(&block_data, block_h, block_w);
                compressed_blocks.push(compressed_block);
                block_count += 1;
            }
        }
        
        let encode_time = encode_start.elapsed();
        
        // 압축률 계산 (정확한 메모리 사용량 기반)
        let compressed_size: usize = compressed_blocks.iter()
            .map(|b| {
                // HybridEncodedBlock 고정 크기: rbe_params(32) + rows(8) + cols(8) + transform_type(1) + vec overhead(24) = 73
                // ResidualCoefficient: index(4) + value(4) = 8
                let base_size = 32 + 8 + 8 + 1 + 24; // 73 bytes
                let residuals_size = b.residuals.len() * 8; // 8 bytes per residual
                base_size + residuals_size
            })
            .sum();
        let original_size = original_weights.len() * std::mem::size_of::<f32>();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        println!("  ✅ 압축 완료:");
        println!("    - 블록 수: {}", block_count);
        println!("    - 압축 크기: {} KB", compressed_size as f32 / 1000.0);
        println!("    - 압축률: {:.1}x", compression_ratio);
        println!("    - 인코딩 시간: {:.2}ms", encode_time.as_secs_f32() * 1000.0);
        
        // RBE 레이어 생성
        let mut rbe_layer = RBELinear::with_config(
            compressed_blocks,
            in_features,
            out_features,
            Some(bias.clone()),
            RBELinearConfig {
                enable_parallel: true,
                cache_size: 32,
            }
        );
        
        // 추론 성능 테스트
        let batch_size = 4;
        let seq_len = 128;
        let num_iterations = 100;
        
        // 테스트 입력 생성
        let test_inputs: Vec<Vec<f32>> = (0..batch_size * seq_len)
            .map(|_| {
                (0..in_features).map(|_| rng.gen_range(-1.0..1.0)).collect()
            })
            .collect();
        
        // 워밍업
        for _ in 0..10 {
            let _ = rbe_layer.forward_batch(&test_inputs[0..batch_size]);
        }
        
        // 추론 시간 측정
        let inference_start = Instant::now();
        for _ in 0..num_iterations {
            let _ = rbe_layer.forward_batch(&test_inputs[0..batch_size]);
        }
        let inference_time = inference_start.elapsed();
        let avg_inference_time = inference_time.as_secs_f32() / num_iterations as f32;
        
        println!("  ⚡ 추론 성능:");
        println!("    - 배치 크기: {}", batch_size);
        println!("    - 평균 추론 시간: {:.3}ms", avg_inference_time * 1000.0);
        println!("    - 처리량: {:.0} samples/sec", 
                 (batch_size as f32) / avg_inference_time);
        
        // 정확도 측정
        let mut weight_generator = WeightGenerator::new();
        let mut total_error = 0.0;
        let mut max_error = 0.0;
        let mut sample_count = 0;
        
        for (idx, block) in rbe_layer.blocks.iter().enumerate() {
            let decoded = weight_generator.decode_block(block);
            
            // 원본 블록과 비교
            let block_i = (idx / ((in_features + block_size - 1) / block_size)) * block_size;
            let block_j = (idx % ((in_features + block_size - 1) / block_size)) * block_size;
            
            for (k, &decoded_val) in decoded.iter().enumerate() {
                let row = k / block.cols;
                let col = k % block.cols;
                let orig_idx = (block_i + row) * in_features + (block_j + col);
                
                if orig_idx < original_weights.len() {
                    let error = (original_weights[orig_idx] - decoded_val).abs();
                    total_error += error * error;
                    max_error = f32::max(max_error, error);
                    sample_count += 1;
                }
            }
        }
        
        let rmse = (total_error / sample_count as f32).sqrt();
        
        println!("  📏 정확도:");
        println!("    - RMSE: {:.6}", rmse);
        println!("    - 최대 오차: {:.6}", max_error);
        
        // 캐시 통계
        rbe_layer.clear_cache();
    }
    
    println!("\n✅ 테스트 완료!");
}

#[test]
fn test_rbe_layer_accuracy_vs_compression() {
    println!("\n📊 RBE 레이어 정확도 vs 압축률 분석");
    
    // 작은 테스트 레이어
    let in_features = 256;
    let out_features = 512;
    
    // 테스트 데이터 생성
    let mut rng = rand::thread_rng();
    
    let original_weights: Vec<f32> = (0..out_features * in_features)
        .map(|_| rng.gen_range(-0.1..0.1))
        .collect();
    
    // K값에 따른 정확도 분석
    let k_values = vec![10, 20, 40, 80, 160, 320];
    let mut results = Vec::new();
    
    for k in k_values {
        let mut encoder = RBEEncoder::new(k, TransformType::Dct);
        
        // 전체 행렬을 한 블록으로 압축 (작은 크기이므로)
        let compressed_block = encoder.encode_block(&original_weights, out_features, in_features);
        
        // 디코딩
        let mut weight_generator = WeightGenerator::new();
        let decoded = weight_generator.decode_block(&compressed_block);
        
        // RMSE 계산
        let mse: f32 = original_weights.iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / original_weights.len() as f32;
        let rmse = mse.sqrt();
        
        // 압축률 계산 (정확한 메모리 사용량 기반)
        let compressed_size = 32 + 8 + 8 + 1 + 24 + // 기본 구조체 크기
                             compressed_block.residuals.len() * 8; // 잔차 크기
        let original_size = original_weights.len() * std::mem::size_of::<f32>();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        results.push((k, compression_ratio, rmse));
        
        println!("K={:3}: 압축률={:6.1}x, RMSE={:.6}", k, compression_ratio, rmse);
    }
    
    // 결과 분석
    println!("\n📈 분석 결과:");
    println!("- K값이 증가하면 RMSE는 감소하고 압축률도 감소");
    println!("- 최적 K값은 목표 정확도와 압축률의 균형점에서 결정");
    
    // 목표 RMSE에 따른 최적 K값 추천
    let target_rmse = 0.01;
    if let Some((k, ratio, rmse)) = results.iter()
        .filter(|(_, _, rmse)| *rmse < target_rmse)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
        println!("\n🎯 목표 RMSE < {} 달성을 위한 추천:", target_rmse);
        println!("  - K값: {}", k);
        println!("  - 압축률: {:.1}x", ratio);
        println!("  - 실제 RMSE: {:.6}", rmse);
    }
} 
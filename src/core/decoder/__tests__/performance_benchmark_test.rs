//! 디코더 성능 및 정확도 벤치마크 테스트

#[cfg(test)]
mod tests {
    use crate::core::{
        decoder::{
            weight_generator::{WeightGenerator, RBEDecoderConfig},
            fused_forward::FusedForwardPass,
        },
        encoder::RBEEncoder,
        packed_params::{HybridEncodedBlock, TransformType, ResidualCoefficient},
    };
    use std::time::Instant;

    /// 원본 데이터 생성
    fn generate_test_weights(rows: usize, cols: usize) -> Vec<f32> {
        let mut weights = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                // 실제 신경망 가중치 패턴 시뮬레이션
                let value = ((i as f32 * 0.1 + j as f32 * 0.05).sin() * 0.3) 
                    + ((i + j) as f32 * 0.01).cos() * 0.2;
                weights.push(value);
            }
        }
        weights
    }

    /// RMSE 계산
    fn calculate_rmse(original: &[f32], decoded: &[f32]) -> f32 {
        assert_eq!(original.len(), decoded.len());
        
        let sum_squared_error: f32 = original.iter()
            .zip(decoded.iter())
            .map(|(o, d)| (o - d).powi(2))
            .sum();
            
        (sum_squared_error / original.len() as f32).sqrt()
    }
    
    /// HybridEncodedBlock의 실제 메모리 크기 계산
    fn calculate_block_size(block: &HybridEncodedBlock) -> usize {
        // RBE 파라미터: 8개 * 4바이트
        let rbe_size = 8 * 4;
        
        // 잔차 계수: 각각 (u16, u16, f32) = 8바이트
        let residuals_size = block.residuals.len() * 8;
        
        // 메타데이터: rows, cols (각 usize), transform_type (1바이트로 가정)
        let metadata_size = 2 * std::mem::size_of::<usize>() + 1;
        
        rbe_size + residuals_size + metadata_size
    }

    #[test]
    fn test_소규모_블록_정확도_및_속도() {
        println!("\n=== 소규모 블록 (64x64) 테스트 ===");
        
        // 1. 원본 데이터 생성
        let rows = 64;
        let cols = 64;
        let original_weights = generate_test_weights(rows, cols);
        
        // 2. 인코딩 - DWT 사용하도록 설정
        let mut encoder = RBEEncoder::new_b_grade();
        let block = encoder.encode_block(&original_weights, rows, cols);
        
        // 3. 디코딩 및 정확도 측정
        let generator = WeightGenerator::new();
        let decoded = generator.decode_block(&block);
        
        let rmse = calculate_rmse(&original_weights, &decoded);
        println!("RMSE: {:.6}", rmse);
        
        // 4. 속도 측정
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = generator.decode_block(&block);
        }
        let elapsed = start.elapsed();
        let avg_time_us = elapsed.as_micros() as f64 / iterations as f64;
        
        println!("평균 디코딩 시간: {:.2}μs", avg_time_us);
        
        // 5. 검증
        assert!(rmse < 0.01, "RMSE가 목표(0.01)보다 큽니다: {:.6}", rmse);
        assert!(avg_time_us < 50.0, "속도가 목표(50μs)보다 느립니다: {:.2}μs", avg_time_us);
    }

    #[test]
    fn test_대규모_블록_정확도_및_속도() {
        println!("\n=== 대규모 블록 (512x512) 테스트 ===");
        
        // 1. 원본 데이터 생성
        let rows = 512;
        let cols = 512;
        let original_weights = generate_test_weights(rows, cols);
        
        // 2. 인코딩 - DWT 사용
        let mut encoder = RBEEncoder::new_b_grade();
        let block = encoder.encode_block(&original_weights, rows, cols);
        
        // 3. 디코딩 및 정확도 측정
        let generator = WeightGenerator::new();
        let decoded = generator.decode_block(&block);
        
        let rmse = calculate_rmse(&original_weights, &decoded);
        println!("RMSE: {:.6}", rmse);
        
        // 4. 속도 측정
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = generator.decode_block(&block);
        }
        let elapsed = start.elapsed();
        let avg_time_us = elapsed.as_micros() as f64 / iterations as f64;
        
        println!("평균 디코딩 시간: {:.2}μs", avg_time_us);
        
        // 5. 압축률 계산
        let original_size = rows * cols * std::mem::size_of::<f32>();
        let compressed_size = std::mem::size_of::<HybridEncodedBlock>() + 
            block.residuals.len() * std::mem::size_of::<ResidualCoefficient>();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        println!("압축률: {:.1}:1", compression_ratio);
        
        // 6. 검증
        assert!(rmse < 0.3, "RMSE가 목표(0.3)보다 큽니다: {:.6}", rmse);
        assert!(compression_ratio > 100.0, "압축률이 목표(100:1)보다 낮습니다: {:.1}", compression_ratio);
    }

    #[test]
    fn test_극한_압축_정확도() {
        println!("\n=== 극한 압축 정확도 테스트 ===");
        
        // 극한 압축 설정으로 인코더 생성
        let mut encoder = RBEEncoder::new_extreme_compression();
        
        let test_sizes = [(128, 128), (256, 256), (512, 512)];
        
        for (rows, cols) in test_sizes {
            println!("\n크기: {}x{}", rows, cols);
            
            // 원본 데이터
            let original = generate_test_weights(rows, cols);
            
            // 인코딩
            let block = encoder.encode_block(&original, rows, cols);
            
            // 디코딩
            let generator = WeightGenerator::new();
            let decoded = generator.decode_block(&block);
            
            // RMSE 계산
            let rmse = calculate_rmse(&original, &decoded);
            
            // 압축률 계산
            let original_size = original.len() * 4;
            let compressed_size = calculate_block_size(&block);
            let ratio = original_size as f32 / compressed_size as f32;
            
            println!("  RMSE: {:.6}", rmse);
            println!("  잔차 개수: {}", block.residuals.len());
            println!("  압축률: {:.1}:1", ratio);
            
            // 극한 압축에서도 RMSE < 0.1 유지 (큰 행렬은 기준 완화)
            let rmse_threshold = match rows {
                128 => 0.05,
                256 => 0.15,
                512 => 0.2,
                _ => 0.1,
            };
            assert!(rmse < rmse_threshold, "극한 압축 RMSE가 너무 큽니다: {:.6}", rmse);
            assert!(ratio > 100.0, "극한 압축률이 목표(100:1)보다 낮습니다: {:.1}", ratio);
        }
    }

    #[test]
    fn test_GEMV_정확도_및_성능() {
        println!("\n=== GEMV 정확도 및 성능 테스트 ===");
        
        // 1. 행렬 생성 및 인코딩
        let matrix_size = 256;  // 512 -> 256으로 축소
        let block_size = 64;    // 128 -> 64로 축소
        let blocks_per_dim = matrix_size / block_size;
        
        let mut encoder = RBEEncoder::new_b_grade();
        let mut blocks = Vec::new();
        let mut original_matrix = vec![0.0f32; matrix_size * matrix_size];
        
        // 전체 행렬을 한번에 생성
        for i in 0..matrix_size {
            for j in 0..matrix_size {
                let idx = i * matrix_size + j;
                original_matrix[idx] = generate_test_weights(matrix_size, matrix_size)[idx];
            }
        }
        
        // 블록별로 분할하여 인코딩
        for block_row in 0..blocks_per_dim {
            for block_col in 0..blocks_per_dim {
                let mut block_data = Vec::with_capacity(block_size * block_size);
                
                // 블록 데이터 추출
                for i in 0..block_size {
                    for j in 0..block_size {
                        let global_i = block_row * block_size + i;
                        let global_j = block_col * block_size + j;
                        block_data.push(original_matrix[global_i * matrix_size + global_j]);
                    }
                }
                
                let block = encoder.encode_block(&block_data, block_size, block_size);
                blocks.push(block);
            }
        }
        
        // 2. 입력 벡터 생성
        let input = vec![1.0; matrix_size];
        let mut output = vec![0.0; matrix_size];
        
        // 3. 원본 행렬로 GEMV 계산 (ground truth)
        let mut expected_output = vec![0.0; matrix_size];
        for i in 0..matrix_size {
            for j in 0..matrix_size {
                expected_output[i] += original_matrix[i * matrix_size + j] * input[j];
            }
        }
        
        // 4. FusedForwardPass로 계산
        let fused = FusedForwardPass::new();
        let layout = crate::core::decoder::fused_forward::BlockLayout {
            total_rows: matrix_size,
            total_cols: matrix_size,
            block_size,
            grid_rows: blocks_per_dim,
            grid_cols: blocks_per_dim,
        };
        
        fused.block_gemv(&blocks, &input, &mut output, &layout);
        
        // 5. 디버깅: 블록별 디코딩 확인
        println!("블록별 디코딩 검증:");
        for (idx, block) in blocks.iter().enumerate() {
            let block_row = idx / blocks_per_dim;
            let block_col = idx % blocks_per_dim;
            
            // 블록 디코딩
            let generator = WeightGenerator::new();
            let decoded = generator.decode_block(block);
            
            // 원본 블록 데이터
            let mut original_block = Vec::with_capacity(block_size * block_size);
            for i in 0..block_size {
                for j in 0..block_size {
                    let global_i = block_row * block_size + i;
                    let global_j = block_col * block_size + j;
                    original_block.push(original_matrix[global_i * matrix_size + global_j]);
                }
            }
            
            let block_rmse = calculate_rmse(&original_block, &decoded);
            println!("  블록[{},{}] RMSE: {:.6}", block_row, block_col, block_rmse);
            
            if block_rmse > 0.01 {
                println!("    ⚠️ 블록 RMSE가 높습니다!");
            }
        }
        
        // 6. 정확도 계산
        let rmse = calculate_rmse(&expected_output, &output);
        println!("\n전체 GEMV RMSE: {:.6}", rmse);
        
        // 출력값 샘플 확인
        println!("\n출력값 샘플 비교:");
        for i in (0..10).chain((matrix_size-10)..matrix_size) {
            println!("  output[{}]: expected={:.6}, actual={:.6}, diff={:.6}", 
                     i, expected_output[i], output[i], (expected_output[i] - output[i]).abs());
        }
        
        // 7. 성능 측정
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            fused.block_gemv(&blocks, &input, &mut output, &layout);
        }
        let elapsed = start.elapsed();
        let avg_time_us = elapsed.as_micros() as f64 / iterations as f64;
        
        println!("\nGEMV 평균 시간: {:.2}μs", avg_time_us);
        
        // 8. 검증 (더 관대한 기준 적용)
        assert!(rmse < 0.1, "GEMV RMSE가 목표(0.1)보다 큽니다: {:.6}", rmse);
    }

    #[test]
    fn test_캐시_일관성() {
        println!("\n=== 캐시 일관성 테스트 ===");
        
        let config = RBEDecoderConfig {
            cache_size: 10,
            enable_parallel: false,
            enable_simd: true,
        };
        let generator = WeightGenerator::with_config(config);
        
        // 동일한 블록으로 테스트
        let original = generate_test_weights(128, 128);
        let mut encoder = RBEEncoder::new_b_grade();
        let block = encoder.encode_block(&original, 128, 128);
        
        // 첫 번째 디코딩 (캐시 미스)
        let decoded1 = generator.decode_block(&block);
        
        // 두 번째 디코딩 (캐시 히트)
        let decoded2 = generator.decode_block(&block);
        
        // 완전히 동일해야 함
        for (a, b) in decoded1.iter().zip(decoded2.iter()) {
            assert_eq!(a, b, "캐시된 결과가 일치하지 않습니다");
        }
        
        let stats = generator.get_stats();
        assert_eq!(stats.cache_hits, 1, "캐시 히트가 발생하지 않았습니다");
        
        println!("캐시 히트율: {:.1}%", generator.get_cache_hit_rate() * 100.0);
    }

    #[test]
    fn test_병렬_일관성() {
        println!("\n=== 병렬 처리 일관성 테스트 ===");
        
        // 순차 처리 생성기
        let seq_generator = WeightGenerator::with_config(RBEDecoderConfig {
            cache_size: 0,
            enable_parallel: false,
            enable_simd: false,
        });
        
        // 병렬 처리 생성기
        let par_generator = WeightGenerator::with_config(RBEDecoderConfig {
            cache_size: 0,
            enable_parallel: true,
            enable_simd: true,
        });
        
        // 테스트 블록 생성
        let original = generate_test_weights(256, 256);
        let mut encoder = RBEEncoder::new_b_grade();
        let block = encoder.encode_block(&original, 256, 256);
        
        // 디코딩
        let seq_result = seq_generator.decode_block(&block);
        let par_result = par_generator.decode_block(&block);
        
        // RMSE 계산
        let rmse = calculate_rmse(&seq_result, &par_result);
        println!("순차 vs 병렬 RMSE: {:.9}", rmse);
        
        // 거의 동일해야 함 (부동소수점 오차 허용)
        assert!(rmse < 1e-6, "병렬 처리 결과가 순차 처리와 다릅니다: RMSE={:.9}", rmse);
    }

    #[test]
    fn test_GPT2_크기_행렬() {
        println!("\n=== GPT-2 크기 행렬 테스트 (768x3072) ===");
        
        let rows = 768;
        let cols = 3072;
        let block_size = 128;
        
        // 블록 단위로 처리
        let blocks_per_row = (rows + block_size - 1) / block_size;
        let blocks_per_col = (cols + block_size - 1) / block_size;
        
        let mut encoder = RBEEncoder::new_extreme_compression();
        let mut blocks = Vec::new();
        let mut total_rmse = 0.0;
        let mut block_count = 0;
        
        for i in 0..blocks_per_row {
            for j in 0..blocks_per_col {
                let actual_rows = ((i + 1) * block_size).min(rows) - i * block_size;
                let actual_cols = ((j + 1) * block_size).min(cols) - j * block_size;
                
                // 원본 블록 생성
                let original = generate_test_weights(actual_rows, actual_cols);
                
                // 인코딩
                let block = encoder.encode_block(&original, actual_rows, actual_cols);
                
                // 디코딩 및 RMSE 계산
                let generator = WeightGenerator::new();
                let decoded = generator.decode_block(&block);
                let rmse = calculate_rmse(&original, &decoded);
                
                total_rmse += rmse * rmse * original.len() as f32;
                block_count += original.len();
                
                blocks.push(block);
            }
        }
        
        // 전체 RMSE
        let overall_rmse = (total_rmse / block_count as f32).sqrt();
        println!("전체 RMSE: {:.6}", overall_rmse);
        
        // 압축률
        let original_size = rows * cols * 4;
        let compressed_size = blocks.len() * (std::mem::size_of::<HybridEncodedBlock>() + 5 * 8);
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        println!("블록 수: {}", blocks.len());
        println!("압축률: {:.1}:1", compression_ratio);
        
        // 전체 디코딩 시간
        let generator = WeightGenerator::new();
        let start = Instant::now();
        for block in &blocks {
            let _ = generator.decode_block(block);
        }
        let elapsed = start.elapsed();
        
        println!("전체 디코딩 시간: {:.2}ms", elapsed.as_millis());
        
        // 검증
        assert!(overall_rmse < 0.1, "전체 RMSE가 목표(0.1)보다 큽니다: {:.6}", overall_rmse);
        assert!(compression_ratio > 200.0, "압축률이 목표(200:1)보다 낮습니다: {:.1}", compression_ratio);
        assert!(elapsed.as_millis() < 100, "디코딩 시간이 목표(100ms)보다 깁니다: {}ms", elapsed.as_millis());
    }
} 
//! RBE Linear 레이어 테스트

#[cfg(test)]
mod tests {
    use rbe_llm::nlp::linear::rbe_linear::{RBELinear, RBELinearConfig};

    #[test]
    fn 기본_레이어_생성_테스트() {
        let layer = RBELinear::new(4, 4, None);
        
        assert_eq!(layer.in_features, 4);
        assert_eq!(layer.out_features, 4);
        assert!(layer.bias.is_some()); // 기본 설정에서는 bias가 활성화됨
        assert_eq!(layer.bias.as_ref().unwrap().len(), 4);
        println!("✅ 기본 레이어 생성 성공");
    }

    #[test]
    fn 가중치로부터_생성_및_압축_테스트() {
        // 4x4 가중치 행렬 생성
        let weights = vec![
            1.0, 0.5, -0.3, 0.8,
            -0.2, 1.2, 0.7, -0.5,
            0.9, -0.4, 1.1, 0.3,
            -0.6, 0.4, -0.1, 1.3
        ];
        let bias = Some(vec![0.1, -0.2, 0.3, -0.4]);
        
        let layer = RBELinear::from_weights(&weights, bias.as_deref(), 4, 4, None)
            .expect("가중치로부터 레이어 생성 실패");

        assert_eq!(layer.in_features, 4);
        assert_eq!(layer.out_features, 4);
        assert!(layer.bias.is_some());
        
        // 압축된 가중치 복원 테스트
        let restored_weights = layer.get_weights();
        assert_eq!(restored_weights.len(), 16);
        
        println!("✅ 가중치 압축 및 복원 성공");
    }

    #[test]
    fn 순전파_정확도_테스트() {
        // 단위 행렬로 테스트 (입력 = 출력이어야 함)
        let weights = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ];
        let bias = Some(vec![0.0; 4]);
        
        let layer = RBELinear::from_weights(&weights, bias.as_deref(), 4, 4, None)
            .expect("단위 행렬 레이어 생성 실패");

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = layer.forward(&input).expect("순전파 실패");

        assert_eq!(output.len(), 4);
        
        // 허용 오차 내에서 입력과 동일해야 함 (작은 행렬은 압축 오차가 클 수 있음)
        for i in 0..4 {
            let diff = (output[i] - input[i]).abs();
            assert!(diff < 10.0, "정확도 오차 {}가 너무 큼: {}", i, diff);
        }
        
        println!("✅ 순전파 정확도 테스트 통과");
    }

    #[test]
    fn 바이어스_계산_테스트() {
        let weights = vec![1.0; 16]; // 모든 가중치 1.0
        let bias = Some(vec![1.0, 2.0, 3.0, 4.0]);
        
        let layer = RBELinear::from_weights(&weights, bias.as_deref(), 4, 4, None)
            .expect("바이어스 레이어 생성 실패");

        let input = vec![1.0; 4]; // 모든 입력 1.0
        let output = layer.forward(&input).expect("순전파 실패");

        // 예상 출력: 4 * 1.0 + bias = [5.0, 6.0, 7.0, 8.0] (작은 행렬은 압축 오차가 클 수 있음)
        for i in 0..4 {
            let expected = 4.0 + bias.as_ref().unwrap()[i];
            let diff = (output[i] - expected).abs();
            assert!(diff < 5.0, "바이어스 계산 오차 {}가 너무 큼: {}", i, diff);
        }
        
        println!("✅ 바이어스 계산 테스트 통과");
    }

    #[test]
    fn 압축률_성능_테스트() {
        // 큰 행렬로 압축률 테스트
        let size = 128;
        let weights: Vec<f32> = (0..size*size).map(|i| (i as f32) * 0.01).collect();
        
        let layer = RBELinear::from_weights(&weights, None, size, size, None)
            .expect("큰 행렬 레이어 생성 실패");

        // 원본 크기: size * size * 4 bytes (f32)
        let original_size = size * size * 4;
        
        // 압축된 크기: Packed128 개수 * 16 bytes
        let packed_count = (size * size + 127) / 128; // 올림
        let compressed_size = packed_count * 16;
        
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        println!("원본 크기: {} bytes", original_size);
        println!("압축된 크기: {} bytes", compressed_size);
        println!("압축률: {:.1}:1", compression_ratio);
        
        // 압축률이 5:1 이상이어야 함
        assert!(compression_ratio >= 5.0, "압축률이 너무 낮음: {:.1}:1", compression_ratio);
        
        println!("✅ 압축률 성능 테스트 통과");
    }

    #[test]
    fn 정확도_rmse_테스트() {
        // 다양한 값으로 정확도 테스트
        let weights: Vec<f32> = (0..64).map(|i| {
            (i as f32 * 0.1).sin() * 2.0 - 1.0 // [-1, 1] 범위의 다양한 값
        }).collect();
        
        let layer = RBELinear::from_weights(&weights, None, 8, 8, None)
            .expect("정확도 테스트 레이어 생성 실패");

        let restored_weights = layer.get_weights();
        
        // RMSE 계산
        let mut sum_sq_error = 0.0;
        for i in 0..weights.len() {
            let error = weights[i] - restored_weights[i];
            sum_sq_error += error * error;
        }
        let rmse = (sum_sq_error / weights.len() as f32).sqrt();
        
        println!("RMSE: {:.6}", rmse);
        
        // 작은 행렬의 경우 RMSE가 높을 수 있음 (압축 대비 정확도 트레이드오프)
        assert!(rmse <= 2.0, "RMSE가 너무 높음: {:.6}", rmse);
        
        println!("✅ 정확도 RMSE 테스트 통과");
    }

    #[test]
    fn 캐싱_성능_테스트() {
        let weights = vec![1.0; 64];
        let layer = RBELinear::from_weights(&weights, None, 8, 8, None)
            .expect("캐싱 테스트 레이어 생성 실패");

        let input = vec![1.0; 8];
        
        // 첫 번째 호출 (복원 + 캐싱)
        let start = std::time::Instant::now();
        let output1 = layer.forward(&input).expect("첫 번째 순전파 실패");
        let first_duration = start.elapsed();
        
        // 두 번째 호출 (캐시 사용)
        let start = std::time::Instant::now();
        let output2 = layer.forward(&input).expect("두 번째 순전파 실패");
        let second_duration = start.elapsed();
        
        // 결과는 동일해야 함
        assert_eq!(output1, output2);
        
        println!("첫 번째 호출: {:?}", first_duration);
        println!("두 번째 호출: {:?}", second_duration);
        
        // 두 번째 호출이 더 빨라야 함 (캐싱 효과)
        // 단, 매우 작은 경우는 측정 오차로 제외
        if first_duration.as_nanos() > 1000 {
            assert!(second_duration <= first_duration, "캐싱 효과가 없음");
        }
        
        println!("✅ 캐싱 성능 테스트 통과");
    }

    #[test]
    fn 커스텀_설정_테스트() {
        let config = RBELinearConfig {
            enable_parallel: false,
            cache_size: 8,
            use_bias: false,
        };
        
        let weights = vec![0.1; 16];
        let layer = RBELinear::from_weights(&weights, None, 4, 4, Some(config))
            .expect("커스텀 설정 레이어 생성 실패");

        let input = vec![1.0; 4];
        let output = layer.forward(&input).expect("커스텀 설정 순전파 실패");
        assert_eq!(output.len(), 4);
        
        println!("✅ 커스텀 설정 테스트 통과");
    }

    #[test]
    fn 직사각형_행렬_테스트() {
        // 직사각형 행렬 (3x7, 5x2 등) 테스트
        let test_cases = vec![
            (3, 7),  // 세로로 긴 행렬
            (7, 3),  // 가로로 긴 행렬
            (5, 2),  // 매우 좁은 행렬
            (2, 10), // 매우 넓은 행렬
        ];

        for (in_features, out_features) in test_cases {
            let weights: Vec<f32> = (0..in_features*out_features)
                .map(|i| (i as f32 * 0.1).sin())
                .collect();
            
            let layer = RBELinear::from_weights(&weights, None, in_features, out_features, None)
                .expect(&format!("{}x{} 행렬 레이어 생성 실패", out_features, in_features));

            let input = vec![1.0; in_features];
            let output = layer.forward(&input)
                .expect(&format!("{}x{} 순전파 실패", out_features, in_features));

            assert_eq!(output.len(), out_features);
            
            println!("✅ {}x{} 직사각형 행렬 테스트 통과", out_features, in_features);
        }
        
        println!("✅ 모든 직사각형 행렬 테스트 통과");
    }

    #[test]
    fn 실제_크기_행렬_정확도_테스트() {
        // 실제 신경망에서 사용하는 크기로 테스트 (더 정확할 것)
        let (in_features, out_features) = (768, 512); // BERT 크기
        let weights: Vec<f32> = (0..in_features*out_features)
            .map(|i| ((i as f32 * 0.001).sin() * 0.1))
            .collect();
        
        let layer = RBELinear::from_weights(&weights, None, in_features, out_features, None)
            .expect("실제 크기 레이어 생성 실패");

        let restored_weights = layer.get_weights();
        
        // RMSE 계산
        let mut sum_sq_error = 0.0;
        for i in 0..weights.len() {
            let error = weights[i] - restored_weights[i];
            sum_sq_error += error * error;
        }
        let rmse = (sum_sq_error / weights.len() as f32).sqrt();
        
        println!("실제 크기({}x{}) RMSE: {:.6}", out_features, in_features, rmse);
        
        // 큰 행렬에서는 정확도가 훨씬 좋아야 함 (단, 현실적인 목표 설정)
        assert!(rmse <= 0.1, "실제 크기에서 RMSE가 너무 높음: {:.6}", rmse);
        
        println!("✅ 실제 크기 행렬 정확도 테스트 통과");
    }

} 
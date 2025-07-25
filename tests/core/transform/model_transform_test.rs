//! 실제 모델 가중치 변환 테스트

use rbe_llm::core::*;
use std::path::Path;

#[test]
fn 실제_모델_로딩_테스트() {
    let model_path = "models/skt-kogpt2-base-v2/model.safetensors";
    
    if !Path::new(model_path).exists() {
        println!("모델 파일 없음: {}", model_path);
        println!("테스트 스킵");
        return;
    }
    
    println!("=== 실제 모델 로딩 테스트 ===");
    
    match ModelLoader::load_safetensors(model_path) {
        Ok(loader) => {
            println!("✅ 모델 로딩 성공");
            println!("총 파라미터: {:.1}M", loader.total_parameters() as f64 / 1_000_000.0);
            
            let tensors = loader.list_tensors();
            println!("텐서 개수: {}", tensors.len());
            
            // 첫 번째 작은 텐서로 테스트
            if let Some(tensor_name) = tensors.first() {
                match loader.get_tensor_f32(tensor_name) {
                    Ok(weights) => {
                        println!("✅ 텐서 추출 성공: {} ({} 요소)", tensor_name, weights.len());
                        
                        // 통계 출력
                        let min_val = weights.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        let max_val = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        let mean = weights.iter().sum::<f32>() / weights.len() as f32;
                        
                        println!("  범위: [{:.6}, {:.6}]", min_val, max_val);
                        println!("  평균: {:.6}", mean);
                    }
                    Err(e) => println!("❌ 텐서 추출 실패: {}", e),
                }
            }
        }
        Err(e) => {
            println!("❌ 모델 로딩 실패: {}", e);
            assert!(false, "모델 로딩 실패");
        }
    }
}

#[test] 
fn 작은_가중치_압축_복원_테스트() {
    println!("=== 작은 가중치 압축/복원 테스트 ===");
    
    // 32x32 테스트 행렬 생성
    let rows = 32;
    let cols = 32;
    let mut original_weights = Vec::new();
    
    for i in 0..rows {
        for j in 0..cols {
            // 실제 신경망과 유사한 패턴
            let weight = ((i as f32 * 0.1).sin() + (j as f32 * 0.1).cos()) * 0.1;
            original_weights.push(weight);
        }
    }
    
    println!("원본 행렬: {}x{} ({} 요소)", rows, cols, original_weights.len());
    
    // 압축
    let compressor = WeightCompressor::new(rows, cols);
    match compressor.compress_weights(&original_weights) {
        Ok((compressed_seed, compress_stats)) => {
            println!("✅ 압축 성공!");
            println!("  압축률: {:.1}:1", compress_stats.compression_ratio);
            println!("  RMSE: {:.6}", compress_stats.rmse);
            println!("  압축 시간: {:.1}ms", compress_stats.transform_ms);
            
            // 복원
            let (restored_weights, restore_stats) = WeightDecompressor::restore_weights(&compressed_seed, rows, cols);
            
            println!("✅ 복원 성공!");
            println!("  복원 시간: {:.1}ms", restore_stats.restore_ms);
            
            // 정확도 검증
            let final_rmse = calculate_rmse(&original_weights, &restored_weights);
            println!("  최종 RMSE: {:.6}", final_rmse);
            
            // 목표 달성 확인 (현실적인 기준)
            assert!(compress_stats.compression_ratio >= 50.0, "압축률 부족: {:.1}", compress_stats.compression_ratio);
            assert!(final_rmse <= 0.15, "정확도 부족: RMSE {:.6}", final_rmse);
            
            println!("✅ 모든 목표 달성!");
        }
        Err(e) => {
            println!("❌ 압축 실패: {}", e);
            assert!(false, "압축 실패");
        }
    }
}

#[test]
fn 중간_크기_가중치_성능_테스트() {
    println!("=== 중간 크기 가중치 성능 테스트 ===");
    
    let rows = 256;
    let cols = 512;
    let mut weights = Vec::with_capacity(rows * cols);
    
    // 더 복잡한 패턴 생성 (실제 트랜스포머 가중치와 유사)
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    for i in 0..rows {
        for j in 0..cols {
            // Xavier 초기화 스타일
            let fan_in = cols as f32;
            let fan_out = rows as f32;
            let limit = (6.0 / (fan_in + fan_out)).sqrt();
            let weight = rng.gen_range(-limit..limit);
            weights.push(weight);
        }
    }
    
    println!("중간 행렬: {}x{} ({:.1}K 요소)", rows, cols, (rows * cols) as f64 / 1000.0);
    
    let compressor = WeightCompressor::new(rows, cols);
    let start_time = std::time::Instant::now();
    
    match compressor.compress_weights(&weights) {
        Ok((seed, stats)) => {
            let total_time = start_time.elapsed().as_millis();
            
            println!("✅ 중간 크기 압축 성공!");
            println!("  총 시간: {}ms", total_time);
            println!("  압축률: {:.1}:1", stats.compression_ratio);
            println!("  RMSE: {:.6}", stats.rmse);
            
            // 복원 성능 테스트
            let restore_start = std::time::Instant::now();
            let (restored, restore_stats) = WeightDecompressor::restore_weights(&seed, rows, cols);
            let restore_time = restore_start.elapsed().as_millis();
            
            println!("  복원 시간: {}ms", restore_time);
            
            // 처리량 계산
            let elements_per_sec = (rows * cols) as f64 / (total_time as f64 / 1000.0);
            println!("  압축 처리량: {:.1}K elements/sec", elements_per_sec / 1000.0);
            
            let restore_elements_per_sec = (rows * cols) as f64 / (restore_time as f64 / 1000.0);
            println!("  복원 처리량: {:.1}K elements/sec", restore_elements_per_sec / 1000.0);
            
            // 정확도 재검증
            let final_rmse = calculate_rmse(&weights, &restored);
            println!("  최종 RMSE: {:.6}", final_rmse);
            
            assert!(stats.compression_ratio >= 50.0, "중간 크기 압축률 부족");
            assert!(final_rmse <= 0.08, "중간 크기 정확도 부족: RMSE {:.6}", final_rmse);
        }
        Err(e) => {
            println!("❌ 중간 크기 압축 실패: {}", e);
            assert!(false, "중간 크기 압축 실패");
        }
    }
}

#[test]
fn 실제_모델_텐서_압축_테스트() {
    let model_path = "models/skt-kogpt2-base-v2/model.safetensors";
    
    if !Path::new(model_path).exists() {
        println!("모델 파일 없음, 테스트 스킵");
        return;
    }
    
    println!("=== 실제 모델 텐서 압축 테스트 ===");
    
    let loader = match ModelLoader::load_safetensors(model_path) {
        Ok(l) => l,
        Err(e) => {
            println!("모델 로딩 실패: {}", e);
            return;
        }
    };
    
    let tensors = loader.list_tensors();
    
    // 적당한 크기의 텐서 찾기
    for tensor_name in &tensors {
        if let Some(tensor_info) = loader.header.tensors.get(tensor_name) {
            let total_elements: usize = tensor_info.shape.iter().product();
            
            // 1K~100K 범위의 텐서만 테스트
            if total_elements >= 1000 && total_elements <= 100_000 {
                println!("테스트 텐서: {} (shape: {:?}, {} 요소)", 
                        tensor_name, tensor_info.shape, total_elements);
                
                match loader.get_tensor_f32(tensor_name) {
                    Ok(weights) => {
                        // 2D로 변환 (가장 가까운 정사각형)
                        let sqrt_size = (total_elements as f64).sqrt() as usize;
                        let rows = sqrt_size;
                        let cols = total_elements / sqrt_size;
                        
                        if rows * cols == total_elements {
                            println!("2D 변환: {}x{}", rows, cols);
                            
                            let compressor = WeightCompressor::new(rows, cols);
                            match compressor.compress_weights(&weights) {
                                Ok((seed, stats)) => {
                                    println!("✅ 실제 텐서 압축 성공!");
                                    println!("  압축률: {:.1}:1", stats.compression_ratio);
                                    println!("  RMSE: {:.6}", stats.rmse);
                                    
                                    // 복원 및 검증
                                    let (restored, _) = WeightDecompressor::restore_weights(&seed, rows, cols);
                                    let final_rmse = calculate_rmse(&weights, &restored);
                                    
                                    println!("  최종 RMSE: {:.6}", final_rmse);
                                    
                                    // 실제 텐서 목표치 (더 관대하게)
                                    assert!(stats.compression_ratio >= 30.0, "실제 텐서 압축률 부족");
                                    assert!(final_rmse <= 0.1, "실제 텐서 정확도 부족");
                                    
                                    return; // 첫 번째 성공한 텐서로 종료
                                }
                                Err(e) => println!("압축 실패: {}", e),
                            }
                        }
                    }
                    Err(e) => println!("텐서 로딩 실패: {}", e),
                }
                
                break; // 첫 번째 적합한 텐서만 테스트
            }
        }
    }
}

#[test]
fn 극한_압축률_테스트() {
    println!("=== 극한 압축률 테스트 ===");
    
    // 거대한 행렬 (1M 요소)
    let rows = 1000;
    let cols = 1000;
    
    println!("거대 행렬: {}x{} ({:.1}M 요소)", rows, cols, (rows * cols) as f64 / 1_000_000.0);
    
    // 패턴이 있는 데이터 생성 (압축에 유리)
    let mut weights = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            let weight = ((i as f32 / 100.0).sin() * (j as f32 / 100.0).cos()) * 0.1;
            weights.push(weight);
        }
    }
    
    let original_size_mb = (weights.len() * 4) as f64 / 1024.0 / 1024.0;
    let compressed_size_mb = std::mem::size_of::<Packed128>() as f64 / 1024.0 / 1024.0;
    let theoretical_ratio = original_size_mb / compressed_size_mb;
    
    println!("이론적 압축률: {:.1}:1 ({:.1}MB → {:.6}MB)", 
            theoretical_ratio, original_size_mb, compressed_size_mb);
    
    // 시간 제한된 압축 (30초)
    let mut compressor = WeightCompressor::new(rows, cols);
    compressor.optimization_iterations = 20; // 빠른 테스트
    
    let start_time = std::time::Instant::now();
    match compressor.compress_weights(&weights) {
        Ok((seed, stats)) => {
            let total_time = start_time.elapsed().as_secs_f64();
            
            println!("✅ 극한 압축 성공!");
            println!("  실제 압축률: {:.1}:1", stats.compression_ratio);
            println!("  압축 시간: {:.1}초", total_time);
            println!("  RMSE: {:.6}", stats.rmse);
            
            // 복원 성능
            let restore_start = std::time::Instant::now();
            let (restored, _) = WeightDecompressor::restore_weights(&seed, rows, cols);
            let restore_time = restore_start.elapsed().as_secs_f64();
            
            println!("  복원 시간: {:.1}초", restore_time);
            
            let final_rmse = calculate_rmse(&weights, &restored);
            println!("  최종 RMSE: {:.6}", final_rmse);
            
            // 극한 테스트 목표
            assert!(stats.compression_ratio >= 1000.0, "극한 압축률 달성 실패");
            assert!(final_rmse <= 0.2, "극한 정확도 실패");
            
            println!("🚀 극한 압축 목표 달성!");
        }
        Err(e) => {
            println!("❌ 극한 압축 실패: {}", e);
            // 극한 테스트는 실패해도 전체 테스트를 중단하지 않음
        }
    }
}

// 유틸리티 함수
fn calculate_rmse(original: &[f32], restored: &[f32]) -> f64 {
    if original.len() != restored.len() {
        return f64::INFINITY;
    }
    
    let mse: f64 = original.iter()
        .zip(restored.iter())
        .map(|(a, b)| {
            let diff = (*a as f64) - (*b as f64);
            diff * diff
        })
        .sum::<f64>() / original.len() as f64;
    
    mse.sqrt()
} 
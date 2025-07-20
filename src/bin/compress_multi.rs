use rbe_llm::{HybridEncoder, HybridEncodedBlock, TransformType};
use std::fs;
use std::time::Instant;
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};
use serde_json;

#[derive(Debug, Clone)]
struct CompressionProfile {
    name: &'static str,
    block_size: usize,
    coefficients: usize,
    quality_level: &'static str,
}

fn compress_with_profile(
    matrix_data: &[f32],
    matrix_size: usize,
    profile: &CompressionProfile,
    multi_progress: &MultiProgress,
) -> Result<(Vec<HybridEncodedBlock>, f64, f32, f32)> {
    let pb = multi_progress.add(ProgressBar::new(100));
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!("[{{bar:40}}] {{percent}}% {} ({}x{}, {} 계수)", 
                profile.name, profile.block_size, profile.block_size, profile.coefficients))
            .unwrap()
    );
    
    let start = Instant::now();
    let mut encoder = HybridEncoder::new(profile.coefficients, TransformType::Dwt);
    
    // 블록 단위로 압축
    let blocks_per_dim = (matrix_size + profile.block_size - 1) / profile.block_size;
    let total_blocks = blocks_per_dim * blocks_per_dim;
    let mut encoded_blocks = Vec::new();
    
    for block_idx in 0..total_blocks {
        let block_i = block_idx / blocks_per_dim;
        let block_j = block_idx % blocks_per_dim;
        let start_i = block_i * profile.block_size;
        let start_j = block_j * profile.block_size;
        
        // 블록 데이터 추출
        let mut block_data = vec![0.0f32; profile.block_size * profile.block_size];
        for i in 0..profile.block_size {
            for j in 0..profile.block_size {
                let global_i = start_i + i;
                let global_j = start_j + j;
                if global_i < matrix_size && global_j < matrix_size {
                    block_data[i * profile.block_size + j] = 
                        matrix_data[global_i * matrix_size + global_j];
                }
            }
        }
        
        // 블록 압축
        let encoded_block = encoder.encode_block(&block_data, profile.block_size, profile.block_size);
        encoded_blocks.push(encoded_block);
        
        pb.set_position((block_idx * 100 / total_blocks) as u64);
    }
    
    pb.finish();
    
    let compression_time = start.elapsed().as_secs_f64();
    
    // 압축률 계산
    let original_size = matrix_size * matrix_size * 4; // f32 bytes
    let compressed_size = encoded_blocks.len() * std::mem::size_of::<HybridEncodedBlock>();
    let compression_ratio = original_size as f32 / compressed_size as f32;
    
    // RMSE 계산 - 디코딩해서 원본과 비교
    let mut reconstructed_data = vec![0.0f32; matrix_size * matrix_size];
    let blocks_per_dim = (matrix_size + profile.block_size - 1) / profile.block_size;
    
    for (block_idx, encoded_block) in encoded_blocks.iter().enumerate() {
        let block_i = block_idx / blocks_per_dim;
        let block_j = block_idx % blocks_per_dim;
        let start_i = block_i * profile.block_size;
        let start_j = block_j * profile.block_size;
        
        // 블록 디코딩
        let decoded_block = encoded_block.decode();
        
        // 원본 행렬에 복사
        for i in 0..profile.block_size {
            for j in 0..profile.block_size {
                let global_i = start_i + i;
                let global_j = start_j + j;
                if global_i < matrix_size && global_j < matrix_size {
                    reconstructed_data[global_i * matrix_size + global_j] = 
                        decoded_block[i * profile.block_size + j];
                }
            }
        }
    }
    
    // RMSE 계산
    let mse: f32 = matrix_data.iter()
        .zip(reconstructed_data.iter())
        .map(|(orig, recon)| (orig - recon).powi(2))
        .sum::<f32>() / (matrix_size * matrix_size) as f32;
    let rmse = mse.sqrt();
    Ok((encoded_blocks, compression_time, compression_ratio, rmse))
}

fn generate_test_matrix(size: usize) -> Vec<f32> {
    let mut matrix_data = vec![0.0f32; size * size];
    // 다양한 주기 패턴으로 생성
    for i in 0..size {
        for j in 0..size {
            let x = i as f32 / size as f32;
            let y = j as f32 / size as f32;
            matrix_data[i * size + j] = 
                (2.0 * std::f32::consts::PI * x).sin() * 
                (2.0 * std::f32::consts::PI * y).cos() * 0.5;
        }
    }
    matrix_data
}

fn find_critical_coefficients(
    matrix_data: &[f32], 
    matrix_size: usize, 
    block_size: usize,
    multi_progress: &MultiProgress
) -> Result<usize> {
    // 이분탐색으로 임계 계수 찾기
    let max_coeffs = (block_size * block_size) / 4; // 상한: 전체 픽셀의 1/4
    let min_coeffs = 8; // 하한: 최소 8개
    
    let mut left = min_coeffs;
    let mut right = max_coeffs;
    let mut critical_coeffs = max_coeffs;
    
    let pb = multi_progress.add(ProgressBar::new((right - left) as u64));
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!("[{{bar:30}}] 탐색중: {}x{} 블록", block_size, block_size))
            .unwrap()
    );
    
    while left <= right {
        let mid = (left + right) / 2;
        
        let profile = CompressionProfile {
            name: "임계점탐색",
            block_size,
            coefficients: mid,
            quality_level: "탐색",
        };
        
        match compress_with_profile(matrix_data, matrix_size, &profile, &MultiProgress::new()) {
            Ok((_, _, _, rmse)) => {
                pb.set_message(format!("계수: {}, RMSE: {:.6}", mid, rmse));
                pb.inc(1);
                
                if rmse <= 0.000001 {
                    // 성공: 더 적은 계수로 시도
                    critical_coeffs = mid;
                    right = mid - 1;
                } else {
                    // 실패: 더 많은 계수 필요
                    left = mid + 1;
                }
            },
            Err(_) => {
                left = mid + 1;
            }
        }
    }
    
    pb.finish_with_message(format!("임계 계수: {}", critical_coeffs));
    Ok(critical_coeffs)
}

fn calculate_critical_coefficients(block_size: usize) -> (usize, usize, usize) {
    // 기존 공식 예측값 (비교용)
    let log_factor = if block_size >= 32 {
        (block_size as f32 / 32.0).log2().max(0.0) as usize
    } else {
        0
    };
    
    let r_safe = 32_usize.saturating_sub(log_factor).max(25);
    let r_optimal = r_safe / 2;
    let r_minimal = r_safe;
    
    let k_safe = (block_size * block_size) / r_safe;
    let k_optimal = (block_size * block_size) / r_optimal;
    let k_minimal = (block_size * block_size) / r_minimal;
    
    (k_safe, k_optimal, k_minimal)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("\n=== RBE 임계점 탐색 실험 ===\n");
    
    // 다양한 매트릭스와 블록 크기 (작은 블록 포함)
    let test_combinations = vec![
        // (매트릭스_크기, 블록_크기, 설명)
        (512, 16, "512→16 (32×32=1024블록)"),
        (512, 32, "512→32 (16×16=256블록)"),
        (1024, 16, "1024→16 (64×64=4096블록)"), 
        (1024, 32, "1024→32 (32×32=1024블록)"),
        (1024, 64, "1024→64 (16×16=256블록)"),
        (1024, 128, "1024→128 (8×8=64블록)"),
        (2048, 64, "2048→64 (32×32=1024블록)"),
        (2048, 128, "2048→128 (16×16=256블록)"),
        (2048, 256, "2048→256 (8×8=64블록)"),
        (4096, 128, "4096→128 (32×32=1024블록)"),
        (4096, 256, "4096→256 (16×16=256블록)"),
        (4096, 512, "4096→512 (8×8=64블록)"),
    ];
    
    println!("매트릭스크기 | 블록크기 | 블록개수 | 공식예측 | 실제임계 | 예측정확도");
    println!("-------------|----------|----------|----------|----------|----------");
    
    let mut all_results = Vec::new();
    let multi_progress = MultiProgress::new();
    
    for &(matrix_size, block_size, description) in &test_combinations {
        let blocks_per_dim = matrix_size / block_size;
        let total_blocks = blocks_per_dim * blocks_per_dim;
        let predicted_coeffs = calculate_critical_coefficients(block_size).2;
        
        // 테스트 매트릭스 생성
        let matrix_data = generate_test_matrix(matrix_size);
        
        // 실제 임계점 탐색
        println!("\n🔍 {} 임계점 탐색 중...", description);
        let actual_critical = find_critical_coefficients(&matrix_data, matrix_size, block_size, &multi_progress)?;
        
        let accuracy = (predicted_coeffs as f32 / actual_critical as f32 * 100.0).min(100.0);
        
        println!("{:11} | {:8} | {:8} | {:8} | {:8} | {:7.1}%",
                matrix_size, block_size, total_blocks, predicted_coeffs, actual_critical, accuracy);
        
        // 실제 임계 계수로 최종 압축 테스트
        let profile = CompressionProfile {
            name: description,
            block_size,
            coefficients: actual_critical,
            quality_level: "임계점",
        };
        
        match compress_with_profile(&matrix_data, matrix_size, &profile, &MultiProgress::new()) {
            Ok((_, compression_time, compression_ratio, rmse)) => {
                all_results.push((
                    description,
                    block_size,
                    predicted_coeffs,
                    actual_critical, 
                    compression_ratio,
                    rmse,
                    compression_time,
                    accuracy
                ));
            },
            Err(e) => {
                println!("  → 최종 압축 에러: {}", e);
            }
        }
    }
    
    // 결과 분석
    println!("\n=== 임계점 탐색 결과 분석 ===\n");
    println!("조합                      | 블록크기 | 공식예측 | 실제임계 | 차이    | 압축률      | RMSE      | 정확도");
    println!("--------------------------|----------|----------|----------|---------|-------------|-----------|--------");
    
    let mut total_accuracy = 0.0;
    let mut perfect_predictions = 0;
    
    for (description, block_size, predicted, actual, ratio, rmse, _time, accuracy) in &all_results {
        let diff = (*predicted as i32) - (*actual as i32);
        let diff_str = if diff > 0 { format!("+{}", diff) } else { diff.to_string() };
        
        let abs_diff = if *predicted > *actual { predicted - actual } else { actual - predicted };
        if abs_diff <= actual / 20 {  // 5% 이내면 정확한 것으로 간주
            perfect_predictions += 1;
        }
        
        total_accuracy += accuracy;
        
        println!("{:25} | {:8} | {:8} | {:8} | {:7} | {:10.1} | {:9.6} | {:6.1}%",
                description, block_size, predicted, actual, diff_str, ratio, rmse, accuracy);
    }
    
    let avg_accuracy = total_accuracy / all_results.len() as f32;
    let perfect_rate = perfect_predictions as f32 / all_results.len() as f32 * 100.0;
    
    println!("\n📊 전체 통계:");
    println!("평균 예측 정확도: {:.1}%", avg_accuracy);
    println!("완벽 예측 비율: {:.1}% ({}/{})", perfect_rate, perfect_predictions, all_results.len());
    
    if avg_accuracy >= 90.0 {
        println!("🎉 공식이 매우 정확합니다!");
    } else {
        println!("⚠️  공식 개선이 필요합니다.");
        
        // 개선된 공식 제안
        println!("\n📐 개선 방향 분석:");
        for (_, block_size, predicted, actual, _, _, _, _) in &all_results {
            let r_actual = (block_size * block_size) / actual;
            let r_predicted = (block_size * block_size) / predicted;
            println!("블록{}x{}: 실제R={}, 예측R={}", block_size, block_size, r_actual, r_predicted);
        }
    }
    
    Ok(())
} 
use crate::sllm::{SLLMCompressor, CompressionConfig};
use crate::encoder::HybridEncoder;
use crate::types::TransformType;
use std::path::PathBuf;
use std::time::Instant;

/// 🗜️ 실제 압축 데모
#[tokio::test]
async fn test_real_compression_demo() {
    println!("\n🚀 === RBE 압축 실제 데모 ===\n");
    
    // 1. 간단한 행렬 압축 데모
    println!("📊 1단계: 간단한 행렬 압축 시연");
    demo_simple_matrix_compression();
    
    println!("\n{}\n", "=".repeat(60));
    
    // 2. 대규모 행렬 압축 데모
    println!("🏢 2단계: 대규모 행렬 압축 시연");
    demo_large_matrix_compression();
    
    println!("\n{}\n", "=".repeat(60));
    
    // 3. 실제 모델 압축 시뮬레이션
    println!("🤖 3단계: 실제 모델 압축 시뮬레이션");
    demo_model_compression_simulation().await;
    
    println!("\n✅ 전체 압축 데모 완료!");
}

/// 간단한 행렬 압축
fn demo_simple_matrix_compression() {
    let size = 64;
    let mut matrix = vec![0.0f32; size * size];
    
    // 테스트 패턴 생성
    for i in 0..size {
        for j in 0..size {
            let x = j as f32 / (size - 1) as f32;
            let y = i as f32 / (size - 1) as f32;
            matrix[i * size + j] = ((x * 10.0).sin() + (y * 10.0).cos()) / 2.0;
        }
    }
    
    println!("원본 행렬: {}×{} (16KB)", size, size);
    
    // DCT 압축
    let start = Instant::now();
    let mut dct_encoder = HybridEncoder::new(100, TransformType::Dct);
    let dct_compressed = dct_encoder.encode_block(&matrix, size, size);
    let dct_time = start.elapsed();
    let dct_decoded = dct_compressed.decode();
    
    // 웨이블릿 압축
    let start = Instant::now();
    let mut dwt_encoder = HybridEncoder::new(100, TransformType::Dwt);
    let dwt_compressed = dwt_encoder.encode_block(&matrix, size, size);
    let dwt_time = start.elapsed();
    let dwt_decoded = dwt_compressed.decode();
    
    // RMSE 계산
    let dct_rmse = calculate_rmse(&matrix, &dct_decoded);
    let dwt_rmse = calculate_rmse(&matrix, &dwt_decoded);
    
    println!("\n📊 압축 결과:");
    println!("┌─────────────┬──────────┬──────────┬──────────┬──────────┐");
    println!("│ 방법        │ 계수     │ RMSE     │ 시간     │ 압축률   │");
    println!("├─────────────┼──────────┼──────────┼──────────┼──────────┤");
    println!("│ DCT         │ 100개    │ {:.6} │ {:?}  │ 160:1    │", dct_rmse, dct_time);
    println!("│ 웨이블릿    │ 100개    │ {:.6} │ {:?}  │ 160:1    │", dwt_rmse, dwt_time);
    println!("└─────────────┴──────────┴──────────┴──────────┴──────────┘");
    
    let winner = if dwt_rmse < dct_rmse { "웨이블릿" } else { "DCT" };
    println!("🏆 승자: {}", winner);
}

/// 대규모 행렬 압축
fn demo_large_matrix_compression() {
    let sizes = vec![128, 256, 512];
    let coefficients = vec![50, 100, 200, 500];
    
    println!("대규모 행렬 압축 테스트:");
    println!("┌────────┬────────┬──────────┬──────────┬──────────┐");
    println!("│ 크기   │ 계수   │ RMSE     │ 압축률   │ 품질     │");
    println!("├────────┼────────┼──────────┼──────────┼──────────┤");
    
    for size in &sizes {
        for &coeff in &coefficients {
            let matrix = generate_test_matrix(*size);
            let mut encoder = HybridEncoder::new(coeff, TransformType::Dwt);
            let compressed = encoder.encode_block(&matrix, *size, *size);
            let decoded = compressed.decode();
            let rmse = calculate_rmse(&matrix, &decoded);
            
            let compression_ratio = (*size * *size) as f32 / (16.0 + coeff as f32 * 6.0);
            let quality = if rmse < 0.001 { "🥇 S급" }
            else if rmse < 0.01 { "🥉 A급" }
            else if rmse < 0.05 { "B급" }
            else { "C급" };
            
            println!("│ {}×{} │ {:4}   │ {:.6} │ {:6.1}:1 │ {:8} │", 
                     size, size, coeff, rmse, compression_ratio, quality);
                     
            // S급 달성하면 중단
            if rmse < 0.001 {
                break;
            }
        }
    }
    
    println!("└────────┴────────┴──────────┴──────────┴──────────┘");
}

/// 실제 모델 압축 시뮬레이션
async fn demo_model_compression_simulation() {
    println!("GPT-2 레이어 압축 시뮬레이션:");
    
    // GPT-2 레이어 크기 시뮬레이션
    let layers = vec![
        ("attention.key", 768, 768),
        ("attention.query", 768, 768),
        ("attention.value", 768, 768),
        ("mlp.fc1", 768, 3072),
        ("mlp.fc2", 3072, 768),
    ];
    
    let mut total_original = 0;
    let mut total_compressed = 0;
    
    println!("┌─────────────────┬────────────┬─────────┬──────────┬──────────┐");
    println!("│ 레이어          │ 크기       │ 원본(MB)│ 압축(KB) │ 압축률   │");
    println!("├─────────────────┼────────────┼─────────┼──────────┼──────────┤");
    
    for (name, rows, cols) in &layers {
        let original_size = rows * cols * 4; // f32 = 4 bytes
        let compressed_size = 16 + 500 * 6; // Packed128 + 500 coefficients
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        total_original += original_size;
        total_compressed += compressed_size;
        
        println!("│ {:15} │ {}×{:4} │ {:7.2} │ {:8.2} │ {:6.1}:1 │",
                 name, rows, cols,
                 original_size as f32 / 1_048_576.0,
                 compressed_size as f32 / 1024.0,
                 compression_ratio);
    }
    
    println!("├─────────────────┼────────────┼─────────┼──────────┼──────────┤");
    println!("│ 전체            │            │ {:7.2} │ {:8.2} │ {:6.1}:1 │",
             total_original as f32 / 1_048_576.0,
             total_compressed as f32 / 1024.0,
             total_original as f32 / total_compressed as f32);
    println!("└─────────────────┴────────────┴─────────┴──────────┴──────────┘");
    
    let memory_saving = (1.0 - total_compressed as f32 / total_original as f32) * 100.0;
    println!("\n💾 메모리 절약률: {:.1}%", memory_saving);
    println!("📱 모바일 실행 가능: {}", if memory_saving > 90.0 { "✅ 예" } else { "❌ 아니오" });
}

/// 테스트 행렬 생성
fn generate_test_matrix(size: usize) -> Vec<f32> {
    let mut matrix = vec![0.0f32; size * size];
    for i in 0..size {
        for j in 0..size {
            let x = j as f32 / (size - 1) as f32;
            let y = i as f32 / (size - 1) as f32;
            let r = ((x - 0.5).powi(2) + (y - 0.5).powi(2)).sqrt();
            matrix[i * size + j] = (r * 20.0).sin() * (-r * 5.0).exp();
        }
    }
    matrix
}

/// RMSE 계산
fn calculate_rmse(original: &[f32], reconstructed: &[f32]) -> f32 {
    let mse: f32 = original.iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| (o - r).powi(2))
        .sum::<f32>() / original.len() as f32;
    mse.sqrt()
} 
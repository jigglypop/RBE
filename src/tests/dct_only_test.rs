use crate::types::*;
use crate::encoder::HybridEncoder;
use std::time::Instant;

/// RMSE 계산 유틸리티 함수
fn calculate_rmse(target: &[f32], predicted: &[f32]) -> f32 {
    let mse: f32 = target.iter().zip(predicted.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f32>() / target.len() as f32;
    mse.sqrt()
}

/// 복잡한 테스트 패턴 생성 (중력장 + 파동)
fn generate_complex_test_pattern(rows: usize, cols: usize) -> Vec<f32> {
    let mut pattern = vec![0.0; rows * cols];
    
    for i in 0..rows {
        for j in 0..cols {
            let x = (j as f32 / cols as f32) * 2.0 - 1.0;
            let y = (i as f32 / rows as f32) * 2.0 - 1.0;
            
            // 중력장 성분
            let r = (x * x + y * y).sqrt().max(0.1);
            let gravity = 1.0 / r;
            
            // 파동 성분
            let wave1 = (3.0 * std::f32::consts::PI * x).sin();
            let wave2 = (2.0 * std::f32::consts::PI * y).cos();
            
            // 조합
            let idx = i * cols + j;
            pattern[idx] = 0.5 * gravity + 0.3 * wave1 + 0.2 * wave2;
        }
    }
    
    // 정규화
    let max_val = pattern.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min_val = pattern.iter().copied().fold(f32::INFINITY, f32::min);
    let range = max_val - min_val;
    
    if range > 0.0 {
        for val in pattern.iter_mut() {
            *val = (*val - min_val) / range;
        }
    }
    
    pattern
}

/// 🎯 DCT Only 순수 성능 테스트 (RBE 없음)
#[test]
fn test_dct_only_pure_performance() {
    println!("🎯 === DCT Only 순수 성능 테스트 (RBE 없음) ===");
    
    let rows = 64;
    let cols = 64;
    let target = generate_complex_test_pattern(rows, cols);
    
    println!("테스트 설정: {}×{} 복잡한 패턴", rows, cols);
    println!("목표: DCT 변환만으로 최고 정밀도 달성");
    
    // 다양한 DCT 계수 개수 테스트
    let dct_coefficients = vec![10, 25, 50, 100, 200];
    
    for &coeff_count in &dct_coefficients {
        println!("\n🔵 === DCT 계수 {}개 테스트 ===", coeff_count);
        let start_time = Instant::now();
        
        // DCT 인코더
        let mut dct_encoder = HybridEncoder::new(coeff_count, TransformType::Dct);
        let compressed = dct_encoder.encode_block(&target, rows, cols);
        let decoded = compressed.decode();
        
        let duration = start_time.elapsed().as_millis();
        let rmse = calculate_rmse(&target, &decoded);
        
        // 압축률 계산
        let original_size = rows * cols * 4; // f32 크기
        let compressed_size = 16; // Packed128 크기
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        // 품질 등급
        let quality = if rmse < 0.001 { "🥇 S급" }
        else if rmse < 0.005 { "🥈 A+급" }
        else if rmse < 0.01 { "🥉 A급" }
        else if rmse < 0.05 { "B급" }
        else if rmse < 0.1 { "C급" }
        else { "D급" };
        
        println!("  계수: {} | RMSE: {:.8} | 품질: {} | 시간: {}ms", 
                 coeff_count, rmse, quality, duration);
        println!("  압축률: {:.1}:1 | 메모리 절약: {:.1}%", 
                 compression_ratio, (1.0 - 1.0/compression_ratio) * 100.0);
                 
        // 목표 달성 여부
        if rmse < 0.001 {
            println!("  🎉 목표 달성! DCT {}계수로 RMSE < 0.001 달성!", coeff_count);
        }
    }
    
    // 최고 성능 DCT 계수 찾기
    println!("\n🔍 === 최적 DCT 계수 탐색 ===");
    let mut best_rmse = f32::INFINITY;
    let mut best_coeff = 0;
    
    // 세밀한 계수 스캔
    let fine_coefficients = vec![150, 300, 500, 750, 1000];
    
    for &coeff_count in &fine_coefficients {
        let mut dct_encoder = HybridEncoder::new(coeff_count, TransformType::Dct);
        let compressed = dct_encoder.encode_block(&target, rows, cols);
        let decoded = compressed.decode();
        let rmse = calculate_rmse(&target, &decoded);
        
        println!("  DCT {}계수: RMSE = {:.8}", coeff_count, rmse);
        
        if rmse < best_rmse {
            best_rmse = rmse;
            best_coeff = coeff_count;
        }
    }
    
    println!("\n🏆 === DCT Only 최종 결과 ===");
    println!("최적 DCT 계수: {}", best_coeff);
    println!("최고 RMSE: {:.8}", best_rmse);
    println!("목표 달성: {}", if best_rmse < 0.001 { "✅ 성공!" } else { "❌ 미달성" });
    
    if best_rmse < 0.001 {
        println!("🎯 DCT만으로 목표 달성! RBE가 필요하지 않습니다!");
    } else {
        println!("DCT 한계 확인. RBE 잔차학습이 필요합니다.");
        println!("RBE로 추가 개선 가능: {:.1}배", best_rmse / 0.001);
    }
}

/// 🎯 웨이블릿 Only 순수 성능 테스트 
#[test]
fn test_wavelet_only_pure_performance() {
    println!("🎯 === 웨이블릿 Only 순수 성능 테스트 ===");
    
    let rows = 64;
    let cols = 64;
    let target = generate_complex_test_pattern(rows, cols);
    
    // 웨이블릿 계수 테스트
    let wavelet_coefficients = vec![10, 25, 50, 100, 200, 500];
    let mut best_rmse = f32::INFINITY;
    let mut best_coeff = 0;
    
    for &coeff_count in &wavelet_coefficients {
        println!("\n🟢 === 웨이블릿 계수 {}개 테스트 ===", coeff_count);
        let start_time = Instant::now();
        
        let mut wavelet_encoder = HybridEncoder::new(coeff_count, TransformType::Dwt);
        let compressed = wavelet_encoder.encode_block(&target, rows, cols);
        let decoded = compressed.decode();
        
        let duration = start_time.elapsed().as_millis();
        let rmse = calculate_rmse(&target, &decoded);
        
        let quality = if rmse < 0.001 { "🥇 S급" }
        else if rmse < 0.01 { "🥉 A급" }
        else if rmse < 0.05 { "B급" }
        else if rmse < 0.1 { "C급" }
        else { "D급" };
        
        println!("  계수: {} | RMSE: {:.8} | 품질: {} | 시간: {}ms", 
                 coeff_count, rmse, quality, duration);
                 
        if rmse < best_rmse {
            best_rmse = rmse;
            best_coeff = coeff_count;
        }
    }
    
    println!("\n🏆 === 웨이블릿 Only 최종 결과 ===");
    println!("최적 웨이블릿 계수: {}", best_coeff);  
    println!("최고 RMSE: {:.8}", best_rmse);
    
    if best_rmse < 0.001 {
        println!("🎯 웨이블릿만으로 목표 달성!");
    }
}

/// 🎯 DCT vs 웨이블릿 직접 비교 테스트
#[test] 
fn test_dct_vs_wavelet_comparison() {
    println!("🎯 === DCT vs 웨이블릿 직접 비교 ===");
    
    let rows = 64;
    let cols = 64;
    let target = generate_complex_test_pattern(rows, cols);
    
    let coeff_count = 100; // 동일한 계수로 비교
    
    // DCT 테스트
    println!("\n🔵 DCT {} 계수:", coeff_count);
    let start_time = Instant::now();
    let mut dct_encoder = HybridEncoder::new(coeff_count, TransformType::Dct);
    let dct_compressed = dct_encoder.encode_block(&target, rows, cols);
    let dct_decoded = dct_compressed.decode();
    let dct_time = start_time.elapsed().as_millis();
    let dct_rmse = calculate_rmse(&target, &dct_decoded);
    
    // 웨이블릿 테스트  
    println!("🟢 웨이블릿 {} 계수:", coeff_count);
    let start_time = Instant::now();
    let mut dwt_encoder = HybridEncoder::new(coeff_count, TransformType::Dwt);
    let dwt_compressed = dwt_encoder.encode_block(&target, rows, cols);
    let dwt_decoded = dwt_compressed.decode();
    let dwt_time = start_time.elapsed().as_millis();
    let dwt_rmse = calculate_rmse(&target, &dwt_decoded);
    
    // 결과 비교
    println!("\n📊 === 비교 결과 ===");
    println!("┌──────────────┬─────────────┬─────────────┬─────────────┐");
    println!("│ 방법         │ RMSE        │ 시간 (ms)   │ 성능        │");
    println!("├──────────────┼─────────────┼─────────────┼─────────────┤");
    println!("│ DCT          │ {:11.8} │ {:11} │ {} │", dct_rmse, dct_time, if dct_rmse < dwt_rmse { "🥇 승리" } else { "🥈" });
    println!("│ 웨이블릿     │ {:11.8} │ {:11} │ {} │", dwt_rmse, dwt_time, if dwt_rmse < dct_rmse { "🥇 승리" } else { "🥈" });
    println!("└──────────────┴─────────────┴─────────────┴─────────────┘");
    
    let winner = if dct_rmse < dwt_rmse { "DCT" } else { "웨이블릿" };
    let improvement = ((dct_rmse.max(dwt_rmse) - dct_rmse.min(dwt_rmse)) / dct_rmse.max(dwt_rmse) * 100.0);
    
    println!("\n🏆 승자: {} ({}% 더 우수)", winner, improvement);
    
    if dct_rmse.min(dwt_rmse) < 0.001 {
        println!("🎯 목표 달성! {}가 RMSE < 0.001 달성!", winner);
    }
} 
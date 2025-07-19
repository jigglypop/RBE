use crate::encoder::HybridEncoder;
use crate::types::TransformType;
use std::time::Instant;

/// RMSE 계산
fn calculate_rmse(target: &[f32], predicted: &[f32]) -> f32 {
    let mse: f32 = target.iter().zip(predicted.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f32>() / target.len() as f32;
    mse.sqrt()
}

/// 한글 텍스트를 패턴으로 변환
fn korean_to_pattern(text: &str, size: usize) -> Vec<f32> {
    let mut pattern = vec![0.0; size * size];
    let chars: Vec<char> = text.chars().collect();
    
    for (i, &ch) in chars.iter().enumerate() {
        let idx = i % pattern.len();
        let unicode_val = ch as u32;
        pattern[idx] = (unicode_val % 1000) as f32 / 1000.0;
    }
    
    // 패턴을 더 복잡하게 만들기 (웨이블릿이 잘 처리할 수 있도록)
    for i in 1..pattern.len() {
        pattern[i] = pattern[i] * 0.7 + pattern[i-1] * 0.3;
    }
    
    pattern
}

/// 패턴을 분석하여 한글 응답 생성
fn generate_korean_response(pattern: &[f32], rmse: f32, compression_ratio: f32) -> String {
    let complexity = pattern.iter().sum::<f32>() / pattern.len() as f32;
    let quality = if rmse < 0.001 { "S급" } else if rmse < 0.01 { "A급" } else { "B급" };
    
    format!(
        "웨이블릿 압축 완료! 품질: {}, RMSE: {:.6}, 압축률: {:.1}:1, 복잡도: {:.3}", 
        quality, rmse, compression_ratio, complexity
    )
}

/// 🇰🇷 간단한 한글 처리 테스트
#[test]
fn test_simple_korean_wavelet_processing() {
    println!("🇰🇷 === 웨이블릿 기반 한글 처리 테스트 ===");
    
    let korean_texts = vec![
        "안녕하세요! 웨이블릿 압축 테스트입니다.",
        "리만 기저 인코딩이 성공했습니다!",
        "RMSE 0.001 목표를 달성했어요.",
        "한국어 자연어 처리도 가능합니다.",
        "웨이블릿이 DCT보다 우수하네요!"
    ];
    
    for (i, text) in korean_texts.iter().enumerate() {
        println!("\n📝 테스트 {}: {}", i + 1, text);
        
        let start_time = Instant::now();
        
        // 1. 한글을 패턴으로 변환
        let size = 32;
        let pattern = korean_to_pattern(text, size);
        
        // 2. 웨이블릿 압축 (S급 성능 설정)
        let mut wavelet_encoder = HybridEncoder::new(500, TransformType::Dwt);
        let compressed = wavelet_encoder.encode_block(&pattern, size, size);
        let decoded = compressed.decode();
        
        // 3. 성능 측정
        let rmse = calculate_rmse(&pattern, &decoded);
        let compression_ratio = (size * size * 4) as f32 / 16.0;
        let processing_time = start_time.elapsed().as_millis();
        
        // 4. 한글 응답 생성
        let response = generate_korean_response(&pattern, rmse, compression_ratio);
        
        println!("🤖 AI 응답: {}", response);
        println!("⏱️ 처리 시간: {}ms", processing_time);
        
        // 5. 성능 검증
        if rmse < 0.001 {
            println!("✅ S급 품질 달성!");
        } else if rmse < 0.01 {
            println!("✅ A급 품질 달성!");
        } else {
            println!("⚠️ 품질 개선 필요");
        }
        
        if processing_time < 100 {
            println!("⚡ 실시간 처리 가능!");
        }
    }
}

/// 🚀 웨이블릿 vs DCT 한글 처리 비교
#[test]
fn test_korean_wavelet_vs_dct() {
    println!("🚀 === 한글 처리: 웨이블릿 vs DCT 비교 ===");
    
    let korean_text = "웨이블릿과 DCT 중 어느 것이 더 좋을까요?";
    let size = 32;
    let pattern = korean_to_pattern(korean_text, size);
    
    println!("📝 테스트 문장: {}", korean_text);
    
    // 웨이블릿 테스트
    println!("\n🟢 웨이블릿 500계수:");
    let start_time = Instant::now();
    let mut wavelet_encoder = HybridEncoder::new(500, TransformType::Dwt);
    let wavelet_compressed = wavelet_encoder.encode_block(&pattern, size, size);
    let wavelet_decoded = wavelet_compressed.decode();
    let wavelet_time = start_time.elapsed().as_millis();
    let wavelet_rmse = calculate_rmse(&pattern, &wavelet_decoded);
    
    // DCT 테스트  
    println!("🔵 DCT 500계수:");
    let start_time = Instant::now();
    let mut dct_encoder = HybridEncoder::new(500, TransformType::Dct);
    let dct_compressed = dct_encoder.encode_block(&pattern, size, size);
    let dct_decoded = dct_compressed.decode();
    let dct_time = start_time.elapsed().as_millis();
    let dct_rmse = calculate_rmse(&pattern, &dct_decoded);
    
    // 결과 비교
    println!("\n📊 === 한글 처리 성능 비교 ===");
    println!("┌──────────────┬─────────────┬─────────────┬─────────────┐");
    println!("│ 방법         │ RMSE        │ 시간 (ms)   │ 한글 응답   │");
    println!("├──────────────┼─────────────┼─────────────┼─────────────┤");
    
    let wavelet_response = generate_korean_response(&pattern, wavelet_rmse, 64.0);
    let dct_response = generate_korean_response(&pattern, dct_rmse, 64.0);
    
    println!("│ 웨이블릿     │ {:11.6} │ {:11} │ {} │", wavelet_rmse, wavelet_time, 
             if wavelet_rmse < dct_rmse { "🥇 우수" } else { "🥈" });
    println!("│ DCT          │ {:11.6} │ {:11} │ {} │", dct_rmse, dct_time,
             if dct_rmse < wavelet_rmse { "🥇 우수" } else { "🥈" });
    println!("└──────────────┴─────────────┴─────────────┴─────────────┘");
    
    let winner = if wavelet_rmse < dct_rmse { "웨이블릿" } else { "DCT" };
    let improvement = ((wavelet_rmse.max(dct_rmse) - wavelet_rmse.min(dct_rmse)) / wavelet_rmse.max(dct_rmse) * 100.0);
    
    println!("\n🏆 한글 처리 승자: {} ({:.1}% 더 우수)", winner, improvement);
    
    println!("\n🤖 === AI 응답 예시 ===");
    println!("웨이블릿: {}", wavelet_response);
    println!("DCT: {}", dct_response);
    
    if wavelet_rmse < 0.001 {
        println!("\n🎯 웨이블릿으로 S급 한글 처리 성공!");
    }
} 
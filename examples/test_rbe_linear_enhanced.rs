//! RBELinear Enhanced128 마이그레이션 테스트
//! 
//! Standard vs Enhanced 압축 모드 비교 및 성능 검증

use rbe_llm::nlp::linear::{RBELinear, RBELinearConfig, RBECompressionMode};
use std::time::Instant;

fn main() {
    println!("🚀 RBELinear Enhanced128 마이그레이션 테스트\n");
    
    // 1. 기본 생성 테스트
    basic_creation_test();
    
    // 2. 압축 모드 비교 테스트
    compression_mode_comparison();
    
    // 3. 성능 비교 테스트
    performance_comparison();
    
    // 4. 업그레이드 테스트
    upgrade_test();
    
    println!("\n✅ 모든 테스트 완료!");
}

fn basic_creation_test() {
    println!("📋 1. 기본 생성 테스트");
    
    let in_features = 128;
    let out_features = 64;
    
    // Enhanced 모드 (기본값)
    let layer_enhanced = RBELinear::new(in_features, out_features, None);
    println!("  Enhanced 모드: {:?}", layer_enhanced.compression_mode());
    
    // Standard 모드
    let layer_standard = RBELinear::new_standard(in_features, out_features, None);
    println!("  Standard 모드: {:?}", layer_standard.compression_mode());
    
    // 명시적 Enhanced 모드
    let layer_explicit = RBELinear::new_enhanced(in_features, out_features, None);
    println!("  명시적 Enhanced 모드: {:?}", layer_explicit.compression_mode());
    
    // 메모리 사용량 비교
    println!("  메모리 사용량:");
    println!("    Enhanced: {} bytes", layer_enhanced.memory_usage());
    println!("    Standard: {} bytes", layer_standard.memory_usage());
    
    println!("  ✅ 기본 생성 테스트 통과\n");
}

fn compression_mode_comparison() {
    println!("🔄 2. 압축 모드 비교 테스트");
    
    let in_features = 32;
    let out_features = 16;
    let input = vec![0.1f32; in_features];
    
    // Standard 모드
    let layer_standard = RBELinear::new_standard(in_features, out_features, None);
    let output_standard = layer_standard.forward(&input).unwrap();
    
    // Enhanced 모드
    let layer_enhanced = RBELinear::new_enhanced(in_features, out_features, None);
    let output_enhanced = layer_enhanced.forward(&input).unwrap();
    
    println!("  Standard 출력 (처음 5개): {:?}", &output_standard[..5]);
    println!("  Enhanced 출력 (처음 5개): {:?}", &output_enhanced[..5]);
    
    // 통계 비교
    let (std_min, std_max, std_mean) = layer_standard.weight_stats();
    let (enh_min, enh_max, enh_mean) = layer_enhanced.weight_stats();
    
    println!("  가중치 통계:");
    println!("    Standard: min={:.3}, max={:.3}, mean={:.3}", std_min, std_max, std_mean);
    println!("    Enhanced: min={:.3}, max={:.3}, mean={:.3}", enh_min, enh_max, enh_mean);
    
    println!("  ✅ 압축 모드 비교 완료\n");
}

fn performance_comparison() {
    println!("⚡ 3. 성능 비교 테스트");
    
    let in_features = 256;
    let out_features = 128;
    let batch_size = 100;
    let iterations = 50;
    
    // 입력 데이터 준비
    let inputs: Vec<Vec<f32>> = (0..batch_size)
        .map(|i| vec![i as f32 * 0.01; in_features])
        .collect();
    
    // Standard 모드 성능 테스트
    let layer_standard = RBELinear::new_standard(in_features, out_features, None);
    let start_standard = Instant::now();
    
    for _ in 0..iterations {
        for input in &inputs {
            let _output = layer_standard.forward(input).unwrap();
        }
    }
    
    let standard_time = start_standard.elapsed();
    
    // Enhanced 모드 성능 테스트
    let layer_enhanced = RBELinear::new_enhanced(in_features, out_features, None);
    let start_enhanced = Instant::now();
    
    for _ in 0..iterations {
        for input in &inputs {
            let _output = layer_enhanced.forward(input).unwrap();
        }
    }
    
    let enhanced_time = start_enhanced.elapsed();
    
    // 성능 분석
    let total_operations = iterations * batch_size;
    let standard_ops_per_sec = total_operations as f64 / standard_time.as_secs_f64();
    let enhanced_ops_per_sec = total_operations as f64 / enhanced_time.as_secs_f64();
    let performance_ratio = enhanced_time.as_nanos() as f64 / standard_time.as_nanos() as f64;
    
    println!("  성능 결과 ({}회 × {} 배치):", iterations, batch_size);
    println!("    Standard: {:.2} ms ({:.0} ops/s)", 
             standard_time.as_secs_f64() * 1000.0, standard_ops_per_sec);
    println!("    Enhanced: {:.2} ms ({:.0} ops/s)", 
             enhanced_time.as_secs_f64() * 1000.0, enhanced_ops_per_sec);
    println!("    속도 비율: {:.2}x (Enhanced/Standard)", performance_ratio);
    
    if performance_ratio <= 1.5 {
        println!("    ✅ 성능 목표 달성! (1.5x 이하)");
    } else {
        println!("    ⚠️  성능 저하 (1.5x 초과)");
    }
    
    println!();
}

fn upgrade_test() {
    println!("🔄 4. 업그레이드 테스트");
    
    let in_features = 64;
    let out_features = 32;
    
    // Standard로 시작
    let mut layer = RBELinear::new_standard(in_features, out_features, None);
    
    println!("  초기 모드: {:?}", layer.compression_mode());
    println!("  초기 메모리: {} bytes", layer.memory_usage());
    
    // Enhanced로 업그레이드
    layer.upgrade_to_enhanced().unwrap();
    
    println!("  업그레이드 후 모드: {:?}", layer.compression_mode());
    println!("  업그레이드 후 메모리: {} bytes", layer.memory_usage());
    
    // 이미 Enhanced인 경우 재업그레이드 시도
    layer.upgrade_to_enhanced().unwrap();
    
    // 기능 확인
    let input = vec![0.5f32; in_features];
    let output = layer.forward(&input).unwrap();
    
    println!("  업그레이드 후 출력 크기: {}", output.len());
    println!("  출력 통계: min={:.3}, max={:.3}", 
             output.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    
    println!("  ✅ 업그레이드 테스트 완료\n");
} 
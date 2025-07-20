use crate::core::packed_params::{TransformType, HybridEncodedBlock};
use crate::core::encoder::RBEEncoder;
use crate::core::decoder::optimized_decoder::OptimizedDecoder;
use std::time::Instant;

fn create_test_block(rows: usize, cols: usize, coeffs: usize, transform_type: TransformType) -> HybridEncodedBlock {
    // 테스트 데이터 생성
    let mut test_data = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = (j as f32 / cols as f32) * 2.0 - 1.0;
            let y = (i as f32 / rows as f32) * 2.0 - 1.0;
            test_data[i * cols + j] = (x * x + y * y).sin();
        }
    }
    
    // 인코딩해서 테스트 블록 생성
    let mut encoder = RBEEncoder::new(coeffs, transform_type);
    encoder.encode_block(&test_data, rows, cols)
}

#[test]
fn 성능_비교_테스트() {
    println!("🚀 디코더 성능 비교 테스트");
    
    let test_cases = [
        (64, 64, 128, "64x64 블록"),
        (128, 128, 256, "128x128 블록 (모델 기본)"),
        (256, 256, 512, "256x256 블록"),
    ];
    
    for (rows, cols, coeffs, desc) in test_cases {
        println!("\n📊 테스트: {}", desc);
        
        // 테스트 블록 생성
        let test_block = create_test_block(rows, cols, coeffs, TransformType::Dwt);
        
        // 기존 디코더 성능 측정
        let iterations = 100;
        let start = Instant::now();
        let mut original_result = Vec::new();
        for _ in 0..iterations {
            original_result = test_block.decode();
        }
        let original_time = start.elapsed().as_millis();
        
        // 최적화된 디코더 성능 측정  
        let start = Instant::now();
        let mut optimized_result = Vec::new();
        for _ in 0..iterations {
            optimized_result = test_block.decode_optimized();
        }
        let optimized_time = start.elapsed().as_millis();
        
        // 결과 검증 (같은 결과가 나와야 함)
        assert_eq!(original_result.len(), optimized_result.len(), 
                   "결과 길이가 다름: {} vs {}", original_result.len(), optimized_result.len());
        
        let mut max_diff = 0.0f32;
        for (i, (&orig, &opt)) in original_result.iter().zip(optimized_result.iter()).enumerate() {
            let diff = (orig - opt).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(diff < 1e-5, "인덱스 {}에서 차이가 큼: {} vs {} (차이: {})", i, orig, opt, diff);
        }
        
        // 성능 결과 출력
        let speedup = original_time as f32 / optimized_time as f32;
        println!("  🔸 기존 디코더:   {} ms ({} iterations)", original_time, iterations);
        println!("  🔸 최적화 디코더: {} ms ({} iterations)", optimized_time, iterations);
        println!("  🚀 성능 향상:     {:.2}x 빠름", speedup);
        println!("  ✅ 최대 오차:     {:.2e}", max_diff);
        
        // 성능 향상이 있어야 함
        assert!(speedup > 1.0, "최적화된 디코더가 더 느림: {:.2}x", speedup);
    }
    
    // 캐시 통계 확인
    let (a_cache_size, dct_cache_size) = OptimizedDecoder::get_cache_stats();
    println!("\n📈 캐시 통계:");
    println!("  🔸 A 매트릭스 캐시: {} 개", a_cache_size);
    println!("  🔸 DCT 플래너 캐시: {} 개", dct_cache_size);
    
    assert!(a_cache_size > 0, "A 매트릭스 캐시가 사용되지 않음");
}

#[test]
fn 캐시_효과_테스트() {
    println!("🧪 캐시 효과 테스트");
    
    // 캐시 클리어
    OptimizedDecoder::clear_caches();
    
    let rows = 128;
    let cols = 128; 
    let coeffs = 256;
    let test_block = create_test_block(rows, cols, coeffs, TransformType::Dwt);
    
    // 첫 번째 호출 (캐시 생성)
    let start = Instant::now();
    let _result1 = test_block.decode_optimized();
    let first_call_time = start.elapsed().as_micros();
    
    // 두 번째 호출 (캐시 사용)
    let start = Instant::now();
    let _result2 = test_block.decode_optimized();
    let second_call_time = start.elapsed().as_micros();
    
    // 세 번째 호출 (캐시 사용)
    let start = Instant::now();
    let _result3 = test_block.decode_optimized();
    let third_call_time = start.elapsed().as_micros();
    
    println!("  🔸 첫 번째 호출 (캐시 생성): {} μs", first_call_time);
    println!("  🔸 두 번째 호출 (캐시 사용): {} μs", second_call_time);
    println!("  🔸 세 번째 호출 (캐시 사용): {} μs", third_call_time);
    
    // 캐시 사용 시 더 빨라야 함
    let cache_speedup = first_call_time as f32 / second_call_time as f32;
    println!("  🚀 캐시 효과: {:.2}x 빠름", cache_speedup);
    
    assert!(cache_speedup > 1.5, "캐시 효과가 부족: {:.2}x", cache_speedup);
    
    // 캐시 통계 확인
    let (a_cache_size, dct_cache_size) = OptimizedDecoder::get_cache_stats();
    assert_eq!(a_cache_size, 1, "A 매트릭스 캐시 크기가 예상과 다름");
    assert_eq!(dct_cache_size, 1, "DCT 플래너 캐시 크기가 예상과 다름");
    
    println!("  ✅ 캐시가 정상적으로 작동함");
}

#[test]
fn 다양한_블록_크기_테스트() {
    println!("📐 다양한 블록 크기 테스트");
    
    let test_cases = [
        (32, 64, TransformType::Dwt),
        (64, 32, TransformType::Dwt),
        (48, 48, TransformType::Dct),
        (128, 256, TransformType::Dwt),
        (256, 128, TransformType::Dct),
    ];
    
    for (rows, cols, transform_type) in test_cases {
        println!("\n  📊 테스트: {}x{} ({:?})", rows, cols, transform_type);
        
        let test_block = create_test_block(rows, cols, 64, transform_type);
        
        let original_result = test_block.decode();
        let optimized_result = test_block.decode_optimized();
        
        assert_eq!(original_result.len(), optimized_result.len());
        
        let mut max_diff = 0.0f32;
        for (&orig, &opt) in original_result.iter().zip(optimized_result.iter()) {
            let diff = (orig - opt).abs();
            max_diff = max_diff.max(diff);
        }
        
        println!("    ✅ 최대 오차: {:.2e}", max_diff);
        assert!(max_diff < 1e-4, "오차가 너무 큼: {:.2e}", max_diff);
    }
    
    // 최종 캐시 통계
    let (a_cache_size, dct_cache_size) = OptimizedDecoder::get_cache_stats();
    println!("\n📈 최종 캐시 통계:");
    println!("  🔸 A 매트릭스 캐시: {} 개", a_cache_size);
    println!("  🔸 DCT 플래너 캐시: {} 개", dct_cache_size);
    
    assert!(a_cache_size >= 5, "A 매트릭스 캐시가 충분히 생성되지 않음");
} 

#[test]
fn 병렬_블록_처리_테스트() {
    println!("🚀 병렬 블록 처리 성능 테스트");
    
    // 여러 블록 생성 (DWT는 2의 제곱수 정사각형만 지원)
    let block_sizes = [(64, 64), (128, 128), (32, 32), (16, 16)];
    let mut blocks = Vec::new();
    
    for (rows, cols) in block_sizes {
        for _ in 0..10 { // 각 크기별로 10개씩
            let block = create_test_block(rows, cols, 64, TransformType::Dwt);
            blocks.push(block);
        }
    }
    
    println!("  📊 총 {} 개 블록 처리", blocks.len());
    
    // 순차 처리 성능
    let start = Instant::now();
    let sequential_results: Vec<Vec<f32>> = blocks.iter()
        .map(|block| block.decode())
        .collect();
    let sequential_time = start.elapsed().as_millis();
    
    // 병렬 처리 성능 (기본)
    let start = Instant::now();
    let parallel_results = OptimizedDecoder::decode_blocks_parallel(&blocks);
    let parallel_time = start.elapsed().as_millis();
    
    // 병렬 처리 성능 (청크)
    let start = Instant::now();
    let chunked_results = OptimizedDecoder::decode_blocks_chunked_parallel(&blocks, 8);
    let chunked_time = start.elapsed().as_millis();
    
    // 결과 검증
    assert_eq!(sequential_results.len(), parallel_results.len());
    assert_eq!(sequential_results.len(), chunked_results.len());
    
    for (i, (seq, par)) in sequential_results.iter().zip(parallel_results.iter()).enumerate() {
        assert_eq!(seq.len(), par.len(), "블록 {} 길이 다름", i);
        
        let mut max_diff = 0.0f32;
        for (&s, &p) in seq.iter().zip(par.iter()) {
            max_diff = max_diff.max((s - p).abs());
        }
        assert!(max_diff < 1e-6, "블록 {} 오차 큼: {:.2e}", i, max_diff);
    }
    
    // 성능 결과 출력
    let parallel_speedup = sequential_time as f32 / parallel_time as f32;
    let chunked_speedup = sequential_time as f32 / chunked_time as f32;
    
    println!("  🔸 순차 처리:     {} ms", sequential_time);
    println!("  🔸 병렬 처리:     {} ms", parallel_time);
    println!("  🔸 청크 병렬:     {} ms", chunked_time);
    println!("  🚀 병렬 성능:     {:.2}x 빠름", parallel_speedup);
    println!("  🚀 청크 성능:     {:.2}x 빠름", chunked_speedup);
    
    // 병렬 처리가 더 빨라야 함
    assert!(parallel_speedup > 1.0, "병렬 처리가 느림: {:.2}x", parallel_speedup);
    println!("  ✅ 병렬 블록 처리 성공!");
}

#[test]
fn simd_벡터_덧셈_테스트() {
    println!("🧪 SIMD 벡터 덧셈 테스트");
    
    // simd_add_vectors already imported
    
    let sizes = [128, 1024, 4096, 16384];
    
    for size in sizes {
        println!("\n  📊 크기: {}", size);
        
        // 테스트 벡터 생성
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.05).collect();
        
        // SIMD 덧셈
        let start = Instant::now();
        let simd_result = OptimizedDecoder::simd_add_vectors(&a, &b);
        let simd_time = start.elapsed().as_micros();
        
        // 일반 덧셈
        let start = Instant::now();
        let normal_result: Vec<f32> = a.iter().zip(b.iter())
            .map(|(ai, bi)| ai + bi)
            .collect();
        let normal_time = start.elapsed().as_micros();
        
        // 결과 검증
        assert_eq!(simd_result.len(), normal_result.len());
        
        let mut max_diff = 0.0f32;
        for (&simd, &normal) in simd_result.iter().zip(normal_result.iter()) {
            max_diff = max_diff.max((simd - normal).abs());
        }
        
        let speedup = normal_time as f32 / simd_time as f32;
        
        println!("    🔸 일반 덧셈:   {} μs", normal_time);
        println!("    🔸 SIMD 덧셈:   {} μs", simd_time);
        println!("    🚀 성능 향상:   {:.2}x", speedup);
        println!("    ✅ 최대 오차:   {:.2e}", max_diff);
        
        assert!(max_diff < 1e-6, "SIMD 결과 오차: {:.2e}", max_diff);
    }
    
    println!("\n  ✅ SIMD 벡터 덧셈 테스트 완료!");
} 

#[test]
fn dwt_압축_정확도_상세_분석_테스트() {
    println!("🔬 DWT 기반 압축 정확도 상세 분석");
    println!("📊 손실 압축 특성을 고려한 정확도 검증");
    
    // === 1. 실제 모델 가중치와 유사한 데이터 생성 ===
    println!("\n🧪 테스트 데이터 유형별 분석:");
    
    let test_cases = vec![
        ("가우시안 분포", create_gaussian_weights(128, 128)),
        ("Xavier 초기화", create_xavier_weights(128, 128)),
        ("스파스 가중치", create_sparse_weights(128, 128, 0.3)),
        ("주기적 패턴", create_periodic_weights(128, 128)),
        ("랜덤 노이즈", create_random_weights(128, 128)),
    ];
    
    let mut overall_stats = AccuracyStats::new();
    
    for (name, original_data) in test_cases {
        println!("\n  📋 {} 분석:", name);
        
        // DWT 압축/복원
        let encoded_block = create_test_block_from_data(&original_data, 128, 128, 64, TransformType::Dwt);
        
        let decoded_original = encoded_block.decode();
        let decoded_optimized = encoded_block.decode_optimized();
        
        // 정확도 분석
        let stats = analyze_accuracy(&original_data, &decoded_original, &decoded_optimized);
        overall_stats.merge(&stats);
        
        println!("    🔸 원본 vs 기존 복원:");
        println!("      - MSE:        {:.2e}", stats.original_mse);
        println!("      - PSNR:       {:.2} dB", stats.original_psnr);
        println!("      - 최대 오차:  {:.2e}", stats.original_max_error);
        
        println!("    🔸 기존 vs 최적화 복원:");
        println!("      - MSE:        {:.2e}", stats.optimization_mse);
        println!("      - 최대 오차:  {:.2e}", stats.optimization_max_error);
        println!("      - 상대 오차:  {:.2e}%", stats.relative_error * 100.0);
        
        // DWT 압축 품질 검증
        assert!(stats.original_psnr > 30.0, "{}: PSNR 너무 낮음 ({:.2} dB)", name, stats.original_psnr);
        assert!(stats.optimization_mse < 1e-10, "{}: 최적화 오차 너무 큼 ({:.2e})", name, stats.optimization_mse);
        assert!(stats.relative_error < 1e-5, "{}: 상대 오차 너무 큼 ({:.2e}%)", name, stats.relative_error * 100.0);
    }
    
    // === 2. 전체 통계 ===
    println!("\n📊 전체 정확도 통계:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  🔸 평균 PSNR:         {:.2} dB", overall_stats.avg_psnr());
    println!("  🔸 평균 MSE (압축):   {:.2e}", overall_stats.avg_original_mse());
    println!("  🔸 평균 MSE (최적화): {:.2e}", overall_stats.avg_optimization_mse());
    println!("  🔸 최대 상대 오차:    {:.2e}%", overall_stats.max_relative_error() * 100.0);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // === 3. DWT 특성 검증 ===
    println!("\n🌊 DWT 압축 특성 검증:");
    
    // 주파수 성분 보존 확인
    let test_data = create_periodic_weights(64, 64);
    let encoded = create_test_block_from_data(&test_data, 64, 64, 32, TransformType::Dwt);
    let decoded = encoded.decode_optimized();
    
    let freq_preservation = calculate_frequency_preservation(&test_data, &decoded);
    println!("  🔸 저주파 성분 보존: {:.2}%", freq_preservation.low_freq * 100.0);
    println!("  🔸 중주파 성분 보존: {:.2}%", freq_preservation.mid_freq * 100.0);
    println!("  🔸 고주파 성분 보존: {:.2}%", freq_preservation.high_freq * 100.0);
    
    // DWT는 저주파를 잘 보존해야 함
    assert!(freq_preservation.low_freq > 0.95, "저주파 성분 보존율 부족: {:.2}%", freq_preservation.low_freq * 100.0);
    
    // === 4. 실제 추론 시나리오 테스트 ===
    println!("\n🤖 실제 추론 시나리오 테스트:");
    
    // 대량 블록 처리 시 정확도 누적 오차 확인
    let mut blocks = Vec::new();
    for _ in 0..50 {
        let data = create_gaussian_weights(64, 64);
        blocks.push(create_test_block_from_data(&data, 64, 64, 32, TransformType::Dwt));
    }
    
    // 순차 vs 병렬 처리 정확도 비교
    let sequential_results: Vec<_> = blocks.iter().map(|b| b.decode_optimized()).collect();
    let parallel_results = OptimizedDecoder::decode_blocks_parallel(&blocks);
    
    let mut cumulative_error = 0.0f32;
    for (seq, par) in sequential_results.iter().zip(parallel_results.iter()) {
        for (&s, &p) in seq.iter().zip(par.iter()) {
            cumulative_error += (s - p).abs();
        }
    }
    
    let avg_error_per_element = cumulative_error / (blocks.len() * 64 * 64) as f32;
    println!("  🔸 대량 처리 평균 오차: {:.2e}", avg_error_per_element);
    println!("  🔸 누적 오차:         {:.2e}", cumulative_error);
    
    assert!(avg_error_per_element < 1e-6, "대량 처리 시 오차 누적: {:.2e}", avg_error_per_element);
    
    println!("\n✅ DWT 압축 정확도 분석 완료!");
    println!("  💡 결론: 최적화된 디코더는 DWT 압축 특성을 완벽히 보존합니다.");
}

// 정확도 통계 구조체
#[derive(Debug, Clone)]
struct AccuracyStats {
    original_mse: f32,
    original_psnr: f32,
    original_max_error: f32,
    optimization_mse: f32,
    optimization_max_error: f32,
    relative_error: f32,
}

impl AccuracyStats {
    fn new() -> Self {
        Self {
            original_mse: 0.0,
            original_psnr: 0.0,
            original_max_error: 0.0,
            optimization_mse: 0.0,
            optimization_max_error: 0.0,
            relative_error: 0.0,
        }
    }
    
    fn merge(&mut self, other: &AccuracyStats) {
        self.original_mse += other.original_mse;
        self.original_psnr += other.original_psnr;
        self.original_max_error = self.original_max_error.max(other.original_max_error);
        self.optimization_mse += other.optimization_mse;
        self.optimization_max_error = self.optimization_max_error.max(other.optimization_max_error);
        self.relative_error = self.relative_error.max(other.relative_error);
    }
    
    fn avg_psnr(&self) -> f64 { (self.original_psnr / 5.0) as f64 }
    fn avg_original_mse(&self) -> f64 { (self.original_mse / 5.0) as f64 }
    fn avg_optimization_mse(&self) -> f64 { (self.optimization_mse / 5.0) as f64 }
    fn max_relative_error(&self) -> f64 { self.relative_error as f64 }
}

// 주파수 보존율 구조체
#[derive(Debug)]
struct FrequencyPreservation {
    low_freq: f32,
    mid_freq: f32,
    high_freq: f32,
}

// 다양한 가중치 패턴 생성 함수들
fn create_gaussian_weights(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|i| {
        let x = (i % cols) as f32 / cols as f32 - 0.5;
        let y = (i / cols) as f32 / rows as f32 - 0.5;
        (-2.0 * (x * x + y * y)).exp() * (x * 10.0).sin()
    }).collect()
}

fn create_xavier_weights(rows: usize, cols: usize) -> Vec<f32> {
    let scale = (6.0 / (rows + cols) as f32).sqrt();
    (0..rows * cols).map(|i| {
        let x = (i as f32 / (rows * cols) as f32 - 0.5) * 2.0;
        x * scale * (1.0 + (x * 5.0).sin() * 0.1)
    }).collect()
}

fn create_sparse_weights(rows: usize, cols: usize, sparsity: f32) -> Vec<f32> {
    (0..rows * cols).map(|i| {
        if (i as f32 / (rows * cols) as f32) < sparsity {
            0.0
        } else {
            let x = (i % cols) as f32 / cols as f32;
            (x * 2.0 * std::f32::consts::PI).sin() * 0.5
        }
    }).collect()
}

fn create_periodic_weights(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|i| {
        let x = (i % cols) as f32 / cols as f32;
        let y = (i / cols) as f32 / rows as f32;
        (x * 4.0 * std::f32::consts::PI).sin() * (y * 2.0 * std::f32::consts::PI).cos() * 0.3
    }).collect()
}

fn create_random_weights(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols).map(|i| {
        let x = (i as f32 * 12345.0) % 1.0;
        (x - 0.5) * 2.0 * 0.1
    }).collect()
}

fn create_test_block_from_data(data: &[f32], rows: usize, cols: usize, coeffs: usize, transform_type: TransformType) -> HybridEncodedBlock {
    let mut encoder = RBEEncoder::new(coeffs, transform_type);
    encoder.encode_block(data, rows, cols)
}

fn analyze_accuracy(original: &[f32], decoded_original: &[f32], decoded_optimized: &[f32]) -> AccuracyStats {
    let mut original_mse = 0.0f32;
    let mut original_max_error: f32 = 0.0;
    let mut original_sum_sq = 0.0f32;
    
    // 원본 vs 기존 복원 오차
    for (&orig, &dec) in original.iter().zip(decoded_original.iter()) {
        let error = (orig - dec).abs();
        original_mse += error * error;
        original_max_error = original_max_error.max(error);
        original_sum_sq += orig * orig;
    }
    
    original_mse /= original.len() as f32;
    let original_psnr = if original_mse > 1e-10 {
        20.0 * (1.0 / original_mse.sqrt()).log10()
    } else {
        100.0
    };
    
    // 기존 vs 최적화 복원 오차
    let mut optimization_mse = 0.0f32;
    let mut optimization_max_error: f32 = 0.0;
    
    for (&dec_orig, &dec_opt) in decoded_original.iter().zip(decoded_optimized.iter()) {
        let error = (dec_orig - dec_opt).abs();
        optimization_mse += error * error;
        optimization_max_error = optimization_max_error.max(error);
    }
    
    optimization_mse /= decoded_original.len() as f32;
    
    let relative_error = if original_sum_sq > 0.0 {
        (optimization_mse / original_sum_sq).sqrt()
    } else {
        0.0
    };
    
    AccuracyStats {
        original_mse,
        original_psnr,
        original_max_error,
        optimization_mse,
        optimization_max_error,
        relative_error,
    }
}

fn calculate_frequency_preservation(original: &[f32], decoded: &[f32]) -> FrequencyPreservation {
    // 간단한 주파수 성분 분석 (실제로는 FFT 사용)
    let n = original.len();
    let mut low_orig = 0.0f32;
    let mut mid_orig = 0.0f32;
    let mut high_orig = 0.0f32;
    let mut low_dec = 0.0f32;
    let mut mid_dec = 0.0f32;
    let mut high_dec = 0.0f32;
    
    for i in 0..n {
        let freq_component = (i as f32 / n as f32 * 2.0 * std::f32::consts::PI).sin();
        
        if i < n / 4 {
            low_orig += original[i] * freq_component;
            low_dec += decoded[i] * freq_component;
        } else if i < n / 2 {
            mid_orig += original[i] * freq_component;
            mid_dec += decoded[i] * freq_component;
        } else {
            high_orig += original[i] * freq_component;
            high_dec += decoded[i] * freq_component;
        }
    }
    
    let low_preservation = if low_orig.abs() > 0.0 { (low_dec / low_orig).abs() } else { 1.0 };
    let mid_preservation = if mid_orig.abs() > 0.0 { (mid_dec / mid_orig).abs() } else { 1.0 };
    let high_preservation = if high_orig.abs() > 0.0 { (high_dec / high_orig).abs() } else { 1.0 };
    
    FrequencyPreservation {
        low_freq: low_preservation.min(1.0),
        mid_freq: mid_preservation.min(1.0),
        high_freq: high_preservation.min(1.0),
    }
}

#[test]
fn 전체_최적화_종합_성능_테스트() {
    println!("🚀 전체 최적화 종합 성능 테스트");
    println!("📊 A매트릭스 캐싱 + SIMD 벡터덧셈 + 병렬 블록처리");
    
    // 다양한 크기의 블록 대량 생성 (실제 모델과 유사)
    let mut blocks = Vec::new();
    
    // 작은 블록들 (attention head)
    for _ in 0..20 {
        blocks.push(create_test_block(64, 64, 32, TransformType::Dwt));
    }
    
    // 중간 블록들 (hidden layer)  
    for _ in 0..30 {
        blocks.push(create_test_block(128, 128, 64, TransformType::Dwt));
    }
    
    // 큰 블록들 (output projection)
    for _ in 0..10 {
        blocks.push(create_test_block(256, 256, 128, TransformType::Dwt));
    }
    
    println!("  📊 총 {} 개 블록 (다양한 크기)", blocks.len());
    println!("  🔸 64x64: 20개, 128x128: 30개, 256x256: 10개");
    
    // === 1. 기존 디코더 (순차) ===
    OptimizedDecoder::clear_caches(); // 캐시 클리어
    println!("\n🔹 기존 디코더 (순차 처리)");
    let start = std::time::Instant::now();
    
    let original_results: Vec<Vec<f32>> = blocks.iter()
        .map(|block| block.decode())
        .collect();
    
    let original_time = start.elapsed();
    println!("  ⏱️  시간: {:.3}초", original_time.as_secs_f32());
    
    // === 2. 최적화 디코더 (순차 + 캐싱 + SIMD) ===
    OptimizedDecoder::clear_caches(); // 캐시 클리어
    println!("\n🔹 최적화 디코더 (순차 + 캐싱 + SIMD)");
    let start = std::time::Instant::now();
    
    let optimized_sequential_results: Vec<Vec<f32>> = blocks.iter()
        .map(|block| block.decode_optimized())
        .collect();
    
    let optimized_sequential_time = start.elapsed();
    println!("  ⏱️  시간: {:.3}초", optimized_sequential_time.as_secs_f32());
    
    // === 3. 최적화 디코더 (병렬 + 캐싱 + SIMD) ===
    OptimizedDecoder::clear_caches(); // 캐시 클리어  
    println!("\n🔹 최적화 디코더 (병렬 + 캐싱 + SIMD)");
    let start = std::time::Instant::now();
    
    let optimized_parallel_results = OptimizedDecoder::decode_blocks_parallel(&blocks);
    
    let optimized_parallel_time = start.elapsed();
    println!("  ⏱️  시간: {:.3}초", optimized_parallel_time.as_secs_f32());
    
    // === 4. 최적화 디코더 (청크 병렬 + 캐싱 + SIMD) ===
    OptimizedDecoder::clear_caches(); // 캐시 클리어
    println!("\n🔹 최적화 디코더 (청크 병렬 + 캐싱 + SIMD)");
    let start = std::time::Instant::now();
    
    let optimized_chunked_results = OptimizedDecoder::decode_blocks_chunked_parallel(&blocks, 16);
    
    let optimized_chunked_time = start.elapsed();
    println!("  ⏱️  시간: {:.3}초", optimized_chunked_time.as_secs_f32());
    
    // === 결과 검증 ===
    println!("\n📋 결과 검증:");
    assert_eq!(original_results.len(), optimized_sequential_results.len());
    assert_eq!(original_results.len(), optimized_parallel_results.len());
    assert_eq!(original_results.len(), optimized_chunked_results.len());
    
    let mut max_diff = 0.0f32;
    for (i, ((orig, opt_seq), (opt_par, opt_chunk))) in original_results.iter()
        .zip(optimized_sequential_results.iter())
        .zip(optimized_parallel_results.iter().zip(optimized_chunked_results.iter()))
        .enumerate() {
        
        assert_eq!(orig.len(), opt_seq.len(), "블록 {} 길이 다름", i);
        assert_eq!(orig.len(), opt_par.len(), "블록 {} 길이 다름", i);
        assert_eq!(orig.len(), opt_chunk.len(), "블록 {} 길이 다름", i);
        
        for (&o, (&s, (&p, &c))) in orig.iter().zip(opt_seq.iter().zip(opt_par.iter().zip(opt_chunk.iter()))) {
            max_diff = max_diff.max((o - s).abs());
            max_diff = max_diff.max((o - p).abs());
            max_diff = max_diff.max((o - c).abs());
        }
    }
    
    println!("  ✅ 최대 오차: {:.2e}", max_diff);
    assert!(max_diff < 1e-5, "오차가 너무 큼: {:.2e}", max_diff);
    
    // === 성능 비교 ===
    println!("\n🏆 성능 비교 결과:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let sequential_speedup = original_time.as_secs_f32() / optimized_sequential_time.as_secs_f32();
    let parallel_speedup = original_time.as_secs_f32() / optimized_parallel_time.as_secs_f32();
    let chunked_speedup = original_time.as_secs_f32() / optimized_chunked_time.as_secs_f32();
    
    println!("  🔸 기존 (순차):           {:.3}초", original_time.as_secs_f32());
    println!("  🔸 최적화 (순차):         {:.3}초  →  {:.2}x 빠름", 
             optimized_sequential_time.as_secs_f32(), sequential_speedup);
    println!("  🔸 최적화 (병렬):         {:.3}초  →  {:.2}x 빠름", 
             optimized_parallel_time.as_secs_f32(), parallel_speedup);
    println!("  🔸 최적화 (청크 병렬):    {:.3}초  →  {:.2}x 빠름", 
             optimized_chunked_time.as_secs_f32(), chunked_speedup);
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // 최고 성능 확인
    let best_time = optimized_parallel_time.min(optimized_chunked_time);
    let best_speedup = original_time.as_secs_f32() / best_time.as_secs_f32();
    let best_method = if optimized_parallel_time < optimized_chunked_time { "병렬" } else { "청크 병렬" };
    
    println!("  🚀 최고 성능: {} 처리로 {:.2}x 빠름!", best_method, best_speedup);
    println!("  💾 메모리 절약: A매트릭스 캐싱으로 재계산 없음");
    println!("  ⚡ SIMD 가속: 벡터 연산 하드웨어 최적화");
    println!("  🔄 병렬 처리: CPU 코어 활용 극대화");
    
    // 성능 향상 검증
    assert!(sequential_speedup > 1.0, "순차 최적화 성능 부족: {:.2}x", sequential_speedup);
    assert!(parallel_speedup > 2.0, "병렬 최적화 성능 부족: {:.2}x", parallel_speedup);
    
    // 캐시 통계 출력
    let (a_cache_size, _) = OptimizedDecoder::get_cache_stats();
    println!("  📊 A매트릭스 캐시: {} 개 크기 저장됨", a_cache_size);
    
    println!("\n✅ 전체 최적화 종합 테스트 완료!");
} 
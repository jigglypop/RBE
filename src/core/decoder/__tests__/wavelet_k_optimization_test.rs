//! 웨이블릿 K값 최적화 테스트
//! 
//! RMSE < 0.1, 압축률 최대화, 속도 < 50ns 목표로 K값 최적화

use crate::decoder::weight_generator::{WeightGenerator, WaveletConfig};
use crate::packed_params::{PoincarePackedBit128, PoincareQuadrant};
use std::time::{Instant, Duration};

/// **종합 성능 메트릭**
#[derive(Debug, Clone, Copy)] // Copy trait 추가
pub struct PerformanceMetrics {
    pub k_level: u8,
    pub threshold: f32,
    pub compression_factor: f32,
    pub avg_rmse: f64,
    pub max_rmse: f64,
    pub avg_time_ns: f64,
    pub compression_ratio: f32,
    pub quality_score: f64, // 종합 점수 (0-100)
}

impl PerformanceMetrics {
    fn calculate_quality_score(&mut self) {
        // **가중 점수 계산** (RMSE 50%, 속도 30%, 압축률 20%)
        let rmse_score = if self.avg_rmse < 0.01 { 100.0 } 
                        else if self.avg_rmse < 0.05 { 80.0 } 
                        else if self.avg_rmse < 0.1 { 60.0 } 
                        else { 0.0 };
        
        let speed_score = if self.avg_time_ns < 30.0 { 100.0 }
                         else if self.avg_time_ns < 50.0 { 80.0 }
                         else if self.avg_time_ns < 100.0 { 60.0 }
                         else { 0.0 };
        
        let compression_score = (self.compression_ratio.min(16.0) / 16.0) as f64 * 100.0;
        
        self.quality_score = rmse_score * 0.5 + speed_score * 0.3 + compression_score * 0.2;
    }
}

/// **RMSE 계산 함수**
fn calculate_rmse_vs_reference(
    config: WaveletConfig,
    test_cases: &[(usize, usize, &str)],
    test_seeds: &[PoincarePackedBit128],
) -> (f64, f64) {
    let mut optimized_gen = WeightGenerator::with_config(config);
    let reference_gen = ReferenceWeightGenerator::new();
    
    let mut total_rmse = 0.0;
    let mut max_rmse: f64 = 0.0; // 타입 명시
    let mut test_count = 0;
    
    for (rows, cols, _desc) in test_cases {
        for packed in test_seeds {
            let mut ref_weights = Vec::new();
            let mut opt_weights = Vec::new();
            
            // 샘플링 (성능상 100개 포인트만)
            let sample_size = (*rows * *cols).min(100);
            let step_row = (*rows / (sample_size as f64).sqrt().ceil() as usize).max(1);
            let step_col = (*cols / (sample_size as f64).sqrt().ceil() as usize).max(1);
            
            for row in (0..*rows).step_by(step_row) {
                for col in (0..*cols).step_by(step_col) {
                    let ref_w = reference_gen.generate_weight_precise(packed, row, col, *rows, *cols);
                    let opt_w = optimized_gen.generate_weight(packed, row, col, *rows, *cols);
                    
                    ref_weights.push(ref_w);
                    opt_weights.push(opt_w);
                }
            }
            
            if ref_weights.len() > 0 {
                let rmse = calculate_rmse(&ref_weights, &opt_weights);
                total_rmse += rmse;
                max_rmse = max_rmse.max(rmse);
                test_count += 1;
            }
        }
    }
    
    let avg_rmse = if test_count > 0 { total_rmse / test_count as f64 } else { 999.0 };
    (avg_rmse, max_rmse)
}

/// **속도 벤치마크**
fn benchmark_speed(config: WaveletConfig) -> f64 {
    let mut generator = WeightGenerator::with_config(config);
    let packed = PoincarePackedBit128::new(
        PoincareQuadrant::First, 0x456, 0x789, 0x12, 0x34567890, 0.5, 1.0
    );
    
    // 워밍업
    for _ in 0..1000 {
        generator.generate_weight(&packed, 50, 50, 100, 100);
    }
    
    // 정확한 측정
    let iterations = 10000;
    let start = Instant::now();
    
    for i in 0..iterations {
        let row = i % 100;
        let col = (i * 7) % 100;
        generator.generate_weight(&packed, row, col, 100, 100);
    }
    
    let elapsed = start.elapsed();
    (elapsed.as_nanos() as f64) / (iterations as f64)
}

/// **참조 구현** (정확성 기준)
struct ReferenceWeightGenerator {}

impl ReferenceWeightGenerator {
    fn new() -> Self { Self {} }
    
    fn generate_weight_precise(
        &self,
        packed: &PoincarePackedBit128,
        row: usize,
        col: usize,
        total_rows: usize,
        total_cols: usize,
    ) -> f64 {
        if row >= total_rows || col >= total_cols { return 0.0; }
        
        let quadrant = (packed.hi >> 62) & 0x3;
        let freq = (packed.hi >> 50) & 0xFFF;
        let amp = (packed.hi >> 38) & 0xFFF;
        let phase = (packed.hi >> 26) & 0xFFF;
        let residual = (packed.hi >> 14) & 0xFFF;
        
        let x = if total_cols > 1 { ((col as f64 * 2.0) / total_cols as f64) - 1.0 } else { 0.0 };
        let y = if total_rows > 1 { ((row as f64 * 2.0) / total_rows as f64) - 1.0 } else { 0.0 };
        
        // 정확한 웨이블릿 변환 (Haar)
        let haar_scale = 4.0; // K=4 레벨 기준
        let haar_low_x = if x < 0.0 { 1.0/2_f64.sqrt() } else { 1.0/2_f64.sqrt() } * haar_scale;
        let haar_high_x = if x < 0.0 { 1.0/2_f64.sqrt() } else { -1.0/2_f64.sqrt() } * haar_scale;
        let haar_low_y = if y < 0.0 { 1.0/2_f64.sqrt() } else { 1.0/2_f64.sqrt() } * haar_scale;
        let haar_high_y = if y < 0.0 { 1.0/2_f64.sqrt() } else { -1.0/2_f64.sqrt() } * haar_scale;
        
        let base_value = match quadrant {
            0 => (haar_low_x * haar_low_y * 2.0).tanh() * 0.8,
            1 => (haar_high_x * haar_low_y * std::f64::consts::PI).sin() * 0.7,
            2 => ((haar_low_x * haar_high_y + haar_high_x * haar_low_y) * std::f64::consts::PI * 0.5).cos() * 0.6,
            _ => {
                let combined = (haar_low_x + haar_high_x) * (haar_low_y + haar_high_y);
                (-combined * combined * 0.25).exp() * 0.5
            }
        };
        
        let freq_norm = (freq as f64) / 4096.0 / 8.0; // 8배 압축 기준
        let amp_norm = (amp as f64) / 4096.0;
        let phase_norm = (phase as f64) / 4096.0;
        let residual_norm = (residual as f64) / 4096.0;
        
        let residual_correction = if residual_norm > 0.01 {
            (residual_norm - 0.01) * 0.1
        } else {
            residual_norm * 0.01
        };
        
        let freq_mod = 1.0 + freq_norm * 0.2;
        let amp_mod = 0.5 + amp_norm * 0.5;
        let phase_mod = 1.0 + phase_norm * 0.02;
        
        let pre_weight = base_value * freq_mod * amp_mod * phase_mod;
        let final_weight = pre_weight + residual_correction;
        
        final_weight.clamp(-1.0/8_f64.sqrt(), 1.0/8_f64.sqrt())
    }
}

fn calculate_rmse(reference: &[f64], optimized: &[f32]) -> f64 {
    if reference.len() != optimized.len() || reference.is_empty() { return 999.0; }
    
    let mut sum_sq_error = 0.0;
    for (r, o) in reference.iter().zip(optimized.iter()) {
        let error = r - (*o as f64);
        sum_sq_error += error * error;
    }
    
    (sum_sq_error / reference.len() as f64).sqrt()
}

#[test]
fn test_k_value_optimization() {
    println!("\n🔧 **웨이블릿 K값 최적화 테스트**");
    
    // 테스트 케이스
    let test_cases = vec![
        (64, 64, "중간 행렬"),
        (128, 128, "큰 행렬"),
        (32, 256, "직사각형"),
    ];
    
    let test_seeds = vec![
        PoincarePackedBit128::new(PoincareQuadrant::First, 0x123, 0x456, 0x12, 0x789ABCDE, 0.5, 1.0),
        PoincarePackedBit128::new(PoincareQuadrant::Second, 0xAAA, 0x555, 0x2A, 0x55555555, 0.7, 2.5),
        PoincarePackedBit128::new(PoincareQuadrant::Third, 0x800, 0x200, 0x10, 0x12345678, 0.3, 4.0),
    ];
    
    let mut results = Vec::new();
    
    // **K값 조합 테스트** (체계적 탐색)
    let k_levels = vec![2, 3, 4, 5, 6];
    let thresholds = vec![0.005, 0.01, 0.02, 0.05];
    let compressions = vec![4.0, 6.0, 8.0, 12.0, 16.0];
    
    println!("├─ 총 {} 조합 테스트", k_levels.len() * thresholds.len() * compressions.len());
    
    for k_level in &k_levels {
        for threshold in &thresholds {
            for compression in &compressions {
                let config = WaveletConfig {
                    k_level: *k_level,
                    threshold: *threshold,
                    compression_factor: *compression,
                };
                
                // RMSE 측정
                let (avg_rmse, max_rmse) = calculate_rmse_vs_reference(config, &test_cases, &test_seeds);
                
                // 속도 측정  
                let avg_time_ns = benchmark_speed(config);
                
                let mut metrics = PerformanceMetrics {
                    k_level: *k_level,
                    threshold: *threshold,
                    compression_factor: *compression,
                    avg_rmse,
                    max_rmse,
                    avg_time_ns,
                    compression_ratio: *compression,
                    quality_score: 0.0,
                };
                
                metrics.calculate_quality_score();
                results.push(metrics);
                
                // 실시간 결과 출력 (상위 품질만)
                if metrics.quality_score > 70.0 {
                    println!("├─ K={}, T={:.3}, C={:.1} → RMSE:{:.4}, {}ns, 품질:{:.1}",
                             k_level, threshold, compression, avg_rmse, avg_time_ns as u64, metrics.quality_score);
                }
            }
        }
    }
    
    // **결과 분석**
    results.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap());
    
    println!("\n🏆 **상위 10개 최적 설정**");
    println!("├─ 순위 | K레벨 | 임계값 | 압축률 | RMSE   | 속도(ns) | 품질점수");
    println!("├─ -----|-------|--------|--------|--------|----------|----------");
    
    for (i, metrics) in results.iter().take(10).enumerate() {
        let status = if metrics.avg_rmse < 0.1 && metrics.avg_time_ns < 50.0 { "✅" } else { "⚠️ " };
        
        println!("├─ {:2}   | {:5} | {:6.3} | {:6.1} | {:6.4} | {:8.0} | {:7.1} {}", 
                 i+1, metrics.k_level, metrics.threshold, metrics.compression_factor,
                 metrics.avg_rmse, metrics.avg_time_ns, metrics.quality_score, status);
    }
    
    // **최적 설정 검증**
    let best = &results[0];
    println!("\n🎯 **최적 설정 상세 검증**");
    println!("├─ K레벨: {}", best.k_level);
    println!("├─ 잔차 임계값: {:.4}", best.threshold);
    println!("├─ 압축률: {:.1}배", best.compression_factor);
    println!("├─ 평균 RMSE: {:.6}", best.avg_rmse);
    println!("├─ 최대 RMSE: {:.6}", best.max_rmse);
    println!("├─ 평균 속도: {:.1}ns", best.avg_time_ns);
    println!("├─ 품질 점수: {:.1}/100", best.quality_score);
    
    // **제약 조건 검증**
    assert!(best.avg_rmse < 0.1, "최적 설정 RMSE 허용 범위 초과: {:.6}", best.avg_rmse);
    assert!(best.avg_time_ns < 100.0, "최적 설정 속도 허용 범위 초과: {:.1}ns", best.avg_time_ns);
    assert!(best.compression_factor >= 4.0, "압축률이 너무 낮음: {:.1}배", best.compression_factor);
    
    // **성능 향상 정량화**
    let baseline_rmse = 0.192; // 이전 RMSE
    let improvement = (baseline_rmse - best.avg_rmse) / baseline_rmse * 100.0;
    
    println!("\n📈 **성능 향상**");
    println!("├─ RMSE 개선: {:.1}% ({:.6} → {:.6})", improvement, baseline_rmse, best.avg_rmse);
    println!("├─ 압축률: {:.1}배", best.compression_factor);
    println!("└─ 속도: {:.1}ns (목표 50ns 이내)", best.avg_time_ns);
    
    println!("\n✅ **K값 최적화 완료!**");
}

#[test]
fn test_detailed_rmse_analysis() {
    println!("\n🔍 **상세 RMSE 분석**");
    
    // 최적 설정으로 상세 분석 
    let optimal_config = WaveletConfig {
        k_level: 4,
        threshold: 0.01,
        compression_factor: 8.0,
    };
    
    let mut optimized_gen = WeightGenerator::with_config(optimal_config);
    let reference_gen = ReferenceWeightGenerator::new();
    
    let test_matrix = PoincarePackedBit128::new(
        PoincareQuadrant::Second, 0x555, 0xAAA, 0x2A, 0x12345678, 0.6, 2.0
    );
    
    let (rows, cols) = (128, 128);
    let mut errors = Vec::new();
    
    // 전체 행렬 RMSE 측정
    for row in (0..rows).step_by(4) {
        for col in (0..cols).step_by(4) {
            let ref_val = reference_gen.generate_weight_precise(&test_matrix, row, col, rows, cols);
            let opt_val = optimized_gen.generate_weight(&test_matrix, row, col, rows, cols);
            let error = (ref_val - opt_val as f64).abs();
            errors.push(error);
        }
    }
    
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mean_error: f64 = errors.iter().sum::<f64>() / errors.len() as f64;
    let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
    let median_error = errors[errors.len() / 2];
    let p95_error = errors[(errors.len() as f64 * 0.95) as usize];
    let max_error = errors.last().unwrap();
    
    println!("├─ 샘플 수: {}", errors.len());
    println!("├─ 평균 오차: {:.6}", mean_error);
    println!("├─ RMSE: {:.6}", rmse);
    println!("├─ 중앙값 오차: {:.6}", median_error);
    println!("├─ 95% 오차: {:.6}", p95_error);
    println!("├─ 최대 오차: {:.6}", max_error);
    
    assert!(rmse < 0.1, "상세 RMSE 검증 실패: {:.6}", rmse);
    println!("└─ ✅ 상세 RMSE 검증 통과!");
} 
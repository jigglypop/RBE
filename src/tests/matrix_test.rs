use crate::matrix::{
    HierarchicalBlockMatrix, QualityLevel, ErrorController
};
use std::time::Instant;

/// 6.2.1 품질 등급 테스트
#[test]
fn test_quality_levels() {
    println!("=== 품질 등급 테스트 ===");
    
    let quality_levels = [
        QualityLevel::Ultra,
        QualityLevel::High,
        QualityLevel::Medium,
        QualityLevel::Low,
    ];
    
    for quality in &quality_levels {
        println!("품질 등급: {:?}", quality);
        println!("  최적 블록 크기: {}×{}", quality.optimal_block_size(), quality.optimal_block_size());
        println!("  목표 PSNR: {:.1} dB", quality.target_psnr());
        println!("  압축률: {:.0}:1", quality.compression_ratio());
        
        // 블록 크기가 유효한 범위인지 확인
        let block_size = quality.optimal_block_size();
        assert!(block_size >= 32 && block_size <= 256, "블록 크기가 유효 범위를 벗어남: {}", block_size);
        
        // PSNR이 합리적인 범위인지 확인
        let psnr = quality.target_psnr();
        assert!(psnr >= 20.0 && psnr <= 60.0, "PSNR이 합리적 범위를 벗어남: {}", psnr);
    }
    
    println!("품질 등급 테스트 성공!");
}

/// 6.2.4 오차 제어 시스템 테스트
#[test]
fn test_error_controller() {
    println!("=== 오차 제어 시스템 테스트 ===");
    
    let mut error_controller = ErrorController::new(1e-3);
    
    // 블록 오차 업데이트
    error_controller.update_block_error((0, 0), 0.001);
    error_controller.update_block_error((0, 1), 0.002);
    error_controller.update_block_error((1, 0), 0.0005);
    error_controller.update_block_error((1, 1), 0.003);
    
    // 전체 오차 계산 검증
    let total_error = error_controller.compute_total_error();
    println!("전체 오차: {:.6}", total_error);
    
    // 오차가 합리적인 범위인지 확인
    assert!(total_error > 0.0 && total_error < 1.0, "전체 오차가 비정상적: {}", total_error);
    
    // 분할 필요성 판단 테스트
    let should_subdivide_high_error = error_controller.should_subdivide((1, 1), 1); // 높은 오차
    let should_subdivide_low_error = error_controller.should_subdivide((1, 0), 1);  // 낮은 오차
    let should_subdivide_max_depth = error_controller.should_subdivide((0, 0), 4);  // 최대 깊이
    
    assert!(should_subdivide_high_error, "높은 오차 블록이 분할되지 않음");
    assert!(!should_subdivide_low_error, "낮은 오차 블록이 불필요하게 분할됨");
    assert!(!should_subdivide_max_depth, "최대 깊이에서 분할이 허용됨");
    
    println!("분할 판단 테스트:");
    println!("  높은 오차 블록 분할: {}", should_subdivide_high_error);
    println!("  낮은 오차 블록 분할: {}", should_subdivide_low_error);
    println!("  최대 깊이 분할: {}", should_subdivide_max_depth);
    
    println!("오차 제어 시스템 테스트 성공!");
}

/// 6.2.2 적응적 블록 분할 테스트
#[test]
fn test_adaptive_block_partition() {
    println!("=== 적응적 블록 분할 테스트 ===");
    
    let rows = 512;
    let cols = 512;
    
    // 테스트용 소스 행렬 생성 (중심에서 바깥으로 방사형 패턴)
    let mut source_matrix = vec![0.0; rows * cols];
    let center_x = cols as f32 / 2.0;
    let center_y = rows as f32 / 2.0;
    
    for i in 0..rows {
        for j in 0..cols {
            let dx = j as f32 - center_x;
            let dy = i as f32 - center_y;
            let distance = (dx * dx + dy * dy).sqrt();
            let value = (distance / (rows as f32 / 4.0)).sin() * 0.5 + 0.5;
            source_matrix[i * cols + j] = value;
        }
    }
    
    // High 품질로 블록 분할 테스트
    let mut block_matrix = HierarchicalBlockMatrix::new(rows, cols, QualityLevel::High);
    
    let start_time = Instant::now();
    block_matrix.adaptive_partition(&source_matrix);
    let partition_time = start_time.elapsed();
    
    println!("분할 완료 시간: {:.2} ms", partition_time.as_secs_f64() * 1000.0);
    
    // 분할 결과 검증
    assert!(!block_matrix.l1_blocks.is_empty(), "L1 블록이 생성되지 않음");
    
    let (memory_bytes, compression_ratio) = block_matrix.memory_usage();
    println!("메모리 사용량: {:.2} KB", memory_bytes as f32 / 1024.0);
    println!("압축률: {:.1}:1", compression_ratio);
    
    // 압축률이 합리적인지 확인
    assert!(compression_ratio > 10.0, "압축률이 너무 낮음: {:.1}", compression_ratio);
    assert!(compression_ratio < 10000.0, "압축률이 너무 높음: {:.1}", compression_ratio);
    
    // 품질 통계 출력
    let quality_stats = block_matrix.quality_statistics();
    quality_stats.print_report();
    
    println!("적응적 블록 분할 테스트 성공!");
}

/// 6.4 병렬 GEMV 연산 테스트
#[test]
fn test_parallel_gemv() {
    println!("=== 병렬 GEMV 연산 테스트 ===");
    
    let rows = 256;
    let cols = 256;
    
    // 간단한 대각선 패턴으로 소스 행렬 생성
    let mut source_matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            if i == j {
                source_matrix[i * cols + j] = 1.0; // 대각선
            } else if (i as i32 - j as i32).abs() == 1 {
                source_matrix[i * cols + j] = 0.5; // 대각선 근처
            } else {
                source_matrix[i * cols + j] = 0.1; // 나머지
            }
        }
    }
    
    // Medium 품질로 블록 행렬 생성
    let mut block_matrix = HierarchicalBlockMatrix::new(rows, cols, QualityLevel::Medium);
    block_matrix.adaptive_partition(&source_matrix);
    
    // 입력 벡터 생성
    let input = (0..cols).map(|i| (i as f32 + 1.0) / cols as f32).collect::<Vec<f32>>();
    let mut output = vec![0.0; rows];
    
    // 병렬 GEMV 실행
    let start_time = Instant::now();
    block_matrix.parallel_gemv(&input, &mut output, 4);
    let gemv_time = start_time.elapsed();
    
    println!("병렬 GEMV 실행 시간: {:.2} ms", gemv_time.as_secs_f64() * 1000.0);
    
    // 결과 검증
    let mut non_zero_count = 0;
    let mut max_value: f32 = 0.0;
    let mut min_value: f32 = f32::INFINITY;
    
    for &value in &output {
        if value.abs() > 1e-6 {
            non_zero_count += 1;
        }
        max_value = max_value.max(value);
        min_value = min_value.min(value);
    }
    
    println!("출력 통계:");
    println!("  0이 아닌 값 개수: {}/{}", non_zero_count, output.len());
    println!("  최대값: {:.6}", max_value);
    println!("  최소값: {:.6}", min_value);
    
    // 결과가 합리적인지 검증
    assert!(non_zero_count > 0, "모든 출력이 0임");
    assert!(max_value.is_finite() && min_value.is_finite(), "출력에 무한대 값이 포함됨");
    assert!(max_value > min_value, "출력 범위가 비정상적");
    
    println!("병렬 GEMV 연산 테스트 성공!");
}

/// 6.3 메모리 효율성 테스트
#[test]
fn test_memory_efficiency() {
    println!("=== 메모리 효율성 테스트 ===");
    
    let test_sizes = [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ];
    
    for &(rows, cols) in &test_sizes {
        println!("\n--- {}×{} 행렬 테스트 ---", rows, cols);
        
        // 무작위 패턴으로 소스 행렬 생성
        let mut source_matrix = vec![0.0; rows * cols];
        for i in 0..rows * cols {
            source_matrix[i] = ((i * 31) % 100) as f32 / 100.0;
        }
        
        // 여러 품질 수준 테스트
        let qualities = [QualityLevel::High, QualityLevel::Medium, QualityLevel::Low];
        
        for &quality in &qualities {
            let mut block_matrix = HierarchicalBlockMatrix::new(rows, cols, quality);
            
            let partition_start = Instant::now();
            block_matrix.adaptive_partition(&source_matrix);
            let partition_time = partition_start.elapsed();
            
            let (memory_bytes, compression_ratio) = block_matrix.memory_usage();
            let quality_stats = block_matrix.quality_statistics();
            
            println!("품질 {:?}:", quality);
            println!("  분할 시간: {:.2} ms", partition_time.as_secs_f64() * 1000.0);
            println!("  메모리: {:.2} KB", memory_bytes as f32 / 1024.0);
            println!("  압축률: {:.1}:1", compression_ratio);
            println!("  PSNR: {:.1} dB", quality_stats.psnr);
            println!("  블록 수: {}", quality_stats.total_blocks);
            
            // 성능 기준 검증
            let original_memory = rows * cols * 4; // f32 크기
            let memory_savings = 1.0 - (memory_bytes as f32 / original_memory as f32);
            
            assert!(memory_savings > 0.5, "메모리 절약률이 50% 미만: {:.1}%", memory_savings * 100.0);
            assert!(compression_ratio > 10.0, "압축률이 너무 낮음: {:.1}", compression_ratio);
            
            println!("  메모리 절약률: {:.1}%", memory_savings * 100.0);
        }
    }
    
    println!("\n메모리 효율성 테스트 성공!");
}

/// 6.5 성능 벤치마크 테스트
#[test]
fn test_performance_benchmark() {
    println!("=== 성능 벤치마크 테스트 ===");
    
    let matrix_sizes = [(512, 512)]; // 테스트 속도를 위해 크기 제한
    
    for &(rows, cols) in &matrix_sizes {
        println!("\n--- {}×{} 성능 벤치마크 ---", rows, cols);
        
        // 복잡한 패턴으로 테스트 행렬 생성
        let mut source_matrix = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let x = i as f32 / rows as f32;
                let y = j as f32 / cols as f32;
                let value = (x * 10.0).sin() * (y * 8.0).cos() + 
                           (x * 5.0 + y * 3.0).sin() * 0.5;
                source_matrix[i * cols + j] = value;
            }
        }
        
        // 벤치마크 실행
        for &quality in &[QualityLevel::High, QualityLevel::Medium] {
            println!("품질 {:?} 벤치마크:", quality);
            
            let mut block_matrix = HierarchicalBlockMatrix::new(rows, cols, quality);
            
            // 분할 성능 측정
            let partition_start = Instant::now();
            block_matrix.adaptive_partition(&source_matrix);
            let partition_time = partition_start.elapsed();
            
            // GEMV 성능 측정
            let input = vec![1.0; cols];
            let mut output = vec![0.0; rows];
            
            let gemv_start = Instant::now();
            block_matrix.parallel_gemv(&input, &mut output, 4);
            let gemv_time = gemv_start.elapsed();
            
            // 결과 통계
            let (memory_bytes, compression_ratio) = block_matrix.memory_usage();
            let quality_stats = block_matrix.quality_statistics();
            
            println!("  분할 시간: {:.2} ms", partition_time.as_secs_f64() * 1000.0);
            println!("  GEMV 시간: {:.2} ms", gemv_time.as_secs_f64() * 1000.0);
            println!("  압축률: {:.1}:1", compression_ratio);
            println!("  PSNR: {:.1} dB", quality_stats.psnr);
            println!("  메모리: {:.2} KB", memory_bytes as f32 / 1024.0);
            
            // 성능 기준 확인
            assert!(partition_time.as_secs_f64() < 1.0, "분할 시간이 너무 오래 걸림");
            assert!(gemv_time.as_secs_f64() < 0.1, "GEMV 시간이 너무 오래 걸림");
            assert!(compression_ratio > 50.0, "압축률이 기대치 미만");
        }
    }
    
    println!("\n성능 벤치마크 테스트 성공!");
}

/// 6.6 확장성 테스트
#[test]
fn test_scalability() {
    println!("=== 확장성 테스트 ===");
    
    let sizes = [
        (64, 64),
        (128, 128),
        (256, 256),
    ];
    
    let mut prev_partition_time = 0.0;
    let mut prev_memory_bytes = 0;
    
    for (i, &(rows, cols)) in sizes.iter().enumerate() {
        println!("크기: {}×{}", rows, cols);
        
        // 테스트 행렬 생성
        let source_matrix: Vec<f32> = (0..rows * cols)
            .map(|idx| ((idx * 7) % 100) as f32 / 100.0)
            .collect();
        
        let mut block_matrix = HierarchicalBlockMatrix::new(rows, cols, QualityLevel::Medium);
        
        // 분할 시간 측정
        let start_time = Instant::now();
        block_matrix.adaptive_partition(&source_matrix);
        let partition_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        let (memory_bytes, compression_ratio) = block_matrix.memory_usage();
        
        println!("  분할 시간: {:.2} ms", partition_time);
        println!("  메모리: {:.2} KB", memory_bytes as f32 / 1024.0);
        println!("  압축률: {:.1}:1", compression_ratio);
        
        if i > 0 {
            let time_scaling = (partition_time / prev_partition_time) as f32;
            let memory_scaling = memory_bytes as f32 / prev_memory_bytes as f32;
            let size_scaling = 4.0f32; // 면적은 4배씩 증가
            
            println!("  시간 확장성: {:.2}x (이론치: {:.2}x)", time_scaling, size_scaling);
            println!("  메모리 확장성: {:.2}x", memory_scaling);
            
            // 확장성이 합리적인 범위인지 확인
            assert!(time_scaling < size_scaling * 2.0, "시간 확장성이 너무 나쁨");
            assert!(memory_scaling < size_scaling, "메모리 확장성이 비효율적");
        }
        
        prev_partition_time = partition_time;
        prev_memory_bytes = memory_bytes;
    }
    
    println!("확장성 테스트 성공!");
}

/// 품질-압축률 트레이드오프 테스트
#[test]
fn test_quality_compression_tradeoff() {
    println!("=== 품질-압축률 트레이드오프 테스트 ===");
    
    let rows = 256;
    let cols = 256;
    
    // 고주파 성분이 많은 복잡한 패턴 생성
    let mut source_matrix = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let x = i as f32 * 0.1;
            let y = j as f32 * 0.1;
            let high_freq = (x * 3.14159).sin() * (y * 2.71828).cos();
            let low_freq = (x * 0.5).sin() * (y * 0.3).cos();
            source_matrix[i * cols + j] = high_freq * 0.7 + low_freq * 0.3;
        }
    }
    
    let qualities = [
        QualityLevel::Ultra,
        QualityLevel::High,
        QualityLevel::Medium,
        QualityLevel::Low,
    ];
    
    println!("품질 등급별 성능 비교:");
    println!("등급\t\t압축률\t\tPSNR\t\t메모리\t\t블록수");
    
    for &quality in &qualities {
        let mut block_matrix = HierarchicalBlockMatrix::new(rows, cols, quality);
        block_matrix.adaptive_partition(&source_matrix);
        
        let (memory_bytes, compression_ratio) = block_matrix.memory_usage();
        let stats = block_matrix.quality_statistics();
        
        println!("{:?}\t\t{:.1}:1\t\t{:.1} dB\t\t{:.1} KB\t\t{}", 
                 quality, 
                 compression_ratio,
                 stats.psnr,
                 memory_bytes as f32 / 1024.0,
                 stats.total_blocks);
        
        // 각 품질 등급이 예상 범위 내인지 확인
        match quality {
            QualityLevel::Ultra => {
                assert!(compression_ratio < 500.0, "Ultra 품질의 압축률이 너무 높음");
                assert!(stats.psnr > 40.0, "Ultra 품질의 PSNR이 너무 낮음");
            },
            QualityLevel::High => {
                assert!(compression_ratio > 100.0 && compression_ratio < 1000.0, "High 품질 압축률 이상");
                assert!(stats.psnr > 30.0, "High 품질의 PSNR이 너무 낮음");
            },
            QualityLevel::Medium => {
                assert!(compression_ratio > 200.0, "Medium 품질 압축률이 너무 낮음");
                assert!(stats.psnr > 20.0, "Medium 품질의 PSNR이 너무 낮음");
            },
            QualityLevel::Low => {
                assert!(compression_ratio > 500.0, "Low 품질 압축률이 기대치 미만");
                assert!(stats.psnr > 15.0, "Low 품질의 PSNR이 너무 낮음");
            },
        }
    }
    
    println!("품질-압축률 트레이드오프 테스트 성공!");
} 
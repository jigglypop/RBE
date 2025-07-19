use crate::core::decoder::FusedForwardPass;
use crate::packed_params::{PoincarePackedBit128, PoincareQuadrant};

#[test]
fn 융합_순전파_생성_테스트() {
    let fused_forward = FusedForwardPass::new();
    // 생성이 성공적으로 되었는지 확인
}

#[test]
fn 융합_GEMV_기본_테스트() {
    let fused_forward = FusedForwardPass::new();
    
    // 간단한 2x2 GEMV 테스트
    let weight_seeds = vec![
        PoincarePackedBit128::new(
            PoincareQuadrant::First, 2048, 2048, 16, 0x12345678, 0.5, 0.3
        )
    ];
    
    let input_vector = vec![1.0, 0.5];
    let mut output_vector = vec![0.0, 0.0];
    
    fused_forward.fused_gemv(&weight_seeds, &input_vector, &mut output_vector, 2, 2);
    
    // 출력이 유한한 값인지 확인
    for (i, &output) in output_vector.iter().enumerate() {
        assert!(output.is_finite(), "출력 {}가 무한대: {}", i, output);
    }
    
    println!("GEMV 결과: {:?}", output_vector);
}

#[test]
fn 블록_기반_융합_GEMV_테스트() {
    let fused_forward = FusedForwardPass::new();
    
    // 2x2 블록 그리드 테스트
    let weight_seeds = vec![
        vec![
            PoincarePackedBit128::new(
                PoincareQuadrant::First, 1024, 1024, 8, 0x11111111, 0.4, 0.2
            ),
            PoincarePackedBit128::new(
                PoincareQuadrant::Second, 2048, 2048, 16, 0x22222222, 0.6, 0.4
            ),
        ],
        vec![
            PoincarePackedBit128::new(
                PoincareQuadrant::Third, 3072, 3072, 24, 0x33333333, 0.8, 0.6
            ),
            PoincarePackedBit128::new(
                PoincareQuadrant::Fourth, 4095, 4095, 32, 0x44444444, 0.9, 0.8
            ),
        ],
    ];
    
    let input_vector = vec![1.0, 0.5, 0.3, 0.7];
    let mut output_vector = vec![0.0, 0.0, 0.0, 0.0];
    
    fused_forward.block_fused_gemv(
        &weight_seeds, 
        &input_vector, 
        &mut output_vector, 
        2, 2, // block_height, block_width
        4, 4  // total_rows, total_cols
    );
    
    // 출력이 유한한 값인지 확인
    for (i, &output) in output_vector.iter().enumerate() {
        assert!(output.is_finite(), "블록 출력 {}가 무한대: {}", i, output);
    }
    
    println!("블록 GEMV 결과: {:?}", output_vector);
}

#[test]
fn 성능_특성_분석_테스트() {
    println!("=== 성능 특성 분석 테스트 ===");
    
    let fused_forward = FusedForwardPass::new();
    
    // 다양한 행렬 크기에서 성능 특성 확인
    let matrix_sizes = vec![(4, 4), (8, 8), (16, 16)];
    
    for (rows, cols) in matrix_sizes {
        println!("행렬 크기: {}x{}", rows, cols);
        
        let weight_seeds = vec![
            PoincarePackedBit128::new(
                PoincareQuadrant::Third, 2048, 2048, 16, 0xDEADBEEF, 0.6, 0.5
            )
        ];
        
        let input_vector = vec![1.0; cols];
        let mut output_vector = vec![0.0; rows];
        
        // 융합 GEMV 시간 측정
        let start_time = std::time::Instant::now();
        fused_forward.fused_gemv(&weight_seeds, &input_vector, &mut output_vector, rows, cols);
        let gemv_time = start_time.elapsed();
        
        // 결과 검증
        for (i, &output) in output_vector.iter().enumerate() {
            assert!(output.is_finite(), "출력 {}가 무한대: {}", i, output);
        }
        
        // 통계 계산
        let sum: f32 = output_vector.iter().sum();
        let mean = sum / output_vector.len() as f32;
        let variance: f32 = output_vector.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum::<f32>() / output_vector.len() as f32;
        let std_dev = variance.sqrt();
        
        println!("  융합 GEMV: {:.2}ms", gemv_time.as_secs_f64() * 1000.0);
        println!("  출력 통계: 평균={:.4}, 표준편차={:.4}", mean, std_dev);
        
        // 통계적 합리성 확인
        assert!(mean.abs() < 10.0, "평균이 과도하게 큼: {:.4}", mean);
        assert!(std_dev >= 0.0, "표준편차가 음수: {:.4}", std_dev);
        assert!(std_dev < 10.0, "표준편차가 너무 큼: {:.4}", std_dev);
    }
    
    println!("모든 성능 특성 분석 테스트 통과!");
}

#[test]
fn 메모리_효율성_검증_테스트() {
    println!("=== 메모리 효율성 검증 테스트 ===");
    
    let matrix_sizes = vec![(4, 4), (8, 8), (16, 16), (32, 32)];
    
    for (rows, cols) in matrix_sizes {
        let total_elements = rows * cols;
        
        // 표준 Dense 행렬 메모리 사용량
        let dense_memory = total_elements * std::mem::size_of::<f32>(); // 4 bytes per f32
        
        // 푸앵카레 압축 메모리 사용량 (128비트 = 16바이트)
        let poincare_memory = std::mem::size_of::<PoincarePackedBit128>(); // 16 bytes
        
        let compression_ratio = dense_memory as f32 / poincare_memory as f32;
        let memory_savings = (1.0 - poincare_memory as f32 / dense_memory as f32) * 100.0;
        
        println!("행렬 크기: {}x{} ({} 원소)", rows, cols, total_elements);
        println!("  Dense 메모리: {} KB", dense_memory / 1024);
        println!("  푸앵카레 메모리: {} bytes", poincare_memory);
        println!("  압축률: {:.1}:1", compression_ratio);
        println!("  메모리 절약: {:.2}%", memory_savings);
        
        // 메모리 효율성 검증
        if total_elements >= 16 { // 작은 행렬은 압축 효과가 적음
            assert!(compression_ratio > 1.0, 
                    "압축률이 너무 낮음: {:.1}", compression_ratio);
        }
        
        // 실제 융합 GEMV로 기능성 확인
        let fused_forward = FusedForwardPass::new();
        let weight_seeds = vec![
            PoincarePackedBit128::new(
                PoincareQuadrant::Third, 2048, 2048, 16, 0x12345678, 0.7, 0.3
            )
        ];
        
        let input_vector = vec![1.0; cols];
        let mut output_vector = vec![0.0; rows];
        
        fused_forward.fused_gemv(&weight_seeds, &input_vector, &mut output_vector, rows, cols);
        
        // 기본 검증
        for &output in &output_vector {
            assert!(output.is_finite(), "메모리 효율성 테스트에서 무한대 출력");
        }
        
        println!("  기능성 검증: 통과");
        println!();
    }
    
    println!("모든 메모리 효율성 테스트 통과!");
} 
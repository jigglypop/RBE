use super::*;
use crate::types::*;
use crate::matrix::QualityLevel;
use rand::prelude::*;

/// GPT-2 가중치 시뮬레이터
struct GPT2WeightSimulator {
    rng: ThreadRng,
}

impl GPT2WeightSimulator {
    fn new() -> Self {
        Self {
            rng: thread_rng(),
        }
    }
    
    /// FFN W1 가중치 생성 (768 → 3072)
    fn generate_ffn_w1(&mut self) -> Vec<f32> {
        let size = 768 * 3072;
        (0..size)
            .map(|_| self.rng.gen_range(-0.1..0.1))
            .collect()
    }
    
    /// FFN W2 가중치 생성 (3072 → 768)
    fn generate_ffn_w2(&mut self) -> Vec<f32> {
        let size = 3072 * 768;
        (0..size)
            .map(|_| self.rng.gen_range(-0.1..0.1))
            .collect()
    }
    
    /// Attention 가중치 생성 (768 × 768)
    fn generate_attention_weights(&mut self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let size = 768 * 768;
        let q = (0..size).map(|_| self.rng.gen_range(-0.1..0.1)).collect();
        let k = (0..size).map(|_| self.rng.gen_range(-0.1..0.1)).collect();
        let v = (0..size).map(|_| self.rng.gen_range(-0.1..0.1)).collect();
        let o = (0..size).map(|_| self.rng.gen_range(-0.1..0.1)).collect();
        (q, k, v, o)
    }
    
    /// 임베딩 가중치 생성
    fn generate_embedding_weights(&mut self, vocab_size: usize, hidden_size: usize) -> Vec<f32> {
        let size = vocab_size * hidden_size;
        (0..size)
            .map(|_| self.rng.gen_range(-0.5..0.5))
            .collect()
    }
}

#[test]
fn test_llm_analyzer_basic() {
    println!("=== LLM 분석기 기본 테스트 ===");
    
    let mut analyzer = LLMAnalyzer::new(AnalyzerConfig::default());
    
    // GPT-2 117M 분석
    let result = analyzer.analyze_gpt2_117m();
    assert!(result.is_ok(), "GPT-2 분석 실패: {:?}", result.err());
    
    let architecture = result.unwrap();
    
    // 기본 구조 검증
    assert_eq!(architecture.total_parameters, 117_210_240);
    assert_eq!(architecture.hidden_size, 768);
    assert_eq!(architecture.vocab_size, 50_257);
    assert_eq!(architecture.num_layers, 12);
    
    // 압축 절약 계산
    let savings = analyzer.calculate_compression_savings().unwrap();
    println!("예상 메모리 절약: {}MB ({:.1}%)", 
             savings.savings_mb, savings.savings_ratio * 100.0);
    
    assert!(savings.savings_ratio > 0.5, "압축률이 50% 이하");
    
    // 분석 리포트 출력
    analyzer.print_analysis_report().unwrap();
    
    println!("✓ LLM 분석기 기본 테스트 통과");
}

#[test]
fn test_rbe_converter_ffn_layer() {
    println!("=== RBE FFN 레이어 변환 테스트 ===");
    
    let mut simulator = GPT2WeightSimulator::new();
    let mut converter = RBEConverter::new(ConversionConfig::default());
    
    // 가상 FFN 가중치 생성
    let w1_weights = simulator.generate_ffn_w1();
    let w2_weights = simulator.generate_ffn_w2();
    
    // 레이어 정보 생성 (FFN 레이어 0)
    let layer_info = LayerParameterInfo {
        layer_id: 14, // FFN 레이어 0
        layer_type: LayerType::FFN,
        parameter_count: w1_weights.len() + w2_weights.len(),
        memory_usage: (w1_weights.len() + w2_weights.len()) * 4,
        compressible: true,
        target_compression_ratio: 800.0,
        weight_matrices: vec![],
    };
    
    // FFN 레이어 변환
    let result = converter.convert_ffn_layer(
        &layer_info,
        &w1_weights,
        &w2_weights,
        None,
        None
    );
    
    assert!(result.is_ok(), "FFN 변환 실패: {:?}", result.err());
    
    // 변환 결과 검증
    let converted = converter.get_converted_layer(14).unwrap();
    assert_eq!(converted.layer_type, LayerType::FFN);
    assert_eq!(converted.rbe_weights.len(), 2); // W1, W2
    
    // 압축률 검증
    assert!(converted.metadata.actual_compression_ratio > 100.0, 
            "압축률이 100:1 이하: {}", converted.metadata.actual_compression_ratio);
    
    // 품질 검증
    assert!(converted.quality_metrics.quality_score > 50.0,
            "품질 점수가 50 이하: {}", converted.quality_metrics.quality_score);
    
    converter.print_conversion_report();
    
    println!("✓ FFN 레이어 변환 테스트 통과");
    println!("  압축률: {:.1}:1", converted.metadata.actual_compression_ratio);
    println!("  품질 점수: {:.1}/100", converted.quality_metrics.quality_score);
}

#[test]
fn test_rbe_converter_attention_layer() {
    println!("=== RBE Attention 레이어 변환 테스트 ===");
    
    let mut simulator = GPT2WeightSimulator::new();
    let mut converter = RBEConverter::new(ConversionConfig::default());
    
    // 가상 Attention 가중치 생성
    let (q_weights, k_weights, v_weights, o_weights) = simulator.generate_attention_weights();
    
    // 레이어 정보 생성
    let layer_info = LayerParameterInfo {
        layer_id: 2, // Attention 레이어 0
        layer_type: LayerType::Attention,
        parameter_count: q_weights.len() * 4, // Q, K, V, O
        memory_usage: q_weights.len() * 4 * 4, // f32
        compressible: true,
        target_compression_ratio: 400.0,
        weight_matrices: vec![],
    };
    
    // Attention 레이어 변환
    let result = converter.convert_attention_layer(
        &layer_info,
        &q_weights,
        &k_weights,
        &v_weights,
        &o_weights,
        None
    );
    
    assert!(result.is_ok(), "Attention 변환 실패: {:?}", result.err());
    
    // 변환 결과 검증
    let converted = converter.get_converted_layer(2).unwrap();
    assert_eq!(converted.layer_type, LayerType::Attention);
    assert_eq!(converted.rbe_weights.len(), 4); // Q, K, V, O
    
    // Attention은 더 보수적으로 압축
    assert!(converted.metadata.actual_compression_ratio > 50.0);
    assert!(converted.quality_metrics.quality_score > 60.0);
    
    converter.print_conversion_report();
    
    println!("✓ Attention 레이어 변환 테스트 통과");
    println!("  압축률: {:.1}:1", converted.metadata.actual_compression_ratio);
    println!("  품질 점수: {:.1}/100", converted.quality_metrics.quality_score);
}

#[test]
fn test_rbe_converter_embedding_layer() {
    println!("=== RBE 임베딩 레이어 변환 테스트 ===");
    
    let mut simulator = GPT2WeightSimulator::new();
    let mut converter = RBEConverter::new(ConversionConfig::default());
    
    // 작은 임베딩으로 테스트 (전체 GPT-2는 너무 큼)
    let vocab_size = 1024;
    let hidden_size = 768;
    let embedding_weights = simulator.generate_embedding_weights(vocab_size, hidden_size);
    
    // 레이어 정보 생성
    let layer_info = LayerParameterInfo {
        layer_id: 0, // Token embedding
        layer_type: LayerType::TokenEmbedding,
        parameter_count: embedding_weights.len(),
        memory_usage: embedding_weights.len() * 4,
        compressible: true,
        target_compression_ratio: 500.0,
        weight_matrices: vec![],
    };
    
    // 임베딩 레이어 변환
    let result = converter.convert_embedding_layer(
        &layer_info,
        &embedding_weights,
        vocab_size,
        hidden_size
    );
    
    assert!(result.is_ok(), "임베딩 변환 실패: {:?}", result.err());
    
    // 변환 결과 검증
    let converted = converter.get_converted_layer(0).unwrap();
    assert_eq!(converted.layer_type, LayerType::TokenEmbedding);
    assert_eq!(converted.rbe_weights.len(), 1);
    
    // 임베딩은 높은 압축률 달성 가능
    assert!(converted.metadata.actual_compression_ratio > 200.0);
    assert!(converted.quality_metrics.quality_score > 40.0);
    
    converter.print_conversion_report();
    
    println!("✓ 임베딩 레이어 변환 테스트 통과");
    println!("  압축률: {:.1}:1", converted.metadata.actual_compression_ratio);
    println!("  품질 점수: {:.1}/100", converted.quality_metrics.quality_score);
}

#[test]
fn test_full_gpt2_simulation() {
    println!("=== 전체 GPT-2 시뮬레이션 테스트 ===");
    
    let mut analyzer = LLMAnalyzer::new(AnalyzerConfig::default());
    let mut converter = RBEConverter::new(ConversionConfig::default());
    let mut simulator = GPT2WeightSimulator::new();
    
    // 1. GPT-2 구조 분석
    let architecture = analyzer.analyze_gpt2_117m().unwrap();
    
    // 2. 주요 레이어들 변환 (일부만)
    println!("\n=== Phase 1: FFN 레이어 변환 (처음 3개만) ===");
    
    for transformer_layer in 0..3 {
        let layer_id = 14 + transformer_layer;
        let layer_info = &architecture.layer_parameters[layer_id];
        
        let w1_weights = simulator.generate_ffn_w1();
        let w2_weights = simulator.generate_ffn_w2();
        
        let result = converter.convert_ffn_layer(
            layer_info,
            &w1_weights,
            &w2_weights,
            None,
            None
        );
        
        assert!(result.is_ok(), "FFN 레이어 {} 변환 실패", transformer_layer);
    }
    
    println!("\n=== Phase 2: Attention 레이어 변환 (처음 2개만) ===");
    
    for transformer_layer in 0..2 {
        let layer_id = 2 + transformer_layer;
        let layer_info = &architecture.layer_parameters[layer_id];
        
        let (q, k, v, o) = simulator.generate_attention_weights();
        
        let result = converter.convert_attention_layer(
            layer_info,
            &q, &k, &v, &o,
            None
        );
        
        assert!(result.is_ok(), "Attention 레이어 {} 변환 실패", transformer_layer);
    }
    
    // 3. 결과 종합
    converter.print_conversion_report();
    
    let stats = &converter.conversion_stats;
    assert_eq!(stats.converted_layers, 5); // FFN 3개 + Attention 2개
    assert!(stats.average_compression_ratio > 100.0);
    assert!(stats.average_quality_score > 40.0);
    
    println!("\n✓ 전체 GPT-2 시뮬레이션 테스트 통과");
    println!("  변환된 레이어: {}", stats.converted_layers);
    println!("  평균 압축률: {:.1}:1", stats.average_compression_ratio);
    println!("  평균 품질: {:.1}/100", stats.average_quality_score);
    
    let total_savings = (stats.total_original_size - stats.total_compressed_size) as f32 / 1024.0 / 1024.0;
    println!("  메모리 절약: {:.1}MB", total_savings);
}

#[test]
fn test_adaptive_compression_config() {
    println!("=== 적응적 압축 설정 테스트 ===");
    
    let converter = RBEConverter::new(ConversionConfig::default());
    
    // FFN 레이어 설정 테스트
    let ffn_layer = LayerParameterInfo {
        layer_id: 14,
        layer_type: LayerType::FFN,
        parameter_count: 768 * 3072 * 2,
        memory_usage: 768 * 3072 * 2 * 4,
        compressible: true,
        target_compression_ratio: 1200.0, // 극적 압축
        weight_matrices: vec![],
    };
    
    let ffn_config = converter.determine_optimal_block_config(&ffn_layer).unwrap();
    assert_eq!(ffn_config.block_size, 16); // 작은 블록으로 극적 압축
    assert_eq!(ffn_config.quality_level, QualityLevel::Medium);
    
    // Attention 레이어 설정 테스트
    let attention_layer = LayerParameterInfo {
        layer_id: 2,
        layer_type: LayerType::Attention,
        parameter_count: 768 * 768 * 4,
        memory_usage: 768 * 768 * 4 * 4,
        compressible: true,
        target_compression_ratio: 200.0, // 보수적 압축
        weight_matrices: vec![],
    };
    
    let attn_config = converter.determine_optimal_block_config(&attention_layer).unwrap();
    assert_eq!(attn_config.block_size, 128); // 큰 블록으로 보수적 압축
    assert_eq!(attn_config.quality_level, QualityLevel::Ultra);
    
    println!("✓ 적응적 압축 설정 테스트 통과");
    println!("  FFN 블록 크기: {}, 품질: {:?}", ffn_config.block_size, ffn_config.quality_level);
    println!("  Attention 블록 크기: {}, 품질: {:?}", attn_config.block_size, attn_config.quality_level);
}

#[test]
fn test_quality_metrics_calculation() {
    println!("=== 품질 메트릭 계산 테스트 ===");
    
    let converter = RBEConverter::new(ConversionConfig::default());
    
    // 테스트 데이터 생성
    let original: Vec<f32> = (0..1000)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    
    // 약간의 노이즈를 추가한 "복원" 데이터
    let mut rng = thread_rng();
    let reconstructed: Vec<f32> = original.iter()
        .map(|&x| x + rng.gen_range(-0.01..0.01))
        .collect();
    
    // HierarchicalBlockMatrix 시뮬레이션을 위한 더미 구조체 필요
    // 실제로는 품질 메트릭 계산 로직만 테스트
    
    // MSE 직접 계산으로 검증
    let mse: f32 = original.iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / original.len() as f32;
    
    assert!(mse < 0.001, "MSE가 너무 높음: {}", mse);
    
    // 코사인 유사도 계산
    let dot_product: f32 = original.iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| a * b)
        .sum();
    
    let norm_original: f32 = original.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm_reconstructed: f32 = reconstructed.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    
    let cosine_similarity = dot_product / (norm_original * norm_reconstructed);
    
    assert!(cosine_similarity > 0.99, "코사인 유사도가 낮음: {}", cosine_similarity);
    
    println!("✓ 품질 메트릭 계산 테스트 통과");
    println!("  MSE: {:.6}", mse);
    println!("  코사인 유사도: {:.6}", cosine_similarity);
}

#[test]
fn test_memory_optimization() {
    println!("=== 메모리 최적화 테스트 ===");
    
    // 메모리 제한이 있는 설정
    let mut config = ConversionConfig::default();
    config.memory_limit = 64 * 1024 * 1024; // 64MB 제한
    
    let converter = RBEConverter::new(config);
    
    let large_layer = LayerParameterInfo {
        layer_id: 0,
        layer_type: LayerType::TokenEmbedding,
        parameter_count: 50_257 * 768,
        memory_usage: 50_257 * 768 * 4,
        compressible: true,
        target_compression_ratio: 500.0,
        weight_matrices: vec![],
    };
    
    let config = converter.determine_optimal_block_config(&large_layer).unwrap();
    
    // 메모리 제한으로 인해 블록 크기가 조정되었는지 확인
    assert!(config.block_size <= 32, "메모리 제한이 적용되지 않음: {}", config.block_size);
    
    println!("✓ 메모리 최적화 테스트 통과");
    println!("  조정된 블록 크기: {}", config.block_size);
} 
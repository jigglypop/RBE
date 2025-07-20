//! 동적 레이아웃 시스템 통합 테스트
//! 
//! 압축 → 저장 → 로딩 → 검증의 전체 파이프라인 테스트

use crate::encoder::{ModelCompressor, CompressionConfig, CompressionProfile, QualityGrade};
use crate::decoder::ModelLoader;
use crate::packed_params::TransformType;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use tempfile::TempDir;
use serde_json::json;

fn 테스트_디렉토리_생성() -> TempDir {
    tempfile::tempdir().expect("임시 디렉토리 생성 실패")
}

fn 테스트용_numpy_헤더_생성(shape: &[usize]) -> Vec<u8> {
    let mut header = Vec::new();
    
    // Magic number
    header.extend_from_slice(b"\x93NUMPY");
    
    // Version
    header.extend_from_slice(&[1, 0]);
    
    // Header string
    let shape_str = format!("({},)", shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "));
    let header_dict = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}",
        shape_str
    );
    
    // Pad to 16-byte boundary
    let mut header_dict = header_dict.into_bytes();
    while (header_dict.len() + 10) % 16 != 0 {
        header_dict.push(b' ');
    }
    header_dict.push(b'\n');
    
    // Header length
    let header_len = header_dict.len() as u16;
    header.extend_from_slice(&header_len.to_le_bytes());
    
    // Header content
    header.extend_from_slice(&header_dict);
    
    header
}

fn 테스트용_numpy_파일_생성(path: &Path, shape: &[usize], data: &[f32]) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    
    // NumPy 헤더 작성
    let header = 테스트용_numpy_헤더_생성(shape);
    file.write_all(&header)?;
    
    // 데이터 작성
    for &value in data {
        file.write_all(&value.to_le_bytes())?;
    }
    
    Ok(())
}

fn 완전한_gpt2_모델_생성(temp_dir: &Path) -> std::io::Result<()> {
    // GPT-2 스타일 모델 메타데이터 생성
    let metadata = json!({
        "transformer.wte.weight": {
            "shape": [1000, 256], // 작은 어휘집
            "file": "transformer_wte_weight.npy"
        },
        "transformer.wpe.weight": {
            "shape": [512, 256], // 위치 임베딩
            "file": "transformer_wpe_weight.npy"
        },
        "transformer.h.0.attn.c_attn.weight": {
            "shape": [256, 768], // QKV 가중치
            "file": "transformer_h_0_attn_c_attn_weight.npy"
        },
        "transformer.h.0.attn.c_proj.weight": {
            "shape": [256, 256], // 출력 투영
            "file": "transformer_h_0_attn_c_proj_weight.npy"
        },
        "transformer.h.0.mlp.c_fc.weight": {
            "shape": [256, 1024], // FFN 확장
            "file": "transformer_h_0_mlp_c_fc_weight.npy"
        },
        "transformer.h.0.mlp.c_proj.weight": {
            "shape": [1024, 256], // FFN 축소
            "file": "transformer_h_0_mlp_c_proj_weight.npy"
        },
        "transformer.ln_f.weight": {
            "shape": [256], // 1D - 스킵됨
            "file": "transformer_ln_f_weight.npy"
        },
        "lm_head.weight": {
            "shape": [256, 1000], // 언어 모델 헤드
            "file": "lm_head_weight.npy"
        }
    });
    
    let metadata_path = temp_dir.join("metadata.json");
    fs::write(metadata_path, serde_json::to_string_pretty(&metadata)?)?;
    
    // 각 가중치 파일 생성
    
    // 토큰 임베딩 (패턴있는 데이터)
    let wte_data: Vec<f32> = (0..256000).map(|i| {
        let row = i / 256;
        let col = i % 256;
        (row as f32 * 0.01 + col as f32 * 0.001).sin()
    }).collect();
    테스트용_numpy_파일_생성(
        &temp_dir.join("transformer_wte_weight.npy"),
        &[1000, 256],
        &wte_data,
    )?;
    
    // 위치 임베딩
    let wpe_data: Vec<f32> = (0..131072).map(|i| {
        let pos = i / 256;
        let dim = i % 256;
        (pos as f32 * 0.1 + dim as f32 * 0.01).cos()
    }).collect();
    테스트용_numpy_파일_생성(
        &temp_dir.join("transformer_wpe_weight.npy"),
        &[512, 256],
        &wpe_data,
    )?;
    
    // 어텐션 가중치
    let attn_data: Vec<f32> = (0..196608).map(|i| (i as f32 * 0.0001).tanh()).collect();
    테스트용_numpy_파일_생성(
        &temp_dir.join("transformer_h_0_attn_c_attn_weight.npy"),
        &[256, 768],
        &attn_data,
    )?;
    
    // 어텐션 투영
    let attn_proj_data: Vec<f32> = (0..65536).map(|i| (i as f32 * 0.0002 - 6.0).sinh().max(-1.0).min(1.0)).collect();
    테스트용_numpy_파일_생성(
        &temp_dir.join("transformer_h_0_attn_c_proj_weight.npy"),
        &[256, 256],
        &attn_proj_data,
    )?;
    
    // FFN 확장
    let fc_data: Vec<f32> = (0..262144).map(|i| ((i as f32 * 0.001).sin() * 0.5)).collect();
    테스트용_numpy_파일_생성(
        &temp_dir.join("transformer_h_0_mlp_c_fc_weight.npy"),
        &[256, 1024],
        &fc_data,
    )?;
    
    // FFN 축소
    let proj_data: Vec<f32> = (0..262144).map(|i| ((i as f32 * 0.0005).cos() * 0.3)).collect();
    테스트용_numpy_파일_생성(
        &temp_dir.join("transformer_h_0_mlp_c_proj_weight.npy"),
        &[1024, 256],
        &proj_data,
    )?;
    
    // 1D 가중치 (스킵됨)
    let ln_data: Vec<f32> = (0..256).map(|i| 1.0 + (i as f32) * 0.001).collect();
    테스트용_numpy_파일_생성(
        &temp_dir.join("transformer_ln_f_weight.npy"),
        &[256],
        &ln_data,
    )?;
    
    // LM 헤드
    let lm_head_data: Vec<f32> = (0..256000).map(|i| (i as f32 * 0.00001).atan()).collect();
    테스트용_numpy_파일_생성(
        &temp_dir.join("lm_head_weight.npy"),
        &[256, 1000],
        &lm_head_data,
    )?;
    
    Ok(())
}

#[test]
fn 전체_파이프라인_통합_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let input_path = temp_dir.path().join("input");
    let output_path = temp_dir.path().join("output");
    
    // 1단계: 테스트 모델 생성
    fs::create_dir_all(&input_path).unwrap();
    완전한_gpt2_모델_생성(&input_path).unwrap();
    
    // 2단계: 모델 압축
    let config = CompressionConfig {
        block_size: 32,
        quality_grade: QualityGrade::C,
        transform_type: TransformType::Dwt,
        profile: CompressionProfile::Fast,
        custom_coefficients: Some(50),
        min_block_count: None,
        rmse_threshold: Some(0.5),
        compression_ratio_threshold: Some(2.0),
    };
    
    let mut compressor = ModelCompressor::new("integration-test", config);
    let compression_result = compressor.compress_from_numpy_dir(&input_path, &output_path);
    
    assert!(compression_result.is_ok(), "압축 실패: {:?}", compression_result.err());
    
    // 3단계: 압축 결과 검증
    assert!(output_path.join("rbe_model.bin").exists());
    assert!(output_path.join("weight_layout.json").exists());
    assert!(output_path.join("model_config.json").exists());
    assert!(output_path.join("compression_report.json").exists());
    
    let layout = compressor.get_layout();
    assert_eq!(layout.model_name, "integration-test");
    assert!(layout.weights.len() >= 6); // 2D 가중치들만
    assert!(layout.total_blocks > 0);
    
    // 4단계: 압축된 모델 로딩
    let mut loader = ModelLoader::load_from_directory(&output_path).unwrap();
    
    // 로더 기본 정보 확인
    assert_eq!(loader.get_layout().model_name, "integration-test");
    assert_eq!(loader.list_weight_names().len(), layout.weights.len());
    
    // 5단계: 개별 가중치 로딩 테스트
    let test_weights = vec![
        "transformer.wte.weight",
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.0.mlp.c_fc.weight",
    ];
    
    for weight_name in &test_weights {
        let loaded_weight = loader.load_weight(weight_name).unwrap();
        assert!(loaded_weight.is_some(), "가중치 {} 로딩 실패", weight_name);
        
        let weight = loaded_weight.unwrap();
        assert_eq!(weight.name, *weight_name);
        assert!(weight.blocks.len() > 0);
        
        // 원본 형태 정보 확인
        assert_eq!(weight.original_shape.len(), 2); // 2D 텐서
        assert!(weight.original_shape[0] > 0);
        assert!(weight.original_shape[1] > 0);
    }
    
    // 6단계: 전체 가중치 일괄 로딩
    let all_weights = loader.load_all_weights().unwrap();
    assert_eq!(all_weights.len(), layout.weights.len());
    
    for weight_name in &test_weights {
        assert!(all_weights.contains_key(*weight_name));
    }
    
    // 7단계: 가중치 복원 및 검증
    for weight_name in &test_weights {
        let reconstructed = loader.decode_and_verify_weight(weight_name);
        assert!(reconstructed.is_ok(), "가중치 {} 복원 실패", weight_name);
        
        let data = reconstructed.unwrap();
        let weight_info = loader.get_weight_info(weight_name).unwrap();
        let expected_size = weight_info.original_shape.iter().product::<usize>();
        
        assert_eq!(data.len(), expected_size);
        
        // 복원된 데이터가 유효한지 확인
        for &value in &data {
            assert!(value.is_finite(), "복원된 데이터에 유효하지 않은 값: {}", value);
        }
    }
}

#[test]
fn 압축_로딩_일관성_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let input_path = temp_dir.path().join("input");
    let output_path = temp_dir.path().join("output");
    
    // 테스트 모델 생성
    fs::create_dir_all(&input_path).unwrap();
    완전한_gpt2_모델_생성(&input_path).unwrap();
    
    // 압축
    let config = CompressionConfig {
        block_size: 32,
        quality_grade: QualityGrade::C,
        transform_type: TransformType::Dct,
        profile: CompressionProfile::Balanced,
        custom_coefficients: Some(100),
        min_block_count: None,
        rmse_threshold: Some(0.2),
        compression_ratio_threshold: Some(3.0),
    };
    
    let mut compressor = ModelCompressor::new("consistency-test", config);
    compressor.compress_from_numpy_dir(&input_path, &output_path).unwrap();
    
    // 로딩
    let mut loader = ModelLoader::load_from_directory(&output_path).unwrap();
    
    // 압축기와 로더의 레이아웃 일관성 확인
    let compression_layout = compressor.get_layout();
    let loading_layout = loader.get_layout();
    
    assert_eq!(compression_layout.model_name, loading_layout.model_name);
    assert_eq!(compression_layout.total_blocks, loading_layout.total_blocks);
    assert_eq!(compression_layout.weights.len(), loading_layout.weights.len());
    
    // 각 가중치 정보 일관성 확인
    for (comp_weight, load_weight) in compression_layout.weights.iter().zip(loading_layout.weights.iter()) {
        assert_eq!(comp_weight.name, load_weight.name);
        assert_eq!(comp_weight.offset_bytes, load_weight.offset_bytes);
        assert_eq!(comp_weight.num_blocks, load_weight.num_blocks);
        assert_eq!(comp_weight.original_shape, load_weight.original_shape);
        assert_eq!(comp_weight.compression_ratio, load_weight.compression_ratio);
        assert_eq!(comp_weight.rmse, load_weight.rmse);
        assert_eq!(comp_weight.transform_type, load_weight.transform_type);
    }
}

#[test]
fn 다중_변환_타입_통합_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let input_path = temp_dir.path().join("input");
    
    // 테스트 모델 생성
    fs::create_dir_all(&input_path).unwrap();
    완전한_gpt2_모델_생성(&input_path).unwrap();
    
    let transform_types = vec![
        (TransformType::Dwt, "dwt"),
        (TransformType::Dct, "dct"),
    ];
    
    for (transform_type, name) in transform_types {
        let output_path = temp_dir.path().join(format!("output_{}", name));
        
        // 압축
        let config = CompressionConfig {
            block_size: 32,
            quality_grade: QualityGrade::C,
            transform_type,
            profile: CompressionProfile::Fast,
            custom_coefficients: Some(75),
            min_block_count: None,
            rmse_threshold: Some(0.3),
            compression_ratio_threshold: Some(2.5),
        };
        
        let mut compressor = ModelCompressor::new(&format!("{}-integration", name), config);
        let result = compressor.compress_from_numpy_dir(&input_path, &output_path);
        assert!(result.is_ok(), "변환 타입 {} 압축 실패", name);
        
        // 로딩 및 검증
        let mut loader = ModelLoader::load_from_directory(&output_path).unwrap();
        
        // 변환 타입 일관성 확인
        for weight_info in loader.get_layout().weights.iter() {
            assert_eq!(weight_info.transform_type, transform_type);
        }
        
        // 샘플 가중치 복원 테스트
        let sample_weight = loader.load_weight("transformer.wte.weight").unwrap();
        assert!(sample_weight.is_some());
        
        let weight = sample_weight.unwrap();
        assert!(weight.blocks.len() > 0);
        
        // 복원 테스트
        let reconstructed = loader.decode_and_verify_weight("transformer.wte.weight");
        assert!(reconstructed.is_ok(), "변환 타입 {} 복원 실패", name);
    }
}

#[test]
fn 대용량_가중치_처리_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let input_path = temp_dir.path().join("input");
    let output_path = temp_dir.path().join("output");
    
    // 큰 가중치 파일 생성
    fs::create_dir_all(&input_path).unwrap();
    
    let metadata = json!({
        "large.weight": {
            "shape": [512, 512], // 262,144 개 요소
            "file": "large_weight.npy"
        }
    });
    
    let metadata_path = input_path.join("metadata.json");
    fs::write(metadata_path, serde_json::to_string_pretty(&metadata).unwrap()).unwrap();
    
    // 패턴이 있는 대용량 데이터 생성
    let large_data: Vec<f32> = (0..262144).map(|i| {
        let row = i / 512;
        let col = i % 512;
        ((row as f32 * 0.1).sin() * (col as f32 * 0.05).cos() * 0.5)
    }).collect();
    
    테스트용_numpy_파일_생성(
        &input_path.join("large_weight.npy"),
        &[512, 512],
        &large_data,
    ).unwrap();
    
    // 압축 (작은 블록으로 세분화)
    let config = CompressionConfig {
        block_size: 32, // 작은 블록으로 많은 블록 생성
        quality_grade: QualityGrade::C,
        transform_type: TransformType::Dwt,
        profile: CompressionProfile::Fast,
        custom_coefficients: Some(80),
        min_block_count: None,
        rmse_threshold: Some(0.3),
        compression_ratio_threshold: Some(4.0),
    };
    
    let mut compressor = ModelCompressor::new("large-test", config);
    let result = compressor.compress_from_numpy_dir(&input_path, &output_path);
    assert!(result.is_ok());
    
    // 많은 블록이 생성되었는지 확인
    let layout = compressor.get_layout();
    assert_eq!(layout.weights.len(), 1);
    
    let large_weight_info = &layout.weights[0];
    assert_eq!(large_weight_info.name, "large.weight");
    
    // 예상 블록 개수: (512/32) * (512/32) = 16 * 16 = 256
    let expected_blocks = (512 / 32) * (512 / 32);
    assert_eq!(large_weight_info.num_blocks, expected_blocks);
    
    // 로딩 및 복원
    let mut loader = ModelLoader::load_from_directory(&output_path).unwrap();
    
    let loaded_weight = loader.load_weight("large.weight").unwrap();
    assert!(loaded_weight.is_some());
    
    let weight = loaded_weight.unwrap();
    assert_eq!(weight.blocks.len(), expected_blocks);
    
    // 복원 데이터 검증
    let reconstructed = loader.decode_and_verify_weight("large.weight").unwrap();
    assert_eq!(reconstructed.len(), 512 * 512);
    
    // 복원된 데이터가 원본과 유사한지 검증 (RMSE 확인)
    let rmse = {
        let mse: f32 = large_data.iter()
            .zip(reconstructed.iter())
            .map(|(orig, recon)| (orig - recon).powi(2))
            .sum::<f32>() / large_data.len() as f32;
        mse.sqrt()
    };
    
    assert!(rmse < 0.5, "RMSE가 너무 큼: {}", rmse);
    println!("대용량 가중치 복원 RMSE: {:.6}", rmse);
}

#[test]
fn 빈_모델_처리_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let input_path = temp_dir.path().join("input");
    let output_path = temp_dir.path().join("output");
    
    // 1D 가중치만 있는 모델 (모두 스킵됨)
    fs::create_dir_all(&input_path).unwrap();
    
    let metadata = json!({
        "norm1.weight": {
            "shape": [256],
            "file": "norm1_weight.npy"
        },
        "norm2.bias": {
            "shape": [256],
            "file": "norm2_bias.npy"
        }
    });
    
    let metadata_path = input_path.join("metadata.json");
    fs::write(metadata_path, serde_json::to_string_pretty(&metadata).unwrap()).unwrap();
    
    // 1D 파일들 생성
    let norm_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    테스트용_numpy_파일_생성(
        &input_path.join("norm1_weight.npy"),
        &[256],
        &norm_data,
    ).unwrap();
    
    테스트용_numpy_파일_생성(
        &input_path.join("norm2_bias.npy"),
        &[256],
        &norm_data,
    ).unwrap();
    
    // 압축 (모든 가중치가 스킵됨)
    let config = CompressionConfig::fast();
    let mut compressor = ModelCompressor::new("empty-test", config);
    let result = compressor.compress_from_numpy_dir(&input_path, &output_path);
    
    assert!(result.is_ok());
    
    // 빈 레이아웃 확인
    let layout = compressor.get_layout();
    assert_eq!(layout.model_name, "empty-test");
    assert_eq!(layout.weights.len(), 0); // 2D 가중치가 없음
    assert_eq!(layout.total_blocks, 0);
    
    // 로딩도 가능해야 함
    let loader = ModelLoader::load_from_directory(&output_path);
    assert!(loader.is_ok());
    
    let loader = loader.unwrap();
    assert_eq!(loader.list_weight_names().len(), 0);
} 
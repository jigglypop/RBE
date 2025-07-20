//! ModelCompressor 테스트

use crate::encoder::{ModelCompressor, CompressionConfig, CompressionProfile, QualityGrade};
use crate::packed_params::TransformType;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use tempfile::TempDir;
use serde_json::json;

fn 테스트_디렉토리_생성() -> TempDir {
    tempfile::tempdir().expect("임시 디렉토리 생성 실패")
}

fn 테스트용_numpy_헤더_생성(shape: &[usize], dtype: &str) -> Vec<u8> {
    // NumPy 헤더 생성 (간단한 버전)
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
    let header = 테스트용_numpy_헤더_생성(shape, "f4");
    file.write_all(&header)?;
    
    // 데이터 작성
    for &value in data {
        file.write_all(&value.to_le_bytes())?;
    }
    
    Ok(())
}

fn 테스트용_메타데이터_생성(temp_dir: &Path) -> std::io::Result<()> {
    let metadata = json!({
        "transformer.wte.weight": {
            "shape": [100, 64],
            "file": "transformer_wte_weight.npy"
        },
        "transformer.h.0.attn.c_attn.weight": {
            "shape": [64, 192],
            "file": "transformer_h_0_attn_c_attn_weight.npy"
        },
        "transformer.h.0.ln_1.weight": {
            "shape": [64],
            "file": "transformer_h_0_ln_1_weight.npy"
        }
    });
    
    let metadata_path = temp_dir.join("metadata.json");
    fs::write(metadata_path, serde_json::to_string_pretty(&metadata)?)?;
    
    // 해당하는 numpy 파일들 생성
    
    // 2D 가중치들 (압축됨)
    let wte_data: Vec<f32> = (0..6400).map(|i| (i as f32) * 0.001).collect();
    테스트용_numpy_파일_생성(
        &temp_dir.join("transformer_wte_weight.npy"),
        &[100, 64],
        &wte_data,
    )?;
    
    let attn_data: Vec<f32> = (0..12288).map(|i| (i as f32) * 0.0001).collect();
    테스트용_numpy_파일_생성(
        &temp_dir.join("transformer_h_0_attn_c_attn_weight.npy"),
        &[64, 192],
        &attn_data,
    )?;
    
    // 1D 가중치 (스킵됨)
    let ln_data: Vec<f32> = (0..64).map(|i| 1.0 + (i as f32) * 0.01).collect();
    테스트용_numpy_파일_생성(
        &temp_dir.join("transformer_h_0_ln_1_weight.npy"),
        &[64],
        &ln_data,
    )?;
    
    Ok(())
}

#[test]
fn 모델_압축기_생성_테스트() {
    let config = CompressionConfig::default();
    let compressor = ModelCompressor::new("test-model", config);
    
    let layout = compressor.get_layout();
    assert_eq!(layout.model_name, "test-model");
    assert_eq!(layout.weights.len(), 0);
    assert_eq!(layout.total_blocks, 0);
}

#[test]
fn 기본_설정_압축_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let input_path = temp_dir.path().join("input");
    let output_path = temp_dir.path().join("output");
    
    // 입력 디렉토리 생성
    fs::create_dir_all(&input_path).unwrap();
    테스트용_메타데이터_생성(&input_path).unwrap();
    
    // 압축 수행
    let config = CompressionConfig {
        block_size: 32,
        quality_grade: QualityGrade::C,
        transform_type: TransformType::Dwt,
        profile: CompressionProfile::Fast,
        custom_coefficients: Some(50), // 적은 계수로 빠른 테스트
        min_block_count: None,
        rmse_threshold: Some(0.5), // 관대한 임계값
        compression_ratio_threshold: Some(2.0), // 낮은 압축률 요구
    };
    
    let mut compressor = ModelCompressor::new("test-compression", config);
    let result = compressor.compress_from_numpy_dir(&input_path, &output_path);
    
    assert!(result.is_ok());
    
    // 출력 파일들 확인
    assert!(output_path.join("rbe_model.bin").exists());
    assert!(output_path.join("weight_layout.json").exists());
    assert!(output_path.join("model_config.json").exists());
    assert!(output_path.join("compression_report.json").exists());
    
    // 압축된 레이아웃 확인
    let layout = compressor.get_layout();
    assert_eq!(layout.model_name, "test-compression");
    assert!(layout.weights.len() >= 2); // 2D 가중치들만 압축됨
    assert!(layout.total_blocks > 0);
    
    // 2D 가중치들이 압축되었는지 확인
    let weight_names: Vec<&str> = layout.weights.iter().map(|w| w.name.as_str()).collect();
    assert!(weight_names.contains(&"transformer.wte.weight"));
    assert!(weight_names.contains(&"transformer.h.0.attn.c_attn.weight"));
    // 1D 가중치는 스킵되어야 함
    assert!(!weight_names.contains(&"transformer.h.0.ln_1.weight"));
}

#[test]
fn 빠른_압축_편의_함수_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let input_path = temp_dir.path().join("input");
    let output_path = temp_dir.path().join("output");
    
    // 입력 디렉토리 생성
    fs::create_dir_all(&input_path).unwrap();
    테스트용_메타데이터_생성(&input_path).unwrap();
    
    // 빠른 압축 수행
    let result = ModelCompressor::compress_fast(
        "fast-test",
        &input_path,
        &output_path,
    );
    
    assert!(result.is_ok());
    
    // 출력 파일들 확인
    assert!(output_path.join("rbe_model.bin").exists());
    assert!(output_path.join("weight_layout.json").exists());
    
    // 레이아웃 파일 내용 확인
    let layout_content = fs::read_to_string(output_path.join("weight_layout.json")).unwrap();
    let layout: serde_json::Value = serde_json::from_str(&layout_content).unwrap();
    
    assert_eq!(layout["model_name"], "fast-test");
    assert!(layout["weights"].as_array().unwrap().len() >= 2);
}

#[test]
fn 커스텀_압축_편의_함수_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let input_path = temp_dir.path().join("input");
    let output_path = temp_dir.path().join("output");
    
    // 입력 디렉토리 생성
    fs::create_dir_all(&input_path).unwrap();
    테스트용_메타데이터_생성(&input_path).unwrap();
    
    // 커스텀 압축 수행
    let result = ModelCompressor::compress_custom(
        "custom-test",
        &input_path,
        &output_path,
        32,   // block_size
        0.1,  // rmse_threshold
        5.0,  // compression_ratio
    );
    
    assert!(result.is_ok());
    
    // 출력 파일들 확인
    assert!(output_path.join("rbe_model.bin").exists());
    assert!(output_path.join("weight_layout.json").exists());
    
    // 모델 설정 파일 확인
    let config_content = fs::read_to_string(output_path.join("model_config.json")).unwrap();
    let config: serde_json::Value = serde_json::from_str(&config_content).unwrap();
    
    assert_eq!(config["compression"]["block_size"], 32);
}

#[test]
fn 메타데이터_없음_오류_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let input_path = temp_dir.path().join("input");
    let output_path = temp_dir.path().join("output");
    
    // 메타데이터 없는 빈 디렉토리
    fs::create_dir_all(&input_path).unwrap();
    
    let config = CompressionConfig::fast();
    let mut compressor = ModelCompressor::new("error-test", config);
    let result = compressor.compress_from_numpy_dir(&input_path, &output_path);
    
    assert!(result.is_err());
}

#[test]
fn 입력_디렉토리_없음_오류_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let nonexistent_path = temp_dir.path().join("nonexistent");
    let output_path = temp_dir.path().join("output");
    
    let config = CompressionConfig::fast();
    let mut compressor = ModelCompressor::new("error-test", config);
    let result = compressor.compress_from_numpy_dir(&nonexistent_path, &output_path);
    
    assert!(result.is_err());
}

#[test]
fn 압축_성능_보고서_생성_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let input_path = temp_dir.path().join("input");
    let output_path = temp_dir.path().join("output");
    
    // 입력 디렉토리 생성
    fs::create_dir_all(&input_path).unwrap();
    테스트용_메타데이터_생성(&input_path).unwrap();
    
    // 압축 수행
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
    
    let mut compressor = ModelCompressor::new("report-test", config);
    compressor.compress_from_numpy_dir(&input_path, &output_path).unwrap();
    
    // 압축 보고서 확인
    let report_path = output_path.join("compression_report.json");
    assert!(report_path.exists());
    
    let report_content = fs::read_to_string(report_path).unwrap();
    let report: serde_json::Value = serde_json::from_str(&report_content).unwrap();
    
    // 보고서 구조 확인
    assert!(report.get("compression_summary").is_some());
    assert!(report.get("layer_statistics").is_some());
    
    let summary = &report["compression_summary"];
    assert_eq!(summary["model_name"], "report-test");
    assert!(summary["total_weights"].as_u64().unwrap() >= 2);
    assert!(summary["total_blocks"].as_u64().unwrap() > 0);
    assert!(summary["overall_compression_ratio"].as_f64().unwrap() > 1.0);
    
    let stats = &report["layer_statistics"];
    assert!(stats.get("largest_layers").is_some());
    assert!(stats.get("best_compression").is_some());
}

#[test]
fn 다양한_압축_프로파일_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let input_path = temp_dir.path().join("input");
    
    // 입력 디렉토리 생성
    fs::create_dir_all(&input_path).unwrap();
    테스트용_메타데이터_생성(&input_path).unwrap();
    
    let profiles = vec![
        (CompressionProfile::UltraFast, "ultrafast"),
        (CompressionProfile::Fast, "fast"),
        (CompressionProfile::Balanced, "balanced"),
    ];
    
    for (profile, name) in profiles {
        let output_path = temp_dir.path().join(format!("output_{}", name));
        
        let config = CompressionConfig {
            block_size: 32,
            quality_grade: QualityGrade::C,
            transform_type: TransformType::Dwt,
            profile,
            custom_coefficients: Some(50),
            min_block_count: None,
            rmse_threshold: Some(1.0), // 매우 관대한 임계값
            compression_ratio_threshold: Some(1.5),
        };
        
        let mut compressor = ModelCompressor::new(&format!("{}-test", name), config);
        let result = compressor.compress_from_numpy_dir(&input_path, &output_path);
        
        assert!(result.is_ok(), "프로파일 {} 압축 실패", name);
        
        // 기본 출력 파일들 확인
        assert!(output_path.join("rbe_model.bin").exists());
        assert!(output_path.join("weight_layout.json").exists());
        
        // 바이너리 파일이 비어있지 않은지 확인
        let binary_size = fs::metadata(output_path.join("rbe_model.bin")).unwrap().len();
        assert!(binary_size > 0, "프로파일 {} 바이너리 파일이 비어있음", name);
    }
}

#[test]
fn 변환_타입별_압축_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let input_path = temp_dir.path().join("input");
    
    // 입력 디렉토리 생성
    fs::create_dir_all(&input_path).unwrap();
    테스트용_메타데이터_생성(&input_path).unwrap();
    
    let transform_types = vec![
        (TransformType::Dwt, "dwt"),
        (TransformType::Dct, "dct"),
    ];
    
    for (transform_type, name) in transform_types {
        let output_path = temp_dir.path().join(format!("output_{}", name));
        
        let config = CompressionConfig {
            block_size: 32,
            quality_grade: QualityGrade::C,
            transform_type,
            profile: CompressionProfile::Fast,
            custom_coefficients: Some(50),
            min_block_count: None,
            rmse_threshold: Some(1.0),
            compression_ratio_threshold: Some(1.5),
        };
        
        let mut compressor = ModelCompressor::new(&format!("{}-test", name), config);
        let result = compressor.compress_from_numpy_dir(&input_path, &output_path);
        
        assert!(result.is_ok(), "변환 타입 {} 압축 실패", name);
        
        // 레이아웃에서 변환 타입 확인
        let layout = compressor.get_layout();
        for weight in &layout.weights {
            assert_eq!(weight.transform_type, transform_type);
        }
    }
}

#[test]
fn 압축_실패_처리_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let input_path = temp_dir.path().join("input");
    let output_path = temp_dir.path().join("output");
    
    // 입력 디렉토리 생성
    fs::create_dir_all(&input_path).unwrap();
    
    // 문제가 있는 메타데이터 생성 (잘못된 파일 경로)
    let metadata = json!({
        "valid.weight": {
            "shape": [32, 32],
            "file": "valid_weight.npy"
        },
        "invalid.weight": {
            "shape": [64, 64],
            "file": "nonexistent_file.npy" // 존재하지 않는 파일
        }
    });
    
    let metadata_path = input_path.join("metadata.json");
    fs::write(metadata_path, serde_json::to_string_pretty(&metadata).unwrap()).unwrap();
    
    // 유효한 파일만 생성
    let valid_data: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001).collect();
    테스트용_numpy_파일_생성(
        &input_path.join("valid_weight.npy"),
        &[32, 32],
        &valid_data,
    ).unwrap();
    
    // 압축 수행 (일부 실패 예상)
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
    
    let mut compressor = ModelCompressor::new("failure-test", config);
    let result = compressor.compress_from_numpy_dir(&input_path, &output_path);
    
    // 전체적으로는 성공해야 함 (일부 실패 허용)
    assert!(result.is_ok());
    
    // 성공한 가중치는 압축되어야 함
    let layout = compressor.get_layout();
    assert!(layout.weights.len() >= 1); // 최소 1개는 성공
    
    let weight_names: Vec<&str> = layout.weights.iter().map(|w| w.name.as_str()).collect();
    assert!(weight_names.contains(&"valid.weight"));
} 
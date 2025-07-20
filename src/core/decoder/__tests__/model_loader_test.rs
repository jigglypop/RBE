//! ModelLoader 테스트

use crate::core::decoder::{RBEModelLoader as ModelLoader, LoadedWeight};
use crate::core::encoder::{ModelLayout, WeightInfo};
use crate::packed_params::{HybridEncodedBlock, TransformType, ResidualCoefficient};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use tempfile::TempDir;

fn 테스트_디렉토리_생성() -> TempDir {
    tempfile::tempdir().expect("임시 디렉토리 생성 실패")
}

fn 테스트용_레이아웃_생성() -> ModelLayout {
    use std::collections::HashMap;
    use crate::core::encoder::weight_mapper::CompressionMetadata;
    
    ModelLayout {
        model_type: "test-model".to_string(),
        total_params: 6400 + 12288, // 100*64 + 64*192
        total_blocks: 5,
        weights: vec![
            WeightInfo {
                name: "transformer.wte.weight".to_string(),
                offset_bytes: 0,
                num_blocks: 2,
                original_shape: vec![100, 64],
                compression_type: "rbe".to_string(),
                compression_ratio: 50.0,
                rmse: Some(0.001),
            },
            WeightInfo {
                name: "transformer.h.0.attn.weight".to_string(),
                offset_bytes: 1024, // 2 * 512 bytes per block
                num_blocks: 3,
                original_shape: vec![64, 192],
                compression_type: "rbe".to_string(),
                compression_ratio: 75.0,
                rmse: Some(0.002),
            },
        ],
        metadata: HashMap::new(),
        compression_config: CompressionMetadata {
            block_size: 32,
            transform_type: "dwt".to_string(),
            coefficients: 200,
            quality_grade: "B".to_string(),
        },
    }
}

fn 테스트용_압축_블록_생성(개수: usize, 시작_값: f32) -> Vec<HybridEncodedBlock> {
    (0..개수).map(|i| HybridEncodedBlock {
        rbe_params: [(시작_값 + i as f32); 8],
        residuals: vec![
            ResidualCoefficient {
                index: (i as u16, i as u16),
                value: (시작_값 + i as f32) * 0.1,
            },
            ResidualCoefficient {
                index: ((i + 1) as u16, i as u16),
                value: (시작_값 + i as f32) * 0.2,
            },
        ],
        rows: 32,
        cols: 32,
        transform_type: TransformType::Dwt,
    }).collect()
}

fn 테스트용_바이너리_파일_생성(temp_dir: &Path, blocks_sets: Vec<Vec<HybridEncodedBlock>>) -> std::io::Result<()> {
    let binary_path = temp_dir.join("rbe_model.bin");
    let mut file = File::create(binary_path)?;
    
    for blocks in blocks_sets {
        for block in blocks {
            let config = bincode::config::standard();
            let block_bytes = bincode::encode_to_vec(&block, config).unwrap();
            file.write_all(&block_bytes)?;
        }
    }
    
    Ok(())
}

fn 테스트용_레이아웃_파일_생성(temp_dir: &Path, layout: &ModelLayout) -> std::io::Result<()> {
    let layout_path = temp_dir.join("weight_layout.json");
    let layout_json = serde_json::to_string_pretty(layout).unwrap();
    fs::write(layout_path, layout_json)
}

#[test]
fn 모델_로더_생성_성공_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let temp_path = temp_dir.path();
    
    // 테스트 파일들 생성
    let layout = 테스트용_레이아웃_생성();
    테스트용_레이아웃_파일_생성(temp_path, &layout).unwrap();
    
    let blocks1 = 테스트용_압축_블록_생성(2, 1.0);
    let blocks2 = 테스트용_압축_블록_생성(3, 10.0);
    테스트용_바이너리_파일_생성(temp_path, vec![blocks1, blocks2]).unwrap();
    
    // ModelLoader 생성
    let loader = ModelLoader::new(temp_path);
    assert!(loader.is_ok());
    
    let loader = loader.unwrap();
    assert_eq!(loader.get_layout().model_type, "test-model");
    assert_eq!(loader.list_weights().len(), 2);
}

#[test]
fn 레이아웃_파일_없음_오류_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let temp_path = temp_dir.path();
    
    // 레이아웃 파일 없이 로더 생성 시도
    let result = ModelLoader::new(temp_path);
    assert!(result.is_err());
}

#[test]
fn 바이너리_파일_없음_오류_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let temp_path = temp_dir.path();
    
    // 레이아웃 파일만 생성
    let layout = 테스트용_레이아웃_생성();
    테스트용_레이아웃_파일_생성(temp_path, &layout).unwrap();
    
    // 바이너리 파일 없이 로더 생성 시도
    let result = ModelLoader::new(temp_path);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("바이너리 파일 없음"));
}

#[test]
fn 개별_가중치_로드_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let temp_path = temp_dir.path();
    
    // 테스트 파일들 생성
    let layout = 테스트용_레이아웃_생성();
    테스트용_레이아웃_파일_생성(temp_path, &layout).unwrap();
    
    let blocks1 = 테스트용_압축_블록_생성(2, 1.0);
    let blocks2 = 테스트용_압축_블록_생성(3, 10.0);
    테스트용_바이너리_파일_생성(temp_path, vec![blocks1.clone(), blocks2.clone()]).unwrap();
    
    // ModelLoader 생성
    let mut loader = ModelLoader::new(temp_path).unwrap();
    
    // 첫 번째 가중치 로드
    let weight1 = loader.load_weight("transformer.wte.weight").unwrap();
    assert!(weight1.is_some());
    
    let loaded_weight1 = weight1.unwrap();
    assert_eq!(loaded_weight1.name, "transformer.wte.weight");
    assert_eq!(loaded_weight1.blocks.len(), 2);
    assert_eq!(loaded_weight1.original_shape, vec![100, 64]);
    assert_eq!(loaded_weight1.block_size, 32);
    assert_eq!(loaded_weight1.rmse, 0.001);
    
    // 두 번째 가중치 로드
    let weight2 = loader.load_weight("transformer.h.0.attn.weight").unwrap();
    assert!(weight2.is_some());
    
    let loaded_weight2 = weight2.unwrap();
    assert_eq!(loaded_weight2.name, "transformer.h.0.attn.weight");
    assert_eq!(loaded_weight2.blocks.len(), 3);
    assert_eq!(loaded_weight2.original_shape, vec![64, 192]);
    assert_eq!(loaded_weight2.block_size, 64);
    assert_eq!(loaded_weight2.rmse, 0.002);
}

#[test]
fn 존재하지_않는_가중치_로드_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let temp_path = temp_dir.path();
    
    // 테스트 파일들 생성
    let layout = 테스트용_레이아웃_생성();
    테스트용_레이아웃_파일_생성(temp_path, &layout).unwrap();
    
    let blocks1 = 테스트용_압축_블록_생성(2, 1.0);
    let blocks2 = 테스트용_압축_블록_생성(3, 10.0);
    테스트용_바이너리_파일_생성(temp_path, vec![blocks1, blocks2]).unwrap();
    
    // ModelLoader 생성
    let mut loader = ModelLoader::new(temp_path).unwrap();
    
    // 존재하지 않는 가중치 로드 시도
    let result = loader.load_weight("nonexistent.weight").unwrap();
    assert!(result.is_none());
}

#[test]
fn 가중치_캐싱_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let temp_path = temp_dir.path();
    
    // 테스트 파일들 생성
    let layout = 테스트용_레이아웃_생성();
    테스트용_레이아웃_파일_생성(temp_path, &layout).unwrap();
    
    let blocks1 = 테스트용_압축_블록_생성(2, 1.0);
    let blocks2 = 테스트용_압축_블록_생성(3, 10.0);
    테스트용_바이너리_파일_생성(temp_path, vec![blocks1, blocks2]).unwrap();
    
    let mut loader = ModelLoader::new(temp_path).unwrap();
    
    // 초기 캐시 상태 확인
    let (cached, total) = loader.cache_info();
    assert_eq!(cached, 0);
    assert_eq!(total, 2);
    
    // 첫 번째 로드 (파일에서 읽기)
    let weight1_first = loader.load_weight("transformer.wte.weight").unwrap();
    assert!(weight1_first.is_some());
    
    // 캐시 상태 확인
    let (cached, total) = loader.cache_info();
    assert_eq!(cached, 1);
    assert_eq!(total, 2);
    
    // 두 번째 로드 (캐시에서 읽기)
    let weight1_second = loader.load_weight("transformer.wte.weight").unwrap();
    assert!(weight1_second.is_some());
    
    // 캐시 데이터와 동일한지 확인
    let first = weight1_first.unwrap();
    let second = weight1_second.unwrap();
    assert_eq!(first.name, second.name);
    assert_eq!(first.blocks.len(), second.blocks.len());
    
    // 캐시 지우기
    loader.clear_cache();
    let (cached, total) = loader.cache_info();
    assert_eq!(cached, 0);
    assert_eq!(total, 2);
}

#[test]
fn 가중치_존재_확인_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let temp_path = temp_dir.path();
    
    // 테스트 파일들 생성
    let layout = 테스트용_레이아웃_생성();
    테스트용_레이아웃_파일_생성(temp_path, &layout).unwrap();
    
    let blocks1 = 테스트용_압축_블록_생성(2, 1.0);
    let blocks2 = 테스트용_압축_블록_생성(3, 10.0);
    테스트용_바이너리_파일_생성(temp_path, vec![blocks1, blocks2]).unwrap();
    
    let loader = ModelLoader::new(temp_path).unwrap();
    
    // 존재하는 가중치들
    assert!(loader.has_weight("transformer.wte.weight"));
    assert!(loader.has_weight("transformer.h.0.attn.weight"));
    
    // 존재하지 않는 가중치
    assert!(!loader.has_weight("nonexistent.weight"));
    assert!(!loader.has_weight("transformer.h.1.attn.weight"));
}

#[test]
fn 가중치_정보_조회_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let temp_path = temp_dir.path();
    
    // 테스트 파일들 생성
    let layout = 테스트용_레이아웃_생성();
    테스트용_레이아웃_파일_생성(temp_path, &layout).unwrap();
    
    let blocks1 = 테스트용_압축_블록_생성(2, 1.0);
    let blocks2 = 테스트용_압축_블록_생성(3, 10.0);
    테스트용_바이너리_파일_생성(temp_path, vec![blocks1, blocks2]).unwrap();
    
    let loader = ModelLoader::new(temp_path).unwrap();
    
    // 첫 번째 가중치 정보 조회
    let weight_info = loader.get_weight_info("transformer.wte.weight");
    assert!(weight_info.is_some());
    
    let info = weight_info.unwrap();
    assert_eq!(info.name, "transformer.wte.weight");
    assert_eq!(info.offset_bytes, 0);
    assert_eq!(info.num_blocks, 2);
    assert_eq!(info.original_shape, vec![100, 64]);
    assert_eq!(info.compression_ratio, 50.0);
    
    // 두 번째 가중치 정보 조회
    let weight_info = loader.get_weight_info("transformer.h.0.attn.weight");
    assert!(weight_info.is_some());
    
    let info = weight_info.unwrap();
    assert_eq!(info.name, "transformer.h.0.attn.weight");
    assert_eq!(info.offset_bytes, 1024);
    assert_eq!(info.num_blocks, 3);
    assert_eq!(info.original_shape, vec![64, 192]);
    assert_eq!(info.compression_ratio, 75.0);
}

#[test]
fn 가중치_이름_목록_조회_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let temp_path = temp_dir.path();
    
    // 테스트 파일들 생성
    let layout = 테스트용_레이아웃_생성();
    테스트용_레이아웃_파일_생성(temp_path, &layout).unwrap();
    
    let blocks1 = 테스트용_압축_블록_생성(2, 1.0);
    let blocks2 = 테스트용_압축_블록_생성(3, 10.0);
    테스트용_바이너리_파일_생성(temp_path, vec![blocks1, blocks2]).unwrap();
    
    let loader = ModelLoader::new(temp_path).unwrap();
    
    let weight_names = loader.list_weight_names();
    assert_eq!(weight_names.len(), 2);
    assert!(weight_names.contains(&"transformer.wte.weight"));
    assert!(weight_names.contains(&"transformer.h.0.attn.weight"));
}

#[test]
fn 가중치_복원_및_검증_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let temp_path = temp_dir.path();
    
    // 단순한 레이아웃으로 테스트
    use std::collections::HashMap;
    use crate::core::encoder::weight_mapper::CompressionMetadata;
    
    let layout = ModelLayout {
        model_type: "decode-test".to_string(),
        total_params: 1024, // 32*32
        total_blocks: 1,
        weights: vec![
            WeightInfo {
                name: "simple.weight".to_string(),
                offset_bytes: 0,
                num_blocks: 1,
                original_shape: vec![32, 32], // 작은 크기로 테스트
                compression_type: "rbe".to_string(),
                compression_ratio: 10.0,
                rmse: Some(0.001),
            },
        ],
        metadata: HashMap::new(),
        compression_config: CompressionMetadata {
            block_size: 32,
            transform_type: "dwt".to_string(),
            coefficients: 100,
            quality_grade: "B".to_string(),
        },
    };
    
    테스트용_레이아웃_파일_생성(temp_path, &layout).unwrap();
    
    let blocks = 테스트용_압축_블록_생성(1, 5.0);
    테스트용_바이너리_파일_생성(temp_path, vec![blocks]).unwrap();
    
    let loader = ModelLoader::new(temp_path).unwrap();
    
    // 가중치 복원
    let result = loader.decode_and_verify_weight("simple.weight");
    assert!(result.is_ok());
    
    let reconstructed = result.unwrap();
    assert_eq!(reconstructed.len(), 32 * 32); // 1024개 요소
    
    // 복원된 데이터가 모두 유효한 값인지 확인
    for value in &reconstructed {
        assert!(value.is_finite());
    }
}

#[test]
fn 모든_가중치_로드_테스트() {
    let temp_dir = 테스트_디렉토리_생성();
    let temp_path = temp_dir.path();
    
    // 테스트 파일들 생성
    let layout = 테스트용_레이아웃_생성();
    테스트용_레이아웃_파일_생성(temp_path, &layout).unwrap();
    
    let blocks1 = 테스트용_압축_블록_생성(2, 1.0);
    let blocks2 = 테스트용_압축_블록_생성(3, 10.0);
    테스트용_바이너리_파일_생성(temp_path, vec![blocks1, blocks2]).unwrap();
    
    let mut loader = ModelLoader::new(temp_path).unwrap();
    
    // 모든 가중치 로드
    let all_weights = loader.load_all_weights();
    assert!(all_weights.is_ok());
    
    let weights = all_weights.unwrap();
    assert_eq!(weights.len(), 2);
    
    assert!(weights.contains_key("transformer.wte.weight"));
    assert!(weights.contains_key("transformer.h.0.attn.weight"));
    
    // 각 가중치의 내용 검증
    let weight1 = weights.get("transformer.wte.weight").unwrap();
    assert_eq!(weight1.blocks.len(), 2);
    assert_eq!(weight1.original_shape, vec![100, 64]);
    
    let weight2 = weights.get("transformer.h.0.attn.weight").unwrap();
    assert_eq!(weight2.blocks.len(), 3);
    assert_eq!(weight2.original_shape, vec![64, 192]);
} 
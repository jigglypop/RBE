//! WeightMapper 테스트

use crate::encoder::{WeightMapper, CompressionMetrics, WeightInfo, ModelLayout};
use crate::packed_params::{HybridEncodedBlock, TransformType, ResidualCoefficient};

fn 테스트용_압축_블록_생성(개수: usize) -> Vec<HybridEncodedBlock> {
    (0..개수).map(|i| HybridEncodedBlock {
        rbe_params: [i as f32; 8],
        residuals: vec![ResidualCoefficient {
            index: (i as u16, i as u16),
            value: i as f32 * 0.1,
        }],
        rows: 32,
        cols: 32,
        transform_type: TransformType::Dwt,
    }).collect()
}

fn 테스트용_메트릭_생성() -> CompressionMetrics {
    CompressionMetrics::new(
        1024 * 4, // 원본 크기 (1024 f32)
        256,      // 압축 크기
        0.001,    // RMSE
        TransformType::Dwt,
        64,       // 블록 크기
        200,      // 계수 개수
    )
}

#[test]
fn 가중치_매퍼_생성_테스트() {
    let mapper = WeightMapper::new_for_compression("test-model");
    let layout = mapper.get_layout();
    
    assert_eq!(layout.model_name, "test-model");
    assert_eq!(layout.total_layers, 0);
    assert_eq!(layout.total_blocks, 0);
    assert_eq!(layout.layout_version, "1.0");
    assert!(layout.weights.is_empty());
}

#[test]
fn 가중치_추가_및_조회_테스트() {
    let mut mapper = WeightMapper::new_for_compression("test-model");
    
    // 테스트 데이터 준비
    let weight_name = "transformer.h.0.attn.weight";
    let original_data = vec![1.0f32; 1024];
    let original_shape = vec![32, 32];
    let compressed_blocks = 테스트용_압축_블록_생성(4);
    let metrics = 테스트용_메트릭_생성();
    
    // 가중치 추가
    let result = mapper.add_compressed_weight(
        weight_name,
        &original_data,
        &original_shape,
        compressed_blocks.clone(),
        metrics.clone(),
    );
    
    assert!(result.is_ok());
    
    // 가중치 조회
    let weight_info = mapper.get_weight_info(weight_name);
    assert!(weight_info.is_some());
    
    let info = weight_info.unwrap();
    assert_eq!(info.name, weight_name);
    assert_eq!(info.num_blocks, 4);
    assert_eq!(info.original_shape, vec![32, 32]);
    assert_eq!(info.compression_ratio, metrics.compression_ratio);
    assert_eq!(info.rmse, metrics.rmse);
    assert_eq!(info.transform_type, TransformType::Dwt);
    assert_eq!(info.block_size, 64);
    assert_eq!(info.coefficients, 200);
}

#[test]
fn 여러_가중치_추가_및_오프셋_테스트() {
    let mut mapper = WeightMapper::new_for_compression("test-model");
    
    // 첫 번째 가중치 추가
    let weight1_blocks = 테스트용_압축_블록_생성(2);
    let weight1_metrics = 테스트용_메트릭_생성();
    
    mapper.add_compressed_weight(
        "weight1",
        &vec![1.0f32; 100],
        &vec![10, 10],
        weight1_blocks.clone(),
        weight1_metrics,
    ).unwrap();
    
    // 두 번째 가중치 추가
    let weight2_blocks = 테스트용_압축_블록_생성(3);
    let weight2_metrics = 테스트용_메트릭_생성();
    
    mapper.add_compressed_weight(
        "weight2",
        &vec![2.0f32; 200],
        &vec![20, 10],
        weight2_blocks.clone(),
        weight2_metrics,
    ).unwrap();
    
    // 오프셋 검증
    let weight1_info = mapper.get_weight_info("weight1").unwrap();
    let weight2_info = mapper.get_weight_info("weight2").unwrap();
    
    assert_eq!(weight1_info.offset_bytes, 0);
    assert_eq!(weight1_info.num_blocks, 2);
    
    let expected_offset = weight1_blocks.len() * std::mem::size_of::<HybridEncodedBlock>();
    assert_eq!(weight2_info.offset_bytes, expected_offset as u64);
    assert_eq!(weight2_info.num_blocks, 3);
    
    // 전체 통계 검증
    let layout = mapper.get_layout();
    assert_eq!(layout.weights.len(), 2);
    assert_eq!(layout.total_blocks, 5);
}

#[test]
fn 가중치_이름_목록_조회_테스트() {
    let mut mapper = WeightMapper::new_for_compression("test-model");
    
    // 여러 가중치 추가
    let names = vec!["weight_a", "weight_b", "weight_c"];
    for name in &names {
        mapper.add_compressed_weight(
            name,
            &vec![1.0f32; 64],
            &vec![8, 8],
            테스트용_압축_블록_생성(1),
            테스트용_메트릭_생성(),
        ).unwrap();
    }
    
    // 이름 목록 조회
    let listed_names = mapper.list_weight_names();
    assert_eq!(listed_names.len(), 3);
    
    for name in &names {
        assert!(listed_names.contains(name));
    }
}

#[test]
fn 읽기_정보_조회_테스트() {
    let mut mapper = WeightMapper::new_for_compression("test-model");
    
    // 가중치 추가
    let blocks = 테스트용_압축_블록_생성(5);
    mapper.add_compressed_weight(
        "test_weight",
        &vec![1.0f32; 256],
        &vec![16, 16],
        blocks,
        테스트용_메트릭_생성(),
    ).unwrap();
    
    // 읽기 정보 조회
    let read_info = mapper.get_read_info("test_weight");
    assert!(read_info.is_some());
    
    let (offset, num_blocks) = read_info.unwrap();
    assert_eq!(offset, 0);
    assert_eq!(num_blocks, 5);
    
    // 존재하지 않는 가중치
    let no_info = mapper.get_read_info("nonexistent");
    assert!(no_info.is_none());
}

#[test]
fn 레이아웃_직렬화_역직렬화_테스트() {
    let mut mapper = WeightMapper::new_for_compression("serialization-test");
    
    // 테스트 가중치 추가
    mapper.add_compressed_weight(
        "test.weight",
        &vec![1.0f32; 100],
        &vec![10, 10],
        테스트용_압축_블록_생성(2),
        테스트용_메트릭_생성(),
    ).unwrap();
    
    // 직렬화
    let json_str = mapper.serialize_layout();
    assert!(json_str.is_ok());
    
    let json = json_str.unwrap();
    assert!(json.contains("serialization-test"));
    assert!(json.contains("test.weight"));
    
    // 역직렬화
    let deserialized_layout = WeightMapper::deserialize_layout(&json);
    assert!(deserialized_layout.is_ok());
    
    let layout = deserialized_layout.unwrap();
    assert_eq!(layout.model_name, "serialization-test");
    assert_eq!(layout.weights.len(), 1);
    assert_eq!(layout.weights[0].name, "test.weight");
    assert_eq!(layout.weights[0].num_blocks, 2);
}

#[test]
fn 레이아웃에서_매퍼_로드_테스트() {
    // 테스트용 레이아웃 생성
    let layout = ModelLayout {
        model_name: "loaded-model".to_string(),
        compression_timestamp: "1234567890".to_string(),
        total_layers: 1,
        total_blocks: 3,
        layout_version: "1.0".to_string(),
        weights: vec![
            WeightInfo {
                name: "weight1".to_string(),
                offset_bytes: 0,
                num_blocks: 1,
                original_shape: vec![10, 10],
                compressed_shape: vec![1, 1],
                compression_ratio: 100.0,
                rmse: 0.001,
                transform_type: TransformType::Dwt,
                block_size: 32,
                coefficients: 100,
            },
            WeightInfo {
                name: "weight2".to_string(),
                offset_bytes: 1000,
                num_blocks: 2,
                original_shape: vec![20, 20],
                compressed_shape: vec![1, 2],
                compression_ratio: 200.0,
                rmse: 0.002,
                transform_type: TransformType::Dct,
                block_size: 64,
                coefficients: 200,
            },
        ],
    };
    
    // 레이아웃에서 매퍼 로드
    let mapper = WeightMapper::load_from_layout(layout);
    
    // 검증
    assert_eq!(mapper.get_layout().model_name, "loaded-model");
    assert_eq!(mapper.list_weight_names().len(), 2);
    
    let weight1_info = mapper.get_weight_info("weight1");
    assert!(weight1_info.is_some());
    assert_eq!(weight1_info.unwrap().offset_bytes, 0);
    
    let weight2_info = mapper.get_weight_info("weight2");
    assert!(weight2_info.is_some());
    assert_eq!(weight2_info.unwrap().offset_bytes, 1000);
    
    // 읽기 정보 검증
    let (offset1, blocks1) = mapper.get_read_info("weight1").unwrap();
    assert_eq!(offset1, 0);
    assert_eq!(blocks1, 1);
    
    let (offset2, blocks2) = mapper.get_read_info("weight2").unwrap();
    assert_eq!(offset2, 1000);
    assert_eq!(blocks2, 2);
}

#[test]
fn 잘못된_입력_처리_테스트() {
    let mut mapper = WeightMapper::new_for_compression("error-test");
    
    // 1D 텐서 (지원되지 않음)
    let result = mapper.add_compressed_weight(
        "1d_weight",
        &vec![1.0f32; 100],
        &vec![100], // 1D shape
        테스트용_압축_블록_생성(1),
        테스트용_메트릭_생성(),
    );
    
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("2D 텐서만 지원됩니다"));
    
    // 3D 텐서 (지원되지 않음)
    let result = mapper.add_compressed_weight(
        "3d_weight",
        &vec![1.0f32; 100],
        &vec![10, 5, 2], // 3D shape
        테스트용_압축_블록_생성(1),
        테스트용_메트릭_생성(),
    );
    
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("2D 텐서만 지원됩니다"));
}

#[test]
fn 압축_메트릭_생성_테스트() {
    let metrics = CompressionMetrics::new(
        1000, // 원본 크기
        100,  // 압축 크기
        0.005, // RMSE
        TransformType::Adaptive,
        128,  // 블록 크기
        500,  // 계수 개수
    );
    
    assert_eq!(metrics.compression_ratio, 10.0); // 1000/100
    assert_eq!(metrics.rmse, 0.005);
    assert_eq!(metrics.transform_type, TransformType::Adaptive);
    assert_eq!(metrics.block_size, 128);
    assert_eq!(metrics.coefficients, 500);
} 
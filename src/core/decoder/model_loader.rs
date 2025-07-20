//! 동적 레이아웃 기반 모델 로더
//! 
//! WeightMapper가 생성한 레이아웃 정보를 사용해서
//! 압축된 RBE 모델을 정확하게 로딩하고 추론에 사용할 수 있도록 구성

use crate::core::packed_params::HybridEncodedBlock;
use anyhow::{Context, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use crate::core::encoder::weight_mapper::{ModelLayout, WeightInfo};

/// 로드된 가중치 타입
pub enum LoadedWeight {
    /// RBE 압축된 가중치
    Compressed(Vec<HybridEncodedBlock>),
    /// 압축되지 않은 원본 가중치
    Raw(Vec<f32>),
}

/// RBE 모델 로더 - 메타데이터 기반 동적 로딩
pub struct RBEModelLoader {
    /// 모델 레이아웃 정보
    layout: ModelLayout,
    /// 압축된 가중치 데이터 (전체)
    weight_data: Vec<u8>,
    /// 캐시된 가중치 (lazy loading)
    cache: std::collections::HashMap<String, LoadedWeight>,
}

impl RBEModelLoader {
    /// 모델 디렉토리에서 로더 생성
    pub fn new(model_dir: &Path) -> Result<Self> {
        // 1. 레이아웃 파일 로드
        let layout_path = model_dir.join("rbe_layout.json");
        let layout_file = File::open(&layout_path)
            .with_context(|| format!("레이아웃 파일을 열 수 없습니다: {}", layout_path.display()))?;
        let layout: ModelLayout = serde_json::from_reader(layout_file)
            .with_context(|| "레이아웃 파일 파싱 실패")?;
        
        // 2. 압축된 가중치 데이터 로드
        let bin_path = model_dir.join("rbe_model.bin");
        let mut weight_data = Vec::new();
        let mut bin_file = File::open(&bin_path)
            .with_context(|| format!("가중치 파일을 열 수 없습니다: {}", bin_path.display()))?;
        bin_file.read_to_end(&mut weight_data)
            .with_context(|| "가중치 파일 읽기 실패")?;
        
        println!("✅ 모델 로드 완료:");
        println!("  - 모델 타입: {}", layout.model_type);
        println!("  - 총 파라미터: {}", layout.total_params);
        println!("  - 가중치 개수: {}", layout.weights.len());
        println!("  - 데이터 크기: {:.2} MB", weight_data.len() as f64 / 1_048_576.0);
        
        Ok(Self {
            layout,
            weight_data,
            cache: std::collections::HashMap::new(),
        })
    }
    
    /// 가중치 정보를 반환
    pub fn get_weight_info(&self, weight_name: &str) -> Result<&WeightInfo> {
        self.layout.weights.iter()
            .find(|w| w.name == weight_name)
            .with_context(|| format!("'{}' 가중치 정보를 찾을 수 없습니다", weight_name))
    }
    
    /// 가중치 로드 (캐시 활용)
    pub fn load(&mut self, weight_name: &str) -> Result<()> {
        // 이미 캐시에 있으면 스킵
        if self.cache.contains_key(weight_name) {
            return Ok(());
        }
        
        // WeightInfo 가져오기 - self.layout은 immutable borrow이므로 안전
        let info = self.layout.weights.iter()
            .find(|w| w.name == weight_name)
            .with_context(|| format!("'{}' 가중치 정보를 찾을 수 없습니다", weight_name))?;
        
        // 압축 타입에 따라 처리
        let loaded = match info.compression_type.as_str() {
            "rbe" => {
                // RBE 압축된 데이터 로드
                let blocks = self.load_rbe_blocks(info)?;
                LoadedWeight::Compressed(blocks)
            },
            "raw" => {
                // 압축되지 않은 원본 데이터 (향후 지원)
                return Err(anyhow::anyhow!("Raw 가중치 로딩은 아직 구현되지 않았습니다"));
            },
            _ => {
                return Err(anyhow::anyhow!("알 수 없는 압축 타입: {}", info.compression_type));
            }
        };
        
        // 캐시에 저장
        self.cache.insert(weight_name.to_string(), loaded);
        Ok(())
    }
    
    /// RBE 압축 블록 로드
    fn load_rbe_blocks(&self, info: &WeightInfo) -> Result<Vec<HybridEncodedBlock>> {
        let start = info.offset_bytes as usize;
        let buffer = &self.weight_data[start..];
        
        // bincode로 역직렬화
        let config = bincode::config::standard();
        let (blocks, _): (Vec<HybridEncodedBlock>, usize) = bincode::decode_from_slice(buffer, config)
            .with_context(|| format!("'{}' 가중치 역직렬화 실패", info.name))?;
        
        // 블록 개수 검증
        if blocks.len() != info.num_blocks {
            return Err(anyhow::anyhow!(
                "'{}' 가중치의 블록 개수 불일치: 예상 {}, 실제 {}",
                info.name, info.num_blocks, blocks.len()
            ));
        }
        
        Ok(blocks)
    }
    
    /// 가중치를 디코딩하여 원본 형태로 반환
    pub fn decode_weight(&mut self, weight_name: &str) -> Result<Vec<f32>> {
        // 캐시에서 먼저 확인
        if let Some(loaded) = self.cache.get(weight_name) {
            // 캐시에 있으면 info를 가져와서 디코딩
            let info = self.layout.weights.iter()
                .find(|w| w.name == weight_name)
                .with_context(|| format!("'{}' 가중치 정보를 찾을 수 없습니다", weight_name))?;
            
            match loaded {
                LoadedWeight::Compressed(blocks) => {
                    let total_elements: usize = info.original_shape.iter().product();
                    let mut decoded_data = Vec::with_capacity(total_elements);
                    
                    for block in blocks {
                        let block_data = block.decode();
                        decoded_data.extend_from_slice(&block_data);
                    }
                    
                    if decoded_data.len() != total_elements {
                        return Err(anyhow::anyhow!(
                            "'{}' 가중치 디코딩 크기 불일치: 예상 {}, 실제 {}",
                            weight_name, total_elements, decoded_data.len()
                        ));
                    }
                    
                    Ok(decoded_data)
                },
                LoadedWeight::Raw(data) => Ok(data.clone()),
            }
        } else {
            // 캐시에 없으면 로드하고 디코딩
            self.load(weight_name)?;
            
            // 재귀 호출 대신 직접 처리
            let info = self.layout.weights.iter()
                .find(|w| w.name == weight_name)
                .with_context(|| format!("'{}' 가중치 정보를 찾을 수 없습니다", weight_name))?;
            
            if let Some(loaded) = self.cache.get(weight_name) {
                match loaded {
                    LoadedWeight::Compressed(blocks) => {
                        let total_elements: usize = info.original_shape.iter().product();
                        let mut decoded_data = Vec::with_capacity(total_elements);
                        
                        for block in blocks {
                            let block_data = block.decode();
                            decoded_data.extend_from_slice(&block_data);
                        }
                        
                        if decoded_data.len() != total_elements {
                            return Err(anyhow::anyhow!(
                                "'{}' 가중치 디코딩 크기 불일치: 예상 {}, 실제 {}",
                                weight_name, total_elements, decoded_data.len()
                            ));
                        }
                        
                        Ok(decoded_data)
                    },
                    LoadedWeight::Raw(data) => Ok(data.clone()),
                }
            } else {
                Err(anyhow::anyhow!("'{}' 가중치 로드 실패", weight_name))
            }
        }
    }
    
    /// 모든 가중치 이름 목록 반환
    pub fn list_weights(&self) -> Vec<String> {
        self.layout.weights.iter()
            .map(|info| info.name.clone())
            .collect()
    }
    
    /// 모델 레이아웃 반환
    pub fn get_layout(&self) -> &ModelLayout {
        &self.layout
    }
    
    /// 압축 통계 출력
    pub fn print_stats(&self) {
        println!("\n📊 모델 통계:");
        println!("  모델 타입: {}", self.layout.model_type);
        println!("  총 파라미터: {}", self.layout.total_params);
        println!("  총 압축 블록: {}", self.layout.total_blocks);
        println!("  가중치 개수: {}", self.layout.weights.len());
        
        // 압축률 통계
        let total_ratio = self.layout.weights.iter()
            .filter_map(|w| Some(w.compression_ratio))
            .sum::<f32>() / self.layout.weights.len() as f32;
        
        println!("  평균 압축률: {:.1}x", total_ratio);
        
        // RMSE 통계
        let weights_with_rmse: Vec<_> = self.layout.weights.iter()
            .filter_map(|w| w.rmse.map(|r| (w.name.as_str(), r)))
            .collect();
        
        if !weights_with_rmse.is_empty() {
            let avg_rmse = weights_with_rmse.iter()
                .map(|(_, r)| r)
                .sum::<f32>() / weights_with_rmse.len() as f32;
            println!("  평균 RMSE: {:.6}", avg_rmse);
        }
        
        // 캐시 상태
        println!("  캐시된 가중치: {}/{}", self.cache.len(), self.layout.weights.len());
    }
    
    /// 캐시된 압축 블록 수 계산
    pub fn compressed_block_count(&self) -> usize {
        let mut count = 0;
        for info in &self.layout.weights {
            if let Some(LoadedWeight::Compressed(blocks)) = self.cache.get(&info.name) {
                count += blocks.len();
            }
        }
        count
    }
    
    /// 메모리 사용량 계산
    pub fn memory_usage(&self) -> (usize, f64) {
        let cache_size: usize = self.cache.iter()
            .map(|(_, w)| match w {
                LoadedWeight::Compressed(blocks) => blocks.len() * std::mem::size_of::<HybridEncodedBlock>(),
                LoadedWeight::Raw(data) => data.len() * std::mem::size_of::<f32>(),
            })
            .sum();
        
        let total_size = self.weight_data.len() + cache_size;
        let mb = total_size as f64 / 1_048_576.0;
        
        (total_size, mb)
    }
} 
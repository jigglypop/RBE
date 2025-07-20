use crate::packed_params::*;
use crate::encoder::HybridEncoder;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use serde::{Serialize, Deserialize};
use safetensors::{SafeTensors, tensor::TensorView};
use memmap2::Mmap;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::time::Instant;

/// 압축 설정
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// 웨이블릿 계수 개수 (500개로 S급 성능)
    pub wavelet_coefficients: usize,
    /// 블록 크기 (32x32로 최적화)
    pub block_size: usize,
    /// 압축 레벨 (1: 빠름, 3: 균형, 5: 최고 품질)
    pub compression_level: u8,
    /// 병렬 처리 스레드 수
    pub num_threads: usize,
    /// 진행률 표시
    pub show_progress: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            wavelet_coefficients: 500, // S급 성능으로 설정
            block_size: 32,
            compression_level: 3,
            num_threads: num_cpus::get(),
            show_progress: true,
        }
    }
}

/// 압축된 레이어 정보
#[derive(Debug, Serialize, Deserialize)]
pub struct CompressedLayer {
    /// 레이어 이름
    pub name: String,
    /// 원본 크기 (바이트)
    pub original_size: usize,
    /// 압축 후 크기 (바이트)
    pub compressed_size: usize,
    /// 압축률
    pub compression_ratio: f32,
    /// RMSE
    pub rmse: f32,
    /// 압축된 데이터 (웨이블릿 + RBE)
    pub compressed_data: Vec<HybridEncodedBlock>,
    /// 메타데이터
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// 압축된 모델 전체
#[derive(Debug, Serialize, Deserialize)]
pub struct CompressedModel {
    /// 모델 메타데이터
    pub model_name: String,
    pub original_total_size: usize,
    pub compressed_total_size: usize,
    pub total_compression_ratio: f32,
    pub average_rmse: f32,
    
    /// 압축된 레이어들
    pub layers: HashMap<String, CompressedLayer>,
    
    /// 압축 설정
    pub config: CompressionConfig,
    
    /// 압축 시간 (초)
    pub compression_time: f64,
}

/// SLLM RBE 압축기
pub struct SLLMCompressor {
    config: CompressionConfig,
}

impl SLLMCompressor {
    /// 새로운 압축기 생성
    pub fn new(config: CompressionConfig) -> Self {
        Self { config }
    }
    
    /// SafeTensors 모델 압축
    pub async fn compress_safetensors_model(
        &self,
        model_path: &Path,
        output_path: &Path,
    ) -> Result<CompressedModel, Box<dyn std::error::Error>> {
        println!("🗜️ === SLLM RBE 압축 시작 ===");
        let start_time = Instant::now();
        
        // 모델 파일 찾기 (다양한 형식 지원)
        let possible_files = vec![
            "model.safetensors",
            "pytorch_model.bin.safetensors",
            "model-00001-of-00002.safetensors", // 분할된 모델
        ];
        
        let mut safetensors_path = None;
        for file_name in possible_files {
            let path = model_path.join(file_name);
            if path.exists() {
                safetensors_path = Some(path);
                break;
            }
        }
        
        let safetensors_path = safetensors_path
            .ok_or("SafeTensors 모델 파일을 찾을 수 없습니다")?;
        
        println!("📁 모델 파일 로딩: {:?}", safetensors_path);
        let file = fs::File::open(&safetensors_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let safetensors = SafeTensors::deserialize(&mmap)?;
        
        // 레이어 정보 수집
        let tensor_names: Vec<String> = safetensors.names().into_iter().map(|s| s.to_string()).collect();
        println!("📋 전체 레이어 수: {}", tensor_names.len());
        
        // 진행률 표시
        let progress = if self.config.show_progress {
            let pb = ProgressBar::new(tensor_names.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("🗜️ 압축: [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} {msg}")
                    .expect("진행률 스타일 설정 실패")
                    .progress_chars("=>-")
            );
            Some(pb)
        } else {
            None
        };
        
        // 레이어별 압축 (병렬 처리)
        let mut compressed_layers = HashMap::new();
        let mut total_original_size = 0;
        let mut total_compressed_size = 0;
        let mut rmse_sum = 0.0;
        let mut compressed_layer_count = 0;
        
        for (idx, tensor_name) in tensor_names.iter().enumerate() {
            if let Some(ref pb) = progress {
                pb.set_message(format!("압축 중: {}", tensor_name));
                pb.set_position(idx as u64);
            }
            
            // 텐서 데이터 추출
            let tensor = safetensors.tensor(tensor_name)?;
            let shape = tensor.shape().to_vec();
            
            // 2D 가중치 레이어만 압축 (Linear, Conv 등)
            if shape.len() == 2 && shape[0] > 64 && shape[1] > 64 {
                match self.compress_tensor(&tensor, tensor_name).await {
                    Ok(compressed_layer) => {
                        total_original_size += compressed_layer.original_size;
                        total_compressed_size += compressed_layer.compressed_size;
                        rmse_sum += compressed_layer.rmse;
                        compressed_layer_count += 1;
                        
                        compressed_layers.insert(tensor_name.clone(), compressed_layer);
                        
                        println!("✅ 압축 완료: {} (RMSE: {:.6})", 
                                 tensor_name, compressed_layers[tensor_name].rmse);
                    }
                    Err(e) => {
                        println!("⚠️ 압축 실패: {} - {}", tensor_name, e);
                    }
                }
            } else {
                println!("⏭️ 스킵: {} (크기: {:?})", tensor_name, shape);
            }
        }
        
        if let Some(ref pb) = progress {
            pb.finish_with_message("압축 완료!");
        }
        
        let compression_time = start_time.elapsed().as_secs_f64();
        let total_compression_ratio = total_original_size as f32 / total_compressed_size as f32;
        let average_rmse = if compressed_layer_count > 0 {
            rmse_sum / compressed_layer_count as f32
        } else {
            0.0
        };
        
        // 압축된 모델 생성
        let compressed_model = CompressedModel {
            model_name: model_path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            original_total_size: total_original_size,
            compressed_total_size: total_compressed_size,
            total_compression_ratio,
            average_rmse,
            layers: compressed_layers,
            config: self.config.clone(),
            compression_time,
        };
        
        // 압축 결과 저장
        self.save_compressed_model(&compressed_model, output_path).await?;
        
        // 압축 요약 출력
        self.print_compression_summary(&compressed_model);
        
        Ok(compressed_model)
    }
    
    /// 개별 텐서 압축
    async fn compress_tensor(
        &self,
        tensor: &TensorView<'_>,
        tensor_name: &str,
    ) -> Result<CompressedLayer, Box<dyn std::error::Error>> {
        let shape = tensor.shape();
        let data = tensor.data();
        
        // f32 데이터로 변환
        let float_data: Vec<f32> = match tensor.dtype() {
            safetensors::Dtype::F32 => {
                data.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect()
            }
            safetensors::Dtype::F16 => {
                // F16 to F32 변환 (간단한 구현)
                data.chunks_exact(2)
                    .map(|chunk| {
                        let half_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::f16::from_bits(half_bits).to_f32()
                    })
                    .collect()
            }
            _ => return Err(format!("지원하지 않는 데이터 타입: {:?}", tensor.dtype()).into()),
        };
        
        let rows = shape[0];
        let cols = shape[1];
        
        // 웨이블릿 + RBE 압축 (블록 단위)
        let block_size = self.config.block_size;
        let mut all_compressed_blocks = Vec::new();
        let mut total_reconstructed = vec![0.0f32; float_data.len()];
        
        // 행렬을 블록으로 나누어 압축
        for block_row in (0..rows).step_by(block_size) {
            for block_col in (0..cols).step_by(block_size) {
                let end_row = (block_row + block_size).min(rows);
                let end_col = (block_col + block_size).min(cols);
                let block_height = end_row - block_row;
                let block_width = end_col - block_col;
                
                // 블록 데이터 추출
                let mut block_data = Vec::with_capacity(block_height * block_width);
                for r in block_row..end_row {
                    for c in block_col..end_col {
                        block_data.push(float_data[r * cols + c]);
                    }
                }
                
                // 블록 압축
                let mut encoder = HybridEncoder::new(
                    self.config.wavelet_coefficients, 
                    TransformType::Dwt
                );
                let compressed_block = encoder.encode_block(&block_data, block_height, block_width);
                let reconstructed_block = compressed_block.decode();
                
                // 복원된 데이터를 전체 행렬에 다시 배치
                for (i, &val) in reconstructed_block.iter().enumerate() {
                    let block_r = i / block_width;
                    let block_c = i % block_width;
                    let global_r = block_row + block_r;
                    let global_c = block_col + block_c;
                    if global_r < rows && global_c < cols {
                        total_reconstructed[global_r * cols + global_c] = val;
                    }
                }
                
                all_compressed_blocks.push(compressed_block);
            }
        }
        
        // RMSE 계산
        let rmse = calculate_rmse(&float_data, &total_reconstructed);
        
        // 압축률 계산
        let original_size = float_data.len() * 4; // f32 = 4 bytes
        let compressed_size = all_compressed_blocks.len() * std::mem::size_of::<HybridEncodedBlock>();
        let compression_ratio = original_size as f32 / compressed_size as f32;
        
        Ok(CompressedLayer {
            name: tensor_name.to_string(),
            original_size,
            compressed_size,
            compression_ratio,
            rmse,
            compressed_data: all_compressed_blocks,
            shape: shape.to_vec(),
            dtype: format!("{:?}", tensor.dtype()),
        })
    }
    
    /// 압축된 모델 저장
    async fn save_compressed_model(
        &self,
        model: &CompressedModel,
        output_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // 출력 디렉토리 생성
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        // JSON 직렬화 및 저장
        let json_data = serde_json::to_string_pretty(model)?;
        fs::write(output_path, json_data)?;
        
        println!("💾 압축된 모델 저장: {:?}", output_path);
        Ok(())
    }
    
    /// 압축 요약 출력
    fn print_compression_summary(&self, model: &CompressedModel) {
        println!("\n🏆 === RBE 압축 요약 ===");
        println!("모델명: {}", model.model_name);
        println!("압축된 레이어 수: {}", model.layers.len());
        println!("원본 크기: {:.2} MB", model.original_total_size as f64 / 1_048_576.0);
        println!("압축 후 크기: {:.2} KB", model.compressed_total_size as f64 / 1024.0);
        println!("압축률: {:.1}:1", model.total_compression_ratio);
        println!("평균 RMSE: {:.6}", model.average_rmse);
        println!("압축 시간: {:.2}초", model.compression_time);
        
        // 품질 등급
        let quality = if model.average_rmse < 0.001 { "🥇 S급" }
        else if model.average_rmse < 0.01 { "🥉 A급" }
        else if model.average_rmse < 0.05 { "B급" }
        else { "C급" };
        
        println!("압축 품질: {}", quality);
        
        // 메모리 절약률
        let memory_saving = (1.0 - 1.0 / model.total_compression_ratio) * 100.0;
        println!("메모리 절약: {:.1}%", memory_saving);
        
        if model.average_rmse < 0.001 {
            println!("🎯 목표 RMSE < 0.001 달성!");
        }
        
        println!("✅ 압축 완료!");
    }
}

/// RMSE 계산 유틸리티
fn calculate_rmse(original: &[f32], reconstructed: &[f32]) -> f32 {
    if original.len() != reconstructed.len() {
        return f32::INFINITY;
    }
    
    let mse: f32 = original.iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| (o - r).powi(2))
        .sum::<f32>() / original.len() as f32;
    
    mse.sqrt()
} 
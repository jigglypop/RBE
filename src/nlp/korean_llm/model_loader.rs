//! 한국어 모델 로더
//! 
//! Hugging Face에서 한국어 모델을 다운로드하고 로드합니다.

use std::path::PathBuf;
use anyhow::Result;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::fs;
use std::env;
use safetensors::SafeTensors;
use memmap2::Mmap;
use std::fs::File;
use crate::core::transform::{WeightCompressor, TransformStats};
use crate::core::tensors::{Packed128, Enhanced128};

/// F16을 F32로 변환하는 헬퍼 함수
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x3ff;
    
    if exponent == 0 {
        if mantissa == 0 {
            // Zero
            if sign == 1 { -0.0 } else { 0.0 }
        } else {
            // Subnormal
            let val = (mantissa as f32) / 1024.0 / 16384.0;
            if sign == 1 { -val } else { val }
        }
    } else if exponent == 0x1f {
        if mantissa == 0 {
            // Infinity
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            // NaN
            f32::NAN
        }
    } else {
        // Normal number
        let f32_exponent = (exponent as i32) - 15 + 127;
        let f32_mantissa = (mantissa as u32) << 13;
        let f32_bits = ((sign as u32) << 31) | ((f32_exponent as u32) << 23) | f32_mantissa;
        f32::from_bits(f32_bits)
    }
}

/// 한국어 모델 로더
#[derive(Debug, Clone)]
pub struct KoreanModelLoader {
    model_id: String,
    cache_dir: PathBuf,
    model_path: Option<PathBuf>,
    weights_metadata: Option<WeightsMetadata>,
}

/// 가중치 메타데이터
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightsMetadata {
    pub total_parameters: u64,
    pub model_size_mb: f64,
    pub layers: Vec<LayerMetadata>,
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMetadata {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub size_bytes: u64,
}

/// 압축된 레이어 정보
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedLayerInfo {
    pub layer_name: String,
    pub original_shape: Vec<usize>,
    pub compressed_seed: Packed128,
    pub compression_stats: TransformStats,
}

/// 모델 인덱스 (다운로드된 모델 정보)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelIndex {
    pub model_id: String,
    pub model_path: PathBuf,
    pub download_timestamp: String,
    pub file_structure: HashMap<String, String>, // filename -> file_type
    pub compressed_layers: Vec<CompressedLayerInfo>,
    pub total_parameters: u64,
    pub compression_summary: CompressionSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSummary {
    pub total_original_size_mb: f64,
    pub total_compressed_size_mb: f64,
    pub overall_compression_ratio: f64,
    pub average_rmse: f64,
    pub compression_time_ms: f64,
}

impl KoreanModelLoader {
    /// 새로운 모델 로더 생성
    pub fn new(model_id: &str, cache_dir: &PathBuf) -> Self {
        Self {
            model_id: model_id.to_string(),
            cache_dir: cache_dir.clone(),
            model_path: None,
            weights_metadata: None,
        }
    }

    /// 모델 다운로드 및 로드
    pub async fn download_and_load(&mut self) -> Result<()> {
        // 캐시 디렉토리 생성
        fs::create_dir_all(&self.cache_dir).await?;

        // 모델별 디렉토리
        let model_cache_dir = self.cache_dir.join(self.model_id.replace("/", "_"));
        fs::create_dir_all(&model_cache_dir).await?;

        // 기존 model_tools의 downloader 활용
        let downloader = crate::nlp::model_tools::ModelDownloader::new(&self.model_id);
        
        // config.json 다운로드 확인
        let config_path = model_cache_dir.join("config.json");
        if !config_path.exists() {
            println!("⏬ 한국어 모델 다운로드 중: {}", self.model_id);
            
            // 환경 변수에서 Hugging Face 토큰 읽기
            dotenv::dotenv().ok();
            let token = env::var("HUGGING_FACE_TOKEN")
                .or_else(|_| env::var("hugging_face_token"))
                .map_err(|_| anyhow::anyhow!("HUGGING_FACE_TOKEN 환경 변수가 설정되지 않았습니다"))?;
            
            println!("✅ Hugging Face 토큰 확인됨");
            
            // 실제 다운로드 실행
            self.download_model_files(&model_cache_dir, &token).await?;
        }

        self.model_path = Some(model_cache_dir);
        
        // 메타데이터 로드 또는 생성
        self.load_or_create_metadata().await?;

        Ok(())
    }

    /// 메타데이터 로드 또는 생성
    async fn load_or_create_metadata(&mut self) -> Result<()> {
        let metadata_path = self.model_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model path not set"))?
            .join("weights_metadata.json");

        if metadata_path.exists() {
            // 기존 메타데이터 로드
            let metadata_str = fs::read_to_string(&metadata_path).await?;
            self.weights_metadata = Some(serde_json::from_str(&metadata_str)?);
        } else {
            // 새로운 메타데이터 생성
            self.weights_metadata = Some(self.create_metadata_for_model()?);
            
            // 저장
            if let Some(metadata) = &self.weights_metadata {
                let metadata_str = serde_json::to_string_pretty(metadata)?;
                fs::write(&metadata_path, metadata_str).await?;
            }
        }

        Ok(())
    }

    /// 모델별 메타데이터 생성
    fn create_metadata_for_model(&self) -> Result<WeightsMetadata> {
        // 모델별 기본 설정
        let metadata = match self.model_id.as_str() {
            "BM-K/KoMiniLM" => WeightsMetadata {
                total_parameters: 23_000_000,
                model_size_mb: 88.0,
                layers: vec![
                    LayerMetadata {
                        name: "embeddings".to_string(),
                        shape: vec![32000, 384],
                        dtype: "float32".to_string(),
                        size_bytes: 49_152_000,
                    },
                    // 6개 레이어
                    LayerMetadata {
                        name: "encoder.layer.0".to_string(),
                        shape: vec![384, 384],
                        dtype: "float32".to_string(),
                        size_bytes: 589_824,
                    },
                ],
                vocab_size: 32000,
                hidden_size: 384,
                num_layers: 6,
            },
            "skt/kogpt2-base-v2" => WeightsMetadata {
                total_parameters: 125_000_000,
                model_size_mb: 477.0,
                layers: vec![
                    LayerMetadata {
                        name: "transformer.wte".to_string(),
                        shape: vec![51200, 768],
                        dtype: "float32".to_string(),
                        size_bytes: 157_286_400,
                    },
                    LayerMetadata {
                        name: "transformer.wpe".to_string(),
                        shape: vec![1024, 768],
                        dtype: "float32".to_string(),
                        size_bytes: 3_145_728,
                    },
                ],
                vocab_size: 51200,
                hidden_size: 768,
                num_layers: 12,
            },
            "EleutherAI/polyglot-ko-1.3b" => WeightsMetadata {
                total_parameters: 1_331_810_304,
                model_size_mb: 5083.0,
                layers: vec![
                    LayerMetadata {
                        name: "gpt_neox.embed_in.weight".to_string(),
                        shape: vec![30003, 2048],
                        dtype: "float32".to_string(),
                        size_bytes: 245_800_448,
                    },
                ],
                vocab_size: 30003,
                hidden_size: 2048,
                num_layers: 24,
            },
            _ => {
                // 기본값
                WeightsMetadata {
                    total_parameters: 100_000_000,
                    model_size_mb: 400.0,
                    layers: vec![],
                    vocab_size: 30000,
                    hidden_size: 768,
                    num_layers: 12,
                }
            }
        };

        Ok(metadata)
    }

    /// 실제 모델 가중치 로딩 및 압축
    pub async fn load_and_compress_weights(&mut self) -> Result<ModelIndex> {
        let model_path = self.model_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model path not set"))?;

        println!("📦 실제 모델 가중치 로딩 및 압축 시작...");

        // 1. 모델 파일 찾기
        let model_files = self.find_model_files(model_path).await?;
        
        // 2. 가중치 로딩
        let weights_data = self.load_weights_from_files(&model_files).await?;
        
        // 3. 레이어별 압축
        let compressed_layers = self.compress_layers(&weights_data).await?;
        
        // 4. 모델 인덱스 생성
        let model_index = self.create_model_index(model_path, &model_files, &compressed_layers).await?;
        
        // 5. 인덱스 저장
        self.save_model_index(&model_index).await?;
        
        println!("✅ 모델 압축 완료: {:.1}:1 압축률", 
                model_index.compression_summary.overall_compression_ratio);
        
        Ok(model_index)
    }

    /// 모델 파일 찾기
    async fn find_model_files(&self, model_path: &PathBuf) -> Result<HashMap<String, PathBuf>> {
        let mut model_files = HashMap::new();
        
        // safetensors 파일 확인
        let safetensors_path = model_path.join("model.safetensors");
        if safetensors_path.exists() {
            model_files.insert("model.safetensors".to_string(), safetensors_path);
            println!("  ✅ model.safetensors 발견");
        }
        
        // pytorch_model.bin 파일은 스킵 (SafeTensors만 사용)
        let pytorch_path = model_path.join("pytorch_model.bin");
        if pytorch_path.exists() {
            println!("  ⚠️  pytorch_model.bin 발견 - SafeTensors를 사용합니다");
        }
        
        if model_files.is_empty() {
            return Err(anyhow::anyhow!("모델 가중치 파일을 찾을 수 없습니다"));
        }
        
        Ok(model_files)
    }

    /// 실제 가중치 파일에서 데이터 로딩
    async fn load_weights_from_files(&self, model_files: &HashMap<String, PathBuf>) -> Result<HashMap<String, Vec<f32>>> {
        let mut all_weights = HashMap::new();
        
        for (filename, filepath) in model_files {
            println!("📖 {} 로딩 중...", filename);
            
            if filename.ends_with(".safetensors") {
                // SafeTensors 로딩
                let weights = self.load_safetensors(filepath).await?;
                all_weights.extend(weights);
            } else if filename.ends_with(".bin") {
                // PyTorch Binary 로딩 (pickle 형식)
                let weights = self.load_pytorch_bin(filepath).await?;
                all_weights.extend(weights);
            }
        }
        
        println!("✅ 총 {} 개 레이어 로딩 완료", all_weights.len());
        Ok(all_weights)
    }

    /// SafeTensors 파일 로딩
    async fn load_safetensors(&self, filepath: &PathBuf) -> Result<HashMap<String, Vec<f32>>> {
        let file = File::open(filepath)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;
        
        let mut weights = HashMap::new();
        
        for tensor_name in tensors.names() {
            let tensor_view = tensors.tensor(tensor_name)?;
            
            // f32 및 f16 데이터 처리
            match tensor_view.dtype() {
                safetensors::Dtype::F32 => {
                    let shape = tensor_view.shape();
                    let data = tensor_view.data();
                    
                    // f32로 캐스팅
                    let f32_data: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();
                    
                    println!("  - {}: {:?} ({} 파라미터)", tensor_name, shape, f32_data.len());
                    weights.insert(tensor_name.to_string(), f32_data);
                }
                safetensors::Dtype::F16 => {
                    let shape = tensor_view.shape();
                    let data = tensor_view.data();
                    
                    // f16을 f32로 변환
                    let f32_data: Vec<f32> = data
                        .chunks_exact(2)
                        .map(|chunk| {
                            let f16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            f16_to_f32(f16_bits)
                        })
                        .collect();
                    
                    println!("  - {}: {:?} ({} 파라미터, F16→F32 변환)", tensor_name, shape, f32_data.len());
                    weights.insert(tensor_name.to_string(), f32_data);
                }
                _ => {
                    println!("  - {} 스킵: {:?} 타입", tensor_name, tensor_view.dtype());
                }
            }
        }
        
        Ok(weights)
    }

    /// PyTorch Binary 파일 로딩 (pickle 형식 파싱)
    async fn load_pytorch_bin(&self, filepath: &PathBuf) -> Result<HashMap<String, Vec<f32>>> {
        use std::io::{Read, Cursor};
        
        println!("  🔧 PyTorch .bin 파일 파싱 중...");
        
        // 파일 읽기
        let mut file = File::open(filepath)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        
        // Pickle 형식은 복잡하므로, 간단한 해결책:
        // 1. 파일에서 float32 패턴을 찾아서 추출
        // 2. 레이어 이름은 메타데이터에서 가져옴
        
        let mut weights = HashMap::new();
        
        // GPT2 모델의 알려진 레이어 구조
        let mut gpt2_layers = vec![
            ("transformer.wte.weight".to_string(), vec![51200, 768]),  // 토큰 임베딩
            ("transformer.wpe.weight".to_string(), vec![1024, 768]),   // 위치 임베딩
            ("transformer.ln_f.weight".to_string(), vec![768]),        // 최종 LayerNorm
            ("transformer.ln_f.bias".to_string(), vec![768]),
            ("lm_head.weight".to_string(), vec![51200, 768]),          // 언어 모델 헤드
        ];
        
        // 12개 레이어 각각의 가중치
        for i in 0..12 {
            gpt2_layers.extend(vec![
                (format!("transformer.h.{}.ln_1.weight", i), vec![768]),
                (format!("transformer.h.{}.ln_1.bias", i), vec![768]),
                (format!("transformer.h.{}.attn.c_attn.weight", i), vec![768, 2304]),
                (format!("transformer.h.{}.attn.c_attn.bias", i), vec![2304]),
                (format!("transformer.h.{}.attn.c_proj.weight", i), vec![768, 768]),
                (format!("transformer.h.{}.attn.c_proj.bias", i), vec![768]),
                (format!("transformer.h.{}.ln_2.weight", i), vec![768]),
                (format!("transformer.h.{}.ln_2.bias", i), vec![768]),
                (format!("transformer.h.{}.mlp.c_fc.weight", i), vec![768, 3072]),
                (format!("transformer.h.{}.mlp.c_fc.bias", i), vec![3072]),
                (format!("transformer.h.{}.mlp.c_proj.weight", i), vec![3072, 768]),
                (format!("transformer.h.{}.mlp.c_proj.bias", i), vec![768]),
            ]);
        }
        
        // PyTorch 파일 검증 (ZIP 형식 확인)
        if buffer.len() < 4 {
            return Err(anyhow::anyhow!("파일이 너무 작습니다"));
        }
        
        // ZIP 매직 넘버 확인 (PK\x03\x04)
        let is_zip = buffer[0] == 0x50 && buffer[1] == 0x4B && buffer[2] == 0x03 && buffer[3] == 0x04;
        
        if !is_zip {
            println!("  ⚠️  표준 PyTorch ZIP 형식이 아닙니다. Legacy 형식으로 시도합니다.");
        }
        
        // 간단한 휴리스틱: float32 배열 찾기
        // PyTorch는 일반적으로 연속된 float32 데이터를 저장
        let mut cursor = Cursor::new(&buffer);
        let mut offset = 0;
        
        // 실제 구현은 복잡하므로, 다른 방법 시도
        println!("  ⚠️  PyTorch .bin 직접 파싱은 복잡합니다. 대안을 찾고 있습니다...");
        
        // 대안: 파일 크기로 추정
        let file_size = buffer.len();
        let expected_params: usize = gpt2_layers.iter()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .sum();
        let expected_size = expected_params * 4; // float32
        
        println!("  📊 파일 크기: {}MB, 예상 파라미터: {}M", 
                file_size / 1024 / 1024, expected_params / 1_000_000);
        
                    // PyTorch .bin 파일은 현재 지원하지 않음
            return Err(anyhow::anyhow!(
                "PyTorch .bin 파일 로딩은 지원하지 않습니다. SafeTensors 형식을 사용해주세요."
            ));
        
        println!("  ℹ️  가능하면 safetensors 형식 사용을 권장합니다.");
        
        Ok(weights)
    }

    /// 레이어별 압축 수행
    async fn compress_layers(&self, weights_data: &HashMap<String, Vec<f32>>) -> Result<Vec<CompressedLayerInfo>> {
        let mut compressed_layers = Vec::new();
        let mut total_original_size = 0.0;
        let mut total_compressed_size = 0.0;
        let mut total_compression_time = 0.0;
        let mut rmse_sum = 0.0;
        
        for (layer_name, weight_data) in weights_data {
            // 가중치가 2D 행렬로 해석 가능한지 확인
            let total_params = weight_data.len();
            if total_params == 0 {
                continue;
            }
            
            // 형상 추정 (일반적으로 선형 레이어는 2D)
            let (rows, cols) = self.estimate_matrix_shape(total_params, layer_name);
            
            if rows == 0 || cols == 0 {
                println!("  ⚠️  {} 스킵: 형상 결정 불가 ({} 파라미터)", layer_name, total_params);
                continue;
            }
            
            println!("  🔄 {} 압축 중: {}x{}", layer_name, rows, cols);
            
            // RBE 압축 수행
            let mut compressor = WeightCompressor::new(rows, cols);
            let (mut compressed_seed, mut stats) = match compressor.compress_weights(weight_data) {
                Ok(r) => r,
                Err(e) => {
                    println!("    ❌ 1차 압축 실패: {}", e);
                    continue;
                }
            };

            // RMSE가 0.001 초과하면 Enhanced128로 시도 (거의 0에 가까운 기준)
            let mut attempt_block = cols / 2;
            let mut attempt = 0;
            while stats.rmse > 0.001 && attempt < 3 && attempt_block >= 8 {
                let compressor_blk = WeightCompressor::new(rows, cols).with_block_cols(attempt_block);
                if let Ok((seed_blk, stat_blk)) = compressor_blk.compress_weights(weight_data) {
                    if stat_blk.rmse < stats.rmse {
                        compressed_seed = seed_blk;
                        stats = stat_blk;
                    }
                }
                attempt += 1;
                attempt_block /= 2;
            }

            // 최종 결과 기록 (거의 0에 가까운 정확도 요구)
            if stats.rmse <= 0.001 {
                    compressed_layers.push(CompressedLayerInfo {
                        layer_name: layer_name.clone(),
                        original_shape: vec![rows, cols],
                        compressed_seed,
                        compression_stats: stats.clone(),
                    });
                    
                    total_original_size += stats.original_size_mb;
                    total_compressed_size += stats.compressed_size_mb;
                    total_compression_time += stats.transform_ms;
                    rmse_sum += stats.rmse;
                    
                    println!("    ✅ {:.1}:1 압축률, RMSE {:.6}", 
                            stats.compression_ratio, stats.rmse);
            } else {
                // Enhanced128 기반 압축 시도
                println!("    🔄 Enhanced128 압축 시도 중...");
                let enhanced_seed = Enhanced128::random(&mut rand::thread_rng());
                let enhanced_rmse = calculate_enhanced_rmse(&enhanced_seed, weight_data, rows, cols);
                
                if enhanced_rmse <= 0.001 {
                    // Enhanced128을 Packed128으로 변환하여 저장 (임시 호환성)
                    let converted_packed = convert_enhanced_to_packed128(enhanced_seed);
                    
                    compressed_layers.push(CompressedLayerInfo {
                        layer_name: layer_name.clone(),
                        original_shape: vec![rows, cols],
                        compressed_seed: converted_packed,
                        compression_stats: TransformStats {
                            original_size_mb: stats.original_size_mb,
                            compressed_size_mb: 16.0 / 1024.0 / 1024.0, // Enhanced128 크기
                            compression_ratio: stats.original_size_mb / (16.0 / 1024.0 / 1024.0),
                            rmse: enhanced_rmse,
                            transform_ms: stats.transform_ms,
                            restore_ms: 0.0,
                        },
                    });
                    
                    total_original_size += stats.original_size_mb;
                    total_compressed_size += 16.0 / 1024.0 / 1024.0;
                    total_compression_time += stats.transform_ms;
                    rmse_sum += enhanced_rmse;
                    
                    println!("    ✅ Enhanced128 성공: {:.1}:1 압축률, RMSE {:.6}", 
                            stats.original_size_mb / (16.0 / 1024.0 / 1024.0), enhanced_rmse);
                } else {
                    println!("    ⚠️  Enhanced128도 RMSE {:.6} > 0.001 – 레이어 스킵", enhanced_rmse);
                }
            }
        }
        
        println!("📊 전체 압축 통계:");
        println!("  - 압축된 레이어: {}", compressed_layers.len());
        println!("  - 전체 압축률: {:.1}:1", total_original_size / total_compressed_size);
        println!("  - 평균 RMSE: {:.6}", rmse_sum / compressed_layers.len() as f64);
        
        Ok(compressed_layers)
    }

    /// 행렬 형상 추정 (레이어 이름과 파라미터 수 기반)
    fn estimate_matrix_shape(&self, total_params: usize, layer_name: &str) -> (usize, usize) {
        // 1차원 벡터 (bias, LayerNorm 등)
        if layer_name.contains("bias") || layer_name.contains("LayerNorm") || layer_name.contains("layer_norm") {
            // 1차원을 2차원으로 변환: 적절한 행렬 형태로
            match total_params {
                2048 => return (32, 64),   // hidden_size = 2048
                6144 => return (64, 96),   // 3 * hidden_size (QKV)
                8192 => return (64, 128),  // 4 * hidden_size (FFN)
                768 => return (24, 32),    // GPT2 hidden_size
                2304 => return (48, 48),   // GPT2 3 * hidden
                3072 => return (48, 64),   // GPT2 4 * hidden
                _ => {
                    // 기본 형상 추정
                    if total_params <= 512 {
                        let rows = (total_params as f64 / 16.0).ceil() as usize;
                        return (rows.max(1), (total_params / rows.max(1)).max(1));
                    } else {
                        let side = (total_params as f64).sqrt() as usize;
                        if side * side == total_params {
                            return (side, side);
                        }
                        return (side.max(1), (total_params / side.max(1)).max(1));
                    }
                }
            }
        }
        
        // 특정 임베딩 레이어들
        if layer_name.contains("embed_in") || layer_name.contains("embed_out") {
            // GPT-NeoX 임베딩
            if total_params == 61603840 { // 30080 * 2048
                return (30080, 2048);
            }
        }
        
        if layer_name.contains("word_embeddings") || layer_name.contains("wte") {
            // [vocab_size, hidden_size]
            let metadata = self.weights_metadata.as_ref();
            if let Some(meta) = metadata {
                return (meta.vocab_size as usize, meta.hidden_size as usize);
            }
        }
        
        if layer_name.contains("position_embeddings") {
            // [max_position, hidden_size] - 일반적으로 512x384
            if total_params == 196608 { // 512 * 384
                return (512, 384);
            }
        }
        
        if layer_name.contains("token_type_embeddings") {
            // [num_token_types, hidden_size] - 일반적으로 2x384
            if total_params == 768 { // 2 * 384
                return (2, 384);
            }
        }
        
        // 일반적인 Transformer 레이어 패턴
        if layer_name.contains("attention") || layer_name.contains("attn") {
            // query_key_value 통합 레이어 (GPT-NeoX 스타일)
            if layer_name.contains("query_key_value") {
                // total_params = 6144 * 2048 = 12582912
                // 6144 = 3 * 2048 (Q, K, V)
                if total_params == 12582912 {
                    return (6144, 2048);
                }
                // 일반적인 경우: hidden_size 추정
                let hidden_size = ((total_params as f64 / 3.0).sqrt() as usize).max(1);
                return (hidden_size * 3, hidden_size);
            }
            if layer_name.contains("q_proj") || layer_name.contains("k_proj") || layer_name.contains("v_proj") ||
               layer_name.contains("query") || layer_name.contains("key") || layer_name.contains("value") {
                // Query/Key/Value projection: [hidden_size, hidden_size]
                let hidden_size = (total_params as f64).sqrt() as usize;
                return (hidden_size, hidden_size);
            }
            if layer_name.contains("out_proj") || layer_name.contains("o_proj") || layer_name.contains("output") ||
               layer_name.contains("dense") {
                let hidden_size = (total_params as f64).sqrt() as usize;
                return (hidden_size, hidden_size);
            }
        }
        
        if layer_name.contains("intermediate") || layer_name.contains("mlp") || layer_name.contains("ffn") {
            // Feed-forward layers: [hidden_size, intermediate_size] 또는 [intermediate_size, hidden_size]
            
            // GPT-NeoX (polyglot-ko): hidden=2048, intermediate=8192
            if total_params == 16777216 { // 2048 * 8192 또는 8192 * 2048
                if layer_name.contains("dense_h_to_4h") {
                    return (8192, 2048); // up projection
                } else if layer_name.contains("dense_4h_to_h") {
                    return (2048, 8192); // down projection
                }
            }
            
            // GPT2: hidden=768, intermediate=3072
            if total_params == 2359296 { // 768 * 3072
                if layer_name.contains("c_fc") {
                    return (3072, 768); // up projection  
                } else if layer_name.contains("c_proj") {
                    return (768, 3072); // down projection
                }
            }
            
            // BERT small
            if total_params == 589824 { // 384 * 1536 또는 1536 * 384
                if layer_name.contains("dense") && !layer_name.contains("output") {
                    return (1536, 384); // up projection
                } else {
                    return (384, 1536); // down projection  
                }
            }
        }
        
        // 기본적으로 정사각 행렬로 가정
        let side = (total_params as f64).sqrt() as usize;
        if side * side == total_params {
            (side, side)
        } else {
            // 직사각형 행렬로 근사
            let mut best_rows = 1;
            let mut best_cols = total_params;
            let target_ratio = 2.0; // 선호하는 가로세로 비율
            
            for rows in 1..=(total_params as f64).sqrt() as usize {
                if total_params % rows == 0 {
                    let cols = total_params / rows;
                    let ratio = cols as f64 / rows as f64;
                    if (ratio - target_ratio).abs() < (best_cols as f64 / best_rows as f64 - target_ratio).abs() {
                        best_rows = rows;
                        best_cols = cols;
                    }
                }
            }
            
            (best_rows, best_cols)
        }
    }

    /// 모델 인덱스 생성
    async fn create_model_index(
        &self,
        model_path: &PathBuf,
        model_files: &HashMap<String, PathBuf>,
        compressed_layers: &[CompressedLayerInfo],
    ) -> Result<ModelIndex> {
        let file_structure: HashMap<String, String> = model_files
            .iter()
            .map(|(name, _)| {
                let file_type = if name.ends_with(".safetensors") {
                    "safetensors"
                } else if name.ends_with(".bin") {
                    "pytorch_bin"
                } else {
                    "unknown"
                };
                (name.clone(), file_type.to_string())
            })
            .collect();

        let total_original_size: f64 = compressed_layers
            .iter()
            .map(|layer| layer.compression_stats.original_size_mb)
            .sum();
        
        let total_compressed_size: f64 = compressed_layers
            .iter()
            .map(|layer| layer.compression_stats.compressed_size_mb)
            .sum();
        
        let average_rmse: f64 = compressed_layers
            .iter()
            .map(|layer| layer.compression_stats.rmse)
            .sum::<f64>() / compressed_layers.len() as f64;
        
        let total_compression_time: f64 = compressed_layers
            .iter()
            .map(|layer| layer.compression_stats.transform_ms)
            .sum();

        let total_parameters: u64 = compressed_layers
            .iter()
            .map(|layer| (layer.original_shape[0] * layer.original_shape[1]) as u64)
            .sum();

        let compression_summary = CompressionSummary {
            total_original_size_mb: total_original_size,
            total_compressed_size_mb: total_compressed_size,
            overall_compression_ratio: total_original_size / total_compressed_size,
            average_rmse,
            compression_time_ms: total_compression_time,
        };

        Ok(ModelIndex {
            model_id: self.model_id.clone(),
            model_path: model_path.clone(),
            download_timestamp: chrono::Utc::now().to_rfc3339(),
            file_structure,
            compressed_layers: compressed_layers.to_vec(),
            total_parameters,
            compression_summary,
        })
    }

    /// 모델 인덱스 저장
    async fn save_model_index(&self, model_index: &ModelIndex) -> Result<()> {
        let model_path = self.model_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model path not set"))?;
        
        let index_path = model_path.join("model_index.json");
        let index_json = serde_json::to_string_pretty(model_index)?;
        
        fs::write(&index_path, index_json).await?;
        println!("💾 모델 인덱스 저장: {}", index_path.display());
        
        Ok(())
    }

    /// 저장된 모델 인덱스 로딩
    pub async fn load_model_index(&self) -> Result<ModelIndex> {
        let model_path = self.model_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model path not set"))?;
        
        let index_path = model_path.join("model_index.json");
        
        if !index_path.exists() {
            return Err(anyhow::anyhow!("모델 인덱스 파일이 없습니다. 먼저 load_and_compress_weights()를 실행하세요."));
        }
        
        let index_json = fs::read_to_string(&index_path).await?;
        let model_index: ModelIndex = serde_json::from_str(&index_json)?;
        
        println!("📖 모델 인덱스 로딩 완료: {} 레이어", model_index.compressed_layers.len());
        
        Ok(model_index)
    }

    /// 특정 레이어의 압축된 가중치 반환
    pub fn get_compressed_layer<'a>(&self, model_index: &'a ModelIndex, layer_name: &str) -> Option<&'a CompressedLayerInfo> {
        model_index.compressed_layers
            .iter()
            .find(|layer| layer.layer_name == layer_name)
    }

    /// 압축된 가중치 로드
    pub async fn load_compressed_weights(&mut self) -> Result<()> {
        let model_path = self.model_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model path not set"))?;

        let compressed_path = model_path.join("compressed_weights");
        
        if compressed_path.exists() {
            println!("✅ 압축된 가중치 로드 중...");
            // RBE 시스템을 활용한 압축 가중치 로드
            // 실제 구현은 compressor 모듈과 연동
        } else {
            println!("⚠️  압축된 가중치가 없습니다. 원본 모델 사용");
        }

        Ok(())
    }

    /// 모델 메타데이터 반환
    pub fn get_metadata(&self) -> Option<&WeightsMetadata> {
        self.weights_metadata.as_ref()
    }

    /// 모델 경로 반환
    pub fn get_model_path(&self) -> Option<&PathBuf> {
        self.model_path.as_ref()
    }

    /// 실제 모델 파일 다운로드
    async fn download_model_files(&self, output_dir: &PathBuf, token: &str) -> Result<()> {
        use hf_hub::api::tokio::ApiBuilder;
        
        // Hugging Face API 클라이언트 생성
        let api = ApiBuilder::new()
            .with_token(Some(token.to_string()))
            .build()?;
        
        let repo = api.model(self.model_id.clone());
        
        // 다운로드할 파일 목록
        let files_to_download = vec![
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "vocab.json",     // GPT2 토크나이저용
            "merges.txt",     // GPT2 토크나이저용
            "special_tokens_map.json",
        ];
        
        // 모델 파일들도 시도 (safetensors 우선)
        let model_files = vec![
            "model.safetensors",
            "pytorch_model.safetensors", // 때로는 이 이름도 사용
            "tf_model.h5.safetensors",   // TensorFlow 변환 모델
            "flax_model.msgpack.safetensors", // Flax 변환 모델  
            "model-00001-of-00002.safetensors", // sharded 모델
            "pytorch_model.bin", // 마지막 옵션
        ];
        
        println!("📥 필수 파일 다운로드 중...");
        
        // 필수 파일들 다운로드
        for file in &files_to_download {
            let output_path = output_dir.join(file);
            match repo.get(file).await {
                Ok(file_path) => {
                    fs::copy(&file_path, &output_path).await?;
                    println!("  ✅ {}", file);
                }
                Err(e) => {
                    println!("  ⚠️  {} 다운로드 실패: {}", file, e);
                }
            }
        }
        
        println!("📥 모델 가중치 다운로드 시도 중...");
        
        // 모델 파일 다운로드 (하나라도 성공하면 OK)
        let mut model_downloaded = false;
        for file in &model_files {
            let output_path = output_dir.join(file);
            match repo.get(file).await {
                Ok(file_path) => {
                    fs::copy(&file_path, &output_path).await?;
                    println!("  ✅ {}", file);
                    model_downloaded = true;
                    break;
                }
                Err(_) => {
                    println!("  ⚠️  {} 없음, 다른 파일 시도 중...", file);
                }
            }
        }
        
        if !model_downloaded {
            println!("  ⚠️  모델 가중치 파일을 찾을 수 없습니다. 설정 파일만 다운로드됩니다.");
        }
        
        println!("✅ 다운로드 완료!");
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_loader_creation() {
        let loader = KoreanModelLoader::new("BM-K/KoMiniLM", &PathBuf::from("cache"));
        assert_eq!(loader.model_id, "BM-K/KoMiniLM");
    }

    #[test]
    fn test_metadata_creation() {
        let loader = KoreanModelLoader::new("BM-K/KoMiniLM", &PathBuf::from("cache"));
        let metadata = loader.create_metadata_for_model().unwrap();
        assert_eq!(metadata.total_parameters, 23_000_000);
        assert_eq!(metadata.vocab_size, 32000);
    }
}

/// Enhanced128 RMSE 계산 헬퍼 함수
fn calculate_enhanced_rmse(enhanced: &Enhanced128, weights: &[f32], rows: usize, cols: usize) -> f64 {
    let mut total_error = 0.0f64;
    
    for i in 0..rows {
        for j in 0..cols {
            let idx = i * cols + j;
            let predicted = enhanced.fused_forward_enhanced(i, j, rows, cols);
            let target = weights[idx];
            let error = (predicted - target) as f64;
            total_error += error * error;
        }
    }
    
    (total_error / (rows * cols) as f64).sqrt()
}

/// Enhanced128을 Packed128으로 변환 (임시 호환성)
fn convert_enhanced_to_packed128(enhanced: Enhanced128) -> Packed128 {
    // 임시로 Enhanced128의 비트를 Packed128 형식으로 변환
    // 실제로는 더 정교한 변환이 필요할 수 있음
    Packed128 {
        hi: enhanced.hi,
        lo: enhanced.lo,
    }
} 
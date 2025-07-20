use std::path::Path;
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde_json::Value;
use std::fs;

/// 간단한 모델 로더 (기존 바이너리 스타일)
pub struct SimpleModelLoader;

impl SimpleModelLoader {
    /// 새로운 모델 로더 생성
    pub fn new() -> Result<Self> {
        Ok(Self)
    }
    
    /// 모델 가중치 로드 (기존 numpy/json 파일 방식)
    pub fn load_model_weights(&self, model_path: &Path) -> Result<HashMap<String, Vec<f32>>> {
        let weights_dir = model_path.join("weights");
        let metadata_path = weights_dir.join("metadata.json");
        
        if !metadata_path.exists() {
            return Err(anyhow!("메타데이터 파일이 없습니다: {:?}", metadata_path));
        }
        
        // 메타데이터 로드
        let metadata_str = fs::read_to_string(&metadata_path)?;
        let metadata: serde_json::Map<String, Value> = serde_json::from_str(&metadata_str)?;
        
        let mut weights = HashMap::new();
        
        println!("📊 로드된 레이어:");
        for (layer_name, layer_info) in &metadata {
            if let Some(shape_arr) = layer_info.get("shape").and_then(|s| s.as_array()) {
                let shape: Vec<usize> = shape_arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect();
                
                // 2D 텐서만 처리
                if shape.len() == 2 {
                    let file_path = weights_dir.join(format!("{}.npy", layer_name));
                    if file_path.exists() {
                        match self.load_numpy_file(&file_path) {
                            Ok(data) => {
                                println!("  ✅ {}: {}×{}", layer_name, shape[0], shape[1]);
                                weights.insert(layer_name.clone(), data);
                            }
                            Err(e) => {
                                println!("  ❌ {}: 로드 실패 - {}", layer_name, e);
                            }
                        }
                    }
                }
            }
        }
        
        println!("✅ 총 {} 레이어 로드 완료", weights.len());
        Ok(weights)
    }
    
    /// NumPy 파일 로드 (간단한 버전)
    fn load_numpy_file(&self, path: &Path) -> Result<Vec<f32>> {
        let mut file = std::fs::File::open(path)?;
        let (shape, _) = self.read_npy_header(&mut file)?;
        
        let total_size: usize = shape.iter().product();
        let mut buffer = vec![0u8; total_size * 4]; // f32 = 4 bytes
        
        std::io::Read::read_exact(&mut file, &mut buffer)?;
        
        let data: Vec<f32> = buffer.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        Ok(data)
    }
    
    /// NumPy 헤더 읽기
    fn read_npy_header(&self, file: &mut std::fs::File) -> Result<(Vec<usize>, usize)> {
        use std::io::Read;
        
        let mut magic = [0u8; 6];
        file.read_exact(&mut magic)?;
        
        if &magic != b"\x93NUMPY" {
            return Err(anyhow!("유효하지 않은 NumPy 파일"));
        }
        
        let mut version = [0u8; 2];
        file.read_exact(&mut version)?;
        
        let header_len = if version[0] == 1 {
            let mut len_bytes = [0u8; 2];
            file.read_exact(&mut len_bytes)?;
            u16::from_le_bytes(len_bytes) as usize
        } else {
            let mut len_bytes = [0u8; 4];
            file.read_exact(&mut len_bytes)?;
            u32::from_le_bytes(len_bytes) as usize
        };
        
        let mut header = vec![0u8; header_len];
        file.read_exact(&mut header)?;
        let header_str = String::from_utf8_lossy(&header);
        
        // shape 추출 (간단한 파싱)
        let shape_start = header_str.find("'shape': (").unwrap_or(0) + 10;
        let shape_end = header_str[shape_start..].find(')').unwrap_or(0) + shape_start;
        let shape_str = &header_str[shape_start..shape_end];
        
        let shape: Vec<usize> = shape_str.split(", ")
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.trim_end_matches(',').parse().ok())
            .collect();
        
        let total_size = shape.iter().product();
        
        Ok((shape, total_size))
    }
    
    /// 모델 설정 로드 (JSON 파일에서)
    pub fn load_model_config(&self, model_path: &Path) -> Result<ModelConfig> {
        let config_file = model_path.join("config.json");
        
        if !config_file.exists() {
            // 기본값 설정
            return Ok(ModelConfig {
                vocab_size: 50257,
                hidden_size: 768,
                num_layers: 12,
                num_heads: 12,
                max_length: 1024,
            });
        }
        
        let config_content = std::fs::read_to_string(config_file)?;
        let config_json: Value = serde_json::from_str(&config_content)?;
        
        Ok(ModelConfig {
            vocab_size: config_json["vocab_size"].as_u64().unwrap_or(50257) as usize,
            hidden_size: config_json["n_embd"].as_u64().unwrap_or(768) as usize,
            num_layers: config_json["n_layer"].as_u64().unwrap_or(12) as usize,
            num_heads: config_json["n_head"].as_u64().unwrap_or(12) as usize,
            max_length: config_json["n_positions"].as_u64().unwrap_or(1024) as usize,
        })
    }
}

/// 모델 설정 구조체
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_length: usize,
}

/// 간단한 토크나이저 (기존 방식)
pub struct SimpleTokenizer {
    vocab: HashMap<u32, String>,
}

impl SimpleTokenizer {
    /// 토크나이저 로드
    pub fn load(model_path: &Path) -> Result<Self> {
        let tokenizer_file = model_path.join("tokenizer.json");
        
        if !tokenizer_file.exists() {
            return Err(anyhow!("tokenizer.json 파일이 없습니다: {:?}", tokenizer_file));
        }
        
        // tokenizers 라이브러리 사용
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| anyhow!("토크나이저 로드 실패: {}", e))?;
        
        let vocab = tokenizer.get_vocab(false);
        let id_to_token: HashMap<u32, String> = vocab.into_iter()
            .map(|(token, id)| (id, token))
            .collect();
        
        Ok(Self {
            vocab: id_to_token,
        })
    }
    
    /// 텍스트를 토큰 ID로 변환 (간단한 구현)
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        // 간단한 해시 기반 토크나이징
        let mut tokens = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        
        for window in chars.chunks(3) {
            let chunk: String = window.iter().collect();
            let hash = chunk.chars().fold(0u32, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u32));
            let token_id = (hash % self.vocab.len() as u32) as i64;
            tokens.push(token_id);
        }
        
        Ok(tokens)
    }
    
    /// 토큰 ID를 텍스트로 변환
    pub fn decode(&self, token_ids: &[i64]) -> Result<String> {
        let tokens: Vec<String> = token_ids.iter()
            .filter_map(|&id| {
                if id >= 0 && id < self.vocab.len() as i64 {
                    self.vocab.get(&(id as u32)).cloned()
                } else {
                    Some("[UNK]".to_string())
                }
            })
            .collect();
        
        Ok(tokens.join(" "))
    }
} 
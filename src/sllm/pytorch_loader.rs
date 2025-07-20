use std::path::Path;
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde_json::Value;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// PyTorch 모델 로더
pub struct PyTorchLoader {
    python_env: Python<'static>,
}

impl PyTorchLoader {
    /// 새로운 PyTorch 로더 생성
    pub fn new() -> Result<Self> {
        // Python 초기화를 간단하게 처리
        Ok(Self {
            python_env: unsafe { std::mem::transmute(0 as *const ()) },
        })
    }
    
    /// PyTorch 모델을 텐서 딕셔너리로 로드
    pub fn load_model_weights(&self, model_path: &Path) -> Result<HashMap<String, Vec<f32>>> {
        Python::with_gil(|py| {
            let model_file = model_path.join("pytorch_model.bin");
            if !model_file.exists() {
                return Err(anyhow!("pytorch_model.bin 파일을 찾을 수 없습니다: {:?}", model_file));
            }
            
            println!("🔄 PyTorch 모델 로딩 중: {:?}", model_file);
            
            // PyTorch로 모델 로드
            let code = format!(r#"
import torch
import numpy as np

# 모델 가중치 로드
state_dict = torch.load(r"{}", map_location='cpu')
weights = {{}}

for name, tensor in state_dict.items():
    if tensor.dim() == 2:  # 2D 텐서만 (Linear layers)
        numpy_tensor = tensor.detach().numpy().astype(np.float32)
        weights[name] = numpy_tensor

print(f"로드된 2D 레이어 수: {{len(weights)}}")
for name, weight in weights.items():
    print(f"  {{name}}: {{weight.shape}}")
"#, model_file.display());
            
            let locals = PyDict::new(py);
            py.run(&code, None, Some(locals))?;
            
            // Python에서 결과 추출
            let weights_dict: &PyDict = locals.get_item("weights").unwrap().downcast()?;
            let mut rust_weights = HashMap::new();
            
            for (name, tensor) in weights_dict.iter() {
                let name_str: String = name.extract()?;
                let numpy_array: PyReadonlyArray2<f32> = tensor.extract()?;
                let array = numpy_array.as_array();
                
                // 2D 배열을 1D 벡터로 변환 (row-major order)
                let flat_data: Vec<f32> = array.iter().cloned().collect();
                rust_weights.insert(name_str, flat_data);
            }
            
            println!("✅ 모델 로딩 완료: {} 레이어", rust_weights.len());
            Ok(rust_weights)
        })
    }
    
    /// 모델 설정 로드
    pub fn load_model_config(&self, model_path: &Path) -> Result<ModelConfig> {
        let config_file = model_path.join("config.json");
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

/// 토크나이저 래퍼
pub struct Tokenizer {
    python_tokenizer: Py<PyAny>,
}

impl Tokenizer {
    /// 토크나이저 로드
    pub fn load(model_path: &Path) -> Result<Self> {
        Python::with_gil(|py| {
            let code = format!(r#"
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(r"{}")
"#, model_path.display());
            
            let locals = PyDict::new(py);
            py.run(&code, None, Some(locals))?;
            
            let tokenizer = locals.get_item("tokenizer").unwrap().into();
            
            Ok(Self {
                python_tokenizer: tokenizer,
            })
        })
    }
    
    /// 텍스트를 토큰 ID로 변환
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        Python::with_gil(|py| {
            let tokenizer = self.python_tokenizer.as_ref(py);
            let encoded = tokenizer.call_method1("encode", (text,))?;
            let token_ids: Vec<i64> = encoded.extract()?;
            Ok(token_ids)
        })
    }
    
    /// 토큰 ID를 텍스트로 변환
    pub fn decode(&self, token_ids: &[i64]) -> Result<String> {
        Python::with_gil(|py| {
            let tokenizer = self.python_tokenizer.as_ref(py);
            let decoded = tokenizer.call_method1("decode", (token_ids,))?;
            let text: String = decoded.extract()?;
            Ok(text)
        })
    }
} 
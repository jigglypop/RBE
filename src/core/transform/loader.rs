//! 실제 모델 가중치 로더

use std::fs;
use std::collections::HashMap;

/// SafeTensors 파일 헤더
#[derive(Debug)]
pub struct SafeTensorHeader {
    pub metadata: HashMap<String, serde_json::Value>,
    pub tensors: HashMap<String, TensorInfo>,
    pub data_offset: usize,
}

/// 텐서 정보
#[derive(Debug)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// 모델 가중치 로더
pub struct ModelLoader {
    pub file_path: String,
    pub data: Vec<u8>,
    pub header: SafeTensorHeader,
}

impl ModelLoader {
    /// safetensors 파일 로드
    pub fn load_safetensors(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        println!("모델 로딩 시작: {}", path);
        
        let data = fs::read(path)?;
        let header = Self::parse_header(&data)?;
        
        println!("텐서 개수: {}", header.tensors.len());
        println!("파일 크기: {:.1} MB", data.len() as f64 / 1024.0 / 1024.0);
        
        Ok(Self {
            file_path: path.to_string(),
            data,
            header,
        })
    }
    
    /// 헤더 파싱
    fn parse_header(data: &[u8]) -> Result<SafeTensorHeader, Box<dyn std::error::Error>> {
        // SafeTensors 형식: [header_length: u64][header: JSON][data]
        if data.len() < 8 {
            return Err("파일이 너무 작음".into());
        }
        
        let header_len = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7]
        ]) as usize;
        
        if data.len() < 8 + header_len {
            return Err("헤더 길이 오류".into());
        }
        
        let header_bytes = &data[8..8 + header_len];
        let header_str = std::str::from_utf8(header_bytes)?;
        let json: serde_json::Value = serde_json::from_str(header_str)?;
        
        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();
        
        if let serde_json::Value::Object(obj) = json {
            for (key, value) in obj {
                if key == "__metadata__" {
                    if let serde_json::Value::Object(meta) = value {
                        for (k, v) in meta {
                            metadata.insert(k, v);
                        }
                    }
                } else if let serde_json::Value::Object(tensor_obj) = value {
                    // 텐서 정보 파싱
                    let dtype = tensor_obj.get("dtype")
                        .and_then(|v| v.as_str())
                        .unwrap_or("F32")
                        .to_string();
                    
                    let shape: Vec<usize> = tensor_obj.get("shape")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter()
                            .filter_map(|v| v.as_u64())
                            .map(|v| v as usize)
                            .collect())
                        .unwrap_or_default();
                    
                    let data_offsets = if let Some(offsets) = tensor_obj.get("data_offsets") {
                        if let Some(arr) = offsets.as_array() {
                            if arr.len() >= 2 {
                                let start = arr[0].as_u64().unwrap_or(0) as usize;
                                let end = arr[1].as_u64().unwrap_or(0) as usize;
                                (start, end)
                            } else {
                                (0, 0)
                            }
                        } else {
                            (0, 0)
                        }
                    } else {
                        (0, 0)
                    };
                    
                    tensors.insert(key, TensorInfo {
                        dtype,
                        shape,
                        data_offsets,
                    });
                }
            }
        }
        
        Ok(SafeTensorHeader {
            metadata,
            tensors,
            data_offset: 8 + header_len,
        })
    }
    
    /// 특정 텐서의 f32 데이터 추출
    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let tensor_info = self.header.tensors.get(name)
            .ok_or(format!("텐서 '{}' 없음", name))?;
        
        if tensor_info.dtype != "F32" {
            return Err(format!("지원하지 않는 타입: {}", tensor_info.dtype).into());
        }
        
        let start = self.header.data_offset + tensor_info.data_offsets.0;
        let end = self.header.data_offset + tensor_info.data_offsets.1;
        
        if end > self.data.len() {
            return Err("데이터 범위 오류".into());
        }
        
        let bytes = &self.data[start..end];
        let float_count = bytes.len() / 4;
        let mut result = Vec::with_capacity(float_count);
        
        for i in 0..float_count {
            let byte_slice = &bytes[i * 4..(i + 1) * 4];
            let float_val = f32::from_le_bytes([
                byte_slice[0], byte_slice[1], 
                byte_slice[2], byte_slice[3]
            ]);
            result.push(float_val);
        }
        
        Ok(result)
    }
    
    /// 모든 텐서 이름 조회
    pub fn list_tensors(&self) -> Vec<String> {
        self.header.tensors.keys().cloned().collect()
    }
    
    /// 총 파라미터 수 계산
    pub fn total_parameters(&self) -> usize {
        self.header.tensors.values()
            .map(|info| info.shape.iter().product::<usize>())
            .sum()
    }
} 
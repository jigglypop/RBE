# RBE 동적 로딩 아키텍처

## 개요

RBE 라이브러리의 핵심 문제는 압축된 가중치를 하드코딩된 순서로만 불러올 수 있다는 점입니다. 이로 인해 GPT-2 모델 압축 후 복원이 실패하는 문제가 발생했습니다. 이 문서는 메타데이터 기반 동적 로딩 시스템을 통한 해결 방안을 제시합니다.

## 문제점 분석

### 현재 시스템의 한계

1. **하드코딩된 가중치 순서**
   - 모든 가중치가 하나의 바이너리 파일로 순차 저장
   - 소스 코드에 미리 정의된 순서와 개수로만 로딩 가능
   - 모델 구조 변경시 즉시 실패

2. **유연성 부재**
   - 특정 레이어만 압축하거나 다른 압축 방식 적용 불가
   - 다양한 모델 아키텍처 지원 어려움

3. **디버깅 어려움**
   - 가중치 불일치 시 어느 부분이 문제인지 파악 불가

## 해결 방안: 메타데이터 기반 동적 로딩

### 핵심 아이디어

압축된 데이터와 해당 데이터의 구조 정보를 분리하여 관리:

- `rbe_model.bin`: 순수 압축 데이터만 저장
- `rbe_layout.json`: 각 가중치의 위치와 정보를 담은 메타데이터

### 시스템 구조

```
압축 프로세스:
1. GPT-2 모델 로드
2. 각 가중치 압축 + 메타데이터 기록
3. rbe_model.bin + rbe_layout.json 생성

복원 프로세스:
1. rbe_layout.json 로드
2. 메타데이터 기반으로 rbe_model.bin 파싱
3. 동적으로 가중치 매핑 및 모델 구성
```

## 구현 상세

### 1. 메타데이터 구조체 정의

```rust
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WeightInfo {
    pub name: String,          // 가중치 이름 (예: "transformer.h.0.attn.c_attn.weight")
    pub offset_bytes: u64,     // 바이너리 파일 내 오프셋
    pub num_blocks: usize,     // HybridEncodedBlock 개수
    pub original_shape: Vec<usize>, // 원본 텐서 shape
    pub compression_type: String,   // "rbe", "quantized" 등
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelLayout {
    pub model_type: String,    // "gpt2"
    pub total_params: usize,   // 전체 파라미터 수
    pub weights: Vec<WeightInfo>, // 모든 가중치 정보
    pub metadata: HashMap<String, String>, // 추가 메타데이터
}
```

### 2. 압축 프로세스 개선

```rust
pub fn compress_gpt2_with_layout(
    model_path: &Path,
    output_dir: &Path,
    config: &CompressionConfig,
) -> Result<()> {
    // 1. 원본 모델 로드
    let model = load_gpt2_model(model_path)?;
    
    // 2. 출력 파일 준비
    let bin_path = output_dir.join("rbe_model.bin");
    let mut bin_file = File::create(&bin_path)?;
    
    let mut layout = ModelLayout {
        model_type: "gpt2".to_string(),
        total_params: 0,
        weights: Vec::new(),
        metadata: HashMap::new(),
    };
    
    let mut current_offset = 0u64;
    
    // 3. 각 가중치 압축 및 메타데이터 생성
    for (name, tensor) in model.named_parameters() {
        // RBE 압축
        let compressed_blocks = compress_tensor(&tensor, config)?;
        
        // 바이너리 직렬화
        let serialized = bincode::serialize(&compressed_blocks)?;
        bin_file.write_all(&serialized)?;
        
        // 메타데이터 기록
        let info = WeightInfo {
            name: name.to_string(),
            offset_bytes: current_offset,
            num_blocks: compressed_blocks.len(),
            original_shape: tensor.shape().to_vec(),
            compression_type: "rbe".to_string(),
        };
        
        layout.weights.push(info);
        layout.total_params += tensor.numel();
        current_offset += serialized.len() as u64;
    }
    
    // 4. 레이아웃 파일 저장
    let layout_path = output_dir.join("rbe_layout.json");
    let layout_json = serde_json::to_string_pretty(&layout)?;
    fs::write(&layout_path, layout_json)?;
    
    println!("✅ 압축 완료:");
    println!("  - 데이터: {}", bin_path.display());
    println!("  - 레이아웃: {}", layout_path.display());
    println!("  - 총 파라미터: {}", layout.total_params);
    
    Ok(())
}
```

### 3. 동적 로딩 시스템

```rust
pub struct RBEModelLoader {
    layout: ModelLayout,
    weight_data: Vec<u8>,
    cache: HashMap<String, LoadedWeight>,
}

impl RBEModelLoader {
    pub fn new(model_dir: &Path) -> Result<Self> {
        // 레이아웃 로드
        let layout_path = model_dir.join("rbe_layout.json");
        let layout: ModelLayout = serde_json::from_reader(File::open(layout_path)?)?;
        
        // 압축 데이터 로드
        let bin_path = model_dir.join("rbe_model.bin");
        let weight_data = fs::read(bin_path)?;
        
        Ok(Self {
            layout,
            weight_data,
            cache: HashMap::new(),
        })
    }
    
    pub fn load_weight(&mut self, name: &str) -> Result<&LoadedWeight> {
        // 캐시 확인
        if self.cache.contains_key(name) {
            return Ok(&self.cache[name]);
        }
        
        // 메타데이터에서 가중치 정보 찾기
        let info = self.layout.weights.iter()
            .find(|w| w.name == name)
            .ok_or_else(|| anyhow!("Weight '{}' not found", name))?;
        
        // 바이너리 데이터에서 해당 부분 추출
        let start = info.offset_bytes as usize;
        let data_slice = &self.weight_data[start..];
        
        // 역직렬화
        let blocks: Vec<HybridEncodedBlock> = bincode::deserialize(data_slice)?;
        
        // 캐시에 저장
        self.cache.insert(name.to_string(), LoadedWeight::Compressed(blocks));
        
        Ok(&self.cache[name])
    }
}
```

## GPT-2 특화 구현

### GPT-2 모델 구조체

```rust
pub struct GPT2Model {
    config: GPT2Config,
    loader: RBEModelLoader,
}

impl GPT2Model {
    pub fn load_compressed(model_dir: &Path) -> Result<Self> {
        let config = GPT2Config::from_json(model_dir.join("config.json"))?;
        let loader = RBEModelLoader::new(model_dir)?;
        
        Ok(Self { config, loader })
    }
    
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        // 임베딩 레이어
        let wte = self.loader.load_weight("transformer.wte.weight")?;
        let wpe = self.loader.load_weight("transformer.wpe.weight")?;
        
        let mut hidden = embed_tokens(input_ids, wte, wpe)?;
        
        // 각 트랜스포머 블록
        for i in 0..self.config.n_layer {
            hidden = self.forward_block(hidden, i)?;
        }
        
        // 최종 레이어 정규화
        let ln_f_weight = self.loader.load_weight("transformer.ln_f.weight")?;
        let ln_f_bias = self.loader.load_weight("transformer.ln_f.bias")?;
        hidden = layer_norm(hidden, ln_f_weight, ln_f_bias)?;
        
        // 언어 모델 헤드
        let lm_head = self.loader.load_weight("lm_head.weight")?;
        let logits = linear(hidden, lm_head)?;
        
        Ok(logits)
    }
    
    fn forward_block(&mut self, input: Tensor, layer_idx: usize) -> Result<Tensor> {
        let prefix = format!("transformer.h.{}", layer_idx);
        
        // 어텐션 블록
        let attn_output = self.forward_attention(input.clone(), &prefix)?;
        let hidden = input + attn_output;
        
        // MLP 블록
        let mlp_output = self.forward_mlp(hidden.clone(), &prefix)?;
        let output = hidden + mlp_output;
        
        Ok(output)
    }
}
```

## 장점

1. **견고성**: 가중치 순서나 구조 변경에 영향받지 않음
2. **유연성**: 특정 레이어만 압축하거나 다른 압축 방식 혼용 가능
3. **확장성**: 새로운 모델 아키텍처 쉽게 추가 가능
4. **디버깅 용이**: 레이아웃 파일로 구조 파악 가능
5. **버전 관리**: 메타데이터에 버전 정보 포함 가능

## 마이그레이션 계획

1. **Phase 1**: WeightMapper와 ModelLoader 리팩토링
2. **Phase 2**: compress_model.rs에 레이아웃 생성 로직 추가
3. **Phase 3**: GPT-2 추론 엔진을 동적 로딩으로 전환
4. **Phase 4**: 기존 압축 모델 마이그레이션 도구 제공 
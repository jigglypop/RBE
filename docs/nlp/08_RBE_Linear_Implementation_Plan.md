# RBE Linear Layer 구현 계획

## Phase 1: 핵심 인프라 구축

### 1.1 블록 관리 시스템
```rust
// src/nlp/linear/block_manager.rs
pub struct BlockInfo {
    pub input_dim: usize,
    pub output_dim: usize,
    pub block_size: (usize, usize),
    pub num_blocks: (usize, usize),
}

pub struct BlockManager {
    info: BlockInfo,
    block_ranges: Vec<((usize, usize), (usize, usize))>, // (input_range, output_range)
}
```

### 1.2 좌표 변환 모듈
```rust
// src/nlp/linear/coordinate_transform.rs
pub fn block_to_poincare(
    block_i: usize, 
    block_j: usize,
    block_size: (usize, usize)
) -> (f32, f32) {
    // (i,j) -> (r,θ) 변환
}

pub fn optimize_block_size(
    weight_shape: (usize, usize),
    target_compression: f32
) -> (usize, usize) {
    // 최적 블록 크기 계산
}
```

### 1.3 RBE 압축기
```rust
// src/nlp/linear/rbe_compressor.rs
pub struct LinearCompressor {
    target_rmse: f32,
    max_iterations: usize,
    adaptive_basis: bool,
}

impl LinearCompressor {
    pub fn compress_weight_matrix(
        &self,
        weight: &Tensor
    ) -> Result<Vec<Packed256>, CompressionError> {
        // 가중치 행렬을 RBE 시드들로 압축
    }
}
```

## Phase 2: RBE Linear Layer 구현

### 2.1 기본 구조
```rust
// src/nlp/linear/rbe_linear.rs
pub struct RBELinear {
    input_dim: usize,
    output_dim: usize,
    weight_seeds: Vec<Packed256>,
    bias: Option<Tensor>,
    block_manager: BlockManager,
    use_streaming: bool,
}

impl RBELinear {
    pub fn new(input_dim: usize, output_dim: usize) -> Self;
    pub fn from_weight_matrix(weight: &Tensor, bias: Option<&Tensor>) -> Result<Self>;
    pub fn forward(&self, input: &Tensor) -> Tensor;
    pub fn forward_streaming(&self, input: &Tensor) -> Tensor;
}
```

### 2.2 순전파 최적화
```rust
impl RBELinear {
    fn compute_block_output(
        &self,
        input_slice: &Tensor,
        seed: &Packed256,
        block_idx: usize
    ) -> Tensor {
        // 블록별 RBE 순전파
    }
    
    fn parallel_forward(&self, input: &Tensor) -> Tensor {
        // 병렬 블록 처리
    }
}
```

## Phase 3: 통합 및 테스트

### 3.1 기존 레이어 교체
```rust
// src/nlp/linear/mod.rs
pub enum LinearLayerType {
    Standard(nn::Linear),
    RBE(RBELinear),
    Hybrid(HybridLinear), // 크기에 따른 자동 선택
}

pub struct AdaptiveLinear {
    layer_type: LinearLayerType,
    threshold_size: usize, // RBE 적용 임계값
}
```

### 3.2 성능 벤치마크
```rust
// tests/nlp/linear/performance_test.rs
#[test]
fn rbe_linear_압축률_테스트() {
    // 다양한 크기의 레이어에서 압축률 검증
}

#[test]
fn rbe_linear_정확도_테스트() {
    // RMSE < 0.001 검증
}

#[test]
fn rbe_linear_속도_테스트() {
    // 추론 속도 비교
}
```

## Phase 4: 고급 최적화

### 4.1 동적 블록 크기 조정
```rust
pub struct AdaptiveBlockManager {
    base_block_size: (usize, usize),
    rmse_threshold: f32,
    adaptation_history: Vec<(usize, f32)>, // (block_size, rmse)
}

impl AdaptiveBlockManager {
    pub fn adjust_block_size(&mut self, current_rmse: f32) -> (usize, usize);
    pub fn suggest_optimal_basis(&self, weight_pattern: &WeightPattern) -> u8;
}
```

### 4.2 메모리 최적화
```rust
pub struct StreamingRBELinear {
    seeds: Vec<Packed256>,
    block_manager: BlockManager,
    cache_policy: CachePolicy,
}

pub enum CachePolicy {
    NoCache,
    LRU(usize), // 캐시 크기
    Adaptive,   // 사용 패턴 기반
}
```

## 구현 순서

### Week 1: 인프라 구축
1. **Day 1-2**: `BlockManager`, `BlockInfo` 구현
2. **Day 3-4**: 좌표 변환 함수 구현 및 테스트
3. **Day 5-7**: `LinearCompressor` 기본 구현

### Week 2: 핵심 레이어 구현
1. **Day 1-3**: `RBELinear` 기본 구조 및 순전파
2. **Day 4-5**: 병렬 처리 및 최적화
3. **Day 6-7**: 통합 테스트 및 디버깅

### Week 3: 성능 최적화
1. **Day 1-2**: 스트리밍 구현
2. **Day 3-4**: 동적 최적화
3. **Day 5-7**: 벤치마크 및 성능 튜닝

### Week 4: 통합 및 검증
1. **Day 1-3**: 기존 NLP 모델과 통합
2. **Day 4-5**: End-to-end 테스트
3. **Day 6-7**: 문서화 및 최종 검증

## 성공 기준

### 기술적 목표
- [ ] 압축률 150:1 달성
- [ ] RMSE < 0.001 달성
- [ ] 추론 속도 원본 대비 90% 이상
- [ ] 메모리 사용량 50% 이하

### 품질 목표
- [ ] 단위 테스트 커버리지 95% 이상
- [ ] 통합 테스트 통과율 100%
- [ ] 벤치마크 테스트 모든 케이스 통과
- [ ] 코드 리뷰 및 문서화 완료

## 위험 요소 및 대응책

### 위험 요소
1. **수치 불안정성**: 그라디언트 폭발, NaN 발생
2. **메모리 부족**: 대형 모델에서의 메모리 초과
3. **성능 저하**: 예상보다 느린 추론 속도

### 대응책
1. **수치 안정성**: 그라디언트 클리핑, 정규화 강화
2. **메모리 관리**: 스트리밍, 지연 로딩 구현
3. **성능 향상**: 병렬 처리, SIMD 최적화

---

이 계획을 바탕으로 단계별 구현을 시작합니다. 
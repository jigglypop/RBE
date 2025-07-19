# Encoder API

## PoincareEncoder

푸앵카레 기반 인코더

```rust
pub struct PoincareEncoder {
    quality_threshold: f32,
    max_iterations: usize,
    dct_planner: DctPlanner<f32>,
    k_coeffs: usize,
}
```

### 함수

- `new(quality_threshold: f32, max_iterations: usize, k_coeffs: usize) -> Self`
- `encode_matrix(&mut self, matrix: &[Vec<f32>]) -> Packed128`
- `analyze_matrix(&self, matrix: &[Vec<f32>]) -> (f32, Vec<(usize, usize)>)`
- `optimize_encoding(&mut self, matrix: &[Vec<f32>], target_error: f32) -> Packed128`

### 내부 구조체

```rust
struct CoordinateCache {
    cache_size: usize,
    normalized_coords: Vec<(f32, f32)>,
    distances: Vec<f32>,
    angles: Vec<f32>,
}
```

## HybridEncoder

하이브리드 인코더

```rust
pub struct HybridEncoder {
    poincare_encoder: PoincareEncoder,
    block_size: usize,
}
```

### 함수

- `new(quality_threshold: f32, block_size: usize) -> Self`
- `encode_blocks(&mut self, matrix: &[Vec<f32>]) -> Vec<Packed128>`
- `adaptive_block_encoding(&mut self, matrix: &[Vec<f32>]) -> Vec<Packed128>`
- `merge_block_encodings(&self, blocks: &[Packed128]) -> Packed128`

## GridCompressor

그리드 압축기

```rust
pub struct GridCompressor {
    compression_level: f32,
}
```

### 함수

- `new(compression_level: f32) -> Self`
- `compress_grid(&self, grid: &[Vec<f32>]) -> Vec<u8>`
- `decompress_grid(&self, data: &[u8], rows: usize, cols: usize) -> Vec<Vec<f32>>`

## AnalysisResults

분석 결과

```rust
pub struct AnalysisResults {
    pub quality_score: f32,
    pub compression_ratio: f32,
    pub critical_points: Vec<(usize, usize)>,
    pub error_distribution: Vec<f32>,
}
```

### 함수

- `new() -> Self`
- `add_metric(&mut self, name: &str, value: f32)`
- `get_summary(&self) -> String` 
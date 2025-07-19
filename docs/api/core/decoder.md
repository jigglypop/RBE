# Decoder API

## HyperbolicCordic

CORDIC 알고리즘 기반 쌍곡함수 계산

```rust
pub struct HyperbolicCordic {
    artanh_table: [f32; CORDIC_ITERATIONS],
    shift_table: [f32; CORDIC_ITERATIONS],
}
```

### 함수

- `new() -> Self`
- `tanh(x: f32) -> f32`
- `artanh(x: f32) -> f32`
- `sinh(x: f32) -> f32`
- `cosh(x: f32) -> f32`

### 상수

- `CORDIC_ITERATIONS: usize = 16`
- `CORDIC_GAIN: f32`
- `POINCARE_BOUNDARY: f32 = 0.99`

## WeightGenerator

가중치 생성기

```rust
pub struct WeightGenerator;
```

### 함수

- `new() -> Self`
- `generate_basis_weights(seed: &Packed128, i: usize, j: usize) -> f32`
- `apply_geometric_transform(weight: f32, transform_type: u8) -> f32`
- `normalize_weights(weights: &mut [f32])`

## FusedForwardPass

융합 순전파 연산

```rust
pub struct FusedForwardPass;
```

### 함수

- `new() -> Self`
- `gemv_fused(matrix: &Packed128, input: &[f32], output: &mut [f32])`
- `block_gemv(blocks: &[Packed128], input: &[f32], output: &mut [f32])`
- `parallel_forward(params: &[Packed128], batch: &[Vec<f32>]) -> Vec<Vec<f32>>`

## BlockDecoder

블록 기반 디코더

```rust
pub struct BlockDecoder {
    block_size: usize,
}
```

### 함수

- `new(block_size: usize) -> Self`
- `decode_block(&self, block: &Packed128) -> Vec<f32>`
- `parallel_decode(&self, blocks: &[Packed128]) -> Vec<Vec<f32>>`

## GridDecoder

그리드 기반 디코더

```rust
pub struct GridDecoder {
    grid_resolution: usize,
}
```

### 함수

- `new(resolution: usize) -> Self`
- `decode_grid(&self, params: &Packed128, rows: usize, cols: usize) -> Vec<Vec<f32>>`
- `interpolate_grid(&self, grid: &[Vec<f32>], x: f32, y: f32) -> f32` 
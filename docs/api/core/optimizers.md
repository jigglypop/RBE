# Optimizers API

## AdamState

Adam 옵티마이저 상태

```rust
pub struct AdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
    config: AdamConfig,
}
```

### 함수

- `new(param_count: usize, config: AdamConfig) -> Self`
- `update(&mut self, params: &mut [f32], gradients: &[f32])`
- `reset(&mut self)`
- `with_config(param_count: usize, config: AdamConfig) -> Self`

## RiemannianAdamState

리만 Adam 옵티마이저 상태

```rust
pub struct RiemannianAdamState {
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
    config: RiemannianAdamConfig,
}
```

### 함수

- `new(param_count: usize, config: RiemannianAdamConfig) -> Self`
- `update(&mut self, params: &mut [f32], gradients: &[f32])`
- `compute_metric_tensor(point: (f32, f32)) -> [[f32; 2]; 2]`
- `mobius_addition(a: (f32, f32), b: (f32, f32)) -> (f32, f32)`
- `exponential_map(v: (f32, f32), x: (f32, f32)) -> (f32, f32)`

## TransformAnalyzer

변환 분석기

```rust
pub struct TransformAnalyzer {
    transform_type: TransformType,
}
```

### 함수

- `new(transform_type: TransformType) -> Self`
- `analyze_dct_performance(&self, data: &[f32]) -> f32`
- `analyze_wavelet_performance(&self, data: &[f32]) -> f32`
- `select_optimal_transform(&self, data: &[f32]) -> TransformType`

## HybridOptimizer

하이브리드 최적화기

```rust
pub struct HybridOptimizer {
    adam_optimizer: AdamState,
    riemannian_optimizer: RiemannianAdamState,
    current_stage: OptimizationStage,
    performance_tracker: PerformanceTracker,
}
```

### 함수

- `new(param_count: usize, config: OptimizerConfig) -> Self`
- `step(&mut self, params: &mut Packed128, gradients: &[f32]) -> f32`
- `switch_stage(&mut self, stage: OptimizationStage)`
- `get_performance_report(&self) -> PerformanceReport`

## Configuration

### OptimizerConfig

```rust
pub struct OptimizerConfig {
    pub optimizer_type: OptimizerType,
    pub learning_rate_schedule: LearningRateSchedule,
    pub gradient_clipping: Option<f32>,
}
```

### AdamConfig

```rust
pub struct AdamConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}
```

### RiemannianAdamConfig

```rust
pub struct RiemannianAdamConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub manifold_eps: f32,
}
```

## Enums

### OptimizerType

```rust
pub enum OptimizerType {
    Adam,
    RiemannianAdam,
    Hybrid,
}
```

### TransformType

```rust
pub enum TransformType {
    DCT,
    Wavelet,
    Fourier,
    Hybrid,
}
```

### LearningRateSchedule

```rust
pub enum LearningRateSchedule {
    Constant,
    Exponential,
    StepDecay,
    Cosine,
}
```

### OptimizationStage

```rust
pub enum OptimizationStage {
    Initial,
    Intermediate,
    Final,
}
``` 
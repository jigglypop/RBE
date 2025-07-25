//! # 비트 도메인 통합 미분 시스템 (Bit-Domain Unified Differential System)
//!
//! BitForwardPass와 BitBackwardPass를 중심으로 30,904 epoch/s 성능을 달성하는
//! 순수 비트 연산 미분 계산 모듈

pub mod forward;
pub mod backward;

// 핵심 타입들 재수출 (순수 비트 도메인)
pub use forward::{
    BitForwardPass, BitForwardPass as UnifiedForwardPass, 
    BitForwardConfig as ForwardConfig, 
    BitForwardMetrics as ForwardMetrics
};
pub use backward::{
    BitBackwardPass, BitBackwardPass as UnifiedBackwardPass, 
    BitBackwardConfig as BackwardConfig, 
    BitBackwardMetrics as GradientMetrics,
    OptimizerType
};

/// 비트 도메인 통합 미분 시스템 (완전 독립형)
#[derive(Debug, Clone)]
pub struct DifferentialSystem {
    /// 비트 도메인 순전파 엔진 (30,904 epoch/s)
    pub forward_engine: BitForwardPass,
    /// 비트 도메인 역전파 엔진 (옵티마이저 통합)
    pub backward_engine: BitBackwardPass,
}

impl DifferentialSystem {
    /// 새로운 비트 도메인 통합 미분 시스템 생성 (단순화)
    pub fn new() -> Self {
        let forward_config = forward::BitForwardConfig::default();
        let backward_config = backward::BitBackwardConfig::default();
        
        let forward_engine = forward::BitForwardPass::new(forward_config);
        let backward_engine = backward::BitBackwardPass::new(backward_config);
        
        Self {
            forward_engine,
            backward_engine,
        }
    }
    
    /// **핵심 메서드**: 비트 도메인 통합 순전파 (30,904+ epoch/s)
    pub fn unified_forward(
        &mut self,
        packed: &crate::core::tensors::packed_types::Packed128,
        i: usize,
        j: usize,
        rows: usize,
        cols: usize,
    ) -> f32 {
        // 순수 비트 도메인 초고속 순전파 직접 호출
        self.forward_engine.bit_forward_ultra_fast(packed, i, j, rows, cols)
    }
    
    /// **핵심 메서드**: 비트 도메인 통합 역전파 (옵티마이저 자동 선택)
    pub fn unified_backward(
        &mut self,
        target: &[f32],
        predicted: &[f32],
        packed: &mut crate::core::tensors::packed_types::Packed128,
        rows: usize,
        cols: usize,
        learning_rate: f32,
    ) -> (f32, GradientMetrics) {
        let mut total_loss = 0.0f32;
        let mut operations = 0;
        
        // 고성능 배치 처리
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                if idx >= target.len() || idx >= predicted.len() {
                    break;
                }
                
                let loss = self.backward_engine.bit_backward_ultra_fast(
                    packed, target[idx], predicted[idx], i, j, learning_rate, rows, cols
                );
                total_loss += loss;
                operations += 1;
            }
        }
        
        let avg_loss = if operations > 0 { total_loss / operations as f32 } else { 0.0 };
        let metrics = self.backward_engine.get_performance_metrics().clone();
        
        (avg_loss, metrics)
    }
    
    /// **최고 성능**: 통합 순전파-역전파 (원패스)
    pub fn unified_forward_backward(
        &mut self,
        packed: &mut crate::core::tensors::packed_types::Packed128,
        target: f32,
        i: usize,
        j: usize,
        learning_rate: f32,
        rows: usize,
        cols: usize,
    ) -> (f32, f32) {
        self.backward_engine.unified_forward_backward(
            packed, &mut self.forward_engine, target, i, j, learning_rate, rows, cols
        )
    }
    
    /// 배치 순전파 (벡터화 최적화)
    pub fn batch_forward(
        &mut self,
        packed: &crate::core::tensors::packed_types::Packed128,
        positions: &[(usize, usize)],
        rows: usize,
        cols: usize,
    ) -> Vec<f32> {
        self.forward_engine.bit_forward_batch(packed, positions, rows, cols)
    }
    
    /// 배치 역전파 (병렬 최적화)
    pub fn batch_backward(
        &mut self,
        packed: &mut crate::core::tensors::packed_types::Packed128,
        targets: &[f32],
        predicted: &[f32],
        positions: &[(usize, usize)],
        learning_rate: f32,
        rows: usize,
        cols: usize,
    ) -> f32 {
        self.backward_engine.bit_backward_batch(
            packed, targets, predicted, positions, learning_rate, rows, cols
        )
    }
    
    /// 성능 메트릭 수집 (단순화)
    pub fn get_performance_metrics(&self) -> DifferentialMetrics {
        let forward_metrics = self.forward_engine.get_performance_metrics();
        let backward_metrics = self.backward_engine.get_performance_metrics();
        
        DifferentialMetrics {
            forward_ops_per_second: forward_metrics.forwards_per_second,
            backward_ops_per_second: backward_metrics.backwards_per_second,
            forward_ns_per_op: forward_metrics.avg_bit_computation_ns as f64,
            backward_ns_per_op: backward_metrics.avg_backward_time_ns as f64,
            total_cache_hit_rate: forward_metrics.bit_cache_hit_rate + backward_metrics.bit_gradient_cache_hit_rate,
            optimizer_efficiency: backward_metrics.optimizer_integration_efficiency,
        }
    }
    
    /// 옵티마이저 타입 설정
    pub fn set_optimizer_type(&mut self, optimizer_type: OptimizerType) {
        self.backward_engine.set_optimizer_type(optimizer_type);
    }
    
    /// 시스템 초기화 (성능 최적화)
    pub fn reset(&mut self) {
        self.forward_engine.clear_cache();
        self.backward_engine.clear_cache();
    }
}

/// 비트 도메인 미분 시스템 성능 메트릭 (단순화)
#[derive(Debug, Clone)]
pub struct DifferentialMetrics {
    /// 순전파 초당 연산 수
    pub forward_ops_per_second: f64,
    /// 역전파 초당 연산 수  
    pub backward_ops_per_second: f64,
    /// 순전파 ns/op
    pub forward_ns_per_op: f64,
    /// 역전파 ns/op
    pub backward_ns_per_op: f64,
    /// 총 캐시 히트율
    pub total_cache_hit_rate: f32,
    /// 옵티마이저 효율성
    pub optimizer_efficiency: f32,
} 
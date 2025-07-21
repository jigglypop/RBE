use crate::core::packed_params::packed_types::Packed128;
use crate::core::optimizers::{
    cycle_differential::{CycleDifferentialSystem, CycleState, DifferentialPhase},
    bit_aware_gradients::{FusedGradientComputer, FieldGradientAnalysis, BitGradientContribution},
};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

/// RBE 특화 그래디언트 구조
#[derive(Debug, Clone)]
pub struct RBEGradient {
    /// Hi 필드 비트별 그래디언트 (64비트)
    pub hi_gradients: Vec<f32>,
    /// Lo 필드 연속 그래디언트 (r, theta)
    pub lo_gradients: (f32, f32),
    /// Hi-Lo 상호작용 그래디언트
    pub interactions: Vec<(u8, u8, f32)>,
    /// 그래디언트 크기 (정규화용)
    pub magnitude: f32,
}

impl RBEGradient {
    pub fn new() -> Self {
        Self {
            hi_gradients: vec![0.0; 64],
            lo_gradients: (0.0, 0.0),
            interactions: Vec::new(),
            magnitude: 0.0,
        }
    }
    
    pub fn from_field_analysis(analysis: &FieldGradientAnalysis) -> Self {
        let hi_gradients = analysis.hi_gradients.iter()
            .map(|contrib| contrib.gradient_value)
            .collect();
        
        let magnitude = analysis.hi_gradients.iter()
            .map(|contrib| contrib.gradient_value.abs())
            .sum::<f32>() + analysis.lo_gradients.0.abs() + analysis.lo_gradients.1.abs();
        
        Self {
            hi_gradients,
            lo_gradients: analysis.lo_gradients,
            interactions: analysis.interaction_gradients.clone(),
            magnitude,
        }
    }
    
    /// 그래디언트 크기 계산
    pub fn compute_magnitude(&mut self) {
        self.magnitude = self.hi_gradients.iter().map(|g| g.abs()).sum::<f32>()
                       + self.lo_gradients.0.abs() + self.lo_gradients.1.abs();
    }
    
    /// 그래디언트 정규화
    pub fn normalize(&mut self) {
        if self.magnitude > 1e-8 {
            let scale = 1.0 / self.magnitude;
            for grad in &mut self.hi_gradients {
                *grad *= scale;
            }
            self.lo_gradients.0 *= scale;
            self.lo_gradients.1 *= scale;
            for (_, _, value) in &mut self.interactions {
                *value *= scale;
            }
            self.magnitude = 1.0;
        }
    }
}

/// 자동미분 연산 노드 ID
pub type NodeId = usize;

/// 역전파 함수 트레이트
pub trait BackwardFunction: Send + Sync + std::fmt::Debug {
    fn apply(&self, grad_output: &RBEGradient) -> Vec<RBEGradient>;
    fn get_inputs(&self) -> Vec<NodeId>;
}

/// RBE 텐서 (자동미분 지원)
#[derive(Debug, Clone)]
pub struct RBETensor {
    /// Packed128 데이터
    pub data: Vec<Packed128>,
    /// 텐서 형태 [batch, seq, hidden]
    pub shape: Vec<usize>,
    /// 그래디언트 계산 필요 여부
    pub requires_grad: bool,
    /// 연산 그래프 노드 ID
    pub node_id: Option<NodeId>,
    /// 현재 그래디언트
    pub gradient: Option<RBEGradient>,
}

impl RBETensor {
    pub fn new(data: Vec<Packed128>, shape: Vec<usize>, requires_grad: bool) -> Self {
        Self {
            data,
            shape,
            requires_grad,
            node_id: None,
            gradient: if requires_grad { Some(RBEGradient::new()) } else { None },
        }
    }
    
    /// 총 요소 개수
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// 배치 크기
    pub fn batch_size(&self) -> usize {
        self.shape.get(0).copied().unwrap_or(1)
    }
    
    /// 시퀀스 길이  
    pub fn seq_len(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1)
    }
    
    /// 히든 차원
    pub fn hidden_dim(&self) -> usize {
        self.shape.get(2).copied().unwrap_or(self.data.len())
    }
    
    /// 그래디언트 초기화
    pub fn zero_grad(&mut self) {
        if let Some(ref mut grad) = self.gradient {
            grad.hi_gradients.fill(0.0);
            grad.lo_gradients = (0.0, 0.0);
            grad.interactions.clear();
            grad.magnitude = 0.0;
        }
    }
}

/// RBE 특화 연산 타입
#[derive(Debug, Clone)]
pub enum RBEOperation {
    /// Packed MatMul 연산
    PackedMatMul {
        input: NodeId,
        weight: NodeId,
    },
    /// 11비트 사이클 상태 전이
    CycleTransition {
        input: NodeId,
        cycle_params: Vec<CycleState>,
    },
    /// 하이브리드 최적화 스텝
    HybridOptimize {
        input: NodeId,
        target: Vec<f32>,
    },
    /// 리만 기하학적 업데이트
    RiemannianUpdate {
        input: NodeId,
        manifold_params: (f32, f32),
    },
}

/// 연산 그래프 노드
#[derive(Debug)]
pub struct ComputationNode {
    pub operation: RBEOperation,
    pub backward_fn: Option<Box<dyn BackwardFunction>>,
    pub output_shape: Vec<usize>,
}

/// RBE 자동미분 연산 그래프
#[derive(Debug)]
pub struct ComputationGraph {
    /// 노드들
    nodes: HashMap<NodeId, ComputationNode>,
    /// 다음 노드 ID
    next_id: NodeId,
    /// 텐서 저장소
    tensors: HashMap<NodeId, RBETensor>,
    /// 실행 순서 (위상 정렬)
    execution_order: Vec<NodeId>,
}

impl ComputationGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            tensors: HashMap::new(),
            execution_order: Vec::new(),
        }
    }
    
    /// 새 노드 추가
    pub fn add_node(&mut self, operation: RBEOperation, output_shape: Vec<usize>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        
        let node = ComputationNode {
            operation,
            backward_fn: None,
            output_shape,
        };
        
        self.nodes.insert(id, node);
        self.execution_order.push(id);
        
        id
    }
    
    /// 텐서 등록
    pub fn register_tensor(&mut self, tensor: RBETensor) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        
        self.tensors.insert(id, tensor);
        id
    }
    
    /// 순전파 실행
    pub fn forward(&mut self, input_id: NodeId) -> Result<RBETensor> {
        // 실행 순서대로 연산 수행
        let mut current_tensor = self.tensors.get(&input_id)
            .ok_or_else(|| anyhow::anyhow!("Input tensor not found"))?
            .clone();
            
        let execution_order = self.execution_order.clone(); // 순서 복사
        for &node_id in &execution_order {
            if let Some(node) = self.nodes.get(&node_id) {
                current_tensor = self.execute_operation(&node.operation, &current_tensor)?;
                current_tensor.node_id = Some(node_id);
                
                // 그래디언트 계산 필요한 경우 역전파 함수 등록
                if current_tensor.requires_grad {
                    self.register_backward_fn(node_id, &current_tensor)?;
                }
            }
        }
        
        Ok(current_tensor)
    }
    
    /// 연산 실행
    fn execute_operation(&self, operation: &RBEOperation, input: &RBETensor) -> Result<RBETensor> {
        match operation {
            RBEOperation::PackedMatMul { input: _, weight } => {
                let weight_tensor = self.tensors.get(weight)
                    .ok_or_else(|| anyhow::anyhow!("Weight tensor not found"))?;
                self.packed_matmul_forward(input, weight_tensor)
            },
            RBEOperation::CycleTransition { input: _, cycle_params } => {
                self.cycle_transition_forward(input, cycle_params)
            },
            RBEOperation::HybridOptimize { input: _, target } => {
                self.hybrid_optimize_forward(input, target)
            },
            RBEOperation::RiemannianUpdate { input: _, manifold_params } => {
                self.riemannian_update_forward(input, manifold_params)
            },
        }
    }
    
    /// Packed MatMul 순전파
    fn packed_matmul_forward(&self, input: &RBETensor, weight: &RBETensor) -> Result<RBETensor> {
        let batch_size = input.batch_size();
        let seq_len = input.seq_len();
        let input_dim = input.hidden_dim();
        let output_dim = weight.hidden_dim();
        
        let mut output_data = Vec::new();
        
        // 배치/시퀀스별 병렬 처리
        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                let input_start = (batch_idx * seq_len + seq_idx) * input_dim;
                let input_slice = &input.data[input_start..input_start + input_dim];
                
                // Packed128 MatMul 연산
                let mut output_vector = vec![Packed128::zero(); output_dim];
                
                for out_idx in 0..output_dim {
                    for in_idx in 0..input_dim {
                        if in_idx < input_slice.len() && out_idx < weight.data.len() {
                            // Hi 필드 연산 (비트 AND)
                            let hi_result = input_slice[in_idx].hi & weight.data[out_idx].hi;
                            
                            // Lo 필드 연산 (부동소수점 곱셈)
                            let lo_result = f32::from_bits(input_slice[in_idx].lo as u32) * 
                                          f32::from_bits(weight.data[out_idx].lo as u32);
                            
                            output_vector[out_idx] = Packed128 {
                                hi: output_vector[out_idx].hi ^ hi_result,
                                lo: (lo_result.to_bits() as u64) | (output_vector[out_idx].lo & 0xFFFFFFFF00000000),
                            };
                        }
                    }
                }
                
                output_data.extend(output_vector);
            }
        }
        
        Ok(RBETensor::new(
            output_data,
            vec![batch_size, seq_len, output_dim],
            input.requires_grad || weight.requires_grad,
        ))
    }
    
    /// 11비트 사이클 전이 순전파
    fn cycle_transition_forward(&self, input: &RBETensor, cycle_params: &[CycleState]) -> Result<RBETensor> {
        let mut cycle_system = CycleDifferentialSystem::new(11);
        let mut output_data = input.data.clone();
        
        // 각 Packed128에 대해 사이클 전이 적용
        for (i, (data, params)) in output_data.iter_mut().zip(cycle_params.iter().cycle()).enumerate() {
            // Hi 필드에서 11비트 추출하여 사이클 적용
            let hi_bits = (data.hi & 0x7FF) as u16; // 하위 11비트
            let gradient_signal = f32::from_bits(data.lo as u32); // Lo 필드에서 그래디언트 신호 추출
            let learning_phase = crate::core::optimizers::cycle_differential::DifferentialPhase::Exploration; // 기본값
            let new_state = cycle_system.apply_differential_cycle(i, gradient_signal, learning_phase);
            
            // 새로운 상태를 Hi 필드에 반영
            data.hi = (data.hi & !0x7FF) | (new_state as u64);
        }
        
        Ok(RBETensor::new(
            output_data,
            input.shape.clone(),
            input.requires_grad,
        ))
    }
    
    /// 하이브리드 최적화 순전파
    fn hybrid_optimize_forward(&self, input: &RBETensor, target: &[f32]) -> Result<RBETensor> {
        let mut grad_computer = FusedGradientComputer::new();
        let mut output_data = input.data.clone();
        
        // 각 요소에 대해 하이브리드 최적화 적용
        for (i, data) in output_data.iter_mut().enumerate() {
            if i < target.len() {
                let predicted = f32::from_bits(data.lo as u32);
                let error = target[i] - predicted;
                
                // Lo 필드 업데이트 (연속 최적화)
                let new_value = predicted + 0.01 * error; // 간단한 업데이트
                data.lo = (data.lo & 0xFFFFFFFF00000000) | (new_value.to_bits() as u64);
                
                // Hi 필드 업데이트 (이산 최적화)
                if error.abs() > 0.1 {
                    data.hi ^= 1; // 간단한 비트 플립
                }
            }
        }
        
        Ok(RBETensor::new(
            output_data,
            input.shape.clone(),
            input.requires_grad,
        ))
    }
    
    /// 리만 기하학적 업데이트 순전파
    fn riemannian_update_forward(&self, input: &RBETensor, manifold_params: &(f32, f32)) -> Result<RBETensor> {
        let (curvature, metric_scale) = *manifold_params;
        let mut output_data = input.data.clone();
        
        // 푸앵카레 볼에서의 기하학적 업데이트
        for data in &mut output_data {
            let r = f32::from_bits((data.lo & 0xFFFFFFFF) as u32);
            let theta = f32::from_bits((data.lo >> 32) as u32);
            
            // 리만 메트릭 적용
            let new_r = r * (1.0 + curvature * metric_scale);
            let new_theta = theta + 0.01 * metric_scale; // 각도 회전
            
            // 푸앵카레 볼 경계 조건 (|z| < 1)
            let new_r = if new_r >= 1.0 { 0.99 } else { new_r };
            
            data.lo = ((new_theta.to_bits() as u64) << 32) | (new_r.to_bits() as u64);
        }
        
        Ok(RBETensor::new(
            output_data,
            input.shape.clone(),
            input.requires_grad,
        ))
    }
    
    /// 역전파 함수 등록
    fn register_backward_fn(&mut self, node_id: NodeId, output: &RBETensor) -> Result<()> {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            let backward_fn: Box<dyn BackwardFunction> = match &node.operation {
                RBEOperation::PackedMatMul { input, weight } => {
                    Box::new(PackedMatMulBackward {
                        input_id: *input,
                        weight_id: *weight,
                        output_shape: output.shape.clone(),
                    })
                },
                RBEOperation::CycleTransition { input, cycle_params } => {
                    Box::new(CycleTransitionBackward {
                        input_id: *input,
                        cycle_states: cycle_params.clone(),
                        output_shape: output.shape.clone(),
                    })
                },
                RBEOperation::HybridOptimize { input, target } => {
                    Box::new(HybridOptimizeBackward {
                        input_id: *input,
                        target: target.clone(),
                        output_shape: output.shape.clone(),
                    })
                },
                RBEOperation::RiemannianUpdate { input, manifold_params } => {
                    Box::new(RiemannianUpdateBackward {
                        input_id: *input,
                        manifold_params: *manifold_params,
                        output_shape: output.shape.clone(),
                    })
                },
            };
            
            node.backward_fn = Some(backward_fn);
        }
        
        Ok(())
    }
    
    /// 역전파 실행
    pub fn backward(&mut self, loss_grad: &RBEGradient) -> Result<()> {
        // 역순으로 그래디언트 전파
        for &node_id in self.execution_order.iter().rev() {
            if let Some(node) = self.nodes.get(&node_id) {
                if let Some(ref backward_fn) = node.backward_fn {
                    let input_grads = backward_fn.apply(loss_grad);
                    
                    // 입력 텐서들에 그래디언트 누적
                    let input_ids = backward_fn.get_inputs();
                    for (input_id, grad) in input_ids.into_iter().zip(input_grads) {
                        if let Some(tensor) = self.tensors.get_mut(&input_id) {
                            if let Some(ref mut tensor_grad) = tensor.gradient {
                                // 그래디언트 누적
                                for (i, &new_grad) in grad.hi_gradients.iter().enumerate() {
                                    if i < tensor_grad.hi_gradients.len() {
                                        tensor_grad.hi_gradients[i] += new_grad;
                                    }
                                }
                                tensor_grad.lo_gradients.0 += grad.lo_gradients.0;
                                tensor_grad.lo_gradients.1 += grad.lo_gradients.1;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
}

// 역전파 함수 구현들

/// Packed MatMul 역전파
#[derive(Debug)]
struct PackedMatMulBackward {
    input_id: NodeId,
    weight_id: NodeId,
    output_shape: Vec<usize>,
}

impl BackwardFunction for PackedMatMulBackward {
    fn apply(&self, grad_output: &RBEGradient) -> Vec<RBEGradient> {
        // 입력과 가중치에 대한 그래디언트 계산
        let mut input_grad = RBEGradient::new();
        let mut weight_grad = RBEGradient::new();
        
        // 간단한 그래디언트 전파 (실제로는 더 복잡한 계산 필요)
        for (i, &out_grad) in grad_output.hi_gradients.iter().enumerate() {
            if i < input_grad.hi_gradients.len() {
                input_grad.hi_gradients[i] = out_grad * 0.5; // 입력 그래디언트
                weight_grad.hi_gradients[i] = out_grad * 0.5; // 가중치 그래디언트
            }
        }
        
        input_grad.lo_gradients = grad_output.lo_gradients;
        weight_grad.lo_gradients = grad_output.lo_gradients;
        
        vec![input_grad, weight_grad]
    }
    
    fn get_inputs(&self) -> Vec<NodeId> {
        vec![self.input_id, self.weight_id]
    }
}

/// 사이클 전이 역전파
#[derive(Debug)]
struct CycleTransitionBackward {
    input_id: NodeId,
    cycle_states: Vec<CycleState>,
    output_shape: Vec<usize>,
}

impl BackwardFunction for CycleTransitionBackward {
    fn apply(&self, grad_output: &RBEGradient) -> Vec<RBEGradient> {
        let mut input_grad = RBEGradient::new();
        
        // 쌍곡함수 미분을 통한 그래디언트 계산
        for (i, &out_grad) in grad_output.hi_gradients.iter().enumerate() {
            if i < self.cycle_states.len() {
                let state = &self.cycle_states[i];
                let hyperbolic_func = state.get_active_function();
                
                // 쌍곡함수 미분값 적용
                let derivative_scale = match hyperbolic_func.evaluate(1.0) {
                    val if val.abs() > 1e-8 => 1.0 / val,
                    _ => 1.0,
                };
                
                input_grad.hi_gradients[i] = out_grad * derivative_scale;
            }
        }
        
        input_grad.lo_gradients = grad_output.lo_gradients;
        
        vec![input_grad]
    }
    
    fn get_inputs(&self) -> Vec<NodeId> {
        vec![self.input_id]
    }
}

/// 하이브리드 최적화 역전파
#[derive(Debug)]
struct HybridOptimizeBackward {
    input_id: NodeId,
    target: Vec<f32>,
    output_shape: Vec<usize>,
}

impl BackwardFunction for HybridOptimizeBackward {
    fn apply(&self, grad_output: &RBEGradient) -> Vec<RBEGradient> {
        let mut input_grad = RBEGradient::new();
        
        // 목표값과의 차이를 이용한 그래디언트 계산
        for (i, &target_val) in self.target.iter().enumerate() {
            if i < input_grad.hi_gradients.len() {
                let error_scale = if target_val.abs() > 1e-8 { 1.0 / target_val } else { 1.0 };
                input_grad.hi_gradients[i] = grad_output.hi_gradients[i] * error_scale;
            }
        }
        
        input_grad.lo_gradients = grad_output.lo_gradients;
        
        vec![input_grad]
    }
    
    fn get_inputs(&self) -> Vec<NodeId> {
        vec![self.input_id]
    }
}

/// 리만 업데이트 역전파
#[derive(Debug)]
struct RiemannianUpdateBackward {
    input_id: NodeId,
    manifold_params: (f32, f32),
    output_shape: Vec<usize>,
}

impl BackwardFunction for RiemannianUpdateBackward {
    fn apply(&self, grad_output: &RBEGradient) -> Vec<RBEGradient> {
        let mut input_grad = RBEGradient::new();
        let (curvature, metric_scale) = self.manifold_params;
        
        // 리만 메트릭을 고려한 그래디언트 변환
        let riemannian_scale = 1.0 + curvature * metric_scale;
        
        for (i, &out_grad) in grad_output.hi_gradients.iter().enumerate() {
            if i < input_grad.hi_gradients.len() {
                input_grad.hi_gradients[i] = out_grad / riemannian_scale;
            }
        }
        
        input_grad.lo_gradients = (
            grad_output.lo_gradients.0 / riemannian_scale,
            grad_output.lo_gradients.1 / riemannian_scale,
        );
        
        vec![input_grad]
    }
    
    fn get_inputs(&self) -> Vec<NodeId> {
        vec![self.input_id]
    }
}

impl Packed128 {
    pub fn zero() -> Self {
        Self { hi: 0, lo: 0 }
    }
} 
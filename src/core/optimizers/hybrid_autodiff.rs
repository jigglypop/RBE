use crate::core::{
    packed_params::packed_types::Packed128,
    optimizers::{
        hybrid::HybridOptimizer,
        auto_diff::{RBEGradient, RBETensor, ComputationGraph, RBEOperation},
        cycle_differential::CycleState,
    },
};
use anyhow::Result;
use std::time::Instant;

/// 자동미분 통합 하이브리드 최적화기
#[derive(Debug)]
pub struct AutoDiffHybridOptimizer {
    /// 기존 하이브리드 최적화기
    base_optimizer: HybridOptimizer,
    /// 연산 그래프
    computation_graph: ComputationGraph,
    /// 성능 메트릭
    performance_metrics: AutoDiffPerformanceMetrics,
    /// 자동미분 활성화 여부
    autodiff_enabled: bool,
}

/// 자동미분 성능 메트릭
#[derive(Debug, Clone)]
pub struct AutoDiffPerformanceMetrics {
    /// 순전파 시간 (마이크로초)
    pub forward_time_us: f64,
    /// 역전파 시간 (마이크로초)
    pub backward_time_us: f64,
    /// 총 최적화 시간 (마이크로초)
    pub total_optimization_time_us: f64,
    /// 기존 대비 속도 향상 배수
    pub speedup_factor: f64,
    /// 메모리 사용량 (바이트)
    pub memory_usage_bytes: usize,
    /// 정확도 개선률 (%)
    pub accuracy_improvement_percent: f64,
    /// 실행 스텝 수
    pub executed_steps: usize,
}

impl AutoDiffPerformanceMetrics {
    pub fn new() -> Self {
        Self {
            forward_time_us: 0.0,
            backward_time_us: 0.0,
            total_optimization_time_us: 0.0,
            speedup_factor: 1.0,
            memory_usage_bytes: 0,
            accuracy_improvement_percent: 0.0,
            executed_steps: 0,
        }
    }
}

impl AutoDiffHybridOptimizer {
    /// 새로운 자동미분 하이브리드 최적화기 생성
    pub fn new(learning_rate: f32, max_cycle_length: usize, autodiff_enabled: bool) -> Self {
        Self {
            base_optimizer: HybridOptimizer::new(learning_rate, max_cycle_length),
            computation_graph: ComputationGraph::new(),
            performance_metrics: AutoDiffPerformanceMetrics::new(),
            autodiff_enabled,
        }
    }
    
    /// 자동미분을 활용한 최적화 스텝
    pub fn step_with_autodiff(
        &mut self,
        packed: &mut Packed128,
        target: &[f32],
        predicted: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<f32> {
        let start_time = Instant::now();
        
        if self.autodiff_enabled {
            self.autodiff_step(packed, target, predicted, rows, cols)
        } else {
            self.manual_step(packed, target, predicted, rows, cols)
        }
        .map(|loss| {
            self.performance_metrics.total_optimization_time_us += 
                start_time.elapsed().as_micros() as f64;
            self.performance_metrics.executed_steps += 1;
            loss
        })
    }
    
    /// 자동미분 기반 최적화 스텝
    fn autodiff_step(
        &mut self,
        packed: &mut Packed128,
        target: &[f32],
        predicted: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<f32> {
        // 1. RBETensor 생성
        let input_tensor = RBETensor::new(
            vec![*packed],
            vec![1, rows, cols],
            true, // requires_grad = true
        );
        
        // 2. 연산 그래프 구성
        let input_id = self.computation_graph.register_tensor(input_tensor);
        
        // 3. 순전파 연산들 추가
        let forward_start = Instant::now();
        
        // 3.1. 11비트 사이클 전이 추가
        let cycle_states = self.generate_cycle_states(rows * cols);
        let cycle_node = self.computation_graph.add_node(
            RBEOperation::CycleTransition {
                input: input_id,
                cycle_params: cycle_states,
            },
            vec![1, rows, cols],
        );
        
        // 3.2. 하이브리드 최적화 추가
        let hybrid_node = self.computation_graph.add_node(
            RBEOperation::HybridOptimize {
                input: cycle_node,
                target: target.to_vec(),
            },
            vec![1, rows, cols],
        );
        
        // 3.3. 리만 기하학적 업데이트 추가
        let riemannian_node = self.computation_graph.add_node(
            RBEOperation::RiemannianUpdate {
                input: hybrid_node,
                manifold_params: (0.1, 0.01), // 곡률과 메트릭 스케일
            },
            vec![1, rows, cols],
        );
        
        // 4. 순전파 실행
        let output = self.computation_graph.forward(input_id)?;
        
        self.performance_metrics.forward_time_us += 
            forward_start.elapsed().as_micros() as f64;
        
        // 5. 손실 함수 계산
        let loss = self.compute_loss(&output, target);
        
        // 6. 역전파 실행
        let backward_start = Instant::now();
        
        let loss_gradient = self.compute_loss_gradient(&output, target);
        self.computation_graph.backward(&loss_gradient)?;
        
        self.performance_metrics.backward_time_us += 
            backward_start.elapsed().as_micros() as f64;
        
        // 7. 파라미터 업데이트
        if !output.data.is_empty() {
            *packed = output.data[0];
        }
        
        // 8. 메모리 사용량 업데이트
        self.update_memory_usage();
        
        Ok(loss)
    }
    
    /// 기존 수동 최적화 스텝 (비교용)
    fn manual_step(
        &mut self,
        packed: &mut Packed128,
        target: &[f32],
        predicted: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<f32> {
        // 기존 HybridOptimizer 사용
        Ok(self.base_optimizer.step(packed, target, predicted, rows, cols))
    }
    
    /// 사이클 상태 생성 (11비트 구조)
    fn generate_cycle_states(&self, count: usize) -> Vec<CycleState> {
        (0..count)
            .map(|i| {
                let bits = (i % 2048) as u16; // 11비트 범위
                CycleState::from_bits(bits)
            })
            .collect()
    }
    
    /// 손실 함수 계산 (MSE)
    fn compute_loss(&self, output: &RBETensor, target: &[f32]) -> f32 {
        let mut total_loss = 0.0;
        let mut count = 0;
        
        for (i, packed_data) in output.data.iter().enumerate() {
            if i < target.len() {
                // Lo 필드에서 예측값 추출
                let predicted = f32::from_bits(packed_data.lo as u32);
                let error = target[i] - predicted;
                total_loss += error * error;
                count += 1;
            }
        }
        
        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }
    
    /// 손실 그래디언트 계산
    fn compute_loss_gradient(&self, output: &RBETensor, target: &[f32]) -> RBEGradient {
        let mut gradient = RBEGradient::new();
        
        for (i, packed_data) in output.data.iter().enumerate() {
            if i < target.len() && i < gradient.hi_gradients.len() {
                // Lo 필드에서 예측값 추출
                let predicted = f32::from_bits(packed_data.lo as u32);
                let error = target[i] - predicted;
                
                // MSE 그래디언트: 2 * (predicted - target) / N
                let grad_value = 2.0 * error / target.len() as f32;
                
                // Hi 필드 그래디언트 (이산)
                gradient.hi_gradients[i] = if error.abs() > 0.1 { grad_value } else { 0.0 };
                
                // Lo 필드 그래디언트 (연속)
                if i == 0 {
                    gradient.lo_gradients.0 = grad_value; // r 성분
                }
                if i == 1 {
                    gradient.lo_gradients.1 = grad_value; // theta 성분
                }
            }
        }
        
        gradient.compute_magnitude();
        gradient
    }
    
    /// 메모리 사용량 업데이트
    fn update_memory_usage(&mut self) {
        // 연산 그래프와 텐서들의 메모리 사용량 추정
        let tensor_memory = std::mem::size_of::<RBETensor>() * 10; // 추정값
        let graph_memory = std::mem::size_of::<ComputationGraph>();
        let gradient_memory = std::mem::size_of::<RBEGradient>() * 5; // 추정값
        
        self.performance_metrics.memory_usage_bytes = 
            tensor_memory + graph_memory + gradient_memory;
    }
    
    /// 성능 메트릭 반환
    pub fn get_performance_metrics(&self) -> &AutoDiffPerformanceMetrics {
        &self.performance_metrics
    }
    
    /// 성능 비교 실행 (자동미분 vs 수동)
    pub fn benchmark_comparison(
        &mut self,
        test_data: &[(Packed128, Vec<f32>, Vec<f32>)], // (packed, target, predicted)
        iterations: usize,
    ) -> Result<BenchmarkResults> {
        let mut autodiff_results = Vec::new();
        let mut manual_results = Vec::new();
        
        println!("🚀 자동미분 vs 수동 최적화 벤치마크 시작 ({}회 반복)", iterations);
        
        // 자동미분 테스트
        println!("📊 자동미분 최적화 테스트...");
        self.autodiff_enabled = true;
        let autodiff_start = Instant::now();
        
        for (i, (mut packed, target, predicted)) in test_data.iter().cloned().enumerate() {
            for _ in 0..iterations {
                let loss = self.step_with_autodiff(
                    &mut packed,
                    &target,
                    &predicted,
                    8, 8 // 8x8 행렬 가정
                )?;
                autodiff_results.push(loss);
            }
            
            if i % 10 == 0 {
                println!("   진행률: {}/{}", i + 1, test_data.len());
            }
        }
        
        let autodiff_total_time = autodiff_start.elapsed();
        
        // 수동 최적화 테스트
        println!("📊 수동 최적화 테스트...");
        self.autodiff_enabled = false;
        let manual_start = Instant::now();
        
        for (i, (mut packed, target, predicted)) in test_data.iter().cloned().enumerate() {
            for _ in 0..iterations {
                let loss = self.step_with_autodiff(
                    &mut packed,
                    &target,
                    &predicted,
                    8, 8
                )?;
                manual_results.push(loss);
            }
            
            if i % 10 == 0 {
                println!("   진행률: {}/{}", i + 1, test_data.len());
            }
        }
        
        let manual_total_time = manual_start.elapsed();
        
        // 결과 분석
        let autodiff_avg_loss = autodiff_results.iter().sum::<f32>() / autodiff_results.len() as f32;
        let manual_avg_loss = manual_results.iter().sum::<f32>() / manual_results.len() as f32;
        
        let speedup = manual_total_time.as_micros() as f64 / autodiff_total_time.as_micros() as f64;
        let accuracy_improvement = ((manual_avg_loss - autodiff_avg_loss) / manual_avg_loss * 100.0).max(0.0);
        
        // 성능 메트릭 업데이트
        self.performance_metrics.speedup_factor = speedup;
        self.performance_metrics.accuracy_improvement_percent = accuracy_improvement as f64;
        
        Ok(BenchmarkResults {
            autodiff_time_ms: autodiff_total_time.as_millis() as f64,
            manual_time_ms: manual_total_time.as_millis() as f64,
            autodiff_avg_loss,
            manual_avg_loss,
            speedup_factor: speedup,
            accuracy_improvement_percent: accuracy_improvement as f64,
            iterations_per_second_autodiff: (test_data.len() * iterations) as f64 / autodiff_total_time.as_secs_f64(),
            iterations_per_second_manual: (test_data.len() * iterations) as f64 / manual_total_time.as_secs_f64(),
        })
    }
    
    /// 정확도 검증 테스트
    pub fn validate_accuracy(
        &mut self,
        test_cases: &[(Vec<f32>, Vec<f32>)], // (input, expected_output)
    ) -> Result<AccuracyResults> {
        let mut total_error = 0.0;
        let mut max_error: f32 = 0.0;
        let mut converged_cases = 0;
        
        self.autodiff_enabled = true;
        
        for (i, (input, expected)) in test_cases.iter().enumerate() {
            let mut packed = Packed128 {
                hi: 0x123456789ABCDEF0,
                lo: input.get(0).unwrap_or(&0.0).to_bits() as u64,
            };
            
            // 여러 스텝으로 수렴 테스트
            let mut final_loss = f32::INFINITY;
            for step in 0..100 {
                let predicted = vec![f32::from_bits(packed.lo as u32)];
                final_loss = self.step_with_autodiff(
                    &mut packed,
                    expected,
                    &predicted,
                    1, 1
                )?;
                
                // 수렴 체크
                if final_loss < 1e-6 {
                    converged_cases += 1;
                    break;
                }
            }
            
            total_error += final_loss;
            max_error = max_error.max(final_loss);
            
            if i % 50 == 0 {
                println!("   정확도 검증: {}/{} (현재 오차: {:.6})", i + 1, test_cases.len(), final_loss);
            }
        }
        
        Ok(AccuracyResults {
            average_error: total_error / test_cases.len() as f32,
            max_error,
            convergence_rate: converged_cases as f64 / test_cases.len() as f64,
            total_test_cases: test_cases.len(),
        })
    }
    
    /// 진단 정보 출력
    pub fn print_diagnostics(&self) {
        println!("\n🔍 자동미분 하이브리드 최적화기 진단 정보:");
        println!("   자동미분 활성화: {}", self.autodiff_enabled);
        println!("   순전파 평균 시간: {:.2}μs", self.performance_metrics.forward_time_us / self.performance_metrics.executed_steps as f64);
        println!("   역전파 평균 시간: {:.2}μs", self.performance_metrics.backward_time_us / self.performance_metrics.executed_steps as f64);
        println!("   총 최적화 시간: {:.2}ms", self.performance_metrics.total_optimization_time_us / 1000.0);
        println!("   속도 향상: {:.2}x", self.performance_metrics.speedup_factor);
        println!("   정확도 개선: {:.2}%", self.performance_metrics.accuracy_improvement_percent);
        println!("   메모리 사용량: {:.2}KB", self.performance_metrics.memory_usage_bytes as f64 / 1024.0);
        println!("   실행 스텝 수: {}", self.performance_metrics.executed_steps);
    }
}

/// 벤치마크 결과
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub autodiff_time_ms: f64,
    pub manual_time_ms: f64,
    pub autodiff_avg_loss: f32,
    pub manual_avg_loss: f32,
    pub speedup_factor: f64,
    pub accuracy_improvement_percent: f64,
    pub iterations_per_second_autodiff: f64,
    pub iterations_per_second_manual: f64,
}

/// 정확도 결과
#[derive(Debug, Clone)]
pub struct AccuracyResults {
    pub average_error: f32,
    pub max_error: f32,
    pub convergence_rate: f64,
    pub total_test_cases: usize,
} 
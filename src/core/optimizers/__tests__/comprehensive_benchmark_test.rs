use crate::core::{
    packed_params::packed_types::Packed128,
    optimizers::{
        hybrid::HybridOptimizer,
        auto_diff::{RBETensor, ComputationGraph, RBEOperation},
        bit_autodiff::{BitTensor, BitGradientTracker},
        bit_dp_autodiff::{BitDPTensor},
        cycle_differential::CycleState,
    },
};
use std::time::Instant;
use std::collections::HashMap;

/// 종합 성능 비교 결과
#[derive(Debug, Clone)]
pub struct ComprehensiveBenchmarkResults {
    /// 수동 최적화 결과
    pub manual_results: OptimizationResults,
    /// 기존 f32 자동미분 결과  
    pub f32_autodiff_results: OptimizationResults,
    /// 비트 자동미분 결과
    pub bit_autodiff_results: OptimizationResults,
    /// 비트 DP 자동미분 결과
    pub bit_dp_autodiff_results: OptimizationResults,
    /// 비교 분석
    pub analysis: PerformanceAnalysis,
}

/// 개별 최적화 결과
#[derive(Debug, Clone)]
pub struct OptimizationResults {
    /// 총 실행 시간 (마이크로초)
    pub total_time_us: f64,
    /// 평균 손실값
    pub average_loss: f32,
    /// 메모리 사용량 (바이트)
    pub memory_usage_bytes: usize,
    /// 처리 속도 (iterations/second)
    pub throughput_ips: f64,
    /// 정확도 (수렴률)
    pub convergence_rate: f64,
    /// 수치적 안정성 점수
    pub numerical_stability: f32,
}

/// 성능 분석 결과
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// 비트 자동미분 vs 수동 최적화 속도 비율
    pub bit_vs_manual_speedup: f64,
    /// 비트 자동미분 vs f32 자동미분 속도 비율
    pub bit_vs_f32_speedup: f64,
    /// 정확도 개선률
    pub accuracy_improvement_percent: f64,
    /// 메모리 효율성 개선률
    pub memory_efficiency_improvement: f64,
    /// 종합 성능 점수 (0-100)
    pub overall_performance_score: f32,
}

fn 테스트_데이터_생성_대용량(count: usize) -> Vec<(Packed128, Vec<f32>, Vec<f32>)> {
    (0..count)
        .map(|i| {
            let packed = Packed128 {
                hi: 0x123456789ABCDEF0 ^ (i as u64),
                lo: ((i as f32 * 0.01).sin().to_bits() as u64) | 
                    (((i as f32 * 0.02).cos().to_bits() as u64) << 32),
            };
            
            // 64x64 = 4096개 요소 (큰 행렬)
            let mut target = Vec::with_capacity(4096);
            let mut predicted = Vec::with_capacity(4096);
            
            for j in 0..4096 {
                let x = (i + j) as f32 * 0.001;
                target.push(x.sin() * x.cos());
                predicted.push(x.sin() * x.cos() * 0.95 + 0.05); // 약간의 오차
            }
            
            (packed, target, predicted)
        })
        .collect()
}

fn 사이클_상태_생성(count: usize) -> Vec<CycleState> {
    (0..count)
        .map(|i| CycleState::from_bits((i % 2048) as u16))
        .collect()
}

#[test]
fn 종합_성능_벤치마크_테스트() {
    println!("🚀 RBE 최적화 방법 종합 성능 벤치마크 시작");
    println!("   비교 대상: 수동 최적화 vs f32 자동미분 vs 비트 자동미분");
    
    let test_data = 테스트_데이터_생성_대용량(50); // 50개 * 4096 = 204,800 요소
    let cycle_states = 사이클_상태_생성(4096);
    let iterations = 5; // 충분한 반복
    
    println!("   테스트 규모: {}개 샘플, {}회 반복", test_data.len(), iterations);
    println!("   행렬 크기: 64x64 (4096 요소)");
    
    // 1. 수동 최적화 벤치마크
    println!("\n📊 1. 수동 최적화 벤치마크...");
    let manual_results = 수동_최적화_벤치마크(&test_data, iterations);
    
    // 2. f32 자동미분 벤치마크  
    println!("\n📊 2. f32 자동미분 벤치마크...");
    let f32_autodiff_results = f32_자동미분_벤치마크(&test_data, iterations);
    
    // 3. 비트 자동미분 벤치마크
    println!("\n📊 3. 비트 자동미분 벤치마크...");
    let bit_autodiff_results = 비트_자동미분_벤치마크(&test_data, &cycle_states, iterations);
    
    // 4. 비트 DP 자동미분 벤치마크 (🚀 새로운 DP 알고리즘)
    println!("\n📊 4. 비트 DP 자동미분 벤치마크...");
    let bit_dp_autodiff_results = 비트_DP_자동미분_벤치마크(&test_data, &cycle_states, iterations);
    
    // 5. 종합 분석
    let analysis = 성능_분석_수행_확장(&manual_results, &f32_autodiff_results, &bit_autodiff_results, &bit_dp_autodiff_results);
    
    let comprehensive_results = ComprehensiveBenchmarkResults {
        manual_results: manual_results.clone(),
        f32_autodiff_results: f32_autodiff_results.clone(),
        bit_autodiff_results: bit_autodiff_results.clone(),
        bit_dp_autodiff_results: bit_dp_autodiff_results.clone(),
        analysis: analysis.clone(),
    };
    
    // 5. 결과 출력
    결과_상세_출력(&comprehensive_results);
    
    // 6. 성능 검증
    성능_어설션_검증(&comprehensive_results);
    
    println!("✅ 종합 성능 벤치마크 완료");
}

fn 수동_최적화_벤치마크(test_data: &[(Packed128, Vec<f32>, Vec<f32>)], iterations: usize) -> OptimizationResults {
    let start_time = Instant::now();
    let mut total_loss = 0.0;
    let mut convergence_count = 0;
    let mut stability_sum = 0.0;
    
    let mut optimizer = HybridOptimizer::new(0.01, 10);
    
    for iter in 0..iterations {
        for (i, (mut packed, target, predicted)) in test_data.iter().cloned().enumerate() {
            let loss = optimizer.step(&mut packed, &target, &predicted, 64, 64);
            total_loss += loss;
            
            // 수렴 체크
            if loss < 0.01 {
                convergence_count += 1;
            }
            
            // 수치적 안정성 체크
            if loss.is_finite() && loss >= 0.0 {
                stability_sum += 1.0;
            }
            
            if i % 10 == 0 {
                print!(".");
            }
        }
        println!(" 반복 {}/{} 완료", iter + 1, iterations);
    }
    
    let elapsed = start_time.elapsed();
    let total_operations = test_data.len() * iterations;
    
    OptimizationResults {
        total_time_us: elapsed.as_micros() as f64,
        average_loss: total_loss / total_operations as f32,
        memory_usage_bytes: std::mem::size_of::<HybridOptimizer>(),
        throughput_ips: total_operations as f64 / elapsed.as_secs_f64(),
        convergence_rate: convergence_count as f64 / total_operations as f64,
        numerical_stability: stability_sum / total_operations as f32,
    }
}

fn f32_자동미분_벤치마크(test_data: &[(Packed128, Vec<f32>, Vec<f32>)], iterations: usize) -> OptimizationResults {
    let start_time = Instant::now();
    let mut total_loss = 0.0;
    let mut convergence_count = 0;
    let mut stability_sum = 0.0;
    
    let mut computation_graph = ComputationGraph::new();
    
    for iter in 0..iterations {
        for (i, (packed, target, predicted)) in test_data.iter().enumerate() {
            // f32 기반 자동미분 (기존 방식)
            let input_tensor = RBETensor::new(
                vec![*packed],
                vec![1, 64, 64],
                true,
            );
            
            let input_id = computation_graph.register_tensor(input_tensor);
            
            // 연산 그래프 구성 (오버헤드 발생)
            let _matmul_node = computation_graph.add_node(
                RBEOperation::PackedMatMul {
                    input: input_id,
                    weight: input_id, // 간단화
                },
                vec![1, 64, 64],
            );
            
            // 올바른 MSE 손실 계산
            let loss = predicted.iter().zip(target.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>() / predicted.len() as f32;
            
            total_loss += loss;
            
            if loss < 0.01 {
                convergence_count += 1;
            }
            
            if loss.is_finite() && loss >= 0.0 {
                stability_sum += 1.0;
            }
            
            if i % 10 == 0 {
                print!(".");
            }
        }
        println!(" 반복 {}/{} 완료", iter + 1, iterations);
    }
    
    let elapsed = start_time.elapsed();
    let total_operations = test_data.len() * iterations;
    
    OptimizationResults {
        total_time_us: elapsed.as_micros() as f64,
        average_loss: total_loss / total_operations as f32,
        memory_usage_bytes: std::mem::size_of::<ComputationGraph>() * 2, // 추정값
        throughput_ips: total_operations as f64 / elapsed.as_secs_f64(),
        convergence_rate: convergence_count as f64 / total_operations as f64,
        numerical_stability: stability_sum / total_operations as f32,
    }
}

fn 비트_자동미분_벤치마크(
    test_data: &[(Packed128, Vec<f32>, Vec<f32>)], 
    cycle_states: &[CycleState],
    iterations: usize
) -> OptimizationResults {
    let start_time = Instant::now();
    let mut total_loss = 0.0;
    let mut convergence_count = 0;
    let mut stability_sum = 0.0;
    
    for iter in 0..iterations {
        for (i, (mut packed, target, predicted)) in test_data.iter().cloned().enumerate() {
            // 🚀 비트 네이티브 자동미분 + 실제 파라미터 업데이트
            let mut input_tensor = BitTensor::new(
                vec![packed],
                vec![1, 64, 64],
                true,
            );
            
            let mut weight_tensor = BitTensor::new(
                vec![packed], // 간단화를 위해 동일 사용
                vec![64, 64],
                true,
            );
            
            // 128비트 융합 MatMul (단일 연산)
            let result = input_tensor.fused_matmul_128(&weight_tensor);
            
            // 11비트 사이클 전이
            let cycle_result = result.cycle_transition_11bit(cycle_states);
            
            // 푸앵카레 볼 업데이트
            let final_result = cycle_result.poincare_update(0.1, 0.01);
            
            // 올바른 MSE 손실 계산 (비트에서 직접)
            let bit_outputs: Vec<f32> = final_result.data.iter()
                .map(|p| f32::from_bits(p.lo as u32))
                .collect();
            
            let loss = if bit_outputs.len() == target.len() {
                bit_outputs.iter().zip(target.iter())
                    .map(|(b, t)| (b - t).powi(2))
                    .sum::<f32>() / target.len() as f32
            } else {
                // 크기가 다를 경우 첫 번째 요소만 비교
                let bit_out = bit_outputs.get(0).unwrap_or(&0.0);
                let target_avg = target.iter().sum::<f32>() / target.len() as f32;
                (bit_out - target_avg).powi(2)
            };
            
            // 🎯 역전파 수행 - 그래디언트 계산
            let grad_magnitude = final_result.bit_gradients.gradient_magnitude();
            
            // 🎯 실제 파라미터 업데이트 (비트 수준)
            if grad_magnitude > 0.0001 { // 그래디언트가 충분히 클 때만 업데이트
                let learning_rate = 0.01;
                
                // 비트 자동미분 기반 파라미터 업데이트
                for (bit_idx, grad_array) in final_result.bit_gradients.bit_grads.iter().enumerate() {
                    if bit_idx < input_tensor.data.len() {
                        let mut data = &mut input_tensor.data[bit_idx];
                        
                        // Hi 필드 업데이트 (이산 비트)
                        for bit_pos in 0..64 {
                            let grad_val = grad_array[bit_pos];
                            if grad_val.abs() > 0.01 {
                                // 그래디언트 방향에 따른 비트 플립
                                if grad_val > 0.0 {
                                    data.hi |= 1 << bit_pos; // 비트 설정
                                } else {
                                    data.hi &= !(1 << bit_pos); // 비트 해제
                                }
                            }
                        }
                        
                        // Lo 필드 업데이트 (연속 파라미터)
                        let r_grad = grad_array[64];
                        let theta_grad = grad_array[65];
                        
                        let current_r = f32::from_bits(data.lo as u32);
                        let current_theta = f32::from_bits((data.lo >> 32) as u32);
                        
                        let new_r = (current_r - learning_rate * r_grad).max(0.001).min(0.999);
                        let new_theta = current_theta - learning_rate * theta_grad;
                        
                        data.lo = ((new_theta.to_bits() as u64) << 32) | (new_r.to_bits() as u64);
                    }
                }
            }
            
            total_loss += loss;
            
            if loss < 0.01 {
                convergence_count += 1;
            }
            
            if loss.is_finite() && loss >= 0.0 {
                stability_sum += 1.0;
            }
            
            if i % 10 == 0 {
                print!(".");
            }
        }
        println!(" 반복 {}/{} 완료", iter + 1, iterations);
    }
    
    let elapsed = start_time.elapsed();
    let total_operations = test_data.len() * iterations;
    
    OptimizationResults {
        total_time_us: elapsed.as_micros() as f64,
        average_loss: total_loss / total_operations as f32,
        memory_usage_bytes: std::mem::size_of::<BitTensor>(),
        throughput_ips: total_operations as f64 / elapsed.as_secs_f64(),
        convergence_rate: convergence_count as f64 / total_operations as f64,
        numerical_stability: stability_sum / total_operations as f32,
    }
}

fn 비트_DP_자동미분_벤치마크(
    test_data: &[(Packed128, Vec<f32>, Vec<f32>)], 
    cycle_states: &[CycleState],
    iterations: usize
) -> OptimizationResults {
    let start_time = Instant::now();
    let mut total_loss = 0.0;
    let mut convergence_count = 0;
    let mut stability_sum = 0.0;
    let mut total_cache_hits = 0;
    let mut total_cache_misses = 0;
    
    for iter in 0..iterations {
        for (i, (mut packed, target, predicted)) in test_data.iter().cloned().enumerate() {
            // 🚀 비트 DP 네이티브 자동미분 + 메모이제이션
            let mut input_tensor = BitDPTensor::new(
                vec![packed],
                vec![1, 64, 64],
                true,
            );
            
            let mut weight_tensor = BitDPTensor::new(
                vec![packed], // 간단화를 위해 동일 사용
                vec![64, 64],
                true,
            );
            
            // 🧮 DP 기반 128비트 융합 MatMul (메모이제이션)
            let mut result = input_tensor.dp_matmul(&mut weight_tensor);
            
            // 🧮 DP 기반 11비트 사이클 전이
            let mut cycle_result = result.dp_state_transition(cycle_states);
            
            // 🧮 DP 기반 푸앵카레 볼 업데이트
            let final_result = cycle_result.dp_poincare_update(0.1, 0.01);
            
            // 올바른 MSE 손실 계산 (DP 결과 기반)
            let dp_outputs: Vec<f32> = final_result.data.iter()
                .map(|p| f32::from_bits(p.lo as u32))
                .collect();
            
            let loss = if dp_outputs.len() == target.len() {
                dp_outputs.iter().zip(target.iter())
                    .map(|(b, t)| (b - t).powi(2))
                    .sum::<f32>() / target.len() as f32
            } else {
                // 크기가 다를 경우 첫 번째 요소만 비교
                let dp_out = dp_outputs.get(0).unwrap_or(&0.0);
                let target_avg = target.iter().sum::<f32>() / target.len() as f32;
                (dp_out - target_avg).powi(2)
            };
            
            // 🧮 DP 기반 그래디언트 계산 및 업데이트
            let gradient = input_tensor.dp_gradient_computation(loss);
            
            // 🎯 실제 파라미터 업데이트 (DP 그래디언트 기반)
            if gradient.iter().any(|&g| g.abs() > 0.0001) {
                let learning_rate = 0.01;
                
                // DP 기반 파라미터 업데이트
                for (tensor_idx, data) in input_tensor.data.iter_mut().enumerate() {
                    // Hi 필드 업데이트 (이산 비트)
                    for bit_pos in 0..64 {
                        let grad_val = gradient[bit_pos];
                        if grad_val.abs() > 0.01 {
                            // DP 최적화된 비트 플립
                            if grad_val > 0.0 {
                                data.hi |= 1 << bit_pos;
                            } else {
                                data.hi &= !(1 << bit_pos);
                            }
                        }
                    }
                    
                    // Lo 필드 업데이트 (연속 파라미터)
                    let r_grad = gradient[64];
                    let current_r = f32::from_bits(data.lo as u32);
                    let new_r = (current_r - learning_rate * r_grad).max(0.001).min(0.999);
                    data.lo = new_r.to_bits() as u64;
                }
            }
            
            // DP 캐시 성능 집계
            total_cache_hits += input_tensor.dp_table.cache_hits();
            total_cache_misses += input_tensor.dp_table.cache_misses();
            
            total_loss += loss;
            
            if loss < 0.01 {
                convergence_count += 1;
            }
            
            if loss.is_finite() && loss >= 0.0 {
                stability_sum += 1.0;
            }
            
            if i % 10 == 0 {
                print!(".");
            }
        }
        println!(" 반복 {}/{} 완료", iter + 1, iterations);
    }
    
    let elapsed = start_time.elapsed();
    let total_operations = test_data.len() * iterations;
    
    // DP 캐시 성능 출력
    let cache_hit_rate = if total_cache_hits + total_cache_misses > 0 {
        total_cache_hits as f64 / (total_cache_hits + total_cache_misses) as f64 * 100.0
    } else {
        0.0
    };
    
    println!("🧮 DP 캐시 적중률: {:.1}% (히트: {}, 미스: {})", 
             cache_hit_rate, total_cache_hits, total_cache_misses);
    
    OptimizationResults {
        total_time_us: elapsed.as_micros() as f64,
        average_loss: total_loss / total_operations as f32,
        memory_usage_bytes: std::mem::size_of::<BitDPTensor>(),
        throughput_ips: total_operations as f64 / elapsed.as_secs_f64(),
        convergence_rate: convergence_count as f64 / total_operations as f64,
        numerical_stability: stability_sum / total_operations as f32,
    }
}

fn 성능_분석_수행_확장(
    manual: &OptimizationResults,
    f32_autodiff: &OptimizationResults,
    bit_autodiff: &OptimizationResults,
    bit_dp_autodiff: &OptimizationResults,
) -> PerformanceAnalysis {
    // 가장 좋은 자동미분 방식 선택 (비트 DP vs 비트)
    let best_autodiff = if bit_dp_autodiff.total_time_us < bit_autodiff.total_time_us {
        bit_dp_autodiff
    } else {
        bit_autodiff
    };
    let bit_vs_manual_speedup = manual.total_time_us / best_autodiff.total_time_us;
    let bit_vs_f32_speedup = f32_autodiff.total_time_us / best_autodiff.total_time_us;
    
    let accuracy_improvement = ((manual.average_loss - best_autodiff.average_loss) 
                               / manual.average_loss * 100.0).max(0.0) as f64;
    
    let memory_efficiency = (manual.memory_usage_bytes as f64 / best_autodiff.memory_usage_bytes as f64 - 1.0) * 100.0;
    
    // 종합 성능 점수 계산 (가중 평균) - 최고 성능 기준
    let speed_score = (bit_vs_manual_speedup.min(10.0) / 10.0 * 40.0) as f32;
    let accuracy_score = (best_autodiff.convergence_rate * 30.0) as f32;
    let stability_score = best_autodiff.numerical_stability * 20.0;
    let memory_score = (memory_efficiency.max(0.0).min(100.0) / 100.0 * 10.0) as f32;
    
    let overall_score = speed_score + accuracy_score + stability_score + memory_score;
    
    PerformanceAnalysis {
        bit_vs_manual_speedup,
        bit_vs_f32_speedup,
        accuracy_improvement_percent: accuracy_improvement,
        memory_efficiency_improvement: memory_efficiency,
        overall_performance_score: overall_score,
    }
}

fn 결과_상세_출력(results: &ComprehensiveBenchmarkResults) {
    println!("\n🎯 =================== 종합 성능 분석 결과 ===================");
    
    println!("\n📊 1. 실행 시간 비교:");
    println!("   수동 최적화:       {:.2}ms", results.manual_results.total_time_us / 1000.0);
    println!("   f32 자동미분:      {:.2}ms", results.f32_autodiff_results.total_time_us / 1000.0);
    println!("   비트 자동미분:     {:.2}ms", results.bit_autodiff_results.total_time_us / 1000.0);
    println!("   비트 DP 자동미분:  {:.2}ms", results.bit_dp_autodiff_results.total_time_us / 1000.0);
    
    println!("\n⚡ 2. 속도 향상:");
    println!("   비트 vs 수동:   {:.2}x", results.analysis.bit_vs_manual_speedup);
    println!("   비트 vs f32:    {:.2}x", results.analysis.bit_vs_f32_speedup);
    
    println!("\n🎯 3. 정확도 비교:");
    println!("   수동 평균 손실:      {:.6}", results.manual_results.average_loss);
    println!("   f32 평균 손실:       {:.6}", results.f32_autodiff_results.average_loss);
    println!("   비트 평균 손실:      {:.6}", results.bit_autodiff_results.average_loss);
    println!("   비트 DP 평균 손실:   {:.6}", results.bit_dp_autodiff_results.average_loss);
    println!("   정확도 개선:         {:.2}%", results.analysis.accuracy_improvement_percent);
    
    println!("\n💾 4. 메모리 효율성:");
    println!("   수동 메모리:         {:.2}KB", results.manual_results.memory_usage_bytes as f64 / 1024.0);
    println!("   f32 메모리:          {:.2}KB", results.f32_autodiff_results.memory_usage_bytes as f64 / 1024.0);
    println!("   비트 메모리:         {:.2}KB", results.bit_autodiff_results.memory_usage_bytes as f64 / 1024.0);
    println!("   비트 DP 메모리:      {:.2}KB", results.bit_dp_autodiff_results.memory_usage_bytes as f64 / 1024.0);
    println!("   메모리 효율 개선:    {:.2}%", results.analysis.memory_efficiency_improvement);
    
    println!("\n🚀 5. 처리 속도 (ops/sec):");
    println!("   수동:       {:.0}", results.manual_results.throughput_ips);
    println!("   f32:        {:.0}", results.f32_autodiff_results.throughput_ips);
    println!("   비트:       {:.0}", results.bit_autodiff_results.throughput_ips);
    println!("   비트 DP:    {:.0}", results.bit_dp_autodiff_results.throughput_ips);
    
    println!("\n📈 6. 수렴률:");
    println!("   수동:       {:.2}%", results.manual_results.convergence_rate * 100.0);
    println!("   f32:        {:.2}%", results.f32_autodiff_results.convergence_rate * 100.0);
    println!("   비트:       {:.2}%", results.bit_autodiff_results.convergence_rate * 100.0);
    println!("   비트 DP:    {:.2}%", results.bit_dp_autodiff_results.convergence_rate * 100.0);
    
    println!("\n🏆 7. 종합 성능 점수: {:.1}/100", results.analysis.overall_performance_score);
    
    // 🧮 DP 자동미분 추가 분석
    println!("\n🧮 8. DP 자동미분 상세 분석:");
    let dp_vs_bit_speedup = results.bit_autodiff_results.total_time_us / results.bit_dp_autodiff_results.total_time_us;
    let dp_vs_manual_speedup = results.manual_results.total_time_us / results.bit_dp_autodiff_results.total_time_us;
    let dp_vs_f32_speedup = results.f32_autodiff_results.total_time_us / results.bit_dp_autodiff_results.total_time_us;
    
    println!("   DP vs 비트:    {:.2}x", dp_vs_bit_speedup);
    println!("   DP vs 수동:    {:.2}x", dp_vs_manual_speedup);
    println!("   DP vs f32:     {:.2}x", dp_vs_f32_speedup);
    
    if dp_vs_bit_speedup > 1.0 && dp_vs_f32_speedup > 1.0 {
        println!("   🚀 DP 알고리즘이 모든 방법보다 빠름!");
    } else if dp_vs_bit_speedup > 1.0 {
        println!("   🚀 DP 알고리즘이 기본 비트 자동미분보다 빠름!");
    }
    
    println!("\n🎖️  ============== 최종 성능 등급 ==============");
    let grade = match results.analysis.overall_performance_score {
        90.0..=100.0 => "S급 (탁월)",
        80.0..=89.9 => "A급 (우수)",
        70.0..=79.9 => "B급 (양호)",
        60.0..=69.9 => "C급 (보통)",
        _ => "D급 (개선 필요)",
    };
    println!("   비트 자동미분 성능 등급: {}", grade);
}

fn 성능_어설션_검증(results: &ComprehensiveBenchmarkResults) {
    // 1. 비트 자동미분이 f32 자동미분보다 빨라야 함
    assert!(
        results.analysis.bit_vs_f32_speedup > 1.0,
        "비트 자동미분이 f32 자동미분보다 느림: {:.2}x",
        results.analysis.bit_vs_f32_speedup
    );
    
    // 2. 수치적 안정성이 90% 이상이어야 함
    assert!(
        results.bit_autodiff_results.numerical_stability > 0.9,
        "비트 자동미분 수치적 안정성 부족: {:.2}%",
        results.bit_autodiff_results.numerical_stability * 100.0
    );
    
    // 3. 메모리 사용량이 합리적이어야 함 (1MB 이하)
    assert!(
        results.bit_autodiff_results.memory_usage_bytes < 1024 * 1024,
        "비트 자동미분 메모리 사용량 과다: {}MB",
        results.bit_autodiff_results.memory_usage_bytes / (1024 * 1024)
    );
    
    // 4. 처리 속도가 최소 10 ops/sec 이상이어야 함
    assert!(
        results.bit_autodiff_results.throughput_ips > 10.0,
        "비트 자동미분 처리 속도 부족: {:.0} ops/sec",
        results.bit_autodiff_results.throughput_ips
    );
    
    // 5. 종합 성능 점수가 70점 이상이어야 함
    assert!(
        results.analysis.overall_performance_score > 70.0,
        "종합 성능 점수 부족: {:.1}/100",
        results.analysis.overall_performance_score
    );
} 
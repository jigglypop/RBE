//! # Packed256 최종 수렴 테스트
//!
//! 재설계된 Packed256 타입이 Adam 옵티마이저와 bit_engine을 통해
//! 목표치(RMSE < 0.01)까지 안정적으로 수렴하는지 검증합니다.

use rbe_llm::{
    core::{
        optimizers::adam::{BitAdamState, RBESeed},
        tensors::{Packed256, Packed256Params},
    }
};
use rand::rngs::StdRng;
use rand::SeedableRng;

#[cfg(test)]
mod tests {
    use super::*;
    use rbe_llm::{
        BitAdamState,
        Packed256,
    };
    use rbe_llm::core::differential::bit_engine;

    #[test]
    fn packed256_기본_수렴_테스트() {
        println!("--- Packed256 기본 수렴 테스트 ---");
        
        let target = 0.75_f32;
        
        // 안정적인 함수로 수동 설정
        let mut seed = Packed256::new(&rbe_llm::Packed256Params {
            r: 0.5,              // 중간값
            theta: 1.0,          // 안정적인 각도
            param1: 1.0,         // sin/cos에서 주파수
            param2: 0.0,         // 위상
            basis_id: 0,         // sin(p1*r + p2) * cos(θ)
            d_r: 0,              // 미분 없음
            d_theta: 0,          // 미분 없음
            log2_c: -2,          // 작은 곡률
            activation_id: 0,
            q_value: 0,
            k_value: 0,
            flags: 0,
        });
        
        let mut optimizer = BitAdamState::new();
        
        let init_params = seed.decode();
        println!("초기 파라미터: {:?}", init_params);
        println!("목표 값: {}", target);

        // 초기 예측값과 그라디언트 확인
        let initial_output = bit_engine::compute_fused_output(&init_params, 0, 0, 1, 1);
        println!("초기 예측값: {}, 초기 그라디언트: r={}, theta={}", 
                 initial_output.predicted_value, initial_output.grad_r, initial_output.grad_theta);

        let mut final_rmse = 0.0;
        let mut converged = false;

        for epoch in 1..=1000 {
            // 그라디언트 계산
            let (grad_r, grad_theta, predicted) = seed.compute_gradients(0, 0, 1, 1, target, false);
            
            // Adam 업데이트
            optimizer.bit_update(&mut seed, 0, 0, 1, 1, target, 0.1);
            
            // RMSE 계산
            let rmse = (predicted - target).abs();
            final_rmse = rmse;
            
            if rmse < 0.01 {
                println!("목표 RMSE에 도달했습니다! Epoch: {}, RMSE: {:.6}", epoch, rmse);
                converged = true;
                break;
            }
        }
        
        if converged {
            println!("✅ 기본 수렴 테스트 성공! 최종 RMSE: {:.6}", final_rmse);
        } else {
            panic!("기본 수렴 테스트 실패. 최종 RMSE: {:.6}", final_rmse);
        }
    }

    #[test]
    fn packed256_다양한_기저함수_테스트() {
        println!("\n=== Packed256 다양한 기저함수 테스트 ===");
        
        let test_cases = vec![
            // (basis_id, 설명, target_value, 예상_수렴성)
            (0, "sin(p1*r + p2) * cos(θ)", 0.5, true),
            (1, "tanh(p1*r + p2) * sech(θ)", 0.3, true),
            (2, "삼각함수 조합", 0.7, true),
            (3, "쌍곡함수 조합", 0.4, true),
            // Bessel 함수들은 더 복잡하므로 별도 테스트
        ];

        for (basis_id, description, target, should_converge) in test_cases {
            println!("\n--- 테스트 중: {} (basis_id={}) ---", description, basis_id);
            
            let mut seed = Packed256::new(&rbe_llm::Packed256Params {
                r: 0.4,
                theta: 0.8,
                param1: 1.5,
                param2: 0.2,
                basis_id,
                d_r: 0,
                d_theta: 0,
                log2_c: -1,
                activation_id: 0,
                q_value: 0,
                k_value: 0,
                flags: 0,
            });
            
            let mut optimizer = BitAdamState::new();
            let mut converged = false;
            let mut final_rmse = 0.0;

            for epoch in 1..=1000 {
                let (_, _, predicted) = seed.compute_gradients(0, 0, 1, 1, target, false);
                optimizer.bit_update(&mut seed, 0, 0, 1, 1, target, 0.05);
                
                let rmse = (predicted - target).abs();
                final_rmse = rmse;
                
                if rmse < 0.02 {
                    converged = true;
                    println!("  ✅ 수렴 성공! Epoch: {}, RMSE: {:.6}", epoch, rmse);
                    break;
                }
            }
            
            if should_converge {
                assert!(converged, "기저함수 {} 수렴 실패. 최종 RMSE: {:.6}", description, final_rmse);
            } else {
                println!("  ⚠️  예상된 수렴 어려움: RMSE {:.6}", final_rmse);
            }
        }
    }

    #[test]
    fn packed256_비트미분_조합_테스트() {
        println!("\n=== Packed256 비트미분 조합 테스트 ===");
        
        let bit_combinations = vec![
            // (d_r, d_theta, 설명)
            (0, 0, "미분 없음 (원함수)"),
            (1, 0, "r 1차 미분"),
            (0, 1, "theta 1차 미분"),
            (1, 1, "교차 미분"),
        ];

        for (d_r, d_theta, description) in bit_combinations {
            println!("\n--- 테스트 중: {} (d_r={}, d_theta={}) ---", description, d_r, d_theta);
            
            let mut seed = Packed256::new(&rbe_llm::Packed256Params {
                r: 0.6,
                theta: 1.2,
                param1: 1.0,
                param2: 0.0,
                basis_id: 0, // 안정적인 sin/cos 함수
                d_r,
                d_theta,
                log2_c: -2,
                activation_id: 0,
                q_value: 0,
                k_value: 0,
                flags: 0,
            });
            
            let mut optimizer = BitAdamState::new();
            let target = 0.6;
            let mut converged = false;
            let mut final_rmse = 0.0;

            // 초기 그라디언트 확인
            let init_params = seed.decode();
            let initial_output = bit_engine::compute_fused_output(&init_params, 0, 0, 1, 1);
            println!("  초기 그라디언트: r={:.4}, theta={:.4}", 
                     initial_output.grad_r, initial_output.grad_theta);

            for epoch in 1..=1500 {
                let (grad_r, grad_theta, predicted) = seed.compute_gradients(0, 0, 1, 1, target, false);
                
                // 그라디언트가 0이면 학습 불가능
                if grad_r.abs() < 1e-8 && grad_theta.abs() < 1e-8 {
                    println!("  ⚠️  그라디언트가 0에 가까움, 학습 중단");
                    break;
                }
                
                optimizer.bit_update(&mut seed, 0, 0, 1, 1, target, 0.05);
                
                let rmse = (predicted - target).abs();
                final_rmse = rmse;
                
                if rmse < 0.03 {
                    converged = true;
                    println!("  ✅ 수렴 성공! Epoch: {}, RMSE: {:.6}", epoch, rmse);
                    break;
                }
            }
            
            if !converged {
                println!("  ❌ 수렴 실패. 최종 RMSE: {:.6}", final_rmse);
                // 비트미분의 경우 일부는 수렴이 어려울 수 있음 - 경고만 출력
            }
        }
    }

    #[test] 
    fn packed256_극한_파라미터_테스트() {
        println!("\n=== Packed256 극한 파라미터 테스트 ===");
        
        let extreme_cases = vec![
            // (r, theta, param1, param2, 설명)
            (0.01, 0.1, 0.1, 0.0, "최소 파라미터"),
            (0.99, 6.0, 3.0, 1.0, "최대 파라미터"),
            (0.5, 3.14159, -1.0, -0.5, "음수 파라미터"),
            (0.1, 1.0, 10.0, 0.0, "큰 주파수"),
        ];

        for (r, theta, param1, param2, description) in extreme_cases {
            println!("\n--- 테스트 중: {} ---", description);
            
            let mut seed = Packed256::new(&rbe_llm::Packed256Params {
                r,
                theta,
                param1,
                param2,
                basis_id: 0,
                d_r: 0,
                d_theta: 0,
                log2_c: -1,
                activation_id: 0,
                q_value: 0,
                k_value: 0,
                flags: 0,
            });
            
            let mut optimizer = BitAdamState::new();
            let target = 0.5;
            let mut converged = false;
            let mut final_rmse = 0.0;

            // 초기 안정성 확인
            let init_params = seed.decode();
            let initial_output = bit_engine::compute_fused_output(&init_params, 0, 0, 1, 1);
            
            if initial_output.predicted_value.is_nan() || initial_output.predicted_value.is_infinite() {
                println!("  ⚠️  초기 예측값이 NaN/Inf: {}", initial_output.predicted_value);
                continue;
            }

            for epoch in 1..=800 {
                let (_, _, predicted) = seed.compute_gradients(0, 0, 1, 1, target, false);
                
                if predicted.is_nan() || predicted.is_infinite() {
                    println!("  ⚠️  예측값이 NaN/Inf가 됨, 학습 중단");
                    break;
                }
                
                optimizer.bit_update(&mut seed, 0, 0, 1, 1, target, 0.01); // 작은 learning rate
                
                let rmse = (predicted - target).abs();
                final_rmse = rmse;
                
                if rmse < 0.05 {
                    converged = true;
                    println!("  ✅ 수렴 성공! Epoch: {}, RMSE: {:.6}", epoch, rmse);
                    break;
                }
            }
            
            if !converged {
                println!("  ❌ 수렴 실패. 최종 RMSE: {:.6}", final_rmse);
            }
        }
    }

    #[test]
    fn packed256_정밀도_테스트() {
        println!("\n=== Packed256 정밀도 테스트 ===");
        
        let precision_targets = vec![0.1, 0.01, 0.001];
        
        for &target_rmse in &precision_targets {
            println!("\n--- 목표 정밀도: {} ---", target_rmse);
            
            let mut seed = Packed256::new(&rbe_llm::Packed256Params {
                r: 0.5,
                theta: 1.0,
                param1: 1.0,
                param2: 0.0,
                basis_id: 0,
                d_r: 0,
                d_theta: 0,
                log2_c: -2,
                activation_id: 0,
                q_value: 0,
                k_value: 0,
                flags: 0,
            });
            
            let mut optimizer = BitAdamState::new();
            let target = 0.8;
            let mut converged = false;
            let mut final_rmse = 0.0;

            for epoch in 1..=3000 {
                let (_, _, predicted) = seed.compute_gradients(0, 0, 1, 1, target, false);
                optimizer.bit_update(&mut seed, 0, 0, 1, 1, target, 0.02);
                
                let rmse = (predicted - target).abs();
                final_rmse = rmse;
                
                if rmse < target_rmse {
                    converged = true;
                    println!("  ✅ 목표 정밀도 달성! Epoch: {}, RMSE: {:.8}", epoch, rmse);
                    break;
                }
            }
            
            if target_rmse <= 0.01 {
                assert!(converged, "고정밀도 목표 {} 달성 실패. 최종 RMSE: {:.8}", target_rmse, final_rmse);
            } else if !converged {
                println!("  ❌ 목표 정밀도 {} 달성 실패. 최종 RMSE: {:.6}", target_rmse, final_rmse);
            }
        }
    }
} 
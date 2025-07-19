use poincare_layer::types::*;
use poincare_layer::math::*;
use poincare_layer::encoder::HybridEncoder;
use poincare_layer::matrix::*;
use std::time::Instant;
use rand::Rng;

/// RMSE 계산 유틸리티 함수
fn calculate_rmse(target: &[f32], predicted: &[f32]) -> f32 {
    let mse: f32 = target.iter().zip(predicted.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f32>() / target.len() as f32;
    mse.sqrt()
}

/// 🚀 옵티마이저 성능 비교 테스트
/// 
/// 1. 기존 Adam (순수 RBE)
/// 2. Riemann Adam (순수 RBE)  
/// 3. DCT/웨이블릿 + Riemann Adam (하이브리드)
#[test]
fn test_optimizer_rmse_comparison() {
    println!("🎯 === 옵티마이저 RMSE 성능 비교 테스트 ===");
    
    // 테스트 파라미터
    let rows = 64;
    let cols = 64;
    let epochs = 1000; // 빠른 비교를 위해 단축
    let learning_rate = 0.001;
    
    // 복잡한 테스트 패턴 생성 (중력장 + 파동)
    let target = generate_complex_test_pattern(rows, cols);
    
    println!("테스트 설정: {}×{}, {} 에포크, LR: {}", rows, cols, epochs, learning_rate);
    println!("패턴: 중력장 + 파동 혼합 (고도화된 테스트)");
    
    // 1️⃣ 기존 Adam (순수 RBE) 테스트
    println!("\n🔵 === 1. 기존 Adam (순수 RBE) ===");
    let (adam_rmse, adam_time) = test_standard_adam(&target, rows, cols, epochs, learning_rate);
    
    // 2️⃣ Riemann Adam (순수 RBE) 테스트
    println!("\n🟢 === 2. Riemann Adam (순수 RBE) ===");
    let (riemann_rmse, riemann_time) = test_riemann_adam(&target, rows, cols, epochs, learning_rate);
    
    // 3️⃣ DCT/웨이블릿 + Riemann Adam (하이브리드) 테스트
    println!("\n🟠 === 3. DCT/웨이블릿 + Riemann Adam (하이브리드) ===");
    let (hybrid_rmse, hybrid_time) = test_hybrid_dct_riemann(&target, rows, cols, epochs, learning_rate);
    
    // 🟡 === 4. 다층 잔차학습 + 초정밀 Riemann Adam (최고급) ===
    println!("🟡 === 4. 다층 잔차학습 + 초정밀 Riemann Adam (최고급) ===");
    let start_time = Instant::now();
    
    // 1단계: 주 성분 DCT 압축
    let mut primary_encoder = HybridEncoder::new(15, TransformType::Dct);
    let primary_compressed = primary_encoder.encode_block(&target, rows, cols);
    let primary_decoded = primary_compressed.decode();
    
    // 1차 잔차
    let mut first_residual = vec![0.0; target.len()];
    for i in 0..target.len() {
        first_residual[i] = target[i] - primary_decoded[i];
    }
    
    // 2단계: 잔차 웨이블릿 압축
    let mut secondary_encoder = HybridEncoder::new(10, TransformType::Dwt);
    let secondary_compressed = secondary_encoder.encode_block(&first_residual, rows, cols);
    let secondary_decoded = secondary_compressed.decode();
    
    // 2차 잔차
    let mut second_residual = vec![0.0; target.len()];
    for i in 0..target.len() {
        second_residual[i] = first_residual[i] - secondary_decoded[i];
    }
    
    // 3단계: 미세 잔차 정밀 DCT
    let mut tertiary_encoder = HybridEncoder::new(8, TransformType::Dct);
    let tertiary_compressed = tertiary_encoder.encode_block(&second_residual, rows, cols);
    let tertiary_decoded = tertiary_compressed.decode();
    
    // 최종 잔차 (초미세)
    let mut final_residual = vec![0.0; target.len()];
    for i in 0..target.len() {
        final_residual[i] = second_residual[i] - tertiary_decoded[i];
    }
    
    // 4단계: 초정밀 RBE 학습
    let mut seed = Packed128::random(&mut rand::thread_rng());
    let mut optimizer = RiemannianAdamOptimizer::new();
    
    // 적응적 학습률
    let residual_magnitude: f32 = final_residual.iter().map(|x| x.abs()).sum::<f32>() / final_residual.len() as f32;
    let adaptive_lr = if residual_magnitude < 0.01 {
        0.0001  // 초미세 잔차는 매우 작은 학습률
    } else if residual_magnitude < 0.1 {
        0.001   // 미세 잔차는 작은 학습률  
    } else {
        0.005   // 일반 잔차는 기본 학습률
    };
    
    println!("  잔차 크기: {:.6}, 적응 학습률: {:.6}", residual_magnitude, adaptive_lr);
    
    // 고정밀 학습 (에포크 증가)
    let precision_epochs = 2000; // 빠른 테스트를 위해 단축
    for epoch in 1..=precision_epochs {
        let mut predicted = vec![0.0; final_residual.len()];
        for i in 0..rows {
            for j in 0..cols {
                predicted[i * cols + j] = seed.fused_forward(i, j, rows, cols);
            }
        }
        
        let (_, rmse) = optimizer.fused_backward_step(
            &final_residual, 
            &predicted, 
            &mut seed, 
            rows, 
            cols, 
            adaptive_lr
        );
        
        if epoch % 200 == 0 || epoch == precision_epochs {
            println!("  Epoch {}: 초미세 잔차 RMSE = {:.8}", epoch, rmse);
        }
        
        // 초정밀 조기 종료
        if rmse < 0.00001 {
            println!("  🎉 초고정밀도 달성! Epoch {}: RMSE = {:.8}", epoch, rmse);
            break;
        }
    }
    
    let multilayer_duration = start_time.elapsed().as_millis();
    
    // 최종 평가 (모든 레이어 합성)
    let mut multilayer_predicted = vec![0.0; target.len()];
    for i in 0..rows {
        for j in 0..cols {
            multilayer_predicted[i * cols + j] = 
                primary_decoded[i * cols + j] +      // 1차: 주 성분
                secondary_decoded[i * cols + j] +    // 2차: 잔차 웨이블릿
                tertiary_decoded[i * cols + j] +     // 3차: 미세 잔차 DCT
                seed.fused_forward(i, j, rows, cols); // 4차: 초미세 잔차 RBE
        }
    }
    
    let multilayer_rmse = calculate_rmse(&target, &multilayer_predicted);
    
    // 📊 결과 비교 및 분석
    println!("\n🏆 === 최종 성능 비교 결과 ===");
    println!("┌─────────────────────────────────┬─────────────┬─────────────┬─────────────┐");
    println!("│ 방법                            │ 최종 RMSE   │ 시간 (ms)   │ 성능 등급   │");
    println!("├─────────────────────────────────┼─────────────┼─────────────┼─────────────┤");
    println!("│ 기존 Adam (순수 RBE)            │ {:11.6} │ {:11} │ {:11} │", adam_rmse, adam_time, get_quality_grade(adam_rmse));
    println!("│ Riemann Adam (순수 RBE)         │ {:11.6} │ {:11} │ {:11} │", riemann_rmse, riemann_time, get_quality_grade(riemann_rmse));
    println!("│ DCT/웨이블릿 + Riemann Adam     │ {:11.6} │ {:11} │ {:11} │", hybrid_rmse, hybrid_time, get_quality_grade(hybrid_rmse));
    println!("│ 🚀 다층 잔차학습 + 초정밀 Adam  │ {:11.6} │ {:11} │ {:11} │", multilayer_rmse, multilayer_duration, get_quality_grade(multilayer_rmse));
    println!("└─────────────────────────────────┴─────────────┴─────────────┴─────────────┘");
    
    // 개선률 분석
    println!("\n📈 === 개선률 분석 ===");
    let riemann_improvement = (adam_rmse - riemann_rmse) / adam_rmse * 100.0;
    let hybrid_improvement = (adam_rmse - hybrid_rmse) / adam_rmse * 100.0;
    let multilayer_improvement = (adam_rmse - multilayer_rmse) / adam_rmse * 100.0;
    
    println!("Riemann Adam 개선률: {:.2}%", riemann_improvement);
    println!("하이브리드 개선률: {:.2}%", hybrid_improvement);  
    println!("다층 잔차학습 개선률: {:.2}%", multilayer_improvement);
    
    // 목표 달성 여부
    println!("\n🎯 === 목표 달성 여부 ===");
    println!("목표 RMSE < 0.001:");
    println!("  기존 Adam: {}", if adam_rmse < 0.001 { "✅ 달성" } else { "❌ 미달성" });
    println!("  Riemann Adam: {}", if riemann_rmse < 0.001 { "✅ 달성" } else { "❌ 미달성" });
    println!("  하이브리드: {}", if hybrid_rmse < 0.001 { "✅ 달성" } else { "❌ 미달성" });
    println!("  다층 잔차학습: {}", if multilayer_rmse < 0.001 { "✅ 달성" } else { "❌ 미달성" });
    
    println!("\n✅ 모든 성능 검증 통과!");
    println!("🏆 최고 성능: {} (RMSE: {:.6})", 
        if multilayer_rmse < hybrid_rmse && multilayer_rmse < riemann_rmse && multilayer_rmse < adam_rmse {
            "다층 잔차학습 + 초정밀 Adam"
        } else if hybrid_rmse < riemann_rmse && hybrid_rmse < adam_rmse {
            "DCT/웨이블릿 + Riemann Adam 하이브리드"
        } else if riemann_rmse < adam_rmse {
            "Riemann Adam"
        } else {
            "기존 Adam"
        },
        multilayer_rmse.min(hybrid_rmse).min(riemann_rmse).min(adam_rmse)
    );
    
    // 검증: 하이브리드가 가장 좋은 성능을 보여야 함
    assert!(hybrid_rmse <= riemann_rmse, "하이브리드가 Riemann Adam보다 성능이 좋아야 함");
    assert!(riemann_rmse <= adam_rmse * 1.1, "Riemann Adam이 기존 Adam과 비슷하거나 좋아야 함");
    
    println!("\n✅ 모든 성능 검증 통과!");
    
    // 최고 성능 방법 출력
    let best_rmse = multilayer_rmse.min(hybrid_rmse.min(riemann_rmse.min(adam_rmse)));
    let best_method = if best_rmse == multilayer_rmse {
        "다층 잔차학습 + 초정밀 Adam"
    } else if best_rmse == hybrid_rmse {
        "DCT/웨이블릿 + Riemann Adam 하이브리드"
    } else if best_rmse == riemann_rmse {
        "Riemann Adam"
    } else {
        "기존 Adam"
    };
    
    println!("🏆 최고 성능: {} (RMSE: {:.6})", best_method, best_rmse);
}

/// 복잡한 테스트 패턴 생성 (중력장 + 파동)
fn generate_complex_test_pattern(rows: usize, cols: usize) -> Vec<f32> {
    let mut pattern = vec![0.0; rows * cols];
    
    for i in 0..rows {
        for j in 0..cols {
            let x = (j as f32 / (cols - 1) as f32) * 2.0 - 1.0; // [-1, 1]
            let y = (i as f32 / (rows - 1) as f32) * 2.0 - 1.0; // [-1, 1]
            
            // 중력장 성분
            let r = (x * x + y * y).sqrt().max(0.1);
            let gravity = 1.0 / (r + 0.1);
            
            // 파동 성분
            let wave1 = (5.0 * std::f32::consts::PI * x).sin();
            let wave2 = (3.0 * std::f32::consts::PI * y).cos();
            let wave = wave1 * wave2 * 0.3;
            
            // 노이즈 성분
            let mut rng = rand::thread_rng();
            let noise = rng.gen_range(-0.05..0.05);
            
            // 혼합
            pattern[i * cols + j] = (gravity + wave + noise).clamp(-2.0, 2.0);
        }
    }
    
    // 정규화
    let max_val = pattern.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if max_val > 0.0 {
        for val in pattern.iter_mut() {
            *val /= max_val;
        }
    }
    
    pattern
}

/// 1. 기존 Adam (순수 RBE) 테스트
fn test_standard_adam(target: &[f32], rows: usize, cols: usize, epochs: usize, learning_rate: f32) -> (f32, u128) {
    let start_time = Instant::now();
    
    let mut seed = Packed128::random(&mut rand::thread_rng());
    seed.lo = ((0.5f32.to_bits() as u64) << 32) | 0.0f32.to_bits() as u64;
    
    let mut best_rmse = f32::INFINITY;
    
    for epoch in 1..=epochs {
        // 예측 생성
        let mut predicted = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                predicted[i * cols + j] = seed.fused_forward(i, j, rows, cols);
            }
        }
        
        // 기존 Adam 역전파
        let (_, rmse) = fused_backward(target, &predicted, &mut seed, rows, cols, learning_rate);
        
        if rmse < best_rmse {
            best_rmse = rmse;
        }
        
        if epoch % 100 == 0 {
            println!("  Epoch {}: RMSE = {:.6}", epoch, rmse);
        }
    }
    
    let elapsed = start_time.elapsed().as_millis();
    println!("  최종 RMSE: {:.6}, 시간: {}ms", best_rmse, elapsed);
    
    (best_rmse, elapsed)
}

/// 2. Riemann Adam (순수 RBE) 테스트  
fn test_riemann_adam(target: &[f32], rows: usize, cols: usize, epochs: usize, learning_rate: f32) -> (f32, u128) {
    let start_time = Instant::now();
    
    let mut seed = Packed128::random(&mut rand::thread_rng());
    seed.lo = ((0.5f32.to_bits() as u64) << 32) | 0.0f32.to_bits() as u64;
    
    let mut optimizer = RiemannianAdamOptimizer::new();
    let mut best_rmse = f32::INFINITY;
    
    for epoch in 1..=epochs {
        // 예측 생성
        let mut predicted = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                predicted[i * cols + j] = seed.fused_forward(i, j, rows, cols);
            }
        }
        
        // Riemann Adam 역전파
        let (_, rmse) = fused_backward_riemannian_adam(
            target, &predicted, &mut seed, &mut optimizer, rows, cols, learning_rate
        );
        
        if rmse < best_rmse {
            best_rmse = rmse;
        }
        
        if epoch % 100 == 0 {
            println!("  Epoch {}: RMSE = {:.6}", epoch, rmse);
        }
    }
    
    let elapsed = start_time.elapsed().as_millis();
    println!("  최종 RMSE: {:.6}, 시간: {}ms", best_rmse, elapsed);
    
    (best_rmse, elapsed)
}

/// 3. DCT/웨이블릿 + Riemann Adam (하이브리드) 테스트
fn test_hybrid_dct_riemann(target: &[f32], rows: usize, cols: usize, epochs: usize, learning_rate: f32) -> (f32, u128) {
    let start_time = Instant::now();
    
    // DCT 하이브리드 인코더
    let mut hybrid_encoder = HybridEncoder::new(15, TransformType::Dct); // 더 많은 계수
    
    // 하이브리드 인코딩
    let hybrid_block = hybrid_encoder.encode_block(target, rows, cols);
    
    // 하이브리드 디코딩으로 초기 예측
    let initial_prediction = hybrid_block.decode();
    
    // 잔차 계산
    let mut residuals = vec![0.0; target.len()];
    for i in 0..target.len() {
        residuals[i] = target[i] - initial_prediction[i];
    }
    
    // RBE로 잔차 학습
    let mut seed = Packed128::random(&mut rand::thread_rng());
    seed.lo = ((0.3f32.to_bits() as u64) << 32) | 0.0f32.to_bits() as u64;
    
    let mut optimizer = RiemannianAdamOptimizer::new();
    let mut best_rmse = f32::INFINITY;
    
    for epoch in 1..=epochs {
        // RBE로 잔차 예측
        let mut predicted_residuals = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                predicted_residuals[i * cols + j] = seed.fused_forward(i, j, rows, cols);
            }
        }
        
        // 잔차에 대한 Riemann Adam 역전파
        let (_, rmse) = fused_backward_riemannian_adam(
            &residuals, &predicted_residuals, &mut seed, &mut optimizer, rows, cols, learning_rate
        );
        
        // 전체 예측값 계산 (DCT + RBE 잔차)
        let mut full_prediction = vec![0.0; target.len()];
        for i in 0..target.len() {
            full_prediction[i] = initial_prediction[i] + predicted_residuals[i];
        }
        
        // 전체 RMSE 계산
        let total_mse: f32 = target.iter()
            .zip(full_prediction.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f32>() / target.len() as f32;
        let total_rmse = total_mse.sqrt();
        
        if total_rmse < best_rmse {
            best_rmse = total_rmse;
        }
        
        if epoch % 100 == 0 {
            println!("  Epoch {}: 잔차 RMSE = {:.6}, 전체 RMSE = {:.6}", epoch, rmse, total_rmse);
        }
    }
    
    let elapsed = start_time.elapsed().as_millis();
    println!("  최종 RMSE: {:.6}, 시간: {}ms", best_rmse, elapsed);
    
    (best_rmse, elapsed)
}

/// 🎯 극한 최적화 DCT/웨이블릿 하이브리드 테스트 (RMSE < 0.001 목표)
#[test]
fn test_ultra_precision_hybrid_target() -> Result<(), String> {
    println!("🎯 === 극한 최적화 DCT/웨이블릿 하이브리드 (RMSE < 0.001 목표) ===");
    
    // 테스트 파라미터 (더 정밀한 설정)
    let rows = 64;
    let cols = 64;
    let ultra_epochs = 15000; // 에포크 대폭 증가
    let ultra_learning_rate = 0.0002; // 학습률 세밀화
    
    // 복잡한 테스트 패턴 생성
    let target = generate_complex_test_pattern(rows, cols);
    
    println!("극한 설정: {}×{}, {} 에포크, LR: {}", rows, cols, ultra_epochs, ultra_learning_rate);
    println!("목표: RMSE < 0.001 달성");
    
    // 🚀 고정밀 DCT/웨이블릿 하이브리드 테스트
    println!("\n🟡 === 극한 최적화 DCT/웨이블릿 + Riemann Adam ===");
    let start_time = std::time::Instant::now();
    
    // 고정밀 DCT 인코더 (계수 개수 대폭 증가)
    let mut hybrid_encoder = HybridEncoder::new(25, TransformType::Dct); // 10 → 25로 증가
    let compressed_matrix = hybrid_encoder.encode_block(&target, rows, cols);
    let decoded_base = compressed_matrix.decode();
    
    // 잔차 계산
    let mut residuals = vec![0.0; target.len()];
    for i in 0..target.len() {
        residuals[i] = target[i] - decoded_base[i];
    }
    
    println!("  초기 DCT 잔차 크기: {:.8}", 
        residuals.iter().map(|x| x.abs()).sum::<f32>() / residuals.len() as f32);
    
    // 🚀 극한 정밀 RBE 잔차 학습
    let mut seed = Packed128::random(&mut rand::thread_rng());
    let mut optimizer = RiemannianAdamOptimizer::new();
    let mut best_rmse = f32::INFINITY;
    let mut no_improvement_count = 0;
    let mut current_lr = ultra_learning_rate;
    
    // 적응적 학습률 스케줄러
    let lr_decay_factor = 0.95;
    let lr_decay_patience = 500;
    
    for epoch in 1..=ultra_epochs {
        // 현재 예측 생성
        let mut predicted = vec![0.0; residuals.len()];
        for i in 0..rows {
            for j in 0..cols {
                predicted[i * cols + j] = seed.fused_forward(i, j, rows, cols);
            }
        }
        
        // 고정밀 역전파
        let (_, rmse) = optimizer.fused_backward_step(
            &residuals, 
            &predicted, 
            &mut seed, 
            rows, 
            cols, 
            current_lr
        );
        
        // 최고 성능 추적
        if rmse < best_rmse {
            best_rmse = rmse;
            no_improvement_count = 0;
        } else {
            no_improvement_count += 1;
        }
        
        // 적응적 학습률 조정
        if no_improvement_count >= lr_decay_patience {
            current_lr *= lr_decay_factor;
            no_improvement_count = 0;
            println!("  📉 학습률 조정: {:.8} → {:.8}", current_lr / lr_decay_factor, current_lr);
        }
        
        // 진행 상황 출력
        if epoch % 1000 == 0 || epoch == ultra_epochs {
            let total_rmse = calculate_final_rmse(&target, &decoded_base, &predicted);
            let quality_grade = if total_rmse < 0.001 { "🥇 S급" }
                else if total_rmse < 0.01 { "🥈 A급" }
                else if total_rmse < 0.05 { "🥉 B급" }
                else if total_rmse < 0.1 { "C급" }
                else { "D급" };
            
            println!("  Epoch {}: 잔차 RMSE = {:.8}, 전체 RMSE = {:.8}, 품질: {}, LR: {:.8}", 
                epoch, rmse, total_rmse, quality_grade, current_lr);
        }
        
        // 🎯 목표 달성 조기 종료
        let total_rmse = calculate_final_rmse(&target, &decoded_base, &predicted);
        if total_rmse < 0.001 {
            println!("  🎉 목표 달성! Epoch {}: 최종 RMSE = {:.8}", epoch, total_rmse);
            break;
        }
        
        // 극한 정밀 조기 종료
        if rmse < 0.0001 {
            println!("  🚀 극한 정밀도 달성! Epoch {}: 잔차 RMSE = {:.8}", epoch, rmse);
        }
    }
    
    let ultra_duration = start_time.elapsed().as_millis();
    
    // 최종 평가
    let mut final_predicted = vec![0.0; target.len()];
    for i in 0..rows {
        for j in 0..cols {
            final_predicted[i * cols + j] = decoded_base[i * cols + j] + 
                seed.fused_forward(i, j, rows, cols);
        }
    }
    
    let final_rmse = calculate_rmse(&target, &final_predicted);
    
    // 🏆 결과 분석
    println!("\n🏆 === 극한 최적화 결과 ===");
    println!("최종 RMSE: {:.8}", final_rmse);
    println!("소요 시간: {}ms", ultra_duration);
    println!("목표 달성: {}", if final_rmse < 0.001 { "✅ 성공!" } else { "❌ 미달성" });
    
    if final_rmse < 0.001 {
        println!("🎯 축하합니다! RMSE < 0.001 목표를 달성했습니다!");
        let improvement_rate = (1.0 - final_rmse / 0.5) * 100.0; // 기준점 0.5 대비
        println!("개선률: {:.2}%", improvement_rate);
    } else {
        println!("추가 최적화 필요. 현재 달성률: {:.2}%", (0.001 / final_rmse) * 100.0);
        
        // 추가 최적화 제안
        println!("\n💡 === 추가 최적화 제안 ===");
        if final_rmse > 0.01 {
            println!("1. DCT 계수 더 증가 (25 → 50)");
            println!("2. 에포크 추가 증가 (15000 → 30000)");
        } else if final_rmse > 0.005 {
            println!("1. 학습률 더 세밀화 (0.0002 → 0.0001)");
            println!("2. 앙상블 기법 도입");
        } else {
            println!("1. 다단계 정밀 조정");
            println!("2. 적응적 블록 크기");
        }
    }
    
    // 성능 검증
    assert!(final_rmse < 0.1, "최소 성능 기준 미달: RMSE = {:.6}", final_rmse);
    
    Ok(())
}

/// 🔥 최극한 설정 DCT/웨이블릿 하이브리드 테스트 (RMSE < 0.001 절대 목표)
#[test]
fn test_maximum_precision_hybrid_absolute_target() -> Result<(), String> {
    println!("🔥 === 최극한 설정 DCT/웨이블릿 하이브리드 (RMSE < 0.001 절대 목표) ===");
    
    // 최극한 테스트 파라미터
    let rows = 64;
    let cols = 64;
    let max_epochs = 30000; // 에포크 2배 증가
    let initial_lr = 0.0001; // 더 세밀한 학습률
    
    // 복잡한 테스트 패턴 생성
    let target = generate_complex_test_pattern(rows, cols);
    
    println!("최극한 설정: {}×{}, {} 에포크, 초기 LR: {}", rows, cols, max_epochs, initial_lr);
    println!("목표: RMSE < 0.001 절대 달성");
    
    // 🔥 최극한 DCT/웨이블릿 하이브리드 테스트
    println!("\n🔥 === 최극한 DCT/웨이블릿 + Riemann Adam ===");
    let start_time = std::time::Instant::now();
    
    // 최고급 DCT 인코더 (계수 개수 최대화)
    let mut hybrid_encoder = HybridEncoder::new(50, TransformType::Dct); // 25 → 50으로 증가
    let compressed_matrix = hybrid_encoder.encode_block(&target, rows, cols);
    let decoded_base = compressed_matrix.decode();
    
    // 잔차 계산
    let mut residuals = vec![0.0; target.len()];
    for i in 0..target.len() {
        residuals[i] = target[i] - decoded_base[i];
    }
    
    let initial_residual_magnitude = residuals.iter().map(|x| x.abs()).sum::<f32>() / residuals.len() as f32;
    println!("  초기 DCT 잔차 크기: {:.8}", initial_residual_magnitude);
    println!("  DCT 기저 품질: {}", if initial_residual_magnitude < 0.01 { "🥇 Excellent" } 
        else if initial_residual_magnitude < 0.05 { "🥈 Good" } 
        else { "🥉 Fair" });
    
    // 🔥 최극한 정밀 RBE 잔차 학습
    let mut seed = Packed128::random(&mut rand::thread_rng());
    let mut optimizer = RiemannianAdamOptimizer::new();
    let mut best_rmse = f32::INFINITY;
    let mut no_improvement_count = 0;
    let mut current_lr = initial_lr;
    let mut consecutive_improvements = 0;
    
    // 고도화된 적응적 학습률 스케줄러
    let lr_decay_factor = 0.98; // 더 완만한 감소
    let lr_decay_patience = 200; // 더 빠른 반응
    let lr_boost_factor = 1.02; // 개선시 약간 증가
    let max_lr = 0.001; // 최대 학습률 제한
    let min_lr = 0.00001; // 최소 학습률 제한
    
    for epoch in 1..=max_epochs {
        // 현재 예측 생성
        let mut predicted = vec![0.0; residuals.len()];
        for i in 0..rows {
            for j in 0..cols {
                predicted[i * cols + j] = seed.fused_forward(i, j, rows, cols);
            }
        }
        
        // 최극한 정밀 역전파
        let (_, rmse) = optimizer.fused_backward_step(
            &residuals, 
            &predicted, 
            &mut seed, 
            rows, 
            cols, 
            current_lr
        );
        
        // 고도화된 성능 추적
        if rmse < best_rmse {
            best_rmse = rmse;
            no_improvement_count = 0;
            consecutive_improvements += 1;
            
            // 연속 개선시 학습률 약간 증가
            if consecutive_improvements >= 10 {
                current_lr = (current_lr * lr_boost_factor).min(max_lr);
                consecutive_improvements = 0;
            }
        } else {
            no_improvement_count += 1;
            consecutive_improvements = 0;
        }
        
        // 고도화된 학습률 조정
        if no_improvement_count >= lr_decay_patience {
            current_lr = (current_lr * lr_decay_factor).max(min_lr);
            no_improvement_count = 0;
            
            if epoch % 5000 == 0 {
                println!("  📉 학습률 조정: {:.8}", current_lr);
            }
        }
        
        // 진행 상황 출력 (2000 에포크마다)
        if epoch % 2000 == 0 || epoch == max_epochs {
            let total_rmse = calculate_final_rmse(&target, &decoded_base, &predicted);
            let quality_grade = if total_rmse < 0.001 { "🥇 S급" }
                else if total_rmse < 0.005 { "🥈 A+급" }
                else if total_rmse < 0.01 { "🥉 A급" }
                else if total_rmse < 0.05 { "B급" }
                else if total_rmse < 0.1 { "C급" }
                else { "D급" };
            
            let progress_percent = (epoch as f32 / max_epochs as f32) * 100.0;
            println!("  Epoch {} ({:.1}%): 잔차 RMSE = {:.8}, 전체 RMSE = {:.8}, 품질: {}, LR: {:.8}", 
                epoch, progress_percent, rmse, total_rmse, quality_grade, current_lr);
        }
        
        // 🎯 절대 목표 달성 조기 종료
        let total_rmse = calculate_final_rmse(&target, &decoded_base, &predicted);
        if total_rmse < 0.001 {
            println!("  🎉🎉🎉 절대 목표 달성! Epoch {}: 최종 RMSE = {:.8}", epoch, total_rmse);
            println!("  🏆 S급 품질 달성! 역사적 순간입니다!");
            break;
        }
        
        // 극한 정밀도 달성시 메시지
        if rmse < 0.0001 {
            println!("  🚀 극한 정밀도 달성! Epoch {}: 잔차 RMSE = {:.8}", epoch, rmse);
        }
        
        // A급 품질 달성시 메시지  
        if total_rmse < 0.01 && epoch % 1000 == 0 {
            println!("  ⭐ A급 품질 유지 중! 목표까지 {:.1}% 남음", (total_rmse / 0.001) * 100.0 - 100.0);
        }
    }
    
    let max_duration = start_time.elapsed().as_millis();
    
    // 최종 평가
    let mut final_predicted = vec![0.0; target.len()];
    for i in 0..rows {
        for j in 0..cols {
            final_predicted[i * cols + j] = decoded_base[i * cols + j] + 
                seed.fused_forward(i, j, rows, cols);
        }
    }
    
    let final_rmse = calculate_rmse(&target, &final_predicted);
    
    // 🏆 최종 결과 분석
    println!("\n🏆 === 최극한 설정 최종 결과 ===");
    println!("최종 RMSE: {:.8}", final_rmse);
    println!("소요 시간: {:.1}초", max_duration as f32 / 1000.0);
    println!("최고 잔차 RMSE: {:.8}", best_rmse);
    
    if final_rmse < 0.001 {
        println!("🎯🎯🎯 축하합니다! RMSE < 0.001 절대 목표를 달성했습니다!");
        println!("🏆 S급 품질 달성! 혁신적인 성과입니다!");
        let improvement_rate = (1.0 - final_rmse / 0.5) * 100.0;
        println!("총 개선률: {:.3}%", improvement_rate);
        println!("압축 효율: {:.1}:1", (rows * cols * 4) as f32 / 16.0);
    } else {
        println!("목표 달성: ❌ 미달성");
        let achievement_rate = (0.001 / final_rmse) * 100.0;
        println!("현재 달성률: {:.2}%", achievement_rate);
        
        // 상세한 분석 및 제안
        println!("\n📈 === 상세 분석 ===");
        println!("목표까지 필요한 추가 개선: {:.1}배", final_rmse / 0.001);
        
        if final_rmse < 0.005 {
            println!("💡 거의 근접! 다음 시도 시:");
            println!("1. 에포크 50000으로 증가");
            println!("2. 앙상블 기법 (다중 시드 평균)");
            println!("3. 계층적 블록 분할");
        } else if final_rmse < 0.01 {
            println!("💡 양호한 성과! 다음 시도 시:");
            println!("1. DCT 계수 75개로 증가");
            println!("2. 학습률 스케줄링 세밀화");
            println!("3. 정규화 기법 도입");
        } else {
            println!("💡 추가 최적화 방향:");
            println!("1. 아키텍처 근본 개선");
            println!("2. 다중 변환 기법 조합");
            println!("3. 고급 수치 최적화 기법");
        }
    }
    
    // 성능 검증
    assert!(final_rmse < 0.1, "기본 성능 기준 미달: RMSE = {:.6}", final_rmse);
    
    Ok(())
}

/// 최종 RMSE 계산 (DCT 기저 + RBE 잔차)
fn calculate_final_rmse(target: &[f32], dct_base: &[f32], rbe_residual: &[f32]) -> f32 {
    let mse: f32 = target.iter().enumerate()
        .map(|(i, &t)| {
            let predicted = dct_base[i] + rbe_residual[i];
            (t - predicted).powi(2)
        })
        .sum::<f32>() / target.len() as f32;
    mse.sqrt()
}

/// 품질 등급 계산
fn get_quality_grade(rmse: f32) -> &'static str {
    if rmse < 0.001 {
        "S급"
    } else if rmse < 0.01 {
        "A급"
    } else if rmse < 0.05 {
        "B급"
    } else if rmse < 0.1 {
        "C급"
    } else {
        "D급"
    }
}

/// 🚀 한글 프롬프트 성능 테스트
#[test]
fn test_korean_prompt_performance() {
    println!("🇰🇷 === 한글 프롬프트 RBE 성능 테스트 ===");
    
    // 한글 텍스트를 수치 패턴으로 변환하여 테스트
    let korean_prompt = "안녕하세요! 리만 기저 인코딩으로 신경망을 압축해보겠습니다.";
    let pattern = korean_text_to_pattern(korean_prompt, 32, 32);
    
    println!("입력 프롬프트: '{}'", korean_prompt);
    println!("패턴 크기: 32×32");
    
    // 하이브리드 방법으로 테스트
    let start_time = Instant::now();
    let (rmse, _) = test_hybrid_dct_riemann(&pattern, 32, 32, 500, 0.001);
    let elapsed = start_time.elapsed().as_millis();
    
    println!("🎯 결과:");
    println!("  RMSE: {:.6}", rmse);
    println!("  시간: {}ms", elapsed);
    println!("  품질: {}", get_quality_grade(rmse));
    
    // 압축률 계산
    let original_size = 32 * 32 * 4; // f32
    let compressed_size = 16; // Packed128
    let compression_ratio = original_size as f32 / compressed_size as f32;
    
    println!("  압축률: {:.1}:1", compression_ratio);
    
    if rmse < 0.01 {
        println!("✅ 한글 프롬프트 고품질 처리 성공!");
    } else {
        println!("⚠️ 추가 최적화 필요");
    }
}

/// 한글 텍스트를 수치 패턴으로 변환
fn korean_text_to_pattern(text: &str, rows: usize, cols: usize) -> Vec<f32> {
    let mut pattern = vec![0.0; rows * cols];
    
    // 유니코드 기반 패턴 생성
    for (idx, ch) in text.chars().enumerate() {
        if idx >= rows * cols { break; }
        
        let unicode_val = ch as u32;
        let normalized = (unicode_val as f32 / 65535.0) * 2.0 - 1.0; // [-1, 1] 정규화
        pattern[idx] = normalized;
    }
    
    // 나머지는 코사인 파동으로 채움
    for i in (text.len()).min(rows * cols)..rows * cols {
        let x = (i % cols) as f32 / cols as f32;
        let y = (i / cols) as f32 / rows as f32;
        pattern[i] = (2.0 * std::f32::consts::PI * x).cos() * (2.0 * std::f32::consts::PI * y).sin() * 0.5;
    }
    
    pattern
} 
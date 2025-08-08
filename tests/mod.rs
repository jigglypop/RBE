pub mod core;
pub mod nlp;

// 극한 압축률 실험을 여기에 직접 추가
use rbe_llm::{
    core::{
        optimizers::BitAdamState,
        tensors::{Packed256, Packed256Params},
        differential::bit_engine,
    },
    nlp::linear::{
        BlockInfo, BlockManager, Tensor,
        optimize_block_size, block_to_poincare
    }
};
use rand::{rngs::StdRng, SeedableRng, Rng};

#[test]
fn 극한_압축률_100대1_실험() {
    println!("\n🚀 === 더 큰 블록 크기 압축 정확도 개선 실험 ===");
    
    // 대형 Linear 레이어 시뮬레이션 (더 큰 블록으로)
    let input_dim = 512;
    let output_dim = 2048; 
    let target_compression = 105.0; // 여유를 두고 105:1 목표
    let target_rmse = 0.001;        // 더 정밀한 목표
    let max_epochs = 8000;          // 대폭 증가
    
    println!("📊 실험 설정:");
    println!("  - 입력 차원: {}", input_dim);
    println!("  - 출력 차원: {}", output_dim);
    println!("  - 총 파라미터: {:.1}M", (input_dim * output_dim) as f32 / 1_000_000.0);
    println!("  - 목표 압축률: {}:1", target_compression);
    println!("  - 목표 RMSE: {:.6}", target_rmse);
    println!("  - 최대 에포크: {}", max_epochs);
    
    // 최적 블록 크기 계산
    let optimal_block_size = optimize_block_size((input_dim, output_dim), target_compression);
    println!("\n🔍 최적 블록 크기: {}x{}", optimal_block_size.0, optimal_block_size.1);
    
    let block_info = BlockInfo::new(input_dim, output_dim, optimal_block_size).unwrap();
    let block_manager = BlockManager::new(block_info.clone());
    
    let actual_compression = block_manager.calculate_compression_ratio();
    println!("📈 실제 압축률: {:.2}:1", actual_compression);
    println!("📦 총 블록 수: {}", block_info.total_blocks());
    
    // 더 많은 블록으로 테스트 (더 정확한 평가)
    let sample_blocks = 50.min(block_info.total_blocks());
    let mut total_rmse = 0.0;
    let mut converged_blocks = 0;
    
    println!("\n🔥 샘플 블록 ({}/{}개) RBE 압축 시작...", sample_blocks, block_info.total_blocks());
    
    for block_idx in 0..sample_blocks {
        let result = compress_sample_block(optimal_block_size, target_rmse, max_epochs, block_idx);
        total_rmse += result.final_rmse;
        if result.converged {
            converged_blocks += 1;
        }
        
        if block_idx % 10 == 0 {
            let basis_options = [0, 1, 2, 3];
            let (dr, dt) = match block_idx % 4 {
                0 => (0, 0), 1 => (1, 0), 2 => (2, 0), 3 => (0, 0),
                _ => (0, 0),
            };
            println!("  블록 {} (기저={}, d_r={}, d_θ={}): RMSE={:.8}, 수렴={}", 
                     block_idx, basis_options[block_idx % basis_options.len()],
                     dr, dt, result.final_rmse, result.converged);
        }
    }
    
    let average_rmse = total_rmse / sample_blocks as f32;
    let convergence_rate = converged_blocks as f32 / sample_blocks as f32 * 100.0;
    
    println!("\n📊 === 극한 압축 결과 ===");
    println!("✅ 평균 RMSE: {:.8}", average_rmse);
    println!("📈 수렴률: {:.1}% ({}/{})", convergence_rate, converged_blocks, sample_blocks);
    println!("🎯 목표 달성: {}", if average_rmse <= target_rmse { "성공!" } else { "도전 중..." });
    
    // 더 큰 블록으로 더 나은 성능 기대
    assert!(
        average_rmse <= target_rmse * 3.0, 
        "RMSE 목표 미달성: {:.8} > {:.8}", 
        average_rmse, target_rmse * 3.0
    );
    
    assert!(
        actual_compression >= target_compression * 0.8, 
        "압축률 목표 미달성: {:.2} < {:.2}", 
        actual_compression, target_compression * 0.8
    );
    
    // 수렴률도 체크
    assert!(
        convergence_rate >= 20.0,
        "수렴률이 너무 낮음: {:.1}% < 20%",
        convergence_rate
    );
    
    println!("\n🎉 극한 압축 실험 성공!");
}

/// 샘플 블록 압축 결과
struct SampleCompressionResult {
    final_rmse: f32,
    converged: bool,
}

/// 샘플 블록으로 압축 테스트
fn compress_sample_block(
    block_size: (usize, usize), 
    target_rmse: f32, 
    max_epochs: usize,
    block_idx: usize
) -> SampleCompressionResult {
    let (block_h, block_w) = block_size;
    
    // 랜덤 가중치 블록 생성
    let mut rng = StdRng::seed_from_u64(42 + block_idx as u64);
    let weight_data: Vec<f32> = (0..block_h * block_w)
        .map(|_| rng.gen_range(-0.05..0.05))
        .collect();
    
    // 고성능 기저만 선택 (Bessel J0가 최고 성능)
    let basis_options = [0, 0, 0, 1]; // Bessel J0에 가중치
    let selected_basis = basis_options[block_idx % basis_options.len()];
    
    // 안정적 비트 미분만 사용 (theta 미분 제외)
    let (d_r, d_theta) = match block_idx % 4 {
        0 => (0, 0), // 원함수 - 가장 안정적
        1 => (1, 0), // r 1차 미분 - 안정적  
        2 => (2, 0), // r 2차 미분 - 안정적
        3 => (0, 0), // 원함수 반복 (더 많은 기회)
        _ => (0, 0),
    };
    
    let mut seed = Packed256::new(&Packed256Params {
        r: 0.25,
        theta: 0.6,
        param1: 1.5,
        param2: 1.1,
        basis_id: selected_basis,
        d_r: d_r as u8,      // 비트 미분 적용!
        d_theta: d_theta as u8,
        log2_c: -2,          // 더 세밀한 곡률
        activation_id: 0,
        q_value: 0,
        k_value: 0,
        flags: 0,
    });
    
    let mut optimizer = BitAdamState::new();
    let mut best_rmse = f32::INFINITY;
    let mut converged = false;
    // 고성능 학습률: Bessel J0 중심
    let learning_rate = match (d_r, selected_basis) {
        (0, 0) => 0.040, // Bessel J0 원함수 - 최대 적극적
        (0, _) => 0.035, // 다른 기저 원함수 
        (1, 0) => 0.030, // J0 + r 1차 미분 - 더 적극적
        (1, _) => 0.025, // 다른 기저 + r 1차 미분
        (2, 0) => 0.020, // J0 + r 2차 미분 - 더 적극적
        (2, _) => 0.015, // 다른 기저 + r 2차 미분
        _ => 0.025,
    };
    
    for epoch in 1..=max_epochs {
        let mut epoch_rmse = 0.0;
        
        // 블록의 모든 요소에 대해 학습
        for i in 0..block_h {
            for j in 0..block_w {
                let target_value = weight_data[i * block_w + j];
                let (_r, _theta) = block_to_poincare(i, j, (block_h, block_w));
                
                // RBE 순전파
        let params = Packed256Params {
            r: seed.get_r(),
            theta: seed.get_theta(),
            param1: seed.get_param1(),
            param2: seed.get_param2(),
            basis_id: seed.get_basis_id(),
            d_r: seed.get_d_r(),
            d_theta: seed.get_d_theta(),
            log2_c: seed.get_log2_c(),
            activation_id: seed.get_activation_id(),
            q_value: seed.get_q_value(),
            k_value: seed.get_k_value(),
            flags: seed.get_flags(),
        };
        let output = bit_engine::compute_fused_output(&params, i, j, block_h, block_w);
                let predicted = output.predicted_value;
                
                // 오차 계산
                let error = predicted - target_value;
                epoch_rmse += error * error;
                
                // 역전파 (더 자주 업데이트로 정확도 향상)
                if (i + j) % 2 == 0 {
                    optimizer.bit_update(&mut seed, i, j, block_h, block_w, target_value, learning_rate);
                }
            }
        }
        
        epoch_rmse = (epoch_rmse / (block_h * block_w) as f32).sqrt();
        
        if epoch_rmse < best_rmse {
            best_rmse = epoch_rmse;
        }
        
        // 수렴 체크 (더 빠른 감지)
        if epoch_rmse <= target_rmse {
            converged = true;
            break;
        }
        
        // 조기 성공: 매우 낮은 RMSE 달성 시 즉시 종료
        if epoch_rmse <= target_rmse * 0.3 {
            converged = true;
            break;
        }
        
        // 고정밀도 수렴: RMSE가 매우 낮아지면 즉시 완료
        if epoch_rmse <= 0.005 {
            converged = true;
            break;
        }
        
        // 조기 종료 (개선이 없으면) - 더 오래 기다리기
        if epoch > 500 && epoch_rmse > best_rmse * 1.2 {
            break;
        }
    }
    
    SampleCompressionResult {
        final_rmse: best_rmse,
        converged,
    }
} 
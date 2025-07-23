use rbe_llm::core::optimizers::{AdamState, AdamBuffer};

#[test]
fn simd_vs_scalar_rmse() {
    let n = 10_000;
    let mut params_scalar: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut params_simd = params_scalar.clone();
    let grads: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002).cos()).collect();

    // 스칼라 버전 - 각 파라미터에 대해 독립적인 Adam 상태
    let mut adam_states: Vec<AdamState> = (0..n).map(|_| AdamState::new()).collect();
    for (i, (p, &g)) in params_scalar.iter_mut().zip(grads.iter()).enumerate() {
        adam_states[i].update(p, g, 1e-3);
    }

    // SIMD 버전
    let mut adam_simd = AdamState::new();
    let mut buf = AdamBuffer::zeroed(n);
    adam_simd.update_batch_simd(&mut params_simd, &grads, &mut buf, 1e-3);

    // RMSE 계산
    let mse: f32 = params_scalar.iter()
        .zip(params_simd.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / n as f32;
    let rmse = mse.sqrt();
    println!("RMSE between scalar and SIMD: {:.6e}", rmse);
    assert!(rmse < 1e-6, "RMSE too high: {:.6e}", rmse);
}

#[test]
fn adam_batch_performance() {
    use std::time::Instant;
    
    let sizes = vec![1000, 10_000, 100_000];
    
    for n in sizes {
        let params = vec![0.1f32; n];
        let grads: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
        
        // 스칼라 버전 - 각 파라미터에 대해 독립적인 Adam 상태
        let mut adam_states: Vec<AdamState> = (0..n).map(|_| AdamState::new()).collect();
        let mut params_scalar = params.clone();
        let start = Instant::now();
        for (i, (p, &g)) in params_scalar.iter_mut().zip(grads.iter()).enumerate() {
            adam_states[i].update(p, g, 1e-3);
        }
        let scalar_time = start.elapsed();
        
        // SIMD 버전
        let mut adam_simd = AdamState::new();
        let mut params_simd = params.clone();
        let mut buf = AdamBuffer::zeroed(n);
        let start = Instant::now();
        adam_simd.update_batch_simd(&mut params_simd, &grads, &mut buf, 1e-3);
        let simd_time = start.elapsed();
        
        let speedup = scalar_time.as_secs_f32() / simd_time.as_secs_f32();
        println!("N={}: scalar={:?}, SIMD={:?}, speedup={:.2}x", 
                 n, scalar_time, simd_time, speedup);
    }
} 
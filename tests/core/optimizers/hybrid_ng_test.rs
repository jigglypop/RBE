use rbe_llm::core::optimizers::{HybridNaturalOptimizer, HybridNGConfig};
use rbe_llm::core::tensors::{Packed256, packed256_types::Packed256Params};
use rbe_llm::core::differential::bit_engine;

fn rmse_from_seeds(seeds: &[Packed256], rows: usize, cols: usize, target: &[f32]) -> f32 {
    let mut mse = 0.0f32;
    for i in 0..rows {
        for j in 0..cols {
            let mut pred_sum = 0.0f32;
            for s in 0..seeds.len() {
                let params = Packed256Params {
                    r: seeds[s].get_r(),
                    theta: seeds[s].get_theta(),
                    param1: seeds[s].get_param1(),
                    param2: seeds[s].get_param2(),
                    basis_id: seeds[s].get_basis_id(),
                    d_r: seeds[s].get_d_r(),
                    d_theta: seeds[s].get_d_theta(),
                    log2_c: seeds[s].get_log2_c(),
                    activation_id: seeds[s].get_activation_id(),
                    q_value: seeds[s].get_q_value(),
                    k_value: seeds[s].get_k_value(),
                    flags: seeds[s].get_flags(),
                };
                pred_sum += bit_engine::compute_fused_output(&params, i, j, rows, cols).predicted_value;
            }
            let e = pred_sum - target[i*cols + j];
            mse += e * e;
        }
    }
    (mse / (rows*cols) as f32).sqrt()
}

#[test]
fn hybrid_ng_single_seed_improves_rmse() {
    let (h, w) = (64usize, 64usize);
    // Ground truth single seed
    let gt = Packed256::new(&Packed256Params {
        r: 0.7, theta: 0.2, param1: std::f32::consts::PI * 1.5, param2: 0.9,
        basis_id: 12, d_r: 0, d_theta: 0, log2_c: -20, activation_id: 160, q_value: 255, k_value: 200, flags: 0,
    });
    // Build target
    let mut target = vec![0.0f32; h*w];
    for i in 0..h { for j in 0..w {
        let params = Packed256Params { r: gt.get_r(), theta: gt.get_theta(), param1: gt.get_param1(), param2: gt.get_param2(), basis_id: gt.get_basis_id(), d_r: gt.get_d_r(), d_theta: gt.get_d_theta(), log2_c: gt.get_log2_c(), activation_id: gt.get_activation_id(), q_value: gt.get_q_value(), k_value: gt.get_k_value(), flags: gt.get_flags() };
        target[i*w + j] = bit_engine::compute_fused_output(&params, i, j, h, w).predicted_value;
    }}
    // Init perturbed seed
    let mut seeds = vec![Packed256::new(&Packed256Params {
        r: (gt.get_r()+0.1).clamp(0.0,0.9999), theta: gt.get_theta()-0.1, param1: gt.get_param1()*0.9, param2: (gt.get_param2()+0.1).clamp(0.0,4.0),
        basis_id: 12, d_r: 0, d_theta: 0, log2_c: -20, activation_id: 150, q_value: 255, k_value: 210, flags: 0,
    })];

    let rmse0 = rmse_from_seeds(&seeds, h, w, &target);
    let opt = HybridNaturalOptimizer::new(HybridNGConfig {
        steps: 200, batch_size: h*w, learning_rate_r: 1e-2, learning_rate_theta: 1e-2, learning_rate_p1: 5e-3, learning_rate_p2: 5e-3, ..Default::default()
    });
    opt.optimize_tile(&mut seeds, &target, h, w);
    let rmse1 = rmse_from_seeds(&seeds, h, w, &target);
    assert!(rmse1 < rmse0, "rmse did not drop: before={:.6} after={:.6}", rmse0, rmse1);
    assert!(rmse1 <= 0.05, "rmse not sufficiently low: {:.6}", rmse1);
}


#[test]
fn hybrid_ng_single_seed_improves_rmse_basis13() {
    let (h, w) = (64usize, 64usize);
    // Ground truth single seed for basis 13 (rank-1 sinusoid)
    let gt = Packed256::new(&Packed256Params {
        r: 0.65, theta: 0.3, param1: std::f32::consts::PI * 1.2, param2: 0.8,
        basis_id: 13, d_r: 0, d_theta: 0, log2_c: -20, activation_id: 0, q_value: 64, k_value: 0, flags: 0,
    });
    // Build target
    let mut target = vec![0.0f32; h*w];
    for i in 0..h { for j in 0..w {
        let params = Packed256Params {
            r: gt.get_r(), theta: gt.get_theta(), param1: gt.get_param1(), param2: gt.get_param2(),
            basis_id: gt.get_basis_id(), d_r: gt.get_d_r(), d_theta: gt.get_d_theta(), log2_c: gt.get_log2_c(),
            activation_id: gt.get_activation_id(), q_value: gt.get_q_value(), k_value: gt.get_k_value(), flags: gt.get_flags(),
        };
        target[i*w + j] = bit_engine::compute_fused_output(&params, i, j, h, w).predicted_value;
    }}
    // Init perturbed seed
    let mut seeds = vec![Packed256::new(&Packed256Params {
        r: (gt.get_r()+0.1).clamp(0.0,0.9999), theta: gt.get_theta()-0.2, param1: gt.get_param1()*0.85, param2: (gt.get_param2()+0.1).clamp(-4.0,4.0),
        basis_id: 13, d_r: 0, d_theta: 0, log2_c: -20, activation_id: 0, q_value: 64, k_value: 0, flags: 0,
    })];

    let rmse0 = rmse_from_seeds(&seeds, h, w, &target);
    let opt = HybridNaturalOptimizer::new(HybridNGConfig {
        steps: 400, batch_size: h*w, learning_rate_r: 5e-3, learning_rate_theta: 1e-2, learning_rate_p1: 1e-3, learning_rate_p2: 2e-3, grad_clip: 0.3, ..Default::default()
    });
    opt.optimize_tile(&mut seeds, &target, h, w);
    let rmse1 = rmse_from_seeds(&seeds, h, w, &target);
    assert!(rmse1 < rmse0, "[b13] rmse did not drop: before={:.6} after={:.6}", rmse0, rmse1);
    assert!(rmse1 <= 0.06, "[b13] rmse not sufficiently low: {:.6}", rmse1);
}


#[test]
fn hybrid_ng_single_seed_improves_rmse_basis13_1000() {
    let (h, w) = (64usize, 64usize);
    let gt = Packed256::new(&Packed256Params {
        r: 0.65, theta: 0.3, param1: std::f32::consts::PI * 1.2, param2: 0.8,
        basis_id: 13, d_r: 0, d_theta: 0, log2_c: -20, activation_id: 0, q_value: 64, k_value: 0, flags: 0,
    });
    let mut target = vec![0.0f32; h*w];
    for i in 0..h { for j in 0..w {
        let params = Packed256Params {
            r: gt.get_r(), theta: gt.get_theta(), param1: gt.get_param1(), param2: gt.get_param2(),
            basis_id: gt.get_basis_id(), d_r: gt.get_d_r(), d_theta: gt.get_d_theta(), log2_c: gt.get_log2_c(),
            activation_id: gt.get_activation_id(), q_value: gt.get_q_value(), k_value: gt.get_k_value(), flags: gt.get_flags(),
        };
        target[i*w + j] = bit_engine::compute_fused_output(&params, i, j, h, w).predicted_value;
    }}
    let mut seeds = vec![Packed256::new(&Packed256Params {
        r: (gt.get_r()+0.1).clamp(0.0,0.9999), theta: gt.get_theta()-0.2, param1: gt.get_param1()*0.85, param2: (gt.get_param2()+0.1).clamp(-4.0,4.0),
        basis_id: 13, d_r: 0, d_theta: 0, log2_c: -20, activation_id: 0, q_value: 64, k_value: 0, flags: 0,
    })];

    let rmse0 = rmse_from_seeds(&seeds, h, w, &target);
    let opt = HybridNaturalOptimizer::new(HybridNGConfig {
        steps: 1000, batch_size: h*w, learning_rate_r: 5e-3, learning_rate_theta: 1e-2, learning_rate_p1: 1e-3, learning_rate_p2: 2e-3, grad_clip: 0.3, ..Default::default()
    });
    opt.optimize_tile(&mut seeds, &target, h, w);
    let rmse1 = rmse_from_seeds(&seeds, h, w, &target);
    assert!(rmse1 < rmse0, "[b13-1000] rmse did not drop: before={:.6} after={:.6}", rmse0, rmse1);
    assert!(rmse1 <= 0.06, "[b13-1000] rmse not sufficiently low: {:.6}", rmse1);
}


use std::fs::read;
use std::env;
use safetensors::SafeTensors;

use rbe_llm::core::optimizers::{HybridNaturalOptimizer, HybridNGConfig};
use rbe_llm::core::tensors::{Packed256, packed256_types::Packed256Params};
use rbe_llm::core::differential::bit_engine;

#[inline]
fn to_params(seed: &Packed256) -> Packed256Params {
    Packed256Params {
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
    }
}

fn rmse_from_seeds(seeds: &[Packed256], rows: usize, cols: usize, target: &[f32]) -> f32 {
    let mut mse = 0.0f64;
    for i in 0..rows { for j in 0..cols {
        let mut pred_sum = 0.0f32;
        for s in 0..seeds.len() {
            pred_sum += bit_engine::compute_fused_output(&to_params(&seeds[s]), i, j, rows, cols).predicted_value;
        }
        let e = (pred_sum - target[i*cols + j]) as f64;
        mse += e*e;
    }}
    (mse / (rows*cols) as f64).sqrt() as f32
}

#[test]
#[ignore]
fn hybrid_ng_llm_single_layer_tile_rmse_and_cr50() -> anyhow::Result<()> {
    // Required env: LLM_WEIGHTS_PATH, TENSOR_KEY
    let weights_path = match env::var("LLM_WEIGHTS_PATH") { Ok(v) => v, Err(_) => return Ok(()) };
    let tensor_key = match env::var("TENSOR_KEY") { Ok(v) => v, Err(_) => return Ok(()) };
    let row0: usize = env::var("TILE_ROW0").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let col0: usize = env::var("TILE_COL0").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let tile_h: usize = env::var("TILE_H").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let tile_w: usize = env::var("TILE_W").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let seeds_k: usize = env::var("SEEDS_K").ok().and_then(|v| v.parse().ok()).unwrap_or(16);
    let steps: usize = env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);

    let bytes = read(&weights_path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let tensor = st.tensor(&tensor_key)?;
    let shape = tensor.shape();
    assert_eq!(shape.len(), 2, "tensor must be 2D, got shape={:?}", shape);
    let (rows_all, cols_all) = (shape[0], shape[1]);

    let tile_h = tile_h.min(rows_all);
    let tile_w = tile_w.min(cols_all);
    let row0 = row0.min(rows_all.saturating_sub(tile_h));
    let col0 = col0.min(cols_all.saturating_sub(tile_w));

    let data_f32: Vec<f32> = match tensor.dtype() {
        safetensors::Dtype::F32 => {
            let raw = tensor.data();
            let ptr = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, raw.len() / 4) };
            ptr.to_vec()
        }
        safetensors::Dtype::F16 => {
            let raw = tensor.data();
            let ptr = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const u16, raw.len() / 2) };
            ptr.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect()
        }
        other => anyhow::bail!("unsupported dtype: {:?}", other),
    };

    let mut tile = vec![0.0f32; tile_h * tile_w];
    for i in 0..tile_h { for j in 0..tile_w {
        let src_idx = (row0 + i) * cols_all + (col0 + j);
        tile[i*tile_w + j] = data_f32[src_idx];
    }}

    let mut seeds: Vec<Packed256> = (0..seeds_k).map(|s| {
        let r = 0.7f32;
        let theta = 0.2 + 0.01 * (s as f32);
        let p1 = std::f32::consts::PI * (1.0 + 0.05 * (s as f32));
        let p2 = 0.8f32;
        Packed256::new(&Packed256Params { r, theta, param1: p1, param2: p2, basis_id: 12, d_r: 0, d_theta: 0, log2_c: -20, activation_id: 160, q_value: 255, k_value: 200, flags: 0 })
    }).collect();

    let rmse0 = rmse_from_seeds(&seeds, tile_h, tile_w, &tile);

    let cfg = HybridNGConfig { steps, batch_size: tile_h*tile_w, learning_rate_r: 5e-3, learning_rate_theta: 1e-2, learning_rate_p1: 1e-3, learning_rate_p2: 2e-3, grad_clip: 0.3, seed: 42, ..Default::default() };
    let opt = HybridNaturalOptimizer::new(cfg);
    opt.optimize_tile(&mut seeds, &tile, tile_h, tile_w);
    let rmse1 = rmse_from_seeds(&seeds, tile_h, tile_w, &tile);

    println!("[TEST][LLM] {} tile=({},{}; {}x{}) k={} steps={} rmse_before={:.8} rmse_after={:.8}", tensor_key, row0, col0, tile_h, tile_w, seeds_k, steps, rmse0, rmse1);

    // CR=50 feasibility for this tile (no residual path in this test)
    let t = tile_h as u32;
    let s_allow = 4.0 * (t*t) as f32 / 50.0;
    let s_use = 32.0 * (seeds_k as f32) + 4.0;
    println!("[TEST][LLM] CR50 allow={:.2}B use={:.2}B", s_allow, s_use);

    assert!(rmse1 < rmse0, "rmse did not drop: before={:.6} after={:.6}", rmse0, rmse1);
    Ok(())
}



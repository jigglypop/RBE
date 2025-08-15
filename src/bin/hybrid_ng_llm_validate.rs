use clap::Parser;
use safetensors::SafeTensors; 
use std::fs::read;
use std::path::PathBuf;

use rbe_llm::core::optimizers::{HybridNaturalOptimizer, HybridNGConfig};
use rbe_llm::core::tensors::{Packed256, packed256_types::Packed256Params};
use rbe_llm::core::differential::bit_engine;

#[derive(Parser, Debug)]
#[command(name = "hybrid-ng-llm-validate", about = "Validate Hybrid NG on a real LLM weight tensor tile (safetensors)")]
struct Args {
    #[arg(long)]
    weights_path: PathBuf,
    #[arg(long)]
    tensor_key: String,
    #[arg(long, default_value_t = 0)]
    row0: usize,
    #[arg(long, default_value_t = 0)]
    col0: usize,
    #[arg(long, default_value_t = 64)]
    tile_h: usize,
    #[arg(long, default_value_t = 64)]
    tile_w: usize,
    #[arg(long, default_value_t = 16)]
    seeds_k: usize,
    #[arg(long, default_value_t = 12)]
    basis_id: u8,
    #[arg(long, default_value_t = 2000)]
    steps: usize,
    #[arg(long, default_value_t = 50.0)]
    cr_target: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// standardize tile: subtract mean and divide by std before optimizing
    #[arg(long, default_value_t = false)]
    standardize: bool,
    // learning rates and clip (optional overrides)
    #[arg(long)]
    lr_r: Option<f32>,
    #[arg(long)]
    lr_theta: Option<f32>,
    #[arg(long)]
    lr_p1: Option<f32>,
    #[arg(long)]
    lr_p2: Option<f32>,
    #[arg(long)]
    clip: Option<f32>,
    /// Optional Huber delta; if set, use Huber loss instead of L2
    #[arg(long)]
    huber_delta: Option<f32>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let bytes = read(&args.weights_path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let tensor = st.tensor(&args.tensor_key)?;

    // Load tensor as f32 matrix [rows, cols]
    let shape = tensor.shape();
    anyhow::ensure!(shape.len() == 2, "tensor must be 2D, got shape={:?}", shape);
    let (rows_all, cols_all) = (shape[0], shape[1]);

    // Support f32 and f16 tensors
    let tile_h = args.tile_h.min(rows_all);
    let tile_w = args.tile_w.min(cols_all);
    let row0 = args.row0.min(rows_all.saturating_sub(tile_h));
    let col0 = args.col0.min(cols_all.saturating_sub(tile_w));

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

    // Extract tile (row-major)
    let mut tile = vec![0.0f32; tile_h * tile_w];
    for i in 0..tile_h { for j in 0..tile_w {
        let src_idx = (row0 + i) * cols_all + (col0 + j);
        tile[i*tile_w + j] = data_f32[src_idx];
    }}

    // Compute tile mean/std for optional standardization
    let mut mean = 0.0f64; let mut var = 0.0f64; let n = (tile_h * tile_w) as f64;
    for &v in &tile { mean += v as f64; }
    mean /= n;
    for &v in &tile { let d = v as f64 - mean; var += d*d; }
    let std = (var / n).sqrt().max(1e-8);
    let mut tile_work = tile.clone();
    if args.standardize {
        for v in &mut tile_work { *v = ((*v as f64 - mean) / std) as f32; }
    }

    // Initialize k seeds
    let mut seeds: Vec<Packed256> = Vec::with_capacity(args.seeds_k);
    // simple deterministic rng based on seed
    let mut state = args.seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut next_f32 = || -> f32 { state = state.wrapping_mul(6364136223846793005).wrapping_add(1); ((state >> 32) as u32) as f32 / (u32::MAX as f32) };
    if args.basis_id == 13 || args.basis_id == 14 {
        // Spectral seeding for rank-1 basis
        let h = tile_h as usize; let w = tile_w as usize;
        let mut row_sum = vec![0.0f64; h];
        let mut col_sum = vec![0.0f64; w];
        for i in 0..h { for j in 0..w { let v = tile_work[i*w + j] as f64; row_sum[i] += v; col_sum[j] += v; } }
        let max_mx = (h/2).max(1).min(16);
        let mut freq_x: Vec<(usize, f64, f32)> = Vec::with_capacity(max_mx);
        for m in 1..=max_mx {
            let omega = 2.0 * std::f64::consts::PI * (m as f64);
            let mut s_sin = 0.0f64; let mut s_cos = 0.0f64;
            for i in 0..h { let r = if h>1 { i as f64/((h as f64)-1.0) } else { 0.0 }; let a=omega*r; s_sin += row_sum[i]*a.sin(); s_cos += row_sum[i]*a.cos(); }
            let mag = (s_sin*s_sin + s_cos*s_cos).sqrt(); let phi = (s_cos.atan2(s_sin) as f32);
            freq_x.push((m, mag, phi));
        }
        freq_x.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        let max_my = (w/2).max(1).min(16);
        let mut freq_y: Vec<(usize, f64, f32)> = Vec::with_capacity(max_my);
        for n in 1..=max_my {
            let mut s_sin = 0.0f64; let mut s_cos = 0.0f64;
            for j in 0..w { let theta = 2.0 * std::f64::consts::PI * (j as f64)/(w as f64); let a=(n as f64)*theta; s_sin += col_sum[j]*a.sin(); s_cos += col_sum[j]*a.cos(); }
            let mag = (s_sin*s_sin + s_cos*s_cos).sqrt(); let phi = (s_cos.atan2(s_sin) as f32);
            freq_y.push((n, mag, phi));
        }
        freq_y.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        for s in 0..args.seeds_k {
            let (mx, _, phi_x) = freq_x[s % freq_x.len()];
            let (ny, _, phi_y) = freq_y[s % freq_y.len()];
            let r = 0.66 + 0.02 * (next_f32() - 0.5) * 2.0;
            let omega_x = (2.0 * std::f32::consts::PI) * (mx as f32);
            let omega_y = (2.0 * std::f32::consts::PI) * (ny as f32);
            let mut seed = Packed256::new(&Packed256Params {
                r: r.clamp(0.0, 0.9999), theta: phi_x, param1: omega_x, param2: 0.2,
                basis_id: args.basis_id, d_r: 0, d_theta: 0, log2_c: -20, activation_id: 0, q_value: 0, k_value: 0, flags: 0
            });
            // For basis 13: internal path multiplies oy by 8; for 14: two_pi*8
            let oy = if args.basis_id == 13 { (omega_y / 8.0).clamp(0.0, 1.0) } else { (omega_y / (2.0*std::f32::consts::PI*8.0)).clamp(0.0, 1.0) };
            seed.set_q_value((oy * 255.0).round().clamp(0.0, 255.0) as u8);
            let sector = (((phi_y.rem_euclid(2.0*std::f32::consts::PI)) / (0.5*std::f32::consts::PI)).round() as i32) & 0b11;
            let mut flags = seed.get_flags(); flags &= !(0b11 << 4); flags |= ((sector as u8) & 0b11) << 4; seed.set_flags(flags);
            seeds.push(seed);
        }
    } else {
        for s in 0..args.seeds_k {
            let r = 0.65 + 0.05 * (next_f32() - 0.5) * 2.0; // ~[0.60,0.70]
            let theta = (2.0 * std::f32::consts::PI) * ((s as f32 + 0.5) / args.seeds_k as f32);
            let p1_low = std::f32::consts::PI * 0.25; let p1_high = std::f32::consts::PI * 6.0;
            let p1 = p1_low + (p1_high - p1_low) * ((s as f32) / (args.seeds_k.max(1) as f32));
            let p2 = 0.05f32;
            seeds.push(Packed256::new(&Packed256Params { r: r.clamp(0.0, 0.9999), theta, param1: p1, param2: p2, basis_id: args.basis_id, d_r: 0, d_theta: 0, log2_c: -20, activation_id: 160, q_value: 255, k_value: 200, flags: 0 }));
        }
    }

    // Baseline RMSE
    let huber_delta = args.huber_delta.unwrap_or(0.0);
    let rmse_from_seeds = |seeds: &Vec<Packed256>| -> f32 {
        let mut mse = 0.0f64;
        for i in 0..tile_h { for j in 0..tile_w {
            let mut pred_sum = 0.0f32;
            for s in 0..seeds.len() {
                pred_sum += bit_engine::compute_fused_output(&to_params(&seeds[s]), i, j, tile_h, tile_w).predicted_value;
            }
            let e = (pred_sum - tile[i*tile_w + j]) as f64;
            if huber_delta > 0.0 {
                let d = huber_delta as f64;
                let ae = e.abs();
                let l = if ae <= d { 0.5*e*e } else { d*(ae - 0.5*d) };
                mse += l*2.0; // scale so that delta->0 recovers L2 magnitude
            } else {
                mse += e*e;
            }
        }}
        ((mse / (tile_h * tile_w) as f64)) as f32
    };

    let rmse0 = rmse_from_seeds(&seeds).sqrt();

    // Optimize
    let cfg = HybridNGConfig {
        steps: args.steps,
        batch_size: tile_h * tile_w,
        learning_rate_r: args.lr_r.unwrap_or(5e-3),
        learning_rate_theta: args.lr_theta.unwrap_or(1e-2),
        learning_rate_p1: args.lr_p1.unwrap_or(1e-3),
        learning_rate_p2: args.lr_p2.unwrap_or(2e-3),
        grad_clip: args.clip.unwrap_or(0.3),
        seed: args.seed,
        ..Default::default()
    };
    let opt = HybridNaturalOptimizer::new(cfg);
    // Optimize against possibly standardized tile
    opt.optimize_tile(&mut seeds, if args.standardize { &tile_work } else { &tile }, tile_h, tile_w);

    // If standardized, rescale amplitudes back to original scale
    if args.standardize {
        for s in &mut seeds { s.set_param2(s.get_param2() * (std as f32)); }
    }

    let rmse1 = rmse_from_seeds(&seeds).sqrt();

    // CR=50 feasibility check for tile
    let t = tile_h as u32; // assume square-ish; use tile_h for budget formula
    let s_allow = 4.0 * (t * t) as f32 / args.cr_target;
    let s_use = 32.0 * (args.seeds_k as f32) + 4.0; // no residual
    let ok = s_use <= s_allow + 1e-3;

    println!("[LLM-Validate] tensor='{}' tile=({},{}; {}x{}) k={} steps={} rmse_before={:.8} rmse_after={:.8} standardized={} std={:.6}",
        args.tensor_key, args.row0, args.col0, tile_h, tile_w, args.seeds_k, args.steps, rmse0, rmse1, args.standardize, std as f32);
    println!("[LLM-Validate] CR_target={} s_allow={:.2}B s_use={:.2}B feasible={}", args.cr_target, s_allow, s_use, ok);

    Ok(())
}

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



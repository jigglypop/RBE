use rbe_llm::core::optimizers::{HybridNaturalOptimizer, HybridNGConfig};
use rbe_llm::core::tensors::{Packed256, packed256_types::Packed256Params};
use rbe_llm::core::differential::bit_engine;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "hybrid-ng-drive", about = "Run Hybrid NG optimizer on a synthetic single-tile target with logging")] 
struct Args {
    #[arg(long, default_value_t = 64)]
    rows: usize,
    #[arg(long, default_value_t = 64)]
    cols: usize,
    #[arg(long, default_value_t = 2000)]
    steps: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn main() {
    let args = Args::parse();
    let (h, w) = (args.rows, args.cols);

    // Ground truth target using basis 12 (separable) for a realistic case
    let gt = Packed256::new(&Packed256Params { 
        r: 0.7, theta: 0.2, param1: std::f32::consts::PI*1.5, param2: 0.9,
        basis_id: 12, d_r: 0, d_theta: 0, log2_c: -20, activation_id: 160, q_value: 255, k_value: 200, flags: 0
    });
    let mut target = vec![0.0f32; h*w];
    for i in 0..h { for j in 0..w { 
        target[i*w + j] = bit_engine::compute_fused_output(&Packed256Params {
            r: gt.get_r(), theta: gt.get_theta(), param1: gt.get_param1(), param2: gt.get_param2(),
            basis_id: gt.get_basis_id(), d_r: gt.get_d_r(), d_theta: gt.get_d_theta(), log2_c: gt.get_log2_c(),
            activation_id: gt.get_activation_id(), q_value: gt.get_q_value(), k_value: gt.get_k_value(), flags: gt.get_flags(),
        }, i, j, h, w).predicted_value; 
    }}

    // Initialize a slightly perturbed seed
    let mut seeds = vec![Packed256::new(&Packed256Params { 
        r: (gt.get_r()+0.1).clamp(0.0,0.9999), theta: gt.get_theta()-0.1, param1: gt.get_param1()*0.9, param2: (gt.get_param2()+0.1).clamp(0.0,4.0),
        basis_id: 12, d_r: 0, d_theta: 0, log2_c: -20, activation_id: 150, q_value: 255, k_value: 210, flags: 0
    })];

    let cfg = HybridNGConfig { 
        steps: args.steps, 
        batch_size: h*w, // full-batch
        learning_rate_r: 5e-3, learning_rate_theta: 5e-3, learning_rate_p1: 1e-3, learning_rate_p2: 2e-3,
        grad_clip: 0.3, seed: args.seed, ..Default::default() };
    let opt = HybridNaturalOptimizer::new(cfg);
    opt.optimize_tile(&mut seeds, &target, h, w);

    // Final RMSE
    let mut mse = 0.0f64;
    for i in 0..h { for j in 0..w { 
        let mut pred_sum = 0.0f32; 
        for s in 0..seeds.len() { 
            pred_sum += bit_engine::compute_fused_output(&Packed256Params { 
                r: seeds[s].get_r(), theta: seeds[s].get_theta(), param1: seeds[s].get_param1(), param2: seeds[s].get_param2(), 
                basis_id: seeds[s].get_basis_id(), d_r: seeds[s].get_d_r(), d_theta: seeds[s].get_d_theta(), log2_c: seeds[s].get_log2_c(),
                activation_id: seeds[s].get_activation_id(), q_value: seeds[s].get_q_value(), k_value: seeds[s].get_k_value(), flags: seeds[s].get_flags(),
            }, i, j, h, w).predicted_value; 
        }
        let e = (pred_sum - target[i*w + j]) as f64; mse += e*e; 
    }}
    let rmse = (mse / (h*w) as f64).sqrt();
    println!("[HybridNG-Drive] final rmse={:.10}", rmse);
}



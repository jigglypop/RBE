use crate::core::differential::bit_engine;
use crate::core::differential::bit_engine::{compute_b12_with_y_overrides, compute_b13_rank1_with_y_overrides, compute_b14_cartesian_with_y_overrides};
use crate::core::tensors::{packed256_types::Packed256Params, Packed256};

/// Optimizer configuration
#[derive(Clone, Debug)]
pub struct HybridNGConfig {
    pub steps: usize,
    pub batch_size: usize,
    pub learning_rate_r: f32,
    pub learning_rate_theta: f32,
    pub learning_rate_p1: f32,
    pub learning_rate_p2: f32,
    pub learning_rate_y: f32,
    pub grad_clip: f32,
    pub seed: u64,
}

impl Default for HybridNGConfig {
    fn default() -> Self {
        Self {
            steps: 100,
            batch_size: 8192,
            learning_rate_r: 2e-3,
            learning_rate_theta: 2e-3,
            learning_rate_p1: 5e-4,
            learning_rate_p2: 1e-3,
            learning_rate_y: 1e-2,
            grad_clip: 0.1,
            seed: 42,
        }
    }
}

/// Simple hybrid natural-gradient optimizer
pub struct HybridNaturalOptimizer {
    cfg: HybridNGConfig,
}

impl HybridNaturalOptimizer {
    pub fn new(cfg: HybridNGConfig) -> Self { Self { cfg } }

    /// Optimize k seeds on a single tile to fit weights (no residual basis)
    /// - seeds: length k, updated in-place
    /// - weights: tile weights in row-major (rows*cols)
    pub fn optimize_tile(
        &self,
        seeds: &mut [Packed256],
        weights: &[f32],
        rows: usize,
        cols: usize,
    ) {
        if rows == 0 || cols == 0 || seeds.is_empty() { return; }
        let total = rows * cols;
        let mut rng = SplitMix64::new(self.cfg.seed);

        // Continuous shadow parameters to avoid Q24.8 rounding during optimization
        let k = seeds.len();
        let mut curr_r: Vec<f32> = (0..k).map(|s| seeds[s].get_r()).collect();
        // Reparameterize r via u = atanh(r) for stable updates in hyperbolic geometry
        let mut curr_u: Vec<f32> = curr_r.iter().map(|&r| {
            // clamp r to avoid INF in atanh
            let rr = r.clamp(-0.9999, 0.9999);
            libm::atanhf(rr)
        }).collect();
        let mut curr_th: Vec<f32> = (0..k).map(|s| seeds[s].get_theta()).collect();
        let mut curr_p1: Vec<f32> = (0..k).map(|s| seeds[s].get_param1()).collect();
        let mut curr_p2: Vec<f32> = (0..k).map(|s| seeds[s].get_param2()).collect();
        // New: continuous y-axis overrides for basis 12/13
        let mut curr_oy: Vec<f32> = (0..k).map(|s| (seeds[s].get_q_value() as f32) / 255.0).collect();
        let mut curr_py: Vec<f32> = (0..k).map(|s| {
            let bits = (seeds[s].get_flags() >> 4) & 0b11;
            match bits { 0 => 0.0, 1 => 0.5*std::f32::consts::PI, 2 => std::f32::consts::PI, _ => 1.5*std::f32::consts::PI }
        }).collect();

        for _step in 0..self.cfg.steps {
            // Sync r from u before each step
            for s in 0..k { curr_r[s] = curr_u[s].tanh(); }

            // Accumulate grads per seed (operate in u-space for r)
            let k = seeds.len();
            let mut sum_grad_r = vec![0.0f32; k]; // kept for readability; used as du accumulator
            let mut sum_grad_th = vec![0.0f32; k]; // will accumulate NG(theta_scale)+phi
            let mut sum_grad_p1 = vec![0.0f32; k];
            let mut sum_grad_p2 = vec![0.0f32; k];
            let mut sum_grad_oy = vec![0.0f32; k];
            let mut sum_grad_py = vec![0.0f32; k];

            let batch = self.cfg.batch_size.min(total);
            // Optional Huber delta: passed through learning_rate_p2 as sentinel? keep L2 here (Huber handled in driver RMSE only)
            if batch == total {
                // full sweep: deterministic, low-variance
                for i in 0..rows { for j in 0..cols {
                    let idx = i*cols + j;
                    // forward sum across seeds
                    let mut pred_sum = 0.0f32;
                    let mut outs: Vec<bit_engine::EngineOutput> = Vec::with_capacity(k);
                    for s in 0..k {
                        let mut params = to_params(&seeds[s]);
                        params.r = curr_r[s]; params.theta = curr_th[s]; params.param1 = curr_p1[s]; params.param2 = curr_p2[s];
                        if params.basis_id == 12 {
                            let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                            let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                            let (out, _goy, _gpy) = compute_b12_with_y_overrides(&params, r_coord, th_coord, curr_oy[s], curr_py[s]);
                            pred_sum += out.predicted_value; outs.push(out);
                        } else if params.basis_id == 13 {
                            let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                            let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                            let (out, _goy, _gpy) = compute_b13_rank1_with_y_overrides(&params, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]);
                            pred_sum += out.predicted_value; outs.push(out);
                        } else if params.basis_id == 14 {
                            let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                            let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                            let (out, _goy, _gpy) = compute_b14_cartesian_with_y_overrides(&params, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]);
                            pred_sum += out.predicted_value; outs.push(out);
                        } else {
                            let out = bit_engine::compute_fused_output(&params, i, j, rows, cols);
                            pred_sum += out.predicted_value; outs.push(out);
                        }
                    }
                    let y = unsafe { *weights.get_unchecked(idx) };
                    let err = pred_sum - y; // optimizer keeps L2 gradient; robust scoring handled at evaluation layer

                    for s in 0..k {
                        let r0 = curr_r[s];
                        let one_minus_r2 = (1.0 - r0 * r0).max(1e-6);
                        let mut scale_theta = (one_minus_r2 * one_minus_r2) / (4.0 * (r0 * r0 + 1e-9));
                        scale_theta = scale_theta.clamp(1e-3, 1e3);
                        // NG scaling only for theta_scale path; phi path remains unscaled
                        let ng_theta = outs[s].grad_theta_scale * scale_theta + outs[s].grad_phi;
                        // chain to u: dL/du = dL/dr * (1 - r^2)
                        sum_grad_r[s] += err * outs[s].grad_r * one_minus_r2;
                        sum_grad_th[s] += err * ng_theta;
                        sum_grad_p1[s] += err * outs[s].grad_p1;
                        sum_grad_p2[s] += err * outs[s].grad_p2;
                        // extra y-axis grads available only if basis 12; approximate via recomputation around stored out
                        if {
                            let b = to_params(&seeds[s]).basis_id;
                            b == 12 || b == 13 || b == 14
                        } {
                            // recompute y-axis grads via specialized path to get goy,gpy
                            let mut p = to_params(&seeds[s]); p.r = curr_r[s]; p.theta = curr_th[s]; p.param1 = curr_p1[s]; p.param2 = curr_p2[s];
                            let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                            let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                            let (_out2, goy, gpy) = if p.basis_id == 12 {
                                compute_b12_with_y_overrides(&p, r_coord, th_coord, curr_oy[s], curr_py[s])
                            } else if p.basis_id == 13 {
                                let (o2, gy, gp) = compute_b13_rank1_with_y_overrides(&p, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]);
                                (o2, gy*8.0, gp)
                            } else { // basis 14
                                let two_pi = 2.0f32*std::f32::consts::PI;
                                let (o2, gy, gp) = compute_b14_cartesian_with_y_overrides(&p, r_coord, th_coord, two_pi*8.0*curr_oy[s], curr_py[s]);
                                (o2, gy*(two_pi*8.0), gp)
                            };
                            sum_grad_oy[s] += err * goy;
                            sum_grad_py[s] += err * gpy;
                        }
                    }
                }}
            } else {
                // stochastic sampling
            for _ in 0..batch {
                let idx = (rng.next_u64() as usize) % total;
                let i = idx / cols;
                let j = idx % cols;

                // forward sum across seeds
                let mut pred_sum = 0.0f32;
                let mut outs: Vec<bit_engine::EngineOutput> = Vec::with_capacity(k);
                for s in 0..k {
                    let mut params = to_params(&seeds[s]);
                    params.r = curr_r[s]; params.theta = curr_th[s]; params.param1 = curr_p1[s]; params.param2 = curr_p2[s];
                    let out = if params.basis_id == 12 {
                        let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                        let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                        let (o,_,_) = compute_b12_with_y_overrides(&params, r_coord, th_coord, curr_oy[s], curr_py[s]);
                        o
                    } else if params.basis_id == 13 {
                        let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                        let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                        let (o,_,_) = compute_b13_rank1_with_y_overrides(&params, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]);
                        o
                    } else if params.basis_id == 14 {
                        let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                        let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                        let (o,_,_) = compute_b14_cartesian_with_y_overrides(&params, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]);
                        o
                    } else {
                        bit_engine::compute_fused_output(&params, i, j, rows, cols)
                    };
                    pred_sum += out.predicted_value;
                    outs.push(out);
                }
                let y = unsafe { *weights.get_unchecked(idx) };
                let err = pred_sum - y; // d(0.5*e^2)/dp = e (Huber not applied in gradient by default)

                // accumulate grads per seed with Riemannian scaling for (r,theta)
                for s in 0..k {
                        let r0 = curr_r[s];
                    let one_minus_r2 = (1.0 - r0 * r0).max(1e-6);
                    let mut scale_theta = (one_minus_r2 * one_minus_r2) / (4.0 * (r0 * r0 + 1e-9));
                    scale_theta = scale_theta.clamp(1e-3, 1e3);
                        let ng_theta = outs[s].grad_theta_scale * scale_theta + outs[s].grad_phi;
                        sum_grad_r[s] += err * outs[s].grad_r * one_minus_r2;
                        sum_grad_th[s] += err * ng_theta;
                    sum_grad_p1[s] += err * outs[s].grad_p1;
                    sum_grad_p2[s] += err * outs[s].grad_p2;
                    }
                }
            }

            // apply updates (normalize by actual sampled batch size)
            let bsz_inv = 1.0f32 / (batch as f32).max(1.0);
            // per-seed averaged gradients
            let mut avg_r: Vec<f32> = Vec::with_capacity(k);
            let mut avg_th: Vec<f32> = Vec::with_capacity(k);
            let mut avg_p1: Vec<f32> = Vec::with_capacity(k);
            let mut avg_p2: Vec<f32> = Vec::with_capacity(k);
            for s in 0..k {
                avg_r.push(sum_grad_r[s] * bsz_inv);
                avg_th.push(sum_grad_th[s] * bsz_inv);
                avg_p1.push(sum_grad_p1[s] * bsz_inv);
                avg_p2.push(sum_grad_p2[s] * bsz_inv);
            }
            // per-seed adaptive inverse scales (avoid cross-seed coupling)
            let mut inv_r_s = vec![1.0f32; k];
            let mut inv_th_s = vec![1.0f32; k];
            let mut inv_p1_s = vec![1.0f32; k];
            let mut inv_p2_s = vec![1.0f32; k];
            for s in 0..k {
                inv_r_s[s] = (1.0 / (avg_r[s].abs() + 1e-8)).min(10.0);
                inv_th_s[s] = (1.0 / (avg_th[s].abs() + 1e-8)).min(50.0);
                inv_p1_s[s] = (1.0 / (avg_p1[s].abs() + 1e-8)).min(10.0);
                inv_p2_s[s] = (1.0 / (avg_p2[s].abs() + 1e-8)).min(10.0);
            }
            // debug aggregates
            let rms = |v: &Vec<f32>| -> f32 {
                if v.is_empty() { return 0.0; }
                let mut acc = 0.0f64; for &x in v { acc += (x as f64)*(x as f64); }
                ((acc / (v.len() as f64)).sqrt()) as f32
            };
            let rms_r = rms(&avg_r); let rms_th = rms(&avg_th); let rms_p1 = rms(&avg_p1); let rms_p2 = rms(&avg_p2);

            let mut dbg_step0: Option<(f32,f32,f32,f32)> = None; // (dr,dth,dp1,dp2) after averaging+clip
            // Helper: wrap angle to [0, 2π)
            let wrap_0_2pi = |theta: f32| -> f32 {
                let two_pi = 2.0f32 * std::f32::consts::PI;
                let t = theta.rem_euclid(two_pi);
                if t < 0.0 { t + two_pi } else { t }
            };

            for s in 0..seeds.len() {
                let mut dr = avg_r[s] * inv_r_s[s];
                let mut dth = avg_th[s] * inv_th_s[s];
                let mut dp1 = avg_p1[s] * inv_p1_s[s];
                let mut dp2 = avg_p2[s] * inv_p2_s[s];
                let mut doy = sum_grad_oy[s] * bsz_inv * 10.0; // modest scale
                let mut dpy = sum_grad_py[s] * bsz_inv * 10.0;
                // clip
                let c = self.cfg.grad_clip;
                dr = dr.clamp(-c, c); dth = dth.clamp(-c, c); dp1 = dp1.clamp(-c, c); dp2 = dp2.clamp(-c, c);
                doy = doy.clamp(-c, c); dpy = dpy.clamp(-c, c);
                if s == 0 { dbg_step0 = Some((dr,dth,dp1,dp2)); }

                let r0 = curr_r[s];
                let th0 = curr_th[s];
                let p1_0 = curr_p1[s];
                let p2_0 = curr_p2[s];

                // update u and immediately reflect r via tanh(u) for this step
                let u_new = curr_u[s] - self.cfg.learning_rate_r * dr;
                curr_u[s] = u_new;
                let r_new = curr_u[s].tanh();
                let th_new = wrap_0_2pi(th0 - self.cfg.learning_rate_theta * dth);
                let p1_new = (p1_0 - self.cfg.learning_rate_p1 * dp1).clamp(-16.0, 16.0);
                let p2_new = (p2_0 - self.cfg.learning_rate_p2 * dp2).clamp(-4.0, 4.0);
                curr_r[s] = r_new;
                curr_th[s] = th_new;
                curr_p1[s] = p1_new;
                curr_p2[s] = p2_new;
                // update y-axis if basis 12
                if to_params(&seeds[s]).basis_id == 12 || to_params(&seeds[s]).basis_id == 13 {
                    curr_oy[s] = (curr_oy[s] - self.cfg.learning_rate_y * doy).clamp(0.0, 1.0);
                    curr_py[s] = wrap_0_2pi(curr_py[s] - self.cfg.learning_rate_y * dpy);
                }
            }

            // Joint amplitude refit (all seeds at once) via normal equations (F^T F) a = F^T y
            // Column-normalized for conditioning: solve on F' = F * D^{-1}, then a = D^{-1} a'
            // Improves RMSE by resolving inter-seed interference in amplitudes
            let k = seeds.len();
            // Base functions with param2=1.0 for each seed
            let mut base_all = vec![0.0f32; total*k];
            for s in 0..k {
                let mut params = to_params(&seeds[s]);
                params.r = curr_r[s]; params.theta = curr_th[s]; params.param1 = curr_p1[s]; params.param2 = 1.0;
                for i in 0..rows { for j in 0..cols {
                    let idx = i*cols + j;
                    if params.basis_id == 12 {
                        let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                        let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                        let (out,_,_) = compute_b12_with_y_overrides(&params, r_coord, th_coord, curr_oy[s], curr_py[s]);
                        base_all[s*total + idx] = out.predicted_value;
                    } else if params.basis_id == 13 {
                        let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                        let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                        let (out,_,_) = compute_b13_rank1_with_y_overrides(&params, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]);
                        base_all[s*total + idx] = out.predicted_value;
                    } else if params.basis_id == 14 {
                        let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                        let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                        let (out,_,_) = compute_b14_cartesian_with_y_overrides(&params, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]);
                        base_all[s*total + idx] = out.predicted_value;
                    } else {
                        base_all[s*total + idx] = bit_engine::compute_fused_output(&params, i, j, rows, cols).predicted_value;
                    }
                }}
            }
            // QR (modified Gram–Schmidt) on F (columns: base_all with param2=1)
            let mut qmat = vec![0.0f64; total * k];  // column-major Q (orthonormal)
            let mut rmat = vec![0.0f64; k * k];      // upper-triangular R
            // Copy F into Q as starting columns
            for s in 0..k {
                for i in 0..total { qmat[s*total + i] = base_all[s*total + i] as f64; }
                // Orthogonalize against previous columns
                for t in 0..s {
                    // r_{t,s} = q_t^T q_s
                    let mut dot = 0.0f64;
                    for i in 0..total { dot += qmat[t*total + i] * qmat[s*total + i]; }
                    rmat[t*k + s] = dot;
                    for i in 0..total { qmat[s*total + i] -= dot * qmat[t*total + i]; }
                }
                // r_{s,s} = ||q_s||
                let mut norm2 = 0.0f64; for i in 0..total { norm2 += qmat[s*total + i] * qmat[s*total + i]; }
                let norm = norm2.sqrt().max(1e-12);
                rmat[s*k + s] = norm;
                for i in 0..total { qmat[s*total + i] /= norm; }
            }
            // c = Q^T y
            let mut c = vec![0.0f64; k];
            for s in 0..k {
                let mut dot = 0.0f64;
                for i in 0..total { dot += qmat[s*total + i] * (unsafe { *weights.get_unchecked(i) } as f64); }
                c[s] = dot;
            }
            // Back solve R a = c
            let mut a = vec![0.0f64; k];
            for i in (0..k).rev() {
                let mut sum = c[i];
                for j in (i+1)..k { sum -= rmat[i*k + j] * a[j]; }
                a[i] = sum / rmat[i*k + i].max(1e-12);
            }
            for s in 0..k { curr_p2[s] = (a[s] as f32).clamp(-4.0, 4.0); }

            // Closed-form phase refit for basis 12 (φx only), with fixed amplitudes
            // r = y - (sum_total - x_part); solve α,β in r ≈ W*(sinC*α + cosC*β), then φ = atan2(β, α)
            // Recompute total prediction with current params
            let mut pred_sum_total = vec![0.0f32; total];
            for i in 0..rows { for j in 0..cols {
                let idx = i*cols + j;
                let mut ssum = 0.0f32;
                for s in 0..k {
                    let mut p = to_params(&seeds[s]);
                    p.r = curr_r[s]; p.theta = curr_th[s]; p.param1 = curr_p1[s]; p.param2 = curr_p2[s];
                    if p.basis_id == 12 {
                        let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                        let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                        let (out,_,_) = compute_b12_with_y_overrides(&p, r_coord, th_coord, curr_oy[s], curr_py[s]);
                        ssum += out.predicted_value;
                    } else if p.basis_id == 13 {
                        let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                        let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                        let (out,_,_) = compute_b13_rank1_with_y_overrides(&p, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]);
                        ssum += out.predicted_value;
                    } else {
                        ssum += bit_engine::compute_fused_output(&p, i, j, rows, cols).predicted_value;
                    }
                }
                pred_sum_total[idx] = ssum;
            }}
            let two_pi = 2.0f32 * std::f32::consts::PI;
            for s in 0..k {
                // Only basis 12 supports φx decomposition here
                let params0 = to_params(&seeds[s]);
                if params0.basis_id != 12 { continue; }
                let omega_x = curr_p1[s];
                let phi_x = curr_th[s];
                let a_x = (params0.k_value as f32) / 255.0;
                let a_y = (params0.activation_id as f32) / 255.0;
                    let omega_y = curr_oy[s];
                    let phi_y = curr_py[s];

                let p2_s = curr_p2[s];
                let r_scale = curr_r[s];
                // Accumulators for normal eq
                let mut s11 = 0.0f64; let mut s22 = 0.0f64; let mut s12 = 0.0f64; let mut t1 = 0.0f64; let mut t2 = 0.0f64;
                for i in 0..rows { for j in 0..cols {
                    let idx = i*cols + j;
                    // coords
                    let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                    let theta_coord = if cols > 0 { two_pi * (j as f32) / (cols as f32) } else { 0.0 };
                    let r_eff = r_scale * r_coord;
                    let theta_eff = theta_coord; // basis 12 uses theta_scale=1
                    // metric
                    let c = 2.0_f32.powi(params0.log2_c as i32);
                    let denom = 1.0 - c * r_eff * r_eff;
                    if denom <= 1e-8 { continue; }
                    let metric = 1.0 / denom;
                    // current x- and y- parts
                    let a_i = omega_x * r_eff;
                    let sin_a = a_i.sin();
                    let cos_a = a_i.cos();
                    let x_weight = (p2_s * a_x * metric) as f64;
                    let x_part_curr = x_weight as f64 * (sin_a as f64 * (phi_x.cos() as f64) + cos_a as f64 * (phi_x.sin() as f64));
                    let _y_part_curr = (p2_s * a_y * metric) as f64 * ((omega_y * theta_eff + phi_y).cos() as f64);
                    let pred_total = pred_sum_total[idx] as f64;
                    let y_true = unsafe { *weights.get_unchecked(idx) } as f64;
                    // residual target for x part only
                    let r_target = y_true - (pred_total - x_part_curr);
                    // design entries
                    let wu = x_weight * (sin_a as f64);
                    let wv = x_weight * (cos_a as f64);
                    s11 += wu * wu; s22 += wv * wv; s12 += wu * wv;
                    t1 += wu * r_target; t2 += wv * r_target;
                }}
                let det = s11*s22 - s12*s12;
                if det.abs() > 1e-18 {
                    let inv11 =  s22 / det;
                    let inv12 = -s12 / det;
                    let inv22 =  s11 / det;
                    let alpha = inv11 * t1 + inv12 * t2; // ≈ cos φx
                    let beta  = inv12 * t1 + inv22 * t2; // ≈ sin φx
                    let new_phi = libm::atan2f(beta as f32, alpha as f32);
                    curr_th[s] = wrap_0_2pi(new_phi);
                    // refresh pred_sum_total to reflect updated φx for consistency
                    for i in 0..rows { for j in 0..cols {
                        let idx = i*cols + j;
                        let mut p = to_params(&seeds[s]);
                        p.r = curr_r[s]; p.theta = curr_th[s]; p.param1 = curr_p1[s]; p.param2 = curr_p2[s];
                        let new_contrib = {
                            if p.basis_id == 12 {
                                let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                let (out,_,_) = compute_b12_with_y_overrides(&p, r_coord, th_coord, curr_oy[s], curr_py[s]);
                                out.predicted_value
                            } else if p.basis_id == 13 {
                                let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                let (out,_,_) = compute_b13_rank1_with_y_overrides(&p, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]);
                                out.predicted_value
                            } else { bit_engine::compute_fused_output(&p, i, j, rows, cols).predicted_value }
                        };
                        // compute old contrib with old phi_x
                        let old_phi = phi_x;
                        let mut p_old = to_params(&seeds[s]);
                        p_old.r = r_scale; p_old.theta = old_phi; p_old.param1 = omega_x; p_old.param2 = p2_s;
                        let old_contrib = {
                            if p_old.basis_id == 12 {
                                let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                let (out,_,_) = compute_b12_with_y_overrides(&p_old, r_coord, th_coord, curr_oy[s], curr_py[s]);
                                out.predicted_value
                            } else if p_old.basis_id == 13 {
                                let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                let (out,_,_) = compute_b13_rank1_with_y_overrides(&p_old, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]);
                                out.predicted_value
                            } else { bit_engine::compute_fused_output(&p_old, i, j, rows, cols).predicted_value }
                        };
                        pred_sum_total[idx] += (new_contrib - old_contrib);
                    }}
                }
            }

            // Closed-form φx refit for basis 13 (rank-1), with fixed amplitudes and current φy
            for s in 0..k {
                let params0 = to_params(&seeds[s]);
                if params0.basis_id != 13 { continue; }
                let omega_x = curr_p1[s];
                let phi_x_old = curr_th[s];
                let p2_s = curr_p2[s];
                let r_scale = curr_r[s];
                let omega_y = 8.0 * curr_oy[s];
                let phi_y = curr_py[s];
                let two_pi = 2.0f32 * std::f32::consts::PI;

                let mut s11 = 0.0f64; let mut s22 = 0.0f64; let mut s12 = 0.0f64; let mut t1 = 0.0f64; let mut t2 = 0.0f64;
                for i in 0..rows { for j in 0..cols {
                    let idx = i*cols + j;
                    let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                    let theta_coord = if cols > 0 { two_pi * (j as f32) / (cols as f32) } else { 0.0 };
                    let r_eff = r_scale * r_coord;
                    let c = 2.0_f32.powi(params0.log2_c as i32);
                    let denom = 1.0 - c * r_eff * r_eff; if denom <= 1e-8 { continue; }
                    let metric = 1.0 / denom;
                    let inner_x = omega_x * r_eff;
                    let sin_x = inner_x.sin();
                    let cos_x = inner_x.cos();
                    let cos_y = (omega_y * theta_coord + phi_y).cos();
                    let w = (p2_s * metric * cos_y) as f64;
                    let pred_total = pred_sum_total[idx] as f64;
                    let y_true = unsafe { *weights.get_unchecked(idx) } as f64;
                    let x_part_curr = w * (sin_x as f64 * (phi_x_old.cos() as f64) + cos_x as f64 * (phi_x_old.sin() as f64));
                    let r_target = y_true - (pred_total - x_part_curr);
                    let u = w * (sin_x as f64);
                    let v = w * (cos_x as f64);
                    s11 += u*u; s22 += v*v; s12 += u*v; t1 += u * r_target; t2 += v * r_target;
                }}
                let det = s11*s22 - s12*s12;
                if det.abs() > 1e-18 {
                    let inv11 =  s22 / det;
                    let inv12 = -s12 / det;
                    let inv22 =  s11 / det;
                    let alpha = inv11 * t1 + inv12 * t2;
                    let beta  = inv12 * t1 + inv22 * t2;
                    let new_phi_x = libm::atan2f(beta as f32, alpha as f32);
                    curr_th[s] = wrap_0_2pi(new_phi_x);
                    // Refresh pred_sum_total efficiently
                    let phi_old = phi_x_old; let phi_new = curr_th[s];
                    for i in 0..rows { for j in 0..cols {
                        let idx = i*cols + j;
                        let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                        let theta_coord = if cols > 0 { two_pi * (j as f32) / (cols as f32) } else { 0.0 };
                        let r_eff = r_scale * r_coord;
                        let c = 2.0_f32.powi(params0.log2_c as i32);
                        let denom = 1.0 - c * r_eff * r_eff; if denom <= 1e-8 { continue; }
                        let metric = 1.0 / denom;
                        let inner_x = omega_x * r_eff;
                        let sin_x = inner_x.sin();
                        let cos_x = inner_x.cos();
                        let cos_y = (omega_y * theta_coord + phi_y).cos();
                        let w = (p2_s * metric * cos_y) as f64;
                        let old_contrib = w * (sin_x as f64 * (phi_old.cos() as f64) + cos_x as f64 * (phi_old.sin() as f64));
                        let new_contrib = w * (sin_x as f64 * (phi_new.cos() as f64) + cos_x as f64 * (phi_new.sin() as f64));
                        pred_sum_total[idx] = (pred_sum_total[idx] as f64 + (new_contrib - old_contrib)) as f32;
                    }}
                }
            }

            // Closed-form φy refit for basis 13/14 with fixed amplitudes and current φx
            // Model: r ≈ w * (α*cos(ωy*v) + β*sin(ωy*v)), φy = atan2(-β, α)
            // where w = p2 * metric * sin_x and v is θ (basis 13) or y (basis 14)
            for s in 0..k {
                let params0 = to_params(&seeds[s]);
                if !(params0.basis_id == 13 || params0.basis_id == 14) { continue; }
                let omega_x = curr_p1[s];
                let phi_x = curr_th[s];
                let p2_s = curr_p2[s];
                let r_scale = curr_r[s];
                let omega_y = 8.0 * curr_oy[s];
                let phi_y_old = curr_py[s];

                // Accumulators
                let mut s11 = 0.0f64; let mut s22 = 0.0f64; let mut s12 = 0.0f64; let mut t1 = 0.0f64; let mut t2 = 0.0f64;
                for i in 0..rows { for j in 0..cols {
                    let idx = i*cols + j;
                    // coords
                    let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                    let theta_coord = if cols > 0 { two_pi * (j as f32) / (cols as f32) } else { 0.0 };
                    // effective inputs per basis
                    let (x_pos, y_arg, r_eff) = if params0.basis_id == 13 {
                        let r_eff = r_scale * r_coord; // θ_eff = theta_coord
                        (r_eff, theta_coord, r_eff)
                    } else {
                        // basis 14: x = r_coord, y = theta/(2π)
                        let x = r_coord;
                        let y = theta_coord / two_pi;
                        (x, y, x)
                    };
                    // metric (consistent with compute paths)
                    let c = 2.0_f32.powi(params0.log2_c as i32);
                    let denom = 1.0 - c * r_eff * r_eff;
                    if denom <= 1e-8 { continue; }
                    let metric = 1.0 / denom;
                    // x part
                    let a_i = omega_x * x_pos + phi_x;
                    let sin_x = a_i.sin();
                    // weight
                    let w = (p2_s * metric * sin_x) as f64;
                    // current total prediction and target for this seed's y part only
                    let pred_total = pred_sum_total[idx] as f64;
                    let y_true = unsafe { *weights.get_unchecked(idx) } as f64;
                    let r_target = y_true - (pred_total - (w * ((omega_y as f64 * y_arg as f64 + phi_y_old as f64).cos())));
                    // design
                    let cy = (omega_y * y_arg).cos() as f64;
                    let sy = (omega_y * y_arg).sin() as f64;
                    let u = w * cy;
                    let v = w * sy;
                    s11 += u*u; s22 += v*v; s12 += u*v;
                    t1 += u * r_target; t2 += v * r_target;
                }}
                let det = s11*s22 - s12*s12;
                if det.abs() > 1e-18 {
                    let inv11 =  s22 / det;
                    let inv12 = -s12 / det;
                    let inv22 =  s11 / det;
                    let alpha = inv11 * t1 + inv12 * t2; // ≈ cos φy
                    let beta  = inv12 * t1 + inv22 * t2; // ≈ (-sin φy) up to sign handled below
                    let new_phi_y = libm::atan2f((-beta) as f32, alpha as f32);
                    curr_py[s] = {
                        let two_pi = 2.0*std::f32::consts::PI; let t = new_phi_y.rem_euclid(two_pi); if t < 0.0 { t + two_pi } else { t }
                    };
                    // Refresh pred_sum_total to reflect updated φy efficiently
                    let phi_old = phi_y_old;
                    let phi_new = curr_py[s];
                    for i in 0..rows { for j in 0..cols {
                        let idx = i*cols + j;
                        let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                        let theta_coord = if cols > 0 { two_pi * (j as f32) / (cols as f32) } else { 0.0 };
                        // recompute w and y_arg
                        let (x_pos, y_arg, r_eff) = if params0.basis_id == 13 {
                            let r_eff = r_scale * r_coord; (r_eff, theta_coord, r_eff)
                        } else { let x = r_coord; let y = theta_coord / two_pi; (x, y, x) };
                        let c = 2.0_f32.powi(params0.log2_c as i32);
                        let denom = 1.0 - c * r_eff * r_eff; if denom <= 1e-8 { continue; }
                        let metric = 1.0 / denom;
                        let sin_x = (omega_x * x_pos + phi_x).sin();
                        let w = (p2_s * metric * sin_x) as f64;
                        let old_contrib = w * ((omega_y as f64 * y_arg as f64 + phi_old as f64).cos());
                        let new_contrib = w * ((omega_y as f64 * y_arg as f64 + phi_new as f64).cos());
                        pred_sum_total[idx] = (pred_sum_total[idx] as f64 + (new_contrib - old_contrib)) as f32;
                    }}
                }
            }

            // 1D Gauss-Newton update for ωy (basis 13/14) with backtracking line search
            // Δωy = - (Σ e*g) / (Σ g^2), g = ∂pred/∂ωy from specialized path
            {
                // recompute pred_sum_total consistent with latest params
                let mut pred_sum_total = vec![0.0f32; total];
                for i in 0..rows { for j in 0..cols {
                    let idx = i*cols + j;
                    let mut ssum = 0.0f32;
                    for s in 0..k {
                        let mut p = to_params(&seeds[s]);
                        p.r = curr_r[s]; p.theta = curr_th[s]; p.param1 = curr_p1[s]; p.param2 = curr_p2[s];
                        if p.basis_id == 12 {
                            let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                            let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                            ssum += compute_b12_with_y_overrides(&p, r_coord, th_coord, curr_oy[s], curr_py[s]).0.predicted_value;
                        } else if p.basis_id == 13 {
                            let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                            let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                            ssum += compute_b13_rank1_with_y_overrides(&p, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]).0.predicted_value;
                        } else if p.basis_id == 14 {
                            let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                            let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                            ssum += compute_b14_cartesian_with_y_overrides(&p, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]).0.predicted_value;
                        } else {
                            ssum += bit_engine::compute_fused_output(&p, i, j, rows, cols).predicted_value;
                        }
                    }
                    pred_sum_total[idx] = ssum;
                }}

                for s in 0..k {
                    let params0 = to_params(&seeds[s]);
                    if !(params0.basis_id == 13 || params0.basis_id == 14) { continue; }
                    let two_pi = 2.0f32 * std::f32::consts::PI;
                    let mut num: f64 = 0.0; let mut den: f64 = 0.0;
                    for i in 0..rows { for j in 0..cols {
                        let idx = i*cols + j;
                        let e = (pred_sum_total[idx] - unsafe { *weights.get_unchecked(idx) }) as f64;
                        let mut p = params0;
                        p.r = curr_r[s]; p.theta = curr_th[s]; p.param1 = curr_p1[s]; p.param2 = curr_p2[s];
                        let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                        let th_coord = if cols > 0 { two_pi * (j as f32) / (cols as f32) } else { 0.0 };
                        let (out, gy, _gpy) = if p.basis_id == 13 {
                            compute_b13_rank1_with_y_overrides(&p, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s])
                        } else {
                            compute_b14_cartesian_with_y_overrides(&p, r_coord, th_coord, two_pi*8.0*curr_oy[s], curr_py[s])
                        };
                        let g = if p.basis_id == 13 { gy as f64 * 1.0 } else { gy as f64 * (two_pi*8.0) as f64 };
                        // Note: above gy is ∂pred/∂(override) where override was given directly; for b14 we scaled input by two_pi*8.0, chain back
                        num += e * g; den += g * g;
                        let _ = out; // silence warning in non-debug
                    }}
                    if den > 1e-18 {
                        let mut delta = (-num / den) as f32; // GN step
                        // backtracking line search
                        let mut scale = 1.0f32; let mut accepted = false;
                        let omega_y0 = 8.0 * curr_oy[s];
                        let phi_y = curr_py[s];
                        // Evaluate baseline rmse
                        let mut se0 = 0.0f64; for idx in 0..total { let e = (pred_sum_total[idx] - unsafe { *weights.get_unchecked(idx) }) as f64; se0 += e*e; }
                        let rmse0 = (se0 / (total as f64)).sqrt();
                        for _try in 0..6 {
                            let omega_y_try = omega_y0 + scale * delta;
                            let oy_try = (omega_y_try / 8.0).clamp(0.0, 1.0);
                            // compute rmse with only this seed's ωy changed
                            let mut se_new = 0.0f64;
                            for i in 0..rows { for j in 0..cols {
                                let idx = i*cols + j;
                                // start with base pred
                                let mut ssum = pred_sum_total[idx] as f64;
                                // remove old y-part of seed s and add new y-part
                                let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                let theta_coord = if cols > 0 { two_pi * (j as f32) / (cols as f32) } else { 0.0 };
                                // compute w and sin_x
                                let r_eff = curr_r[s] * r_coord;
                                let c = 2.0_f32.powi(params0.log2_c as i32);
                                let denom = 1.0 - c * r_eff * r_eff; if denom <= 1e-8 { continue; }
                                let metric = 1.0 / denom;
                                let sin_x = (curr_p1[s] * r_eff + curr_th[s]).sin();
                                let w = (curr_p2[s] * metric * sin_x) as f64;
                                // basis-dependent y-arg
                                let y_arg = if params0.basis_id == 13 { theta_coord as f64 } else { (theta_coord / two_pi) as f64 };
                                let old_oy = 8.0 * curr_oy[s];
                                let old_contrib = w * ((old_oy as f64 * y_arg + phi_y as f64).cos());
                                let new_contrib = w * ((omega_y_try as f64 * y_arg + phi_y as f64).cos());
                                ssum += new_contrib - old_contrib;
                                let e = ssum as f64 - (unsafe { *weights.get_unchecked(idx) } as f64);
                                se_new += e*e;
                            }}
                            let rmse_new = (se_new / (total as f64)).sqrt();
                            if rmse_new < rmse0 { accepted = true; curr_oy[s] = oy_try; break; }
                            scale *= 0.5;
                        }
                        if accepted {
                            // refresh pred_sum_total for next seeds after acceptance
                            for i in 0..rows { for j in 0..cols {
                                let idx = i*cols + j;
                                let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                let theta_coord = if cols > 0 { two_pi * (j as f32) / (cols as f32) } else { 0.0 };
                                let r_eff = curr_r[s] * r_coord;
                                let c = 2.0_f32.powi(params0.log2_c as i32);
                                let denom = 1.0 - c * r_eff * r_eff; if denom <= 1e-8 { continue; }
                                let metric = 1.0 / denom;
                                let sin_x = (curr_p1[s] * r_eff + curr_th[s]).sin();
                                let w = (curr_p2[s] * metric * sin_x) as f64;
                                let y_arg = if params0.basis_id == 13 { theta_coord as f64 } else { (theta_coord / two_pi) as f64 };
                                let old_oy = omega_y0 as f64; let new_oy = (8.0*curr_oy[s]) as f64;
                                let old_contrib = w * ((old_oy * y_arg + phi_y as f64).cos());
                                let new_contrib = w * ((new_oy * y_arg + phi_y as f64).cos());
                                pred_sum_total[idx] = (pred_sum_total[idx] as f64 + (new_contrib - old_contrib)) as f32;
                            }}
                        }
                    }
                }
            }

            // Periodic Gauss-Newton refinement for (r, theta, p1) with LM line search
            if ((_step + 1) % 25) == 0 {
                // Recompute total prediction with current continuous params (fresh residual)
                let mut pred_sum_total = vec![0.0f32; total];
                for i in 0..rows { for j in 0..cols {
                    let idx = i*cols + j;
                    let mut ssum = 0.0f32;
                    for s in 0..k {
                        let mut p = to_params(&seeds[s]);
                        p.r = curr_r[s]; p.theta = curr_th[s]; p.param1 = curr_p1[s]; p.param2 = curr_p2[s];
                        if p.basis_id == 12 {
                            let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                            let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                            let (out,_,_) = compute_b12_with_y_overrides(&p, r_coord, th_coord, curr_oy[s], curr_py[s]);
                            ssum += out.predicted_value;
                        } else if p.basis_id == 13 {
                            let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                            let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                            let (out,_,_) = compute_b13_rank1_with_y_overrides(&p, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]);
                            ssum += out.predicted_value;
                        } else if p.basis_id == 14 {
                            let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                            let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                            let (out,_,_) = compute_b14_cartesian_with_y_overrides(&p, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]);
                            ssum += out.predicted_value;
                        } else {
                            ssum += bit_engine::compute_fused_output(&p, i, j, rows, cols).predicted_value;
                        }
                    }
                    pred_sum_total[idx] = ssum;
                }}

                // Current RMSE
                let mut se_curr = 0.0f64;
                for idx in 0..total {
                    let e = (pred_sum_total[idx] - unsafe { *weights.get_unchecked(idx) }) as f64;
                    se_curr += e*e;
                }
                let rmse_curr = (se_curr / (total as f64)).sqrt();

                let mut gn_dbg: Option<(f32,f32,f32,f64,f64,f32)> = None; // (dr,dth,dp1,rmse0,rmse1,scale)
                for s in 0..k {
                    let mut num_r: f64 = 0.0; let mut den_r: f64 = 0.0;
                    // Joint GN for (theta, p1): accumulate J^T J and J^T e
                    let mut a11: f64 = 0.0; // dth*dth
                    let mut a12: f64 = 0.0; // dth*dp1
                    let mut a22: f64 = 0.0; // dp1*dp1
                    let mut b1: f64 = 0.0;  // e*dth
                    let mut b2: f64 = 0.0;  // e*dp1
                    let mut params = to_params(&seeds[s]);
                    params.r = curr_r[s]; params.theta = curr_th[s]; params.param1 = curr_p1[s]; params.param2 = curr_p2[s];
                    for i in 0..rows { for j in 0..cols {
                        let idx = i*cols + j;
                        let e = (pred_sum_total[idx] - unsafe { *weights.get_unchecked(idx) }) as f64;
                        let out = if params.basis_id == 12 {
                            let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                            let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                            compute_b12_with_y_overrides(&params, r_coord, th_coord, curr_oy[s], curr_py[s]).0
                        } else if params.basis_id == 13 {
                            let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                            let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                            compute_b13_rank1_with_y_overrides(&params, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]).0
                        } else {
                            bit_engine::compute_fused_output(&params, i, j, rows, cols)
                        };
                        let dr = out.grad_r as f64;
                        let dth = out.grad_theta as f64;
                        let dp1 = out.grad_p1 as f64;
                        num_r += e * dr; den_r += dr * dr;
                        a11 += dth*dth; a12 += dth*dp1; a22 += dp1*dp1;
                        b1 += e*dth; b2 += e*dp1;
                    }}
                    // Proposed GN deltas
                    let mut d_r = (num_r / (den_r + 1e-18)) as f32;
                    // Solve 2x2 (with small LM damping)
                    let lambda = 1e-6f64;
                    let a11d = a11 + lambda; let a22d = a22 + lambda;
                    let det = a11d*a22d - a12*a12;
                    let (mut d_th, mut d_p1) = if det.abs() > 1e-24 {
                        let inv11 =  a22d / det;
                        let inv12 = -a12  / det;
                        let inv22 =  a11d / det;
                        let dth = (inv11*b1 + inv12*b2) as f32;
                        let dp1 = (inv12*b1 + inv22*b2) as f32;
                        (dth, dp1)
                    } else { ((b1/(a11+1e-18)) as f32, (b2/(a22+1e-18)) as f32) };
                    d_r = d_r.clamp(-0.01, 0.01);
                    d_th = d_th.clamp(-0.01, 0.01);
                    d_p1 = d_p1.clamp(-0.02, 0.02);

                    // Line search: backoff until RMSE improves
                    let (r0,th0,p10) = (curr_r[s], curr_th[s], curr_p1[s]);
                    let mut scale = 1.0f32;
                    let mut accepted = false;
                    let mut best_rmse = rmse_curr;
                    for _try in 0..6 {
                        let r_try = (r0 - scale*d_r).clamp(0.0, 0.9999);
                        let th_try = wrap_0_2pi(th0 - scale*d_th);
                        let p1_try = (p10 - scale*d_p1).clamp(-16.0, 16.0);
                        // compute rmse if only this seed is changed
                        let mut se_new = 0.0f64;
                        for i in 0..rows { for j in 0..cols {
                            let idx = i*cols + j;
                            // prediction with seed s swapped
                            let mut ssum = pred_sum_total[idx];
                            // remove old contribution of s
                            let mut p_old = to_params(&seeds[s]);
                            p_old.r = r0; p_old.theta = th0; p_old.param1 = p10; p_old.param2 = curr_p2[s];
                            ssum -= if p_old.basis_id == 12 {
                                let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                compute_b12_with_y_overrides(&p_old, r_coord, th_coord, curr_oy[s], curr_py[s]).0.predicted_value
                            } else if p_old.basis_id == 13 {
                                let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                compute_b13_rank1_with_y_overrides(&p_old, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]).0.predicted_value
                            } else if p_old.basis_id == 14 {
                                let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                compute_b14_cartesian_with_y_overrides(&p_old, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]).0.predicted_value
                            } else {
                                bit_engine::compute_fused_output(&p_old, i, j, rows, cols).predicted_value
                            };
                            // add new contribution of s
                            let mut p_new = p_old; p_new.r = r_try; p_new.theta = th_try; p_new.param1 = p1_try;
                            ssum += if p_new.basis_id == 12 {
                                let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                compute_b12_with_y_overrides(&p_new, r_coord, th_coord, curr_oy[s], curr_py[s]).0.predicted_value
                            } else if p_new.basis_id == 13 {
                                let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                compute_b13_rank1_with_y_overrides(&p_new, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]).0.predicted_value
                            } else if p_new.basis_id == 14 {
                                let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                compute_b14_cartesian_with_y_overrides(&p_new, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]).0.predicted_value
                            } else {
                                bit_engine::compute_fused_output(&p_new, i, j, rows, cols).predicted_value
                            };
                            let e = (ssum - unsafe { *weights.get_unchecked(idx) }) as f64;
                            se_new += e*e;
                        }}
                        let rmse_new = (se_new / (total as f64)).sqrt();
                        if rmse_new < best_rmse { best_rmse = rmse_new; accepted = true; curr_r[s] = r_try; curr_th[s] = th_try; curr_p1[s] = p1_try; break; }
                        scale *= 0.5;
                    }
                    if s == 0 { gn_dbg = Some((d_r,d_th,d_p1, rmse_curr, best_rmse, scale)); }
                    // If accepted for any s, pred_sum_total is stale; refresh for next seed to keep consistency
                    if accepted {
                        for i in 0..rows { for j in 0..cols {
                            let idx = i*cols + j;
                            let mut ssum = 0.0f32;
                            for t in 0..k {
                                let mut p = to_params(&seeds[t]);
                                p.r = curr_r[t]; p.theta = curr_th[t]; p.param1 = curr_p1[t]; p.param2 = curr_p2[t];
                                ssum += if p.basis_id == 12 {
                                    let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                    let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                    compute_b12_with_y_overrides(&p, r_coord, th_coord, curr_oy[t], curr_py[t]).0.predicted_value
                                } else if p.basis_id == 13 {
                                    let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                    let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                    compute_b13_rank1_with_y_overrides(&p, r_coord, th_coord, 8.0*curr_oy[t], curr_py[t]).0.predicted_value
                                } else if p.basis_id == 14 {
                                    let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                    let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                    compute_b14_cartesian_with_y_overrides(&p, r_coord, th_coord, 8.0*curr_oy[t], curr_py[t]).0.predicted_value
                                } else {
                                    bit_engine::compute_fused_output(&p, i, j, rows, cols).predicted_value
                                };
                            }
                            pred_sum_total[idx] = ssum;
                        }}
                    }
                }
                if let Some((dr,dt,dp1,rm0,rm1,sc)) = gn_dbg {
                    println!("[HybridNG][GN] step={} d(r,th,p1)=[{:.3e},{:.3e},{:.3e}] rmse {:.6} -> {:.6} scale={:.2}", _step+1, dr, dt, dp1, rm0, rm1, sc);
                }
            }

            // Optional: For separable basis (id=12), refit a_x, a_y (k_value, activation_id)
            for s in 0..k {
                let params0 = to_params(&seeds[s]);
                if params0.basis_id == 12 {
                    // Build base params with current continuous values
                    let mut base = params0;
                    base.r = curr_r[s]; base.theta = curr_th[s]; base.param1 = curr_p1[s]; base.param2 = curr_p2[s];
                    // x-only: a_x=1, a_y=0
                    let mut px = base; px.k_value = 255; px.activation_id = 0;
                    // y-only: a_x=0, a_y=1
                    let mut py = base; py.k_value = 0; py.activation_id = 255;

                    let mut s_xx: f64 = 0.0; let mut s_xy: f64 = 0.0; let mut s_yy: f64 = 0.0;
                    let mut b_x: f64 = 0.0; let mut b_y: f64 = 0.0;
                    for i in 0..rows { for j in 0..cols {
                        let idx = i*cols + j;
                        let mut sum_others = 0.0f32; for t in 0..k { if t!=s { sum_others += (base_all[t*total + idx] * curr_p2[t]); } }
                        let r = (unsafe { *weights.get_unchecked(idx) } - sum_others) as f64;
                        let bx = bit_engine::compute_fused_output(&px, i, j, rows, cols).predicted_value as f64;
                        let by = bit_engine::compute_fused_output(&py, i, j, rows, cols).predicted_value as f64;
                        s_xx += bx*bx; s_xy += bx*by; s_yy += by*by; b_x += bx*r; b_y += by*r;
                    }}
                    // Solve 2x2 normal eq
                    let det = s_xx*s_yy - s_xy*s_xy;
                    if det.abs() > 1e-18 {
                        let ax = (( b_x*s_yy - s_xy*b_y) / det).clamp(0.0, 1.0);
                        let ay = (( s_xx*b_y - s_xy*b_x) / det).clamp(0.0, 1.0);
                        let k_new = (ax * 255.0).round() as u8;
                        let act_new = (ay * 255.0).round() as u8;
                        seeds[s].set_k_value(k_new);
                        seeds[s].set_activation_id(act_new);
                    }
                }
            }

            // Optional: step-wise debug prints + early stopping
            if (_step % 10 == 0) || (_step + 1 == self.cfg.steps) {
                // recompute RMSE after amplitude refit using current seeds
                let mut se_sum = 0.0f64;
                for i in 0..rows { for j in 0..cols {
                    let idx = i*cols + j;
                        let mut pred_sum = 0.0f32;
                        for s in 0..k {
                            let mut params = to_params(&seeds[s]);
                            params.r = curr_r[s]; params.theta = curr_th[s]; params.param1 = curr_p1[s]; params.param2 = curr_p2[s];
                            let val = if params.basis_id == 12 {
                                let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                compute_b12_with_y_overrides(&params, r_coord, th_coord, curr_oy[s], curr_py[s]).0.predicted_value
                            } else if params.basis_id == 13 {
                                let r_coord = if rows > 1 { i as f32 / ((rows as f32) - 1.0) } else { 0.0 };
                                let th_coord = if cols > 0 { 2.0*std::f32::consts::PI*(j as f32)/(cols as f32) } else { 0.0 };
                                compute_b13_rank1_with_y_overrides(&params, r_coord, th_coord, 8.0*curr_oy[s], curr_py[s]).0.predicted_value
                            } else {
                                bit_engine::compute_fused_output(&params, i, j, rows, cols).predicted_value
                            };
                            pred_sum += val;
                        }
                    let e = (pred_sum - unsafe { *weights.get_unchecked(idx) }) as f64;
                    se_sum += e*e;
                }}
                let rmse = (se_sum / (total as f64)).sqrt();
                let s0 = 0usize;
                let r0 = curr_r[s0];
                let th0 = curr_th[s0];
                let p10 = curr_p1[s0];
                let p20 = curr_p2[s0];
                if let Some((dr0,dth0,dp10,dp20)) = dbg_step0 {
                    let inv_r_dbg = inv_r_s[0]; let inv_th_dbg = inv_th_s[0]; let inv_p1_dbg = inv_p1_s[0]; let inv_p2_dbg = inv_p2_s[0];
                    println!(
                        "[HybridNG] step={} rmse={:.8} seed0[r={:.5}, th={:.5}, p1={:.5}, p2={:.5}] d(avg)=[{:.3e},{:.3e},{:.3e},{:.3e}] rms=[{:.3e},{:.3e},{:.3e},{:.3e}] inv=[{:.2},{:.2},{:.2},{:.2}] batch={} clip={:.3}",
                        _step+1, rmse, r0, th0, p10, p20, dr0, dth0, dp10, dp20, rms_r, rms_th, rms_p1, rms_p2, inv_r_dbg, inv_th_dbg, inv_p1_dbg, inv_p2_dbg, batch, self.cfg.grad_clip
                    );
                } else {
                    println!(
                        "[HybridNG] step={} rmse={:.8} seed0[r={:.5}, th={:.5}, p1={:.5}, p2={:.5}] batch={} clip={:.3}",
                        _step+1, rmse, r0, th0, p10, p20, batch, self.cfg.grad_clip
                    );
                }
                if rmse < 1.0e-5 {
                    println!("[HybridNG] early stop at step {} (rmse {:.8})", _step+1, rmse);
                    break;
                }
            }
        }

        // Write back final parameters once (quantized)
        for s in 0..seeds.len() {
            seeds[s].set_r(curr_r[s]);
            seeds[s].set_theta(curr_th[s]);
            seeds[s].set_param1(curr_p1[s]);
            seeds[s].set_param2(curr_p2[s]);
            // Quantize y-axis overrides back into q_value/flags for basis 12/13/14
            let params0 = to_params(&seeds[s]);
            if params0.basis_id == 12 || params0.basis_id == 13 || params0.basis_id == 14 {
                let oy_q = (curr_oy[s].clamp(0.0, 1.0) * 255.0).round().clamp(0.0, 255.0) as u8;
                seeds[s].set_q_value(oy_q);
                // encode phi_y into 2-bit flags[5:4]
                let py = curr_py[s].rem_euclid(2.0*std::f32::consts::PI);
                let sector = ((py / (0.5*std::f32::consts::PI)).round() as i32) & 0b11;
                let mut flags = seeds[s].get_flags();
                flags &= !(0b11 << 4);
                flags |= ((sector as u8) & 0b11) << 4;
                seeds[s].set_flags(flags);
            }
        }

        // Final amplitude refit after quantization (use exact deployed path)
        let total = rows * cols;
        let k = seeds.len();
        if k > 0 {
            let mut base_all = vec![0.0f32; total*k];
            for s in 0..k {
                // Build base column with param2=1 using the deployed compute path
                let mut params = to_params(&seeds[s]);
                params.param2 = 1.0;
                for i in 0..rows { for j in 0..cols {
                    let idx = i*cols + j;
                    base_all[s*total + idx] = bit_engine::compute_fused_output(&params, i, j, rows, cols).predicted_value;
                }}
            }
            // Normal equations via modified Gram–Schmidt QR
            let mut qmat = vec![0.0f64; total * k];
            let mut rmat = vec![0.0f64; k * k];
            for s in 0..k {
                for i in 0..total { qmat[s*total + i] = base_all[s*total + i] as f64; }
                for t in 0..s {
                    let mut dot = 0.0f64; for i in 0..total { dot += qmat[t*total + i] * qmat[s*total + i]; }
                    rmat[t*k + s] = dot; for i in 0..total { qmat[s*total + i] -= dot * qmat[t*total + i]; }
                }
                let mut norm2 = 0.0f64; for i in 0..total { norm2 += qmat[s*total + i] * qmat[s*total + i]; }
                let norm = norm2.sqrt().max(1e-12);
                rmat[s*k + s] = norm; for i in 0..total { qmat[s*total + i] /= norm; }
            }
            let mut c = vec![0.0f64; k];
            for s in 0..k {
                let mut dot = 0.0f64; for i in 0..total { dot += qmat[s*total + i] * (unsafe { *weights.get_unchecked(i) } as f64); }
                c[s] = dot;
            }
            let mut a = vec![0.0f64; k];
            for i in (0..k).rev() {
                let mut sum = c[i]; for j in (i+1)..k { sum -= rmat[i*k + j] * a[j]; }
                a[i] = sum / rmat[i*k + i].max(1e-12);
            }
            for s in 0..k { seeds[s].set_param2((a[s] as f32).clamp(-4.0, 4.0)); }
        }
    }
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

// Simple fast PRNG for sampling indices (SplitMix64)
struct SplitMix64 { state: u64 }
impl SplitMix64 {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        let mut z = { self.state = self.state.wrapping_add(0x9E3779B97F4A7C15); self.state };
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rmse_from_seeds(seeds: &[Packed256], rows: usize, cols: usize, target: &[f32]) -> f32 {
        let mut mse = 0.0f32;
        for i in 0..rows {
            for j in 0..cols {
                let mut pred_sum = 0.0f32;
                for s in 0..seeds.len() {
                    let out = bit_engine::compute_fused_output(&to_params(&seeds[s]), i, j, rows, cols);
                    pred_sum += out.predicted_value;
                }
                let e = pred_sum - target[i*cols + j];
                mse += e * e;
            }
        }
        (mse / (rows*cols) as f32).sqrt()
    }

    #[test]
    fn optimize_tile_recovers_single_seed() {
        let (h, w) = (64usize, 64usize);
        // ground-truth single seed
        let gt = Packed256::new(&Packed256Params { r: 0.7, theta: 0.2, param1: std::f32::consts::PI*1.5, param2: 0.9, basis_id: 12, d_r: 0, d_theta: 0, log2_c: -20, activation_id: 160, q_value: 255, k_value: 200, flags: 0 });
        // target from gt
        let mut target = vec![0.0f32; h*w];
        for i in 0..h { for j in 0..w {
            target[i*w + j] = bit_engine::compute_fused_output(&to_params(&gt), i, j, h, w).predicted_value;
        }}
        // init seed with perturbation
        let mut seeds = vec![Packed256::new(&Packed256Params { r: (gt.get_r()+0.1).clamp(0.0,0.9999), theta: gt.get_theta()-0.1, param1: gt.get_param1()*0.9, param2: (gt.get_param2()+0.1).clamp(0.0,4.0), basis_id: 12, d_r: 0, d_theta: 0, log2_c: -20, activation_id: 150, q_value: 255, k_value: 210, flags: 0 })];
        let rmse0 = rmse_from_seeds(&seeds, h, w, &target);
        let opt = HybridNaturalOptimizer::new(HybridNGConfig{ steps: 200, batch_size: h*w, learning_rate_r: 1e-2, learning_rate_theta: 1e-2, learning_rate_p1: 5e-3, learning_rate_p2: 5e-3, ..Default::default() });
        opt.optimize_tile(&mut seeds, &target, h, w);
        let rmse1 = rmse_from_seeds(&seeds, h, w, &target);
        assert!(rmse1 < rmse0, "rmse did not drop: before={:.6} after={:.6}", rmse0, rmse1);
        assert!(rmse1 <= 0.05, "rmse not sufficiently low: {:.6}", rmse1);
    }

    #[test]
    fn storage_budget_feasible_cr50_examples() {
        // S_allow = 4*T^2 / CR; S_use = 32*k + (m*b/8) + 4
        let cr = 50.0f32;
        let t = 256u32; let s_allow = 4.0 * (t*t) as f32 / cr; // ≈ 5242.88 B
        // Example: k=16, b=0 (no residual), should be feasible
        let s_use = 32.0*16.0 + 4.0; // 516 B
        assert!(s_use <= s_allow + 1e-3);
        // Example: k=1, b=4, m=9600 (p≈98), should be feasible within 5.2KB
        let s_use2 = 32.0*1.0 + (9600.0*4.0/8.0) + 4.0; // 4800+36 ≈ 4836 B
        assert!(s_use2 <= s_allow + 1e-3);
    }
}



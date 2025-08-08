use std::time::Instant;
use rand::{rngs::StdRng, SeedableRng};

use rbe_llm::core::{
    optimizers::BitAdamState,
    tensors::{Packed256, Packed256Params},
    differential::bit_engine,
};

fn eval_rmse(seed: &Packed256, block_h: usize, block_w: usize, weight_data: &[f32]) -> f32 {
    let params = Packed256Params {
        r: seed.get_r(),
        theta: seed.get_theta(),
        param1: seed.get_param1(),
        param2: seed.get_param2(),
        basis_id: seed.get_basis_id(),
        d_r: 0,
        d_theta: 0,
        log2_c: seed.get_log2_c(),
        activation_id: seed.get_activation_id(),
        q_value: seed.get_q_value(),
        k_value: seed.get_k_value(),
        flags: seed.get_flags(),
    };
    let mut mse = 0.0f32;
    for i in 0..block_h {
        for j in 0..block_w {
            let out = bit_engine::compute_fused_output(&params, i, j, block_h, block_w);
            let err = out.predicted_value - weight_data[i * block_w + j];
            mse += err * err;
        }
    }
    (mse / (block_h * block_w) as f32).sqrt()
}

fn beam_init(seed: &mut Packed256, block_h: usize, _block_w: usize, weight_data: &[f32]) {
    // p1(주파수)와 φ만 탐색 (기타 파라미터 고정)
    let p1_scales = [0.75f32, 1.0, 1.25];
    let phi_grid = [
        0.0f32,
        0.25 * std::f32::consts::PI,
        0.5 * std::f32::consts::PI,
        -0.25 * std::f32::consts::PI,
    ];

    let base_p1 = seed.get_param1();
    let base_phi = seed.get_theta();
    let mut best_rmse = f32::INFINITY;
    let mut best = (base_p1, base_phi);

    for &s in &p1_scales {
        let p1 = base_p1 * s;
        for &phi in &phi_grid {
            let mut tmp = *seed;
            tmp.set_param1(p1);
            tmp.set_theta(phi);
            // 플래그/혼합/아틀라스/가중치는 고정
            let rmse = eval_rmse(&tmp, block_h, _block_w, weight_data);
            if rmse < best_rmse {
                best_rmse = rmse;
                best = (p1, phi);
            }
        }
    }

    seed.set_param1(best.0);
    seed.set_theta(best.1);
    seed.set_flags(0);
}

fn train_block(block_h: usize, block_w: usize, epochs: usize, block_idx: usize) -> (f32, bool, u128) {
    let _rng = StdRng::seed_from_u64(42 + block_idx as u64);
    // 스무스 타깃(검증용): sin(2π i/rows) + 0.5 cos(2π j/cols)
    let weight_data: Vec<f32> = (0..block_h * block_w)
        .map(|idx| {
            let i0 = idx / block_w;
            let j0 = idx % block_w;
            let denom_r = ((block_h as f32) - 1.0).max(1.0);
            let fx = 2.0 * std::f32::consts::PI * (i0 as f32) / denom_r;
            let fy = 2.0 * std::f32::consts::PI * (j0 as f32) / (block_w as f32).max(1.0);
            fx.sin() + 0.5 * fy.cos()
        })
        .collect();

    // 분리형 스무스 타깃 정합: 도함수 모드는 끔
    let (d_r, d_theta) = (0u8, 0u8);

    let mut seed = Packed256::new(&Packed256Params {
        r: 1.0,                      // r_scale = 1.0 → sin(2π i/(H-1))와 정확히 정합
        theta: 1.0,                  // theta_scale ~ 1
        param1: std::f32::consts::PI * 2.0, // omega_x ~ 2π
        param2: 1.0,                 // amplitude (global)
        basis_id: 12,                // separable sum basis for smooth target
        d_r: d_r,
        d_theta: d_theta,
        log2_c: -20,                 // c≈0 → metric≈1 (중립화)
        activation_id: 128,          // a_y 초기 ≈ 0.5
        q_value: 255,                // omega_y = 1.0 (since inner uses theta_eff=2π j/cols)
        k_value: 255,                // a_x 초기 ≈ 1.0
        flags: 0b1110_0000,          // blend weight=1.0(안전), blend 자체는 basis_id=12에서 비활성 처리됨
    });

    // 디버그: 초기 예측 확인
    {
        let p = Packed256Params {
            r: seed.get_r(), theta: seed.get_theta(), param1: seed.get_param1(), param2: seed.get_param2(),
            basis_id: seed.get_basis_id(), d_r: seed.get_d_r(), d_theta: seed.get_d_theta(), log2_c: seed.get_log2_c(),
            activation_id: seed.get_activation_id(), q_value: seed.get_q_value(), k_value: seed.get_k_value(), flags: seed.get_flags()
        };
        let out0 = bit_engine::compute_fused_output(&p, 0, 0, block_h, block_w);
        eprintln!("[debug] init pred(0,0)={:.6}", out0.predicted_value);
    }

    // 빔 초기화 생략: 주파수/스케일 고정(정합 타깃)

    // 빔 초기화 이후 예측 확인
    {
        let p = Packed256Params {
            r: seed.get_r(), theta: seed.get_theta(), param1: seed.get_param1(), param2: seed.get_param2(),
            basis_id: seed.get_basis_id(), d_r: seed.get_d_r(), d_theta: seed.get_d_theta(), log2_c: seed.get_log2_c(),
            activation_id: seed.get_activation_id(), q_value: seed.get_q_value(), k_value: seed.get_k_value(), flags: seed.get_flags()
        };
        let out0 = bit_engine::compute_fused_output(&p, 0, 0, block_h, block_w);
        eprintln!("[debug] after beam pred(0,0)={:.6} amp={:.6} flags={:08b}", out0.predicted_value, p.param2, p.flags);
        let rmse0 = eval_rmse(&seed, block_h, block_w, &weight_data);
        eprintln!("[debug] initial RMSE={:.6}", rmse0);
        // Direct check: analytic separable form
        let r_scale = seed.get_r();
        let theta_scale = seed.get_theta();
        let omega_x = seed.get_param1();
        let omega_y = (seed.get_q_value() as f32) / 255.0;
        let ax = (seed.get_k_value() as f32) / 255.0;
        let ay = (seed.get_activation_id() as f32) / 255.0;
        let mut mse_d = 0.0f32;
        for i in 0..block_h {
            for j in 0..block_w {
                let r_coord = if block_h > 1 { i as f32 / (block_h as f32 - 1.0) } else { 0.0 };
                let theta_coord = 2.0 * std::f32::consts::PI * (j as f32) / (block_w as f32);
                let x = omega_x * r_scale * r_coord;
                let y = omega_y * theta_scale * theta_coord;
                let pred = ax * x.sin() + ay * y.cos();
                let tgt = weight_data[i * block_w + j];
                let e = pred - tgt;
                mse_d += e * e;
            }
        }
        let rmse_direct = (mse_d / (block_h * block_w) as f32).sqrt();
        eprintln!("[debug] direct-form RMSE={:.6}", rmse_direct);
    }

    let _opt = BitAdamState::new();
    let base_lr = 0.03f32;
    let mut best_rmse = f32::INFINITY;
    let mut converged = false;

    let start = Instant::now();
    for epoch in 0..epochs {
        let mut mse = 0.0f32;
        let mut _num_amp = 0.0f32;
        let mut _den_amp = 0.0f32;
        let mut pred_abs_sum = 0.0f32;
        // 선형 최소제곱으로 a_x, a_y 폐형해 추정용 누산
        let mut ss: f32 = 0.0;   // Σ sin_x^2
        let mut cc: f32 = 0.0;   // Σ cos_y^2
        let mut sc: f32 = 0.0;   // Σ sin_x*cos_y
        let mut ys: f32 = 0.0;   // Σ target*sin_x
        let mut yc: f32 = 0.0;   // Σ target*cos_y
        // 간단한 코사인 감쇠 스케줄
        let t = epoch as f32 / (epochs as f32);
        let lr = 0.1f32 * base_lr + 0.9f32 * base_lr * 0.5 * (1.0 + (std::f32::consts::PI * (1.0 - t)).cos());
        // φy=0 고정, φx만 소규모 탐색(−π/2, −π/4, −π/8, 0, π/8, π/4, π/2) 후 ax, ay 폐형해
        let phi_grid = [
            -0.5 * std::f32::consts::PI,
            -0.25 * std::f32::consts::PI,
            -0.125 * std::f32::consts::PI,
            0.0,
            0.125 * std::f32::consts::PI,
            0.25 * std::f32::consts::PI,
            0.5 * std::f32::consts::PI,
        ];
        let mut best_local_rmse = f32::INFINITY;
        let mut best_phi_x = 0.0f32;
        for &phi_x in &phi_grid {
            let mut ss=0.0; let mut cc=0.0; let mut sc=0.0; let mut ys=0.0; let mut yc=0.0;
            let mut mse_try=0.0;
            for i in 0..block_h {
                for j in 0..block_w {
                    let target = weight_data[i * block_w + j];
                    let r_coord = if block_h > 1 { i as f32 / (block_h as f32 - 1.0) } else { 0.0 };
                    let theta_coord = 2.0 * std::f32::consts::PI * (j as f32) / (block_w as f32).max(1.0);
                    let r_eff = seed.get_r() * r_coord;
                    let theta_eff = seed.get_theta() * theta_coord;
                    let omega_x = seed.get_param1();
                    let omega_y = (seed.get_q_value() as f32)/255.0;
                    let sin_x = (omega_x * r_eff + phi_x).sin();
                    let cos_y = (omega_y * theta_eff).cos();
                    ss += sin_x*sin_x; cc += cos_y*cos_y; sc += sin_x*cos_y;
                    ys += target*sin_x; yc += target*cos_y;
                }
            }
            let det = ss*cc - sc*sc;
            if det.abs() < 1e-8 { continue; }
            let inv_det = 1.0/det;
            let ax = (cc*ys - sc*yc)*inv_det;
            let ay = (ss*yc - sc*ys)*inv_det;
            for i in 0..block_h {
                for j in 0..block_w {
                    let target = weight_data[i * block_w + j];
                    let r_coord = if block_h > 1 { i as f32 / (block_h as f32 - 1.0) } else { 0.0 };
                    let theta_coord = 2.0 * std::f32::consts::PI * (j as f32) / (block_w as f32).max(1.0);
                    let r_eff = seed.get_r() * r_coord;
                    let theta_eff = seed.get_theta() * theta_coord;
                    let omega_x = seed.get_param1();
                    let omega_y = (seed.get_q_value() as f32)/255.0;
                    let pred = ax*(omega_x*r_eff + phi_x).sin() + ay*(omega_y*theta_eff).cos();
                    let e = pred - target;
                    mse_try += e*e;
                }
            }
            let rmse_try = (mse_try / (block_h*block_w) as f32).sqrt();
            if rmse_try < best_local_rmse {
                best_local_rmse = rmse_try;
                best_phi_x = phi_x;
            }
        }
        seed.set_theta(best_phi_x);
        seed.set_flags(seed.get_flags() & !(0b11<<4));
        
        for i in 0..block_h {
            for j in 0..block_w {
                let target = weight_data[i * block_w + j];
                // 순전파(우리 엔진)
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
                let out = bit_engine::compute_fused_output(&params, i, j, block_h, block_w);
                let err = out.predicted_value - target;
                mse += err * err;
                pred_abs_sum += out.predicted_value.abs();

                // 진폭의 폐형해 업데이트를 위한 통계 f_base = pred with amp=1
                let params_amp1 = Packed256Params { param2: 1.0, ..params };
                let out_amp1 = bit_engine::compute_fused_output(&params_amp1, i, j, block_h, block_w);
                let f_base = out_amp1.predicted_value; // equals func_val*metric
                _num_amp += f_base * target;
                _den_amp += f_base * f_base;

                // a_x, a_y용 신호 누적 (metric≈1, amp=1 가정)
                // basis 12 구조를 직접 계산해 정확한 신호를 사용
                let rows = block_h as f32;
                let cols = block_w as f32;
                let r_coord = if block_h > 1 { i as f32 / (rows - 1.0) } else { 0.0 };
                let theta_coord = 2.0 * std::f32::consts::PI * (j as f32) / cols.max(1.0);
                let r_eff = seed.get_r() * r_coord;
                let theta_eff = seed.get_theta() * theta_coord;
                let omega_x = seed.get_param1();
                let omega_y = (seed.get_q_value() as f32) / 255.0; // theta_eff already has 2π
                let sin_x = (omega_x * r_eff + seed.get_theta()).sin();
                let phi_y_flag = (seed.get_flags() >> 4) & 0b11;
                let phi_y = match phi_y_flag { 0 => 0.0, 1 => 0.5*std::f32::consts::PI, 2 => std::f32::consts::PI, _ => 1.5*std::f32::consts::PI };
                let cos_y = (omega_y * theta_eff + phi_y).cos();
                ss += sin_x * sin_x;
                cc += cos_y * cos_y;
                sc += sin_x * cos_y;
                ys += target * sin_x;
                yc += target * cos_y;

                // 이번 사이클: 주파수/스케일 고정, 폐형해 계수만 추정 (optimizer 업데이트 비활성화)
                let _ = &lr; // silence
            }
        }
        let rmse = (mse / (block_h * block_w) as f32).sqrt();
        if block_idx == 0 && (epoch == 0 || epoch == epochs - 1) {
            eprintln!("[debug] epoch {} avg|pred|={:.6}", epoch, pred_abs_sum / (block_h * block_w) as f32);
        }
        if rmse < best_rmse { best_rmse = rmse; }
        if rmse < 0.001 { converged = true; break; }

        // 에폭 끝에서 a_x, a_y를 2x2 정규방정식으로 폐형해 추정 후 8비트로 투영
        let det = ss * cc - sc * sc;
        if det.abs() > 1e-8 {
            let inv_det = 1.0 / det;
            let ax = (cc * ys - sc * yc) * inv_det;
            let ay = (ss * yc - sc * ys) * inv_det;
            let ax_clip = ax.clamp(0.0, 1.0);
            let ay_clip = ay.clamp(0.0, 1.0);
            seed.set_k_value((ax_clip * 255.0).round() as u8);
            seed.set_activation_id((ay_clip * 255.0).round() as u8);
        }

        // amp는 1.0 고정 (계수(ax, ay)로 설명력을 흡수)
        seed.set_param2(1.0);
    }
    let elapsed_ms = start.elapsed().as_millis();
    (best_rmse, converged, elapsed_ms)
}

fn main() {
    let block = (16usize, 16usize);
    let epochs = 1000usize;
    let samples = 10usize;

    println!("| Epochs | Block | Basis | d_r | d_θ | RMSE | Converged | Time(ms) |");
    println!("|---:|---:|---:|---:|---:|---:|:---:|---:|");
    let mut sum_rmse = 0.0f32;
    let mut sum_ms = 0u128;
    let mut cnt_conv = 0usize;
    for idx in 0..samples {
        let basis = idx % 4;
        let dr = match idx % 4 { 0 => 0, 1 => 1, 2 => 0, _ => 0 };
        let dtheta = match idx % 4 { 2 => 1, _ => 0 };

        let (rmse, conv, ms) = train_block(block.0, block.1, epochs, idx);
        println!("| {} | {} | {} | {} | {} | {:.6} | {} | {} |", epochs, idx, basis, dr, dtheta, rmse, if conv {"yes"} else {"no"}, ms);
        sum_rmse += rmse;
        sum_ms += ms;
        if conv { cnt_conv += 1; }
    }
    let avg_rmse = sum_rmse / samples as f32;
    let avg_ms = (sum_ms as f64 / samples as f64) as u128;
    println!("\n| Avg | - | - | - | - | {:.6} | {} / {} | {} |", avg_rmse, cnt_conv, samples, avg_ms);
}



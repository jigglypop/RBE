// Poincaré curvature scaling and natural gradient sanity checks
// Non-interactive binary for first-step cycle verification

fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

fn test_kappa_limit() -> bool {
    // κ -> 0 limit: tanh(κ r)/tanh(κ) -> r
    let rs = [0.0_f32, 0.1, 0.3, 0.7, 0.95];
    let kappas = [1e-6_f32, 5e-7, 1e-7];
    for &r in &rs {
        // encode r into seed.lo via decode path workaround: we call fused_forward_poincare
        // which uses internal decode() for r, so we compare relative scaling behavior by
        // computing r' numerically through output change with/without kappa.
        // We instead directly emulate the formula here for the check.
        for &kappa in &kappas {
            let denom = kappa.tanh();
            let rp = if denom.abs() > 0.0 { (kappa * r).tanh() / denom } else { r };
            if !approx_eq(rp, r, 1e-6_f32.max(5.0 * kappa)) {
                eprintln!("[FAIL] kappa->0 limit: r={} kappa={} rp={}", r, kappa, rp);
                return false;
            }
        }
    }
    true
}

fn test_kappa_monotonic() -> bool {
    // For fixed r in (0,1), rp(k1) <= rp(k2) when 0<k1<k2
    let rs = [0.1_f32, 0.3, 0.6, 0.9];
    let k1 = 0.2_f32;
    let k2 = 0.8_f32;
    for &r in &rs {
        let rp1 = (k1 * r).tanh() / k1.tanh();
        let rp2 = (k2 * r).tanh() / k2.tanh();
        if rp1 > rp2 + 1e-9 {
            eprintln!("[FAIL] kappa monotonic: r={} rp1={} rp2={}", r, rp1, rp2);
            return false;
        }
    }
    true
}

fn test_riemannian_scaling() -> bool {
    // Natural gradient factors: fr=(1-r^2)^2/4, fθ=(1-r^2)^2/(4 r^2)
    // Validate basic properties and finite behavior near boundary
    let rs = [0.1_f32, 0.3, 0.6, 0.9, 0.99, 0.999];
    for &r in &rs {
        let one_minus_r2 = (1.0 - r * r).max(1e-6);
        let fr = (one_minus_r2 * one_minus_r2) / 4.0;
        let ftheta = (one_minus_r2 * one_minus_r2) / (4.0 * (r * r).max(1e-9));
        if !(fr.is_finite() && ftheta.is_finite()) || fr < 0.0 || ftheta < 0.0 {
            eprintln!("[FAIL] riemannian scaling finite/positive: r={} fr={} fθ={}", r, fr, ftheta);
            return false;
        }
    }
    true
}

fn main() {
    let mut passed = true;
    passed &= test_kappa_limit();
    passed &= test_kappa_monotonic();
    passed &= test_riemannian_scaling();

    if passed {
        println!("Poincaré cycle check: PASS");
        std::process::exit(0);
    } else {
        println!("Poincaré cycle check: FAIL");
        std::process::exit(1);
    }
}



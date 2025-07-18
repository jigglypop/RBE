#!/usr/bin/env python
import numpy as np
import time

# Generate gravitational potential matrix (1/r normalized)
def generate_gravity_matrix(rows, cols):
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv**2 + yv**2)
    r[r < 1e-6] = 1e-6
    phi = 1.0 / r
    return phi / phi.max()

# Compute raw and predicted values given parameters r and theta
def compute_raw_and_mask(r_val, theta_val, xv, yv):
    dist = np.sqrt(xv**2 + yv**2)
    raw = r_val - dist * r_val + theta_val
    pred = np.clip(raw, 0.0, 1.0)
    mask = (raw > 0.0) & (raw < 1.0)
    return pred, mask, dist

# Stochastic training using sampled pixels
def train_rbe_stochastic(size, epochs=100, lr=0.05, sample_size=50000, seed=0):
    np.random.seed(seed)
    rows = cols = size
    target = generate_gravity_matrix(rows, cols)
    # Precompute coordinate grids once
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    xv, yv = np.meshgrid(x, y)
    # initialize parameters
    r_val = np.random.uniform(0.5, 1.0)
    theta_val = np.random.uniform(-0.5, 0.5)
    N = rows * cols
    # Flatten arrays for sampling
    flat_xv = xv.flatten()
    flat_yv = yv.flatten()
    flat_target = target.flatten()
    for epoch in range(epochs):
        # sample indices
        idx = np.random.choice(N, size=sample_size, replace=False)
        xv_s = flat_xv[idx]
        yv_s = flat_yv[idx]
        target_s = flat_target[idx]
        dist = np.sqrt(xv_s**2 + yv_s**2)
        raw = r_val - dist * r_val + theta_val
        pred = np.clip(raw, 0.0, 1.0)
        err = pred - target_s
        mask = (raw > 0.0) & (raw < 1.0)
        dp_dr = (1.0 - dist) * mask
        dp_dtheta = 1.0 * mask
        grad_r = 2.0 * np.mean(err * dp_dr)
        grad_theta = 2.0 * np.mean(err * dp_dtheta)
        # update
        r_val -= lr * grad_r
        theta_val -= lr * grad_theta
        r_val = np.clip(r_val, 0.0, 1.0)
        # compute full RMSE periodically
        if (epoch+1) % (epochs//5) == 0 or epoch == 0:
            # compute RMSE on full matrix for logging
            full_pred, _, _ = compute_raw_and_mask(r_val, theta_val, xv, yv)
            rmse = np.sqrt(np.mean((full_pred - target)**2))
            print(f"Epoch {epoch+1}/{epochs}, rmse={rmse:.5f}, r={r_val:.4f}, theta={theta_val:.4f}")
    # final evaluation
    final_pred, _, _ = compute_raw_and_mask(r_val, theta_val, xv, yv)
    rmse_final = np.sqrt(np.mean((final_pred - target)**2))
    compression_ratio = (rows * cols) / 2.0
    return r_val, theta_val, rmse_final, compression_ratio

if __name__ == '__main__':
    # test 4096 x 4096
    size = 4096
    start = time.time()
    r_val, theta_val, rmse, ratio = train_rbe_stochastic(size, epochs=60, lr=0.1, sample_size=100000, seed=42)
    elapsed = time.time() - start
    print(f"Done: r={r_val:.4f}, theta={theta_val:.4f}, RMSE={rmse:.6f}, compression ratio={ratio:.2f}, time={elapsed:.2f}s")

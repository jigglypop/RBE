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

# Compute predicted matrix from parameters r_fp32, theta_fp32
def compute_pred_matrix(r_val, theta_val, rows, cols):
    # normalized coords
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv**2 + yv**2)
    # compute raw value
    raw = r_val - dist * r_val + theta_val
    # clamp between 0 and 1
    return np.clip(raw, 0.0, 1.0)

# Compute loss and gradient
# returns mse, grad_r, grad_theta
def loss_and_grad(r_val, theta_val, target):
    rows, cols = target.shape
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    xv, yv = np.meshgrid(x, y)
    dist = np.sqrt(xv**2 + yv**2)
    raw = r_val - dist * r_val + theta_val
    # apply clamp
    pred = np.clip(raw, 0.0, 1.0)
    err = pred - target
    mse = np.mean(err**2)
    # derivative is zero where clamp saturates
    mask = (raw > 0.0) & (raw < 1.0)
    # grad of prediction w.r.t r is 1 - dist
    dp_dr = (1.0 - dist) * mask
    dp_dtheta = (1.0) * mask
    # gradient of mse: 2/N * sum(err * dp_dr)
    grad_r = 2.0 * np.mean(err * dp_dr)
    grad_theta = 2.0 * np.mean(err * dp_dtheta)
    return mse, grad_r, grad_theta

# Training function
def train_rbe_on_gravity(size, epochs=200, lr=0.05):
    rows = cols = size
    target = generate_gravity_matrix(rows, cols)
    # initialize parameters r and theta
    r_val = np.random.uniform(0.5, 1.0)
    theta_val = np.random.uniform(-0.5, 0.5)
    for epoch in range(epochs):
        mse, grad_r, grad_theta = loss_and_grad(r_val, theta_val, target)
        # update
        r_val -= lr * grad_r
        theta_val -= lr * grad_theta
        # optional clamp r between 0 and 1
        r_val = np.clip(r_val, 0.0, 1.0)
        # debug: print periodically
        if (epoch+1) % (epochs//5) == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, MSE={mse:.6f}, r={r_val:.4f}, theta={theta_val:.4f}")
    # compute final
    final_pred = compute_pred_matrix(r_val, theta_val, rows, cols)
    rmse = np.sqrt(np.mean((final_pred - target)**2))
    compression_ratio = (rows * cols) / 2.0  # 2 floats r and theta
    return r_val, theta_val, rmse, compression_ratio

if __name__ == '__main__':
    for size in [64, 128, 256]:
        print(f"\nTraining RBE-style model on {size}x{size} gravitational pattern")
        start = time.time()
        r_val, theta_val, rmse, ratio = train_rbe_on_gravity(size, epochs=200, lr=0.05)
        elapsed = time.time() - start
        print(f"Final r={r_val:.4f}, theta={theta_val:.4f}, RMSE={rmse:.6f}, compression ratio={ratio:.2f}, time={elapsed:.2f}s")

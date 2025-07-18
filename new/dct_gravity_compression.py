#!/usr/bin/env python
import numpy as np
from scipy.fftpack import dct, idct
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

# 2D DCT-II (scipy)
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# 2D inverse DCT (IDCT-III) for DCT-II input
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

# Evaluate DCT compression for given size and K coefficients
def evaluate(size, K):
    mat = generate_gravity_matrix(size, size)
    t0 = time.time()
    coeff = dct2(mat)
    dct_time = time.time() - t0
    # Flatten and sort coefficients by magnitude
    flat = coeff.flatten()
    idx = np.argsort(np.abs(flat))[::-1]
    # Keep top K coefficients
    mask = np.zeros_like(flat, dtype=bool)
    mask[idx[:K]] = True
    sparse = np.zeros_like(flat)
    sparse[mask] = flat[mask]
    sparse_coeff = sparse.reshape(coeff.shape)
    # Reconstruct
    t0 = time.time()
    recon = idct2(sparse_coeff)
    idct_time = time.time() - t0
    rmse = np.sqrt(np.mean((recon - mat) ** 2))
    compression_ratio = (size * size) / K
    return {
        'size': size,
        'K': K,
        'dct_time': dct_time,
        'idct_time': idct_time,
        'rmse': rmse,
        'compression_ratio': compression_ratio
    }

if __name__ == '__main__':
    for size in [256, 512, 1024]:
        for K in [10, 50, 200]:
            res = evaluate(size, K)
            print(res)

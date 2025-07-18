#!/usr/bin/env python
import numpy as np
import time

# Convert 2D matrix into a 3D tensor (grid_rows, grid_cols, block_size*block_size)
def to_tensor(matrix, block_size):
    rows, cols = matrix.shape
    grid_rows = rows // block_size
    grid_cols = cols // block_size
    depth = block_size * block_size
    tensor = np.zeros((grid_rows, grid_cols, depth), dtype=np.float32)
    for i in range(grid_rows):
        for j in range(grid_cols):
            block = matrix[i*block_size:(i+1)*block_size,
                           j*block_size:(j+1)*block_size]
            tensor[i, j] = block.flatten()
    return tensor

# Mode-n product of a tensor and a matrix (used in HOSVD)
def mode_n_product(tensor, matrix, mode):
    orig_shape = tensor.shape
    order = [mode] + [i for i in range(len(orig_shape)) if i != mode]
    tensor_perm = np.transpose(tensor, order)
    In = orig_shape[mode]
    rest = int(np.prod([orig_shape[i] for i in range(len(orig_shape)) if i != mode]))
    tensor_mat = tensor_perm.reshape(In, rest)
    result_mat = matrix @ tensor_mat
    new_shape = [matrix.shape[0]] + [orig_shape[i] for i in range(len(orig_shape)) if i != mode]
    result_tensor = result_mat.reshape(new_shape)
    inv_order = np.argsort(order)
    result_tensor = np.transpose(result_tensor, inv_order)
    return result_tensor

# HOSVD compression: returns core tensor and list of factor matrices
def hosvd_compress(tensor, ranks):
    modes = len(tensor.shape)
    factors = []
    core = tensor.copy()
    for mode in range(modes):
        orig_shape = tensor.shape
        order = [mode] + [i for i in range(modes) if i != mode]
        tensor_perm = np.transpose(tensor, order)
        In = orig_shape[mode]
        rest = int(np.prod([orig_shape[i] for i in range(modes) if i != mode]))
        unfold = tensor_perm.reshape(In, rest)
        U, S, Vt = np.linalg.svd(unfold, full_matrices=False)
        r = ranks[mode]
        factor = U[:, :r].T  # shape (r, In)
        factors.append(factor)
        core = mode_n_product(core, factor, mode)
    return core, factors

# Reconstruct tensor from core and factor matrices
def hosvd_reconstruct(core, factors):
    tensor_rec = core.copy()
    for mode, factor in enumerate(factors):
        tensor_rec = mode_n_product(tensor_rec, factor.T, mode)
    return tensor_rec

# Generate a radial gradient pattern matrix
def generate_radial_matrix(rows, cols):
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    xv, yv = np.meshgrid(x, y)
    return (1 - np.sqrt(xv**2 + yv**2) / np.sqrt(2)).clip(min=0).astype(np.float32)

# Evaluate compression for a given matrix size
def evaluate(size, block_size=16, ranks=(2,2,5)):
    rows = cols = size
    matrix = generate_radial_matrix(rows, cols)
    # Time to form tensor
    t0 = time.time()
    tensor = to_tensor(matrix, block_size)
    tensor_time = time.time() - t0
    # Compress
    t0 = time.time()
    core, factors = hosvd_compress(tensor, ranks)
    compress_time = time.time() - t0
    # Reconstruct tensor
    t0 = time.time()
    recon_tensor = hosvd_reconstruct(core, factors)
    reconstruct_time = time.time() - t0
    # Convert back to 2D matrix
    grid_rows, grid_cols, depth = recon_tensor.shape
    recon_matrix = np.zeros_like(matrix)
    for i in range(grid_rows):
        for j in range(grid_cols):
            block_flat = recon_tensor[i, j]
            block = block_flat.reshape((block_size, block_size))
            recon_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = block
    # RMSE
    rmse = np.sqrt(np.mean((matrix - recon_matrix)**2))
    # Compression ratio (number of original floats / number of floats in compressed representation)
    original_floats = rows * cols
    core_size = np.prod(core.shape)
    factor1_size = ranks[0] * (rows // block_size)
    factor2_size = ranks[1] * (cols // block_size)
    factor3_size = ranks[2] * (block_size*block_size)
    compressed_floats = core_size + factor1_size + factor2_size + factor3_size
    compression_ratio = original_floats / compressed_floats
    return {
        'size': size,
        'tensor_time': tensor_time,
        'compress_time': compress_time,
        'reconstruct_time': reconstruct_time,
        'rmse': rmse,
        'compression_ratio': compression_ratio
    }

if __name__ == '__main__':
    sizes = [256, 512, 1024, 2048]
    results = []
    for s in sizes:
        res = evaluate(s)
        results.append(res)
        print(f"Matrix {s}x{s}: tensor {res['tensor_time']:.3f}s, compress {res['compress_time']:.3f}s, "
              f"reconstruct {res['reconstruct_time']:.3f}s, RMSE {res['rmse']:.4f}, "
              f"compression ratio {res['compression_ratio']:.2f}")

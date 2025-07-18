# 6. Matrix Synthesis and Operations

## Introduction

In the Riemannian Basis Encoding (RBE) paradigm, large matrices, such as the weight matrices in neural networks, are not stored explicitly. Instead, they are represented implicitly by a compact set of parameters encapsulated in the `Packed128` data structure. This document elucidates the process by which these parameters are used to synthesize matrix elements on-the-fly and perform standard linear algebra operations without ever materializing the full matrix in memory. This "fused" or "encoded" operational model is the key to RBE's immense memory savings and computational efficiency.

## On-the-Fly Matrix Synthesis

The core principle is that any element \( W_{ij} \) of a matrix \( W \) can be generated at any time from a given set of `Packed128` parameters. This synthesis is governed by a deterministic function, `Generate(i, j, params)`, where `(i, j)` are the matrix coordinates.

The synthesis process integrates the dual components of the `Packed128` structure:

1.  **Continuous Core (`lo`)**: This 64-bit segment contains floating-point values that define the matrix's low-frequency, continuous base pattern. These parameters typically include coefficients for radial basis functions, frequencies, phases, and amplitudes that create a smooth, differentiable manifold. For example, a basic pattern could be a linear combination of basis functions:
    \[ W_{\text{base}}(i, j) = \sum_{k=1}^{N} p_k \cdot \phi_k(i, j) \]
    where \( p_k \) are the parameters from `lo` and \( \phi_k \) are basis functions (e.g., cosines, radial distance).

2.  **Discrete State (`hi`)**: This 64-bit segment acts as a state machine. Its bits select and modulate the core pattern, introducing high-frequency, non-linear details. Each combination of bits can map to a different function or a different state of a periodic function (e.g., `00` for `sin`, `01` for `cos`), effectively switching the behavior of the `Generate` function at different coordinates.

The final synthesized value is a composition of these two components:
\[ W_{ij} = \text{Generate}(i, j, \text{params}) = \text{Modulate}(\text{state}_{\text{hi}}, W_{\text{base}}(i, j)) \]

## Fused Operations: The Encoded GEMM

The primary advantage of RBE is performing matrix operations without decoding. A standard matrix-vector multiplication, \( \mathbf{y} = W\mathbf{x} \), is transformed from a memory-bound operation into a compute-bound one.

Instead of reading \( W \) from memory, each element \( y_i \) of the output vector \( \mathbf{y} \) is computed directly:
\[ y_i = \sum_{j=1}^{M} W_{ij} \cdot x_j = \sum_{j=1}^{M} \text{Generate}(i, j, \text{params}) \cdot x_j \]

This is termed a **Fused RBE-GEMM**. The generation of \( W_{ij} \) and its multiplication with \( x_j \) happen in a single, fused kernel. This approach dramatically reduces memory bandwidth requirements. On hardware where memory access is the bottleneck (common in GPUs and on-device accelerators), this can lead to **faster-than-dense** performance, as the computational overhead of the `Generate` function is less than the latency of fetching the full matrix from memory.

## Backpropagation in the Encoded Domain

Learning is achieved by updating the `Packed128` parameters based on the loss gradient \( \frac{\partial L}{\partial W_{ij}} \), again without instantiating \( W \).

### Continuous Parameter Update (`lo`)

The gradients for the continuous parameters in `lo` are calculated using the standard chain rule. The gradient for a parameter \( p_k \) is the sum of its influence over all matrix elements:
\[ \frac{\partial L}{\partial p_k} = \sum_{i,j} \frac{\partial L}{\partial W_{ij}} \frac{\partial W_{ij}}{\partial p_k} \]
Since `Generate` is a differentiable function with respect to the `lo` parameters, this update is straightforward.

### Discrete State Update (`hi`): State-Transition Differentiation

The `hi` bits, representing discrete states, are not updated via conventional gradients. Instead, we employ **State-Transition Differentiation**. The gradient \( \frac{\partial L}{\partial W_{ij}} \) provides a signal indicating the desired direction of change for the function at \( (i,j) \).

For example, if the current state `00` corresponds to `sin(x)` and the gradient indicates that the output should be closer to `cos(x)` (the derivative), the learning algorithm performs a direct bitwise state transition: it flips the bits in `hi` from `00` to `01`. This is a "differentiable bit operation"—a discrete, deterministic update rule guided by the continuous gradient, as detailed in `05_Math.md`.

This dual-update mechanism allows the model to learn both the smooth, underlying structure (via `lo`) and the complex, stateful logic (via `hi`) of the target function, achieving unparalleled representational power within a compressed format. 
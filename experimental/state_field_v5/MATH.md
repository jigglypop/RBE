# RBE v5: quantized-exact coefficient execution with a Riemannian update

This adds a coefficient backend to RBE. It does not change v3's signed-XOR
operator identity and does not claim that a low-parameter Clifford family spans
all dense matrices. Quantization is explicitly authorized for this stage.

## 1. Exact integer contraction, not FP32 emulation

Let q be a b-bit two's-complement integer. Define c_p=2^p for p<b-1 and
c_(b-1)=-2^(b-1). Then q=sum_p c_p bit_p(q). For up to 32 entries, bit-plane
words A_p,B_q give

    sum_j a_j*b_j = sum_p,q c_p*c_q popcount(A_p & B_q).

This is an integer identity, including minimum signed values. The transpose of
a 32x32 bit matrix is a five-stage mask/shift butterfly. It only permutes bits:
no rounding, floating evaluation, approximate lookup, or transposed model copy.
The shader keeps the columns in workgroup memory and reuses them for VJPs.
Equal-significance popcounts are combined before wide accumulation.

    Y = Qw Qx^T
    dX = Qw^T Qg^T
    dW = Qg^T Qx

W[M,N], X[B,N], G[B,M]. Outputs have shapes [M,B], [N,B], [M,N]. There is no
new basis-term list. These are fixed-size arrays determined at compilation.
Generic expressivity costs M*N independent coefficients; there is no claim of
universal dense expressivity at the old small parameter budget.

All outputs are signed int64 represented by two uint32 words. For b<=16 and
summation length <=32, |sum|<=32*2^30=2^35. Tiled reductions up to length 4096
have |sum|<=2^42. Intermediate bit-significance sums are also far below int64.
Therefore the modular two-word operations are the ordinary exact integer result
in the stated domain, not silently overflowing approximations. Partial reduction
uses the same shader with opcode 2.

The alternative INTEGER_MAC path also reads the same packed model but decodes
integer microtiles to workgroup memory and uses imulExtended. It is integer-only,
not AND-only. Neither path evaluates or emulates floating-point arithmetic.
The DENSE_REFERENCE benchmark instead starts from int32 storage and executes
same-output wide integer products. It is not an optimized vendor GEMM.

## 2. Quantization error versus execution error

Let What=Delta_w Qw, Xhat=Delta_x Qx, Ghat=Delta_g Qg, where Delta=2^-s.
The exact output codes represent What Xhat^T, What^T Ghat^T, Ghat^T Xhat in
real dyadic arithmetic, without an additional output rounding. Output scale is
part of the API contract, not an implicit cast back to narrow integers.

For nearest rounding without clipping, |W-What|<=Delta_w/2 and
|X-Xhat|<=Delta_x/2. Thus

    |(WX^T-What Xhat^T)_ib|
    <= (Delta_x/2) sum_j |W_ij|
     + (Delta_w/2) sum_j |X_bj|
     + N Delta_w Delta_x/4.

The same bilinear bound applies to the VJPs, with their own operands and scales.
This is a numerical guarantee, NOT a guarantee of unchanged task accuracy.
The converter rejects nonfinite input and clipping. Subsequent activation or
upstream-gradient requantization must be counted separately, as in the local
training mode. Reference GEMM/FMA bit patterns are not the target contract.

A classifier's top-1 is certified for a particular input if its original gap
between largest and second-largest logits exceeds twice a valid per-logit error
bound. This certifies those inputs, not unobserved inputs or arbitrary models.

The derivative of an integer rounding map is not this VJP. VJPs are derivatives
of the real-valued bilinear operation at quantized operands. Using them to train
quantized weights is a declared surrogate/quantization-aware update.

## 3. Actual geometry: product of Poincare balls

In geometry mode each weight row z=Qw/2^sw is a point inside the radius-0.9
subset of a curvature -1 Poincare ball. This is a weight-coordinate manifold;
it is not a claim that ordinary linear activations are Mobius matvecs.

    g_z = 4/(1-||z||^2)^2 I
    grad_R L = ((1-||z||^2)^2/4) grad_E L.

v3 signed-XOR frames P preserve ||z|| and hence the Poincare metric. CPU tests
verify exact integer frame/inverse and norm preservation; the GPU optimizer uses
the metric explicitly, rather than just attaching a geometric name to a bit dot.

Let U=2^(2sw), S=sum_j Qw_j^2, D=U-S. The shader computes

    gamma = RNE( 2^16 * D^2 / (4 U^2) ).

All numerator arithmetic is integer. For 1<=sw<=15, U<=2^30 and D^2<=2^60.
The Q0.16 metric approximation satisfies

    |gamma/2^16 - (1-||z||^2)^2/4| <= 2^-17.

This error is explicitly additional metric quantization, not swept into a claim
of exact continuous Riemannian optimization. It contributes at most
2^-17 |grad_E| before step-size multiplication. The row update uses a local
coordinate retraction, z+v, NOT a claimed exact exponential map.

For step size 2^-lr, Qx fractional sx and Qg fractional sg, weight-code step is

    shift = 16+sx+sg+lr-sw,
    delta_Qw = RNE( gamma * dW_integer / 2^shift ).

The guard checks representability and S_new<=floor(0.81 U). If necessary the
step is halved, up to 16 attempts. Rejection leaves the row unchanged. The guard
is a discrete trust-region/backtracking device, not a theorem of convergence.
Initial invalid rows and unsupported scales are rejected by the builder.
No projection to this ball is silently applied when converting arbitrary weights.
Generic converted matrices may use the exact contraction without this optimizer.

## 4. Integer error feedback: retain sub-grid steps

With residual R in numerator units, update using A=gamma*dW+R. On an accepted
unbacktracked step,

    step = RNE(A/2^shift)
    R_new = A-step*2^shift
    Qw_new = Qw-step.

The equality R_new+step*2^shift=A is exact, and |R_new|<=2^(shift-1).
Small updates therefore do not disappear repeatedly. A backtracked or rejected
step explicitly resets its residual; the cumulative identity is not claimed
across that reset. Residuals require 64 additional bits per weight during
training. They are optimizer state, NOT included in inference compression ratios.
A floating-point master weight is not required. The live integer arena is fixed.

## 5. One command / one dispatch: exact scope

A 32-bit command packs opcode, three widths, and M,N,B (each <=32). It is the
first word of a 16-word descriptor with buffer offsets, scales and flags. The
whole model obviously does not fit in this one command word.

Opcode 0: with upstream already available, compute Y, dX, dW in one dispatch.
Opcode 1: local linear/ReLU + squared-error target: compute forward, quantized
upstream, VJPs, metric, guarded update and optional feedback in one dispatch.
Opcode 2: exact int64 reduction of partial results. Opcode 3 is reserved.

One workgroup owns each local job, so shared-memory barriers are sufficient.
SSBO accesses are coherent and buffer barriers precede dependent reads.
Global layers need one tile dispatch + one reduction dispatch; arbitrary deep
network backward cannot use an upstream that has not yet been computed.
The implementation does not spin-wait across workgroups or pretend a global
barrier exists. Feedback training jobs own their weight storage; no optimizer
updates race shared weights between jobs.

## 6. Storage and measured scope

Packed matrix bytes = 24 + 4*b*M*ceil(N/32). At aligned N this approaches b bits
per weight. Arbitrary quantized matrices are preserved; no tied coefficients are
assumed and no pruning takes place. Narrow edge tiles lose padding efficiency.
Two's-complement bit payloads, dtype/scale metadata and dimensions are checked.
Raw FP32 weights are NOT preserved once quantized. State precision and numerical
precision must not be conflated.

Reported timings are ANGLE/SwiftShader software Vulkan, NOT physical GPU.
The AND/popcount implementation is slower than the simple wide-integer reference
in the current timing tests. Integer-MAC cached tiles reduce much of this overhead
but do not establish a general speedup. The accuracy contract is stronger than the
performance evidence. No whole-LLM compression, throughput or accuracy is claimed.

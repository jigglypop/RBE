# RBE State Field v5 — quantized-exact bitfield shader and Riemannian updates

Goal: packed weights + fixed-size states + one reusable integer shader for forward,
input VJP, weight VJP and local Riemannian training, with no additional contraction
error beyond explicitly chosen quantization. This supplements v3/v4; it is not a
claim that a compact Clifford family magically compresses arbitrary dense weights.

## Implemented

`shaders/bitfield.comp` operates on ONE uint32 SSBO arena. A 32-bit instruction
inside each 16-word descriptor declares the operation, precision and tile shape.
The packed weights, input and upstream gradient share a bitplane format. The
AND/popcount path computes exact signed products by their bit significances and
uses a five-stage bit transpose for both reverse contractions. The alternative
`INTEGER_MAC=1` caches quantized integer codes in workgroup memory and uses native
extended integer products. No shader path uses float/double, sin/cos/exp, or an
FP32 emulation routine. Integer multiplication is used in the metric/fast path;
"integer-only" does not mean only XOR is sufficient for arbitrary arithmetic.

A <=32x32 matrix with batch<=32 fits a local workgroup job. With a local squared
error target, ONE dispatch performs forward, optional ReLU, upstream quantization,
input and weight gradients, Poincare metric, radius backtracking, packed-weight
update and error feedback. Fixed 8->8 state width was tested for 800 steps without
new buffer allocation. Generic larger/rectangular layers use the SAME shader for
a tile dispatch followed by an exact reduction dispatch. They require available
upstream gradients; there is no global workgroup barrier or fictitious one-pass
backprop through an arbitrary deep network.

## Results recorded in this environment

- `cpu_tests.json`: 23,595 packed roundtrip coefficients, 480 signed bitplane dot
  checks, 3,904 nearest-even cases, 1,598 integer Poincare factors, 250 frame
  isometry checks and malformed input guards.
- `shader_tests.json`: 9 compiled shader specializations; 137,826 exact int64
  forward/VJP outputs, including sums up to 2^35, plus 1,150 exact metric/update
  values. Tested 2..16-bit mixed widths, extrema and non-square tails.
- `tiled_tests.json`: 17,878 additional exact outputs for general rectangular
  matrices. One packed model copy, no global transposed model buffer.
- `train_feedback.json`: 12-bit weights, 8-bit input, 12-bit upstream, integer
  Riemannian/ReLU training. Synthetic loss 0.0059870 -> 1.10147e-8 in 800 steps.
  Without error feedback it stalls at 2.53452e-5. Residual feedback costs int64
  optimizer storage per weight. Full arena with feedback: 9,604 bytes.
- `digits.json`: sklearn digits, fixed stratified split 1,347 train / 450 test.
  A trained classifier INCLUDING its intercept was converted and all logits were
  computed by the integer shader. FP32 accuracy 95.333%; 8/12/16-bit weights kept
  all 450 top-1 predictions. Four bits dropped to 92.667%. This classifier was
  fit by sklearn, not by the Riemannian shader. It is NOT an LLM benchmark.
- 16-bit digit conversion additionally passes a per-example error/margin
  certificate for all 450 held-out inputs; 8-bit certificates cover 430/450.
  The remaining 8-bit predictions agree empirically, not by that loose bound.

## Exactness contract

Quantized coefficients are stored exactly; int64 contraction/VJP accumulation is
exact within validated width/size bounds. Dyadic output scales are returned as
metadata. This is NOT unchanged raw FP32 weights or equality to an arbitrary
hardware GEMM's rounding graph. Quantization, upstream requantization, Q0.16 metric
rounding and optimizer projection/rounding are declared separately. Training uses
the real-bilinear VJP at quantized operands, not a false derivative of rounding.
See `MATH.md` for equations, bounds and the fixed-width contract.

## Compression and speed

A 256x256 FP32 matrix (262,144 bytes) becomes 65,560 bytes at 8 bits (3.9985:1),
98,328 at 12 bits (2.6660:1), 131,096 at 16 bits (1.9996:1), including headers.
These are quantized storage ratios, not lossless raw-FP32 compression. The small
trained digit classifier has edge-padding overhead: 2,600 -> 984 bytes at 8 bits
(2.6423:1), including all affine parameters and the 24-byte file header.
Optimizer buffers, activations and transient transpose workspace are not model
weights; their sizes are reported separately, not hidden in compression numbers.

All current shader runs use ANGLE / SwiftShader SOFTWARE Vulkan. The AND/popcount
path is slower than the same-output simple int32-storage reference here. Cached
integer-MAC is faster than AND/popcount, especially at 12/16 bits, but is not
consistently faster than that reference. No physical-GPU speedup is claimed.
`benchmark.json` publishes every timing, including slower cases. Benchmark compile,
pack/plan, allocation and transfer exclusions are specified. A hardware run against
an optimized integer GEMM remains necessary before making speed claims.

## Run

Python 3.11+, numpy; sklearn is needed only for the external-data test. The local
archive includes the previously supplied v3 `src/gl_base.py` EGL adapter. The Git
publication contains the shader, goals, math and evidence, but not the Python
helpers or that adapter: a helper-file write was blocked by a connector security
decision. The commands below apply to the complete local implementation archive.
No alternative upload of blocked helper files was attempted.
No system install or driver changes are performed by these scripts.

```
python tests/test_cpu.py
python tests/test_shaders.py --angle-dir /path/to/angle
python tests/test_tiled.py --angle-dir /path/to/angle
python tools/train.py --feedback --angle-dir /path/to/angle
python tools/train.py --feedback --integer-mac --angle-dir /path/to/angle
python tools/verify_digits.py --angle-dir /path/to/angle
python tools/bench.py --angle-dir /path/to/angle
python tools/audit.py
```

Software correctness runs must pass `--software`, e.g. `--software --angle-dir
/usr/lib/chromium`. Software adapters are rejected otherwise. Windows/Cargo/full
Rust-model integration and physical GPUs were not verified in this environment.

## Next algorithm priorities

1. Keep the exact quantized contract, but choose bitplane versus packed native
   integer kernels by device and bit width. Do not infer speed from instruction
   counts or compression. Investigate subgroup transpose and hardware packed
   integer dot products on a physical GPU, with this scalar-int64 oracle.
2. Connect v3 signed-XOR/2x2 structured operators to these integer coefficient
   kernels, so structure eliminates products rather than merely emulating them.
   Account for basis-transform rounding rather than calling it raw-bit lossless.
3. Select precision from a declared activation error budget and separate locked
   test sets; keep 12/16 bits for sensitive blocks. Four-bit accuracy is NOT
   generally sufficient. Measure task-level loss and complete-model bytes.
4. Extend multi-tile optimizer scheduling with globally reduced weight gradients.
   The local single-dispatch optimizer must not update duplicated shared weights.

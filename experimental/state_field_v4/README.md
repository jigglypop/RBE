# RBE v4: full linear expressivity and truly lossless weight words

## What improved

The activation width stays N. For N=8, v3 left multiplication spans 8 linear
map directions; two-sided maps span 32; coupling the two central sectors spans
all 64. Equivalently the complete signed-XOR basis P(a,b) spans every real NxN
matrix. Full expressivity requires up to N^2 independent coefficients; it is NOT
full dense expressivity for the old N-parameter budget. The explicit complete
basis implementation is a small proof/training oracle, not a fast GPU kernel.

The v3 real block isomorphism is not bit-invertible after every stage rounds to
f32. A regression counterexample retains 16777216 but changes a component 1 to
0.5 on roundtrip. v4 lossless storage NEVER applies that rounded transform.

## Three exact word-storage formats

`wordcodec.py` works on uint32 bit patterns, not approximate float comparisons.
A XOR diagonal is W[j XOR a,j]. If its words differ only by a parity-controlled
sign, store one coefficient word and one mask. When all masks follow the
Clifford signature, omit the masks too. Every unmatched diagonal stays RAW.
A whole-matrix RAW escape wins whenever compressed metadata would be larger.
NaN payloads, signed zero, subnormals and infinities roundtrip unchanged.
No pruning, epsilon fitting, INT4 conversion or missing tensor escape is used.

Actual matrix storage includes the 24-byte header:

| 64x64 synthetic input | Raw bytes | Stored bytes | Ratio |
|---|---:|---:|---:|
| exact Clifford-tied |16384|280|58.51:1|
| exact general signed-XOR diagonals |16384|536|30.57:1|
| quarter of diagonals unstructured |16384|4632|3.54:1|
| random f32 |16384|16408|0.9985:1|

These inputs were constructed for mechanism testing, not measured pretrained
LLM layers. Generic weights are not assumed to have these ties.

`modelpack.py` preserves every tensor in a local NPZ checkpoint. Unsupported
shapes and dtypes stay verbatim. The six-tensor synthetic checkpoint measured
49,920 raw tensor bytes -> 18,676 file bytes (2.67295:1), including its complete
index and all biases. All dtype, shape and payload bytes roundtripped.
Architecture and external code are not included unless present as tensors.

```sh
python tests/test_cpu.py
python src/modelpack.py /path/to/checkpoint.npz --output checkpoint.rbemod
```

## Shader contract

`wordlinear.comp` reads encoded words directly; no full dense weight buffer is
created. Bind input at 0, encoded payload at 1, output at 3. p=(N,batch,kind,
signature). Inputs/outputs use [state][batch] order. Define TRANSPOSE=0 for
forward, 1 for input VJP. STRICT_BITS=1 inserts v3/shaders/fp32_bits.glsl at the
marked position and uses only uint variables for numerical arithmetic.
This shader does NOT implement optimizer steps or compressed weight VJPs.

Strict output identity is only against ascending-j, separate binary32 multiply
and add with nearest-even and canonical NaN. It is not identity with arbitrary
BLAS/FMA/reassociated execution. Mathematical-real equality is a different
contract. The native path is faster but tested with a numerical tolerance.

The EGL/ctypes host adapter was not published. Local tests used the supplied
v3 adapter and ANGLE/SwiftShader: 17,728 exact decoded words and 4,928 same-order
strict forward/input-VJP outputs passed. The independent CPU word codec and
complete-basis tests are runnable directly from this tree.

## Speed is separate from compression

At N=64,batch=256 on software SwiftShader, dense native was 4.763ms, packed
native 4.479ms, packed uint 25.267ms (medians; dispatch+completion, allocation
excluded). The packed storage did not remove the N^2 coefficient products in
this general same-order shader. No physical GPU or end-to-end LLM speedup is
claimed. The faster v3 block path has a narrower numerical identity contract.

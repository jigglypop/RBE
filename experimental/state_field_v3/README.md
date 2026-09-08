# RBE closed matrix layout v3

Published core of the previously supplied v3 experiment. State width is fixed;
Cl(2,1) is faithfully represented by TWO 2x2 real blocks. It is not an arbitrary
8x8 matrix parameterization. v4 supplies the expressivity analysis.

## Included

- `src/layout.py`: signed-XOR frames, composition, transpose, exact algebraic
  block map, reverse-mode derivatives and safe nonlinear frame rewriting.
- `src/frame_api.*`: C ABI for 32/64-bit payload routing with range/alias checks.
- `shaders/`: standalone GLSL ES 3.10 frame, block, layout and uint binary32
  arithmetic kernels. MODE and STRICT_BITS are compile-time definitions.
- CPU and linked C++ tests; local shader revalidation results.

The EGL/ctypes host adapter is NOT included in this publication. Reported shader
runs used the prior local adapter, on software ANGLE/SwiftShader, not hardware.
This omission is not evidence that the standalone kernels were hardware-tested.

## Run

```sh
python tests/test_cpu.py
python tests/test_rewrite.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Install NumPy from requirements.txt first. Tests write results/*.json.

## Verified scope

The current revalidation passed 17,472 frame compositions, 64 basis products,
16,384 mixed integer coefficients, 400 rewrite programs and 124,000 C++ word
checks. Local software shader revalidation passed 1,190,378 strict binary32
comparisons under canonical NaN semantics, plus native VJP comparisons.

Algebraic exactness does NOT mean a rounded f32 change of basis preserves every
original word. v4 includes a counterexample and a separate true word codec.
STRICT_BITS emulates specified binary32 multiply/add; it is not infinite-real
arithmetic and does not promise equality with FMA or a different reduction tree.

Historical software-renderer benchmark at batch 16384: direct full product
3.684ms, resident native blocks 0.802ms, uint blocks 3.791ms. Those are kernel
measurements from the supplied v3 results, not a new hardware measurement or an
LLM speedup. Repeated layout conversions can erase the benefit.

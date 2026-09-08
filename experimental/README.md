# RBE state-field experiments: published 2026-09-09

`state_field_v3` contains the signed-XOR frame algebra, Cl(2,1) 2x2 block
layout, C ABI, standalone GLSL kernels and CPU/C++ regression tests.
`state_field_v4` adds complete linear-map expressivity tests, bit-exact word
compression with RAW escape, whole-NPZ tensor packing and direct encoded-word
forward/input-VJP shaders.

These are experimental cores, not a replacement for the existing Rust model.
No pretrained LLM accuracy, whole-model throughput or physical GPU speedup is
claimed. The EGL/ctypes host adapter is not part of this publication; local
shader measurements used the prior supplied adapter with ANGLE/SwiftShader.
The standalone shader sources, binding contracts and measured results are
included. CPU/C++ tests run without that adapter.

```sh
python experimental/state_field_v3/tests/test_cpu.py
python experimental/state_field_v3/tests/test_rewrite.py
cmake -S experimental/state_field_v3 -B build-rbe-state -DCMAKE_BUILD_TYPE=Release
cmake --build build-rbe-state
ctest --test-dir build-rbe-state --output-on-failure
python experimental/state_field_v4/tests/test_cpu.py
```

Only the new experimental subtree is changed. v1/v2 archives are historical
inputs to v3, not separately installed here.

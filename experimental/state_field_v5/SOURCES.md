# Sources and provenance

The repository base was remotely verified as jigglypop/RBE main
1d5e34f30234313f65c2720c489169340c17daf3. v4 wordlinear.comp supplies forward
and input VJP and emulates FP32 in strict mode; v5 adds a different quantized
integer coefficient backend with weight VJP and local optimizer.

Primary references consulted (not evidence that this implementation is novel):

- Bécigneul and Ganea, Riemannian Adaptive Optimization Methods,
  https://arxiv.org/abs/1810.00760 . Product-manifold optimization and Poincare
  geometry background. The integer metric quantization and discrete guard here
  require their own tests; the paper's convergence results are NOT transferred.
- Wu et al., Training and Inference with Integers in Deep Neural Networks,
  https://arxiv.org/abs/1802.04680 . Integer training is prior art.
- Li and Gupta, Bit-serial Weight Pools,
  https://arxiv.org/abs/2201.11651 . Bit-serial execution is prior art.
- Khronos OpenGL ES Shading Language reference (integer functions, barriers),
  https://registry.khronos.org/OpenGL/specs/es/3.2/GLSL_ES_Specification_3.20.html .
  The implementation targets ES 3.10; all used operations were compiled on ES3.1.
- sklearn optical handwritten digits dataset documentation,
  https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset .
  The experimental classifier and its split are specified in tools/verify_digits.py.

All results/ JSON files are outputs of local execution, not sourced performance
claims. Physical GPU performance and whole-LLM task accuracy remain unmeasured.
The unchanged EGL/ctypes adapter is from the user's v3 implementation archive.

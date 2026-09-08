# Closed frame and block contracts

For N=2^d and a,b in [0,N), define

P(a,b,s)e_j = (-1)^(s + parity(b & j)) e_(j XOR a).

Then P(a,b,s)P(c,d,t)=P(a XOR c,b XOR d,
s XOR t XOR parity(b & c)). Transpose/inverse retains a,b and XORs the sign
with parity(a & b). These are exact discrete identities on the labels.
The C ABI routes payload words without numerical conversion, including signed
zero and NaN payload. A SUM of frames is generally not ONE frame.

For Cl(2,1), e1=diag(1,-1), e2=[[0,1],[1,0]], and e3 is either +e1e2 or
-e1e2. Both representations together are faithful. Multiplying the generator
matrices in blade order constructs the explicit 8x8 map H in src/layout.py.
H^T H=4I. If B=Hx is reshaped as two 2x2 matrices,

B(xy)=(B_+(x)B_+(y), B_-(x)B_-(y)).

The inverse is H^T/4; the adjoint is H^T, not H^T/4. For a coefficient-space
output cotangent g, the block cotangent is Hg/4. Matrix product reverse-mode is
G_A=G B^T and G_B=A^T G; returning to coefficient gradients uses H^T.
A matching coefficient-coordinate SGD trajectory uses a factor 4 in the block
learning rate. The supplied tests distinguish these scales.

Only diagonal character frames D_b=P(0,b,0) satisfy D_b(xy)=D_b(x)D_b(y).
They can be deferred across Clifford square. A global minus disappears under
square. A general XOR-address frame cannot be deferred: the rewrite compiler
emits it before the barrier and tests an exact counterexample.

This is exact real algebra, not a promise of f32 evaluation-order invariance.
Ordinary numerical coefficient arithmetic remains necessary. The uint kernels
implement binary32 nearest-even add/multiply, with subnormals and canonical NaN;
flags, other rounding modes and NaN payload propagation in arithmetic are not
implemented. Lossless STORAGE of arbitrary word payload is a separate v4 API.

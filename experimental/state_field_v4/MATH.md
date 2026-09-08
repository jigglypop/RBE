# Expressivity, lossless storage, and execution are separate contracts

## Complete finite-dimensional matrix basis

P(a,b)e_j=(-1)^parity(b&j) e_(j XOR a), a,b=0,...,N-1, N=2^d.
For a!=c, P(a,b) and P(c,f) have disjoint nonzero matrix entries. For a=c,
their Frobenius product is sum_j (-1)^parity((b XOR f)&j), which is N if b=f
and zero otherwise. Hence N^2 orthogonal matrices form a basis of M_N(R):

W = sum_(a,b) theta_(a,b) P(a,b),
theta_(a,b) = Tr(P(a,b)^T W)/N.

An explicit matrix unit is
E_(k,j) = (1/N) sum_b (-1)^parity(b&j) P(k XOR j,b).
Activation width is N throughout; the number of independent coefficients is
N^2 in the universal case. Compact coefficient families remain strict subsets.
The numerical finite-basis oracle limits N<=64 to avoid accidental large basis
allocations. Real coefficient transforms are not the word-level encoder.

For Cl(2,1)=M2(R) direct-sum M2(R), left multiplication has dimension 8.
The span of left-right actions is End(M2) direct-sum End(M2), dimension 32.
The e3-sign automorphism interchanges central sectors. Adding its composed
left-right actions gives all 64 real linear maps on the 8-dimensional state.
The test constructs both spans and independently verifies their dimensions.

## Exact word codec

For a XOR diagonal a, seek u_a,b_a such that
bits(W[j XOR a,j]) = u_a XOR (parity(b_a&j)<<31).
No tolerance is allowed. Infer b from unit-bit columns, then verify ALL words.
The compact Clifford mode also checks every b_a against the signature formula.
Unmatched data use verbatim words. Choose the shortest candidate including the
header. No arithmetic acts on original coefficient words during compression.
The decoder validates dimensions, lengths, offsets and allocation limits.

A universally injective lossless encoder cannot shorten every m-bit string:
there are 2^m possible inputs and only 2^m-1 strings of length below m.
Geometry does not change this count. Large ratios require actual repeated
structure or other redundancy, not merely an invertible matrix arrangement.
RAW escape keeps full representability, not guaranteed compression.

## Execution identity

A f32 change of basis plus its inverse can round away low bits despite an exact
real isomorphism. Reassociating sums or replacing mul+add with FMA can likewise
change output bits. Strict mode therefore keeps the original input-index order
and emulates separate binary32 mul/add. Its NaNs are canonical; arbitrary NaN
payload preservation applies to storage and routing, not arithmetic.
The same-order dense and encoded paths read identical coefficient words and
apply the same scalar operations, so their specified output words agree.
This does not prove equality with a different hardware GEMM execution graph.

For model-level accounting, sum all weights, biases and other tensors plus
actual format metadata. If fraction f is compressible by r and the rest stays
raw, ideal payload compression is 1/((1-f)+f/r), before metadata. Kernel speed
likewise does not equal end-to-end speed; unaccelerated work remains.

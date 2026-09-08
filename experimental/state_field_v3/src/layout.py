"""RBE v3: exact matrix-label closure and a faithful Cl(2,1) block layout.

Discrete labels are not trainable real numbers. General coefficients remain
binary32/binary64. The block map is an isomorphism of the v2 algebra over R,
not a claim that different floating point evaluation orders are bit identical.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MASK15 = (1 << 15) - 1

@dataclass(frozen=True)
class Frame:
    """P|j> = (-1)**(s + parity(b&j)) |j XOR a>, packed in 31 bits."""
    a: int = 0
    b: int = 0
    s: int = 0
    def __post_init__(self):
        if not (0 <= self.a <= MASK15 and 0 <= self.b <= MASK15 and self.s in (0, 1)):
            raise ValueError('a,b need 15 bits; sign needs one bit')
    def pack(self) -> int:
        return self.a | (self.b << 15) | (self.s << 30)
    @classmethod
    def unpack(cls, word: int) -> 'Frame':
        if not 0 <= int(word) < (1 << 31):
            raise ValueError('reserved bit31 must be zero')
        return cls(int(word)&MASK15, (int(word)>>15)&MASK15, (int(word)>>30)&1)
    def compose(self, rhs: 'Frame') -> 'Frame':
        """self @ rhs (rhs acts first). All arithmetic is discrete/exact."""
        return Frame(self.a ^ rhs.a, self.b ^ rhs.b,
                     self.s ^ rhs.s ^ ((self.b & rhs.a).bit_count() & 1))
    def transpose(self) -> 'Frame':
        return Frame(self.a, self.b, self.s ^ ((self.a & self.b).bit_count() & 1))
    inverse = transpose
    def dense(self, d: int) -> np.ndarray:
        self.validate_d(d)
        n=1<<d; m=np.zeros((n,n),dtype=np.int64)
        for j in range(n): m[j ^ self.a,j]=1-2*(self.s ^ ((self.b&j).bit_count()&1))
        return m
    def validate_d(self, d: int):
        if not 1 <= d <= 15 or (self.a | self.b) >= (1 << d):
            raise ValueError('label must fit d=1..15')
    def route_words(self, words: np.ndarray, d: int, width: int=1) -> np.ndarray:
        """Exact sign/permutation on binary32 or binary64 payloads, no arithmetic.
        Flat input storage is [batch, state, word(lo[,hi])]. NaN payload retained.
        """
        self.validate_d(d); n=1<<d
        if width not in (1,2): raise ValueError('width must be 1 or 2')
        x=np.asarray(words,dtype=np.uint32)
        if x.size==0 or x.size%(n*width): raise ValueError('wrong carrier size')
        x=x.reshape(-1,n,width)
        j=np.arange(n,dtype=np.uint32)^np.uint32(self.a)
        y=x[:,j,:].copy()
        for i in range(n):
            flip=self.s ^ ((self.b & int(j[i])).bit_count() & 1)
            y[:,i,width-1] ^= np.uint32(flip << 31)
        return y.reshape(-1)

def blade_frame(a: int, negative: int, d: int) -> Frame:
    if not 1<=d<=15 or not 0<=a<(1<<d) or not 0<=negative<(1<<d):
        raise ValueError('invalid blade/signature')
    mask=a & negative
    for i in range(d):
        if (a>>i)&1: mask ^= (1<<i)-1
    return Frame(a,mask)

def fuse(frames) -> Frame:
    """frames listed in execution order; stop at nonlinear/additive barriers."""
    result=Frame()
    for f in frames: result=f.compose(result)
    return result

# Generator representation with e1^2=e2^2=+I, e3^2=-I.
_I=np.eye(2,dtype=np.int64)
_Z=np.array([[1,0],[0,-1]],dtype=np.int64)
_X=np.array([[0,1],[1,0]],dtype=np.int64)
_J=_Z@_X

def _build_H():
    columns=[]
    for blade in range(8):
        reps=[]
        for s in (1,-1):
            m=_I.copy()
            for i,e in enumerate((_Z,_X,s*_J)):
                if (blade>>i)&1: m=m@e
            reps.extend(m.reshape(-1))
        columns.append(reps)
    return np.array(columns,dtype=np.int64).T
H=_build_H() # B=H x; H.T H=4I. Two row-major 2x2 blocks.

def to_blocks(x):
    x=np.asarray(x)
    if x.shape[-1]!=8: raise ValueError('Cl(2,1) has exactly 8 coefficients')
    return (x@H.T).reshape(*x.shape[:-1],2,2,2)

def from_blocks(b):
    b=np.asarray(b)
    if b.shape[-3:]!=(2,2,2): raise ValueError('expected (...,2,2,2)')
    return (b.reshape(*b.shape[:-3],8)@H)/4

def product(a,b):
    return from_blocks(to_blocks(a)@to_blocks(b))

def product_vjp(a,b,g):
    """Gradient in ORIGINAL coefficient coordinates; adjoint != inverse."""
    A,B=to_blocks(a),to_blocks(b)
    G=to_blocks(g)/4 # adjoint of inverse H^T/4
    ga=G@np.swapaxes(B,-1,-2)
    gb=np.swapaxes(A,-1,-2)@G
    return ga.reshape(*ga.shape[:-3],8)@H, gb.reshape(*gb.shape[:-3],8)@H

def as_soa(blocks, dtype=np.float32):
    a=np.asarray(blocks,dtype=dtype).reshape(-1,8)
    return np.ascontiguousarray(a.T).reshape(-1)

def from_soa(words, batch, dtype=np.float32):
    a=np.asarray(words,dtype=dtype)
    if a.size!=8*batch: raise ValueError('SoA size mismatch')
    return a.reshape(8,batch).T.reshape(batch,2,2,2)

def gp_oracle(x,y,negative=4):
    """Independent generator-word insertion reduction, not the block formula."""
    x=np.asarray(x);y=np.asarray(y)
    if x.shape!=y.shape or x.shape[-1]!=8: raise ValueError('matching (...,8) required')
    out=np.zeros_like(x,dtype=np.result_type(x,y))
    for a in range(8):
        for b in range(8):
            word=[i for i in range(3) if (a>>i)&1]; sign=1
            for j in range(3):
                if not ((b>>j)&1): continue
                pos=len(word)
                while pos>0 and word[pos-1]>j: pos-=1;sign=-sign
                if pos>0 and word[pos-1]==j:
                    word.pop(pos-1)
                    if (negative>>j)&1: sign=-sign
                else: word.insert(pos,j)
            dest=sum(1<<i for i in word)
            out[...,dest]+=sign*x[...,a]*y[...,b]
    return out

def compile_program(ops):
    """Fuse frame matrices; defer diagonal character frames across x*x.

    Exact for the v2 Clifford square, because D_b(xy)=(D_b x)(D_b y).
    A global minus also disappears on a square. General XOR-address frames
    are NOT automorphisms and must be materialized before this barrier.
    Nonlinear operations other than 'square' are deliberately unsupported.
    """
    pending=Frame();result=[]
    for op in ops:
        if isinstance(op,Frame):
            pending=op.compose(pending)
        elif op=='square':
            if pending.a==0:
                pending=Frame(0,pending.b,0) # overall sign squared away
                result.append('square')
            else:
                if pending!=Frame():result.append(pending)
                result.append('square');pending=Frame()
        else:
            raise ValueError('unknown operation; no unsafe rewrite across arbitrary nonlinearities')
    if pending!=Frame():result.append(pending)
    return result

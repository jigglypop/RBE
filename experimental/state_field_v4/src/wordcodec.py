"""Bit-exact binary32 matrix storage. No floating-point arithmetic in codec.

Each XOR diagonal is either a signed-generator word or verbatim uint32 data.
A Clifford-specific form also omits derivable masks. RAW escape accepts every
word pattern, including signed zero and NaN payload. This is not a universal
compression guarantee and does not change floating-point evaluation order.
"""
from __future__ import annotations
from dataclasses import dataclass
import struct
import numpy as np

HEADER=struct.Struct('<8sIIII')
MAGIC=b'RBEWORD4'
RAW,MIXED,CLIFFORD=0,1,2
SIGN=np.uint32(0x80000000)


def dimension(n: int) -> int:
    if not isinstance(n,(int,np.integer)) or not 2<=n<=32768 or n&(n-1):
        raise ValueError('dimension must be a power of two in [2,32768]')
    return int(n).bit_length()-1


def mask_for_blade(a: int, negative: int, d: int) -> int:
    b=a&negative
    for i in range(d):
        if a&(1<<i): b^=(1<<i)-1
    return b


def clifford_words(coeff,negative: int) -> np.ndarray:
    c=np.asarray(coeff)
    if c.dtype!=np.uint32 or c.ndim!=1: raise TypeError('uint32 coefficients required')
    n=c.size;d=dimension(n)
    if not 0<=negative<n: raise ValueError('invalid signature')
    p=np.array([j.bit_count()&1 for j in range(n)],dtype=np.uint32)
    j=np.arange(n,dtype=np.uint32);w=np.empty((n,n),dtype=np.uint32)
    for a in range(n):
        w[j^a,j]=c[a]^(p[j&mask_for_blade(a,negative,d)]<<np.uint32(31))
    return w


@dataclass(frozen=True)
class Packet:
    n: int
    kind: int
    auxiliary: int
    words: np.ndarray

    def __post_init__(self):
        dimension(self.n)
        w=np.asarray(self.words)
        if w.dtype!=np.uint32 or w.ndim!=1: raise TypeError('one-dimensional uint32 payload required')
        if self.kind==RAW:
            if self.auxiliary!=0 or w.size!=self.n*self.n: raise ValueError('raw shape')
        elif self.kind==CLIFFORD:
            if not 0<=self.auxiliary<self.n or w.size!=self.n: raise ValueError('Clifford shape')
        elif self.kind==MIXED:
            if self.auxiliary!=0 or w.size<2*self.n: raise ValueError('mixed shape')
            for a in range(self.n):
                tag=int(w[2*a])
                if tag&0x80000000:
                    off=tag&0x7fffffff
                    if off<2*self.n or off+self.n>w.size or w[2*a+1]!=0:
                        raise ValueError('raw diagonal offset')
                elif tag>=self.n: raise ValueError('invalid parity mask')
        else: raise ValueError('unknown format')
        # Immutable snapshot: descriptors cannot be changed after validation.
        w=np.ascontiguousarray(w).copy();w.flags.writeable=False
        object.__setattr__(self,'words',w)

    @property
    def nbytes(self): return HEADER.size+self.words.nbytes

    def to_bytes(self):
        return HEADER.pack(MAGIC,self.n,self.kind,self.auxiliary,self.words.size)+self.words.astype('<u4',copy=False).tobytes()

    @classmethod
    def from_bytes(cls,data: bytes,max_decoded_bytes=64*1024*1024):
        if len(data)<HEADER.size: raise ValueError('truncated header')
        magic,n,kind,aux,count=HEADER.unpack_from(data)
        if magic!=MAGIC or len(data)!=HEADER.size+4*count: raise ValueError('invalid length/magic')
        dimension(n)
        if 4*n*n>max_decoded_bytes: raise ValueError('decoded-size safety limit')
        return cls(n,kind,aux,np.frombuffer(data,dtype='<u4',offset=HEADER.size).astype(np.uint32))

    def word(self,i: int,j: int) -> int:
        if not (0<=i<self.n and 0<=j<self.n): raise IndexError('matrix coordinate')
        if self.kind==RAW: return int(self.words[i*self.n+j])
        a=i^j
        if self.kind==CLIFFORD:
            b=mask_for_blade(a,self.auxiliary,dimension(self.n));v=int(self.words[a])
        else:
            tag=int(self.words[2*a])
            if tag&0x80000000: return int(self.words[(tag&0x7fffffff)+j])
            b=tag;v=int(self.words[2*a+1])
        return v^(((b&j).bit_count()&1)<<31)

    def decode(self,max_decoded_bytes=64*1024*1024):
        if 4*self.n*self.n>max_decoded_bytes: raise ValueError('decoded-size safety limit')
        return np.array([self.word(i,j) for i in range(self.n) for j in range(self.n)],dtype=np.uint32).reshape(self.n,self.n)


def encode(matrix: np.ndarray) -> Packet:
    w=np.asarray(matrix)
    if w.dtype!=np.uint32 or w.ndim!=2 or w.shape[0]!=w.shape[1]:
        raise TypeError('square uint32 matrix required; use float32.view(uint32)')
    n=w.shape[0];d=dimension(n)
    candidates=[Packet(n,RAW,0,np.ascontiguousarray(w).reshape(-1))]
    j=np.arange(n,dtype=np.uint32)
    parity=np.array([i.bit_count()&1 for i in range(n)],dtype=np.uint32)
    descriptors=np.zeros(2*n,dtype=np.uint32);raw=[];masks=[];coeff=[]
    for a in range(n):
        v=w[j^a,j];v0=int(v[0]);b=0
        for i in range(d): b|=((int(v[1<<i])^v0)>>31)<<i
        predicted=np.uint32(v0)^(parity[j&b]<<np.uint32(31))
        exact=np.array_equal(v,predicted)
        masks.append(b if exact else None);coeff.append(v0)
        if exact: descriptors[2*a:2*a+2]=(b,v0)
        else:
            descriptors[2*a]=0x80000000|(2*n+len(raw));raw.extend(map(int,v))
    candidates.append(Packet(n,MIXED,0,np.concatenate([descriptors,np.array(raw,dtype=np.uint32)])))
    if all(b is not None for b in masks):
        negative=sum(((masks[1<<i]>>i)&1)<<i for i in range(d))
        if all(masks[a]==mask_for_blade(a,negative,d) for a in range(n)):
            candidates.append(Packet(n,CLIFFORD,negative,np.array(coeff,dtype=np.uint32)))
    return min(candidates,key=lambda p:p.nbytes)

"""Complete signed-XOR matrix basis; real-algebra proof, not bit-exact codec.

Using all N^2 coefficients removes the v3 left-action restriction without
increasing activation width N. It also removes the guaranteed parameter saving.
"""
from __future__ import annotations
import numpy as np


def frame(a,b,n):
    if n<2 or n&(n-1) or not (0<=a<n and 0<=b<n): raise ValueError('frame size')
    j=np.arange(n);p=np.zeros((n,n),dtype=np.int64)
    p[j^a,j]=[1-2*((b&int(k)).bit_count()&1) for k in j]
    return p


def basis(n):
    if n>64: raise ValueError('explicit proof oracle limited to N<=64')
    return np.array([frame(a,b,n) for a in range(n) for b in range(n)])


def coefficients(w):
    w=np.asarray(w,dtype=np.float64)
    if w.ndim!=2 or w.shape[0]!=w.shape[1]: raise ValueError('square matrix')
    n=w.shape[0];return np.einsum('kij,ij->k',basis(n),w)/n


def reconstruct(theta,n):
    t=np.asarray(theta,dtype=np.float64)
    if t.shape!=(n*n,): raise ValueError('N^2 coefficients required')
    return np.einsum('k,kij->ij',t,basis(n))


def forward(theta,x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2: raise ValueError('batch,state input')
    n=x.shape[1];return x@reconstruct(theta,n).T


def backward(theta,x,g):
    x=np.asarray(x,dtype=np.float64);g=np.asarray(g,dtype=np.float64)
    if x.shape!=g.shape or x.ndim!=2: raise ValueError('gradient shape')
    n=x.shape[1]
    return g@reconstruct(theta,n),np.einsum('bi,kij,bj->k',g,basis(n),x)

from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT.parent/'state_field_v3/src'))
from wordcodec import *
import expressivity as ex
from layout import H,blade_frame,gp_oracle
rng=np.random.default_rng(20260909)
report={};n=8;B=ex.basis(n);flat=B.reshape(n*n,-1)
assert np.array_equal(flat@flat.T,n*np.eye(n*n,dtype=int))
left=np.array([blade_frame(a,4,3).dense(3) for a in range(8)])
assert np.linalg.matrix_rank(left.reshape(8,-1))==8
right=np.array([np.column_stack([gp_oracle(e,f) for e in np.eye(8,dtype=int)]) for f in np.eye(8,dtype=int)])
lr=np.array([a@b for a in left for b in right]);swap=ex.frame(0,4,8)
assert np.linalg.matrix_rank(lr.reshape(64,-1))==32
assert np.linalg.matrix_rank(np.concatenate([lr,lr@swap]).reshape(128,-1))==64
report['linear_map_dimensions']={'left_only':8,'left_and_right_span':32,'cross_sector_coupled_span':64,'complete_signed_xor_basis':64,'activation_width':8}
# Integer targets divided by N=8 have exact dyadic coefficients here.
exact=0
for _ in range(40):
 w=rng.integers(-100,101,(8,8));t=ex.coefficients(w)
 assert np.array_equal(ex.reconstruct(t,8),w);exact+=64
report['exact_integer_matrix_entries']=exact
# Complete span fits an arbitrary teacher on a full-rank calibration input.
w=rng.normal(size=(8,8));x=np.eye(8);teacher=x@w.T;t=np.zeros(64)
_,dt=ex.backward(t,x,ex.forward(t,x)-teacher);t-=dt/8
report['complete_teacher_max_error']=float(np.max(np.abs(ex.forward(t,x)-teacher)))
assert report['complete_teacher_max_error']<1e-14
projection=np.einsum('aij,ij->a',left,w)/8
wl=np.einsum('a,aij->ij',projection,left)
report['same_teacher_left_only_relative_error']=float(np.linalg.norm(wl-w)/np.linalg.norm(w))
# Input and coefficient reverse-mode derivatives.
x=rng.normal(size=(5,8));g=rng.normal(size=(5,8));t=rng.normal(size=64)
gx,gt=ex.backward(t,x,g);err=0.;h=1e-5
for k in range(64):
 p=t.copy();p[k]+=h;m=t.copy();m[k]-=h
 fd=np.sum((ex.forward(p,x)-ex.forward(m,x))*g)/(2*h)
 err=max(err,abs(fd-gt[k]))
for b in range(5):
 for k in range(8):
  p=x.copy();p[b,k]+=h;m=x.copy();m[b,k]-=h
  fd=np.sum((ex.forward(t,p)-ex.forward(t,m))*g)/(2*h)
  err=max(err,abs(fd-gx[b,k]))
report['complete_vjp_max_error']=err;assert err<1e-7
# The real H is invertible; rounding each stage to f32 is NOT bit-invertible.
z=np.array([2**24,1,0,0,0,0,0,0],dtype=np.float32)
z1=(H@z).astype(np.float32);z2=(H.T@z1/4).astype(np.float32)
assert not np.array_equal(z.view(np.uint32),z2.view(np.uint32))
report['float_basis_lossless_counterexample']={'original':z.tolist(),'roundtrip':z2.tolist()}
checks=0;rows=[]
for n in (8,64,256):
 d=dimension(n);c=rng.normal(size=n).astype(np.float32).view(np.uint32)
 tied=clifford_words(c,1<<(d-1))
 # General signed-XOR family, not tied to one Clifford signature.
 general=tied.copy();j=np.arange(n)
 for a in range(n):
  mask=int(rng.integers(n));general[j^a,j]=c[a]^np.array([((mask&int(v)).bit_count()&1)<<31 for v in j],dtype=np.uint32)
 mixed=tied.copy()
 for a in range(n//4):mixed[j^a,j]=rng.integers(0,2**32,n,dtype=np.uint32)
 for name,w in [('clifford_exact',tied),('general_frame_exact',general),('mixed_quarter_raw',mixed),('random_float',rng.normal(size=(n,n)).astype(np.float32).view(np.uint32)),('random_bits',rng.integers(0,2**32,(n,n),dtype=np.uint32))]:
  p=encode(w);q=Packet.from_bytes(p.to_bytes());assert np.array_equal(q.decode(),w)
  assert p.nbytes<=w.nbytes+HEADER.size
  checks+=w.size
  rows.append({'n':n,'source':name,'mode':p.kind,'raw_bytes':w.nbytes,'stored_bytes':p.nbytes,'ratio':w.nbytes/p.nbytes})
# Special payloads: signed zero, subnormals, infinities, both NaN signs/payloads.
c=np.array([0,0x80000000,1,0x807fffff,0x7f800000,0xff800000,0x7fc01234,0xff801234],dtype=np.uint32)
w=clifford_words(c,4);p=encode(w);assert np.array_equal(Packet.from_bytes(p.to_bytes()).decode(),w);checks+=64
# Invalid packets are rejected rather than indexing outside the payload.
invalid=0
for data in (b'',p.to_bytes()[:-1],p.to_bytes()+b'!',b'BADMAGIC'+p.to_bytes()[8:]):
 try: Packet.from_bytes(data)
 except (ValueError,TypeError): invalid+=1
 else: raise AssertionError('invalid packet accepted')
report['lossless_roundtrip_words']=checks;report['invalid_packets_rejected']=invalid;report['compression']=rows
report['passed']=True
(ROOT/'results').mkdir(exist_ok=True);(ROOT/'results/cpu_tests.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

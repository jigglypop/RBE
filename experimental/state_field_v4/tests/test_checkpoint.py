from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from modelpack import pack,unpack
from wordcodec import clifford_words
rng=np.random.default_rng(998);n=64
c=rng.normal(size=n).astype(np.float32).view(np.uint32)
w0=clifford_words(c,32).view(np.float32);w1=w0.copy().view(np.uint32);j=np.arange(n)
for a in range(n):
 b=int(rng.integers(n));w1[j^a,j]=c[a]^np.array([((b&int(v)).bit_count()&1)<<31 for v in j],dtype=np.uint32)
tensors={}
for i,w in enumerate([w0,w1.view(np.float32),rng.normal(size=(n,n)).astype(np.float32)]):
 tensors[f'layer{i}.weight']=w;tensors[f'layer{i}.bias']=rng.normal(size=n).astype(np.float32)
data,report=pack(tensors);decoded=unpack(data)
assert all(tensors[k].dtype==decoded[k].dtype and tensors[k].shape==decoded[k].shape and tensors[k].tobytes()==decoded[k].tobytes() for k in tensors)
report['bitwise_roundtrip']=True;report['scope']='synthetic six-tensor checkpoint; not pretrained LLM or model accuracy'
# Non-square, opposite-endian, integer, empty and special f32 words stay exact.
extra={'empty':np.zeros((0,3),np.float32),'integer':np.arange(15,dtype=np.int64),'opposite_endian':np.arange(12,dtype='>f8').reshape(3,4),'special':np.array([0,0x80000000,1,0x7fc01234,0xff801234],dtype=np.uint32).view(np.float32)}
z,_=pack(extra);back=unpack(z)
assert all(back[k].dtype==v.dtype and back[k].shape==v.shape and back[k].tobytes()==v.tobytes() for k,v in extra.items())
for bad in (np.array([object()],dtype=object),np.zeros(1,dtype=[('a','i4'),('b','f4')])):
 try:pack({'bad':bad})
 except ValueError:pass
 else:raise AssertionError('unsupported tensor silently accepted')
(ROOT/'results/checkpoint.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

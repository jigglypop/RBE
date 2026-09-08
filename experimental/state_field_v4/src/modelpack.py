"""Pack every tensor of a local NPZ checkpoint without dropping unsupported ones.
Reports count actual header/index/payload bytes, not only free parameters.
NPZ is an interchange format here; architecture/code/optimizer state is not
invented. Object arrays are rejected. No pickle or remote downloads are used.
"""
from pathlib import Path
import json,struct,argparse
import numpy as np
from wordcodec import encode,Packet
MAGIC=b'RBEMOD04'

def pack(tensors):
 entries=[];payload=[];offset=0
 for name,a in tensors.items():
  a=np.asarray(a)
  if not isinstance(name,str) or a.dtype.hasobject or a.dtype.fields is not None: raise ValueError('string names and nonobject, nonstructured arrays required')
  mode='raw';data=a.tobytes(order='C')
  if a.dtype==np.dtype('float32') and a.ndim==2 and a.shape[0]==a.shape[1]:
   n=a.shape[0]
   if 2<=n<=32768 and not n&(n-1):
    candidate=encode(a.view(np.uint32)).to_bytes()
    if len(candidate)<len(data):data=candidate;mode='rbe'
  entries.append({'name':name,'shape':list(a.shape),'dtype':a.dtype.str,'mode':mode,'offset':offset,'stored_bytes':len(data),'raw_bytes':a.nbytes})
  payload.append(data);offset+=len(data)
 index=json.dumps(entries,separators=(',',':'),ensure_ascii=True).encode()
 data=MAGIC+struct.pack('<Q',len(index))+index+b''.join(payload)
 return data,{'raw_tensor_bytes':sum(e['raw_bytes'] for e in entries),'stored_file_bytes':len(data),'ratio':sum(e['raw_bytes'] for e in entries)/len(data),'tensors':entries}

def unpack(data,max_bytes=512*1024*1024):
 if len(data)<16 or data[:8]!=MAGIC:raise ValueError('bad model header')
 size=struct.unpack_from('<Q',data,8)[0]
 if size>len(data)-16:raise ValueError('truncated index')
 entries=json.loads(data[16:16+size]);body=memoryview(data)[16+size:];out={};total=0
 for e in entries:
  name=e['name'];shape=e['shape'];dtype=np.dtype(e['dtype'])
  if not isinstance(name,str) or name in out or dtype.hasobject or any(not isinstance(x,int) or x<0 for x in shape):raise ValueError('invalid tensor metadata')
  count=1
  for x in shape:count*=x
  need=count*dtype.itemsize;total+=need
  if total>max_bytes or need!=e['raw_bytes']:raise ValueError('decoded size limit')
  start=e['offset'];end=start+e['stored_bytes']
  if start<0 or end<start or end>len(body):raise ValueError('invalid tensor offset')
  chunk=bytes(body[start:end])
  if e['mode']=='rbe':
   if dtype!=np.dtype('float32'):raise ValueError('RBE type')
   pkt=Packet.from_bytes(chunk,max_decoded_bytes=max_bytes)
   if shape!=[pkt.n,pkt.n]:raise ValueError('RBE shape')
   raw=pkt.decode(max_decoded_bytes=max_bytes).tobytes()
  elif e['mode']=='raw':raw=chunk
  else:raise ValueError('unknown tensor mode')
  if len(raw)!=need:raise ValueError('raw length')
  out[name]=np.frombuffer(raw,dtype=dtype).reshape(shape).copy()
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 with np.load(args.checkpoint,allow_pickle=False) as f:tensors={k:f[k] for k in f.files}
 data,report=pack(tensors);decoded=unpack(data)
 if any(decoded[k].dtype!=a.dtype or decoded[k].shape!=a.shape or decoded[k].tobytes()!=a.tobytes() for k,a in tensors.items()):raise RuntimeError('bitwise roundtrip failed')
 args.output.write_bytes(data);report['bitwise_roundtrip']=True;print(json.dumps(report,indent=2))

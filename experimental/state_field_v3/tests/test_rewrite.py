from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from layout import Frame,compile_program,gp_oracle
rng=np.random.default_rng(3173);ROOT=Path(__file__).resolve().parents[1]

def execute(ops,x):
    x=np.asarray(x,dtype=object)
    for op in ops:
        x=gp_oracle(x,x) if op=='square' else op.dense(3)@x
    return x

checked=0
for diagonal_only in (False,True):
    for _ in range(200):
        ops=[]
        for _ in range(4):
            for _ in range(4):
                ops.append(Frame(0 if diagonal_only else int(rng.integers(8)),int(rng.integers(8)),int(rng.integers(2))))
            ops.append('square')
        x=rng.integers(-1,2,8).astype(object)
        folded=compile_program(ops)
        assert np.array_equal(execute(ops,x),execute(folded,x));checked+=8
        if diagonal_only:assert sum(isinstance(op,Frame) for op in folded)<=1
# Exact counterexample: arbitrary left action must not be moved past a square.
x=np.array([1,2,3,4,5,6,7,8],dtype=object)
f=Frame(1,0,0)
assert not np.array_equal(gp_oracle(f.dense(3)@x,f.dense(3)@x),f.dense(3)@gp_oracle(x,x))
assert compile_program([f,'square'])==[f,'square']
r={'programs':400,'exact_integer_comparisons':checked,
   'diagonal_frame_passes':{'before':16,'after_max':1},'squares_preserved':4,
   'xor_frame_barrier_counterexample':True,'arithmetic':'unbounded integer oracle; not a numerical range promise',
   'passed':True}
(ROOT/'results/rewrite.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

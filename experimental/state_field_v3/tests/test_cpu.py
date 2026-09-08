from pathlib import Path
import sys,json,itertools
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from layout import *
ROOT=Path(__file__).resolve().parents[1]
rng=np.random.default_rng(9132026)
report={'frame_pair_compositions':0,'frame_inverses':0,'associativity_random':0,
        'blade_frame_relations':0,'block_basis_products':0,'block_random_coefficients':0}
for d in range(1,4):
    labels=[Frame(a,b,s) for a in range(1<<d) for b in range(1<<d) for s in range(2)]
    dense={f:f.dense(d) for f in labels}
    for f in labels:
        assert np.array_equal(dense[f].T,f.transpose().dense(d));report['frame_inverses']+=1
        for h in labels:
            assert np.array_equal(dense[f]@dense[h],f.compose(h).dense(d))
            report['frame_pair_compositions']+=1
for _ in range(10000):
    f,h,k=[Frame(int(rng.integers(32768)),int(rng.integers(32768)),int(rng.integers(2))) for _ in range(3)]
    assert f.compose(h).compose(k)==f.compose(h.compose(k))
    assert f.compose(f.inverse())==Frame();assert Frame.unpack(f.pack())==f
    report['associativity_random']+=1
# All signatures through d=4, checking Clifford anticommutation and squares.
for d in range(1,5):
    for neg in range(1<<d):
        for i in range(d):
            a=blade_frame(1<<i,neg,d)
            assert a.compose(a)==Frame(0,0,(neg>>i)&1)
            for j in range(d):
                b=blade_frame(1<<j,neg,d)
                if i!=j:
                    ab=a.compose(b);ba=b.compose(a)
                    assert ab.a==ba.a and ab.b==ba.b and ab.s!=ba.s
                report['blade_frame_relations']+=1
assert np.array_equal(H.T@H,4*np.eye(8,dtype=int))
basis=np.eye(8,dtype=np.int64)
for a in basis:
    for b in basis:
        assert np.array_equal(product(a,b),gp_oracle(a,b))
        report['block_basis_products']+=1
x=rng.integers(-100,101,(2048,8));w=rng.integers(-100,101,(2048,8))
assert np.array_equal(product(x,w),gp_oracle(x,w))
assert np.array_equal(from_blocks(to_blocks(x)),x)
report['block_random_coefficients']=2048*8
# Left regular 8x8 becomes diag(A+ tensor I2, A- tensor I2) after H conjugation.
a=rng.normal(size=8);la=np.column_stack([gp_oracle(a,e) for e in basis]);ab=to_blocks(a)
red=np.zeros((8,8));red[:4,:4]=np.kron(ab[0],np.eye(2));red[4:,4:]=np.kron(ab[1],np.eye(2))
report['regular_representation_error']=float(np.max(np.abs(H@la@H.T/4-red)))
assert report['regular_representation_error']<1e-14
# Do not confuse conversion inverse and its adjoint.
maxerr=0
for _ in range(24):
    a,b,g=rng.normal(size=(3,8));ga,gb=product_vjp(a,b,g);h=1e-5
    for side,grad in [(0,ga),(1,gb)]:
        for i in range(8):
            v=[a.copy(),b.copy()];v[side][i]+=h;u=np.dot(g,product(*v))
            v[side][i]-=2*h;l=np.dot(g,product(*v))
            maxerr=max(maxerr,abs(float((u-l)/(2*h)-grad[i])))
assert maxerr<2e-8
report['vjp_central_difference_max_error']=maxerr
# Nonlinear chain and adjoint through two matrix products (all original coords).
x,w1,w2,g=rng.normal(scale=.2,size=(4,8));h=product(w1,x);z=product(h,h)
gw2,gz=product_vjp(w2,z,g);dh1,dh2=product_vjp(h,h,gz)
gw1,gx=product_vjp(w1,x,dh1+dh2);errs=[]
for idx,analytic in [(0,gx),(1,gw1),(2,gw2)]:
    for j in range(8):
        def loss(vals):
            xx,aa,bb=vals;hh=product(aa,xx);return np.dot(g,product(bb,product(hh,hh)))
        v=[x.copy(),w1.copy(),w2.copy()];v[idx][j]+=1e-5;plus=loss(v)
        v[idx][j]-=2e-5;minus=loss(v);errs.append(abs((plus-minus)/2e-5-analytic[j]))
report['nonlinear_chain_vjp_max_error']=float(max(errs));assert max(errs)<1e-8
# Same SGD trajectory under the non-unit-normalized H change of basis.
a,b,g=rng.normal(size=(3,8));lr=.01;ga,_=product_vjp(a,b,g)
GA=(to_blocks(g)/4)@to_blocks(b).swapaxes(-1,-2)
new_coeff=a-lr*ga
new_from_block=from_blocks(to_blocks(a)-4*lr*GA)
report['sgd_basis_scaling_max_error']=float(np.max(np.abs(new_coeff-new_from_block)))
assert report['sgd_basis_scaling_max_error']<1e-14
# Semantic guard: a SUM of frames generally is not one frame.
f=Frame(1,0);S=np.eye(2)+f.dense(1)
assert not any(np.array_equal(S,q.dense(1)) for q in [Frame(a,b,s) for a,b,s in itertools.product(range(2),repeat=3)])
report['sum_not_one_label_guard']=True
report['H']=H.tolist();report['passed']=True
(ROOT/'results/cpu_tests.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

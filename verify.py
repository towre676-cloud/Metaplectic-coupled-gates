"""Replay SO(5)_2 four-spinor resource and fusion-space calculations.

Exact identities use Q(zeta_5), with rational coefficients (SymPy domain).
Numerics independently check Wigner values and the published braid matrices.
Run: python verify.py; dependencies: numpy, sympy.
Source: Cui--Wang, JMP 56 032202 (2015), Sec. III C.
This does not certify the user's six-spinor F/R data or gate injection.
"""
import json
from pathlib import Path
from collections import deque
import numpy as np
import sympy as sp

zsym = sp.exp(2*sp.pi*sp.I/5)
K = sp.QQ.algebraic_field(zsym)
z = K.from_sympy(zsym)
zero, one = K.zero, K.one
sqrt5 = one+2*(z+z**4)
assert sqrt5*sqrt5 == K(5)
def eye(n): return [[one if i==j else zero for j in range(n)] for i in range(n)]
def mm(a,b):
    return [[sum((a[i][k]*b[k][j] for k in range(len(b))),zero)
             for j in range(len(b[0]))] for i in range(len(a))]
def adj(a):
    def conj(x):
        # Domain conversion is exact, including complex conjugation.
        return K.from_sympy(sp.conjugate(K.to_sympy(x)))
    return [[conj(a[j][i]) for j in range(len(a))] for i in range(len(a[0]))]
def diag(exps):
    return [[z**(exps[i]%5) if i==j else zero for j in range(5)] for i in range(5)]
def powm(a,n):
    if n<0: a,n=adj(a),-n
    r=eye(len(a))
    for _ in range(n): r=mm(r,a)
    return r
def prod(*xs):
    r=eye(len(xs[0]))
    for x in xs:r=mm(r,x)
    return r
def projectively_equal(a,b):
    ij=next((i,j) for i in range(len(b)) for j in range(len(b[0])) if b[i][j]!=zero)
    c=a[ij[0]][ij[1]]/b[ij[0]][ij[1]]
    return c!=zero and all(a[i][j]==c*b[i][j] for i in range(len(b)) for j in range(len(b[0])))
def numeric(a):
    return np.array([[complex(K.to_sympy(v).evalf()) for v in row] for row in a])

# A common scalar -i is omitted from all three published generators.
A=diag([1,0,1,-1,-1]); C=diag([1,-1,-1,1,0])
g=[0,-1,1,1,-1]
B=[[z**(g[(i-j)%5]%5)/sqrt5 for j in range(5)] for i in range(5)]
I=eye(5)
for t in [A,B,C]: assert mm(adj(t),t)==I
assert prod(A,B,A)==prod(B,A,B)
assert prod(B,C,B)==prod(C,B,C)
assert mm(A,C)==mm(C,A)
H=prod(powm(A,-1),powm(C,-1),powm(B,2),powm(A,-1),powm(C,-1))
F=[[z**((i*j)%5)/sqrt5 for j in range(5)] for i in range(5)]
assert projectively_equal(H,F)
Z=mm(A,adj(C)); X=prod(A,B,powm(A,-2),powm(C,2),adj(mm(A,B)))
X0=[[one if i==(j+1)%5 else zero for j in range(5)] for i in range(5)]
assert projectively_equal(X,X0)
assert projectively_equal(Z,diag(list(range(5))))

# Exact fusion ring, labels: 0=1,1=z,2=y1,3=y2,4=s,5=s'.
fusion={}
def rule(a,b,cs): fusion[a,b]=fusion[b,a]=cs
for a in range(6):rule(0,a,[a])
rule(1,1,[0]); rule(1,2,[2]); rule(1,3,[3]);rule(1,4,[5]);rule(1,5,[4])
rule(2,2,[0,1,3]);rule(3,3,[0,1,2]);rule(2,3,[2,3])
for a in [2,3]:
    for b in [4,5]: rule(a,b,[4,5])
rule(4,4,[0,2,3]);rule(5,5,[0,2,3]);rule(4,5,[1,2,3])
def tensor(v,w):
    r=[0]*6
    for a,va in enumerate(v):
        for b,wb in enumerate(w):
            for c in fusion[a,b]:r[c]+=va*wb
    return r
powers=[[1,0,0,0,0,0]]
for n in range(8):powers.append(tensor(powers[-1],[0,0,0,0,1,0]))
assert powers[3]==[0,0,0,0,3,2]
assert powers[4]==[3,2,5,5,0,0]
assert powers[6][0]==13
assert powers[8][2]==125
six_pair_ranks=[powers[4][a] for a in [0,2,3]]
assert six_pair_ranks==[3,5,5]

# Resource: coherent complement of a computational basis projector.
# Translate the excluded basis index from 1 to 0 with braid-realized X^-1.
psi=np.array([0,1,1,1,1],complex)/2
omega=np.exp(2j*np.pi/5)
W=np.array([[sum(omega**(2*p*y)*psi[(q-y)%5]*psi[(q+y)%5].conjugate()
                        for y in range(5)).real/5 for p in range(5)] for q in range(5)])
def exact_w(q,p):
    return sum((z**((2*p*y)%5)/20 for y in range(5)
                if (q-y)%5 and (q+y)%5),zero)
exactW=[[exact_w(q,p) for p in range(5)] for q in range(5)]
assert all(exactW[0][p]==K(sp.Rational(-1,20)) for p in range(1,5))
total=sum((v for row in exactW for v in row),zero)
assert total==one
neg=sum((exactW[q][p] for q in range(5) for p in range(5) if W[q,p]<-1e-10),zero)
assert neg==-sqrt5/5
assert abs(W.sum()-1)<1e-12

# Independent enumeration of the 30 single-qupit pure stabilizer states:
# 5 computational states plus 25 quadratic chirps.
stabilizers=[np.eye(5,dtype=complex)[:,j] for j in range(5)]
for a in range(5):
    for b in range(5):stabilizers.append(np.array([omega**((a*j*j+b*j)%5) for j in range(5)])/np.sqrt(5))
fmax=max(abs(np.vdot(v,psi))**2 for v in stabilizers)
assert abs(fmax-.8)<1e-12

# Stronger injection obstruction: none of the six stabilizer bases is flat
# on M. Any one-qupit Clifford followed by computational readout chooses
# one of these bases. Thus Clifford pre-rotation cannot fix SUM+Z injection.
overlap_exact=[K(0)]+[K(sp.Rational(1,4))]*4
for a in range(5):
    for b in range(5):
        amp=sum((z**((-a*j*j-b*j)%5)/(2*sqrt5) for j in range(1,5)),zero)
        amp_conj=K.from_sympy(sp.conjugate(K.to_sympy(amp)))
        overlap_exact.append(amp*amp_conj)
assert all(x!=K(sp.Rational(1,5)) for x in overlap_exact)
flat_bases=0
for start in range(0,30,5):
    flat_bases+=all(x==K(sp.Rational(1,5)) for x in overlap_exact[start:start+5])
assert flat_bases==0

# Standard SUM + Z readout injection: K_b[j,j] = psi[b-j].
# It has a zero diagonal entry on every branch, so no 5D unitary.
injection=[np.diag([psi[(b-j)%5] for j in range(5)]) for b in range(5)]
assert all(np.linalg.matrix_rank(t)==4 for t in injection)
assert np.allclose(sum(t.conj().T@t for t in injection),np.eye(5))

# Projective numerical closure independently checks the 3000-element image.
def key(u):
    flat=u.ravel(); k=np.argmax(np.abs(flat)>1e-8);u=u/(flat[k]/abs(flat[k]))
    return tuple(np.round(u.real,9).ravel())+tuple(np.round(u.imag,9).ravel())
gens=[numeric(t) for t in [A,B,C]]
seen={key(np.eye(5)):np.eye(5,dtype=complex)}; queue=deque(seen.values())
while queue:
    u=queue.popleft()
    for t in gens:
        v=u@t;k=key(v)
        if k not in seen:
            seen[k]=v;queue.append(v)
            assert len(seen)<=3100,'Closure exceeded expected order; inspect phase hashing'
assert len(seen)==3000

result={
 'status':'PASS',
 'exact_checks':['unitarity of three B4 generators','two Artin relations and far commutation',
                 'published Fourier, X, Z braid words up to phase','fusion powers',
                 'Wigner normalization and negative mass'],
 'projective_braid_image_numeric':len(seen),
 'fusion_powers':{str(i):powers[i] for i in [3,4,6,8]},
 'six_vacuum_dimension':13,'six_triple_decomposition':[9,4],
 'six_pair_charge_projector_ranks':six_pair_ranks,
 'resource_success_probability':'4/5', 'expected_fresh_attempts':'5/4',
 'resource_amplitudes':[0,.5,.5,.5,.5],
 'wigner_table':W.tolist(), 'negative_mass':'1/sqrt(5)',
 'wigner_L1':'1+2/sqrt(5)','max_stabilizer_fidelity_numeric':float(fmax),
 'flat_stabilizer_bases_exact':flat_bases,
 'one_copy_Clifford_rotated_SUM_Z_injection':'No unitary branch on the full five-dimensional qupit',
 'naive_SUM_Z_injection_branch_ranks':[4]*5,
 'limitations':['Four-spinor calculation is in the published Cui-Wang category/gauge, not yet matched to AFT/DAZ lifts.',
                'Coherent vacuum/nonvacuum measurement is assumed; resolving y1/y2 then deleting the record is a different channel.',
                'Resource state does not establish a unitary gate, deterministic injection, or universality.',
                'No coupled six-spinor braid/measurement search has been performed.']}
Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

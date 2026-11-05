"""Conditional six-spinor Gaussian candidate, NOT a physical gate certificate.

Missing bridges: gauge/charge-tree match to the audited AFT/DAZ datum, and a
lawful implementation of the ancilla |+><+| return measurement.
Run python coupled_candidate.py (also replays verify.py).
"""
from verify import K,z,one,zero,sqrt5,eye,mm,adj,diag,prod,numeric
from pathlib import Path
import json
import numpy as np

# Local Weil matrices D and T=F D^-1 F^dagger. All entries in Q(zeta_5).
D=diag([j*j for j in range(5)])
F=[[z**((i*j)%5)/sqrt5 for j in range(5)] for i in range(5)]
T=prod(F,adj(D),adj(F))
assert prod(D,T,D)==prod(T,D,T)
# On |x,y>, G1=D1, G2=T1, G3=zeta^((x+y)^2), G4=T2, G5=D2.
# Every cross braid identity is a 5x5 shifted-D identity at fixed y/x.
for shift in range(5):
    Ds=diag([(j+shift)**2 for j in range(5)])
    assert prod(Ds,T,Ds)==prod(T,Ds,T)
# Far commutation follows from diagonal form or separate tensor factors.
# Every generator preserves simultaneous inversion (x,y)->(-x,-y),
# whose even sector has dimension (25+1)/2=13.
v=np.eye(5)
E=np.column_stack([(v[:,1]+v[:,4])/np.sqrt(2),(v[:,2]+v[:,3])/np.sqrt(2)])
a=E@np.array([1,1])/np.sqrt(2)
G3=np.diag([complex(zv) for zv in [np.exp(2j*np.pi*((x+y)**2%5)/5) for x in range(5) for y in range(5)]])
J=np.kron(E,a[:,None])
return_map=np.kron(np.eye(5),a.conj()[None,:])
L=return_map@G3@J
assert np.linalg.norm((np.eye(5)-E@E.conj().T)@L)<1e-12

# The conditional ancilla-return branch is diagonal on the phi data pair.
k0=(sqrt5-z)/4;k1=(sqrt5-z**4)/4
assert k0*k1==(K(7)+sqrt5)/32
ratio=k1/k0
# All roots of unity in Q(zeta_5) are +/- zeta_5^j.
assert all(ratio != sign*z**j for sign in [1,-1] for j in range(5))
assert np.allclose(E.conj().T@L,numeric([[k0,zero],[zero,k1]]))

# Replace the ideal return projector with the already-recorded local filter
# A^n = P_minus + 5^-n P_plus, using ancilla |minus> instead of |plus>.
# First project both triple totals to the even (s,s) sector. Cross braiding
# produces local odd-odd leakage, so this projector must be counted.
a00=(one+z**4)/2
b00=(z+z**4)/2
assert (a00+b00)/2==k0
l0=(one-z)/4;l1=-(one-z**4)/4
p=(K(7)+sqrt5)/32; q=(K(5)-sqrt5)/32
assert l0*(-l1)==q
# A's 3x3 embedding follows from the recoupled vacuum vector
# (1,sqrt(2),sqrt(2))/sqrt(5). Its phi compression is exactly [[3,-2],[-2,3]]/5.
import sympy as sp
vac=sp.Matrix([1,sp.sqrt(2),sp.sqrt(2)])/sp.sqrt(5)
Q23=sp.eye(3)-vac*vac.T
assert Q23[1:3,1:3]==sp.Matrix([[3,-2],[-2,3]])/5
assert p+q==K(sp.Rational(3,8))

# Independent 25-dimensional numerical replay for n=0,...,5.
minus=E@np.array([1,-1])/np.sqrt(2)
Jminus=np.kron(E,minus[:,None])
parity=np.eye(5)[:,[0,4,3,2,1]]
even=(np.eye(5)+parity)/2
SS=np.kron(even,even)
localA=np.array([[3,-2],[-2,3]],float)/5
finite=[]
pn=float((7+np.sqrt(5))/32);qn=float((5-np.sqrt(5))/32)
for n in range(6):
    t=5.**(-n)
    ancfilter=E@np.linalg.matrix_power(localA,n)@E.conj().T
    state=np.kron(np.eye(5),ancfilter)@SS@G3@Jminus
    # Project onto each ancilla phi charge basis and compare exact formula.
    branches=[]
    for idx in range(2):
        bra=E[:,idx].conj()[None,:]
        branches.append(E.conj().T@np.kron(np.eye(5),bra)@state)
    kk=numeric([[k0,zero],[zero,k1]])
    ll=numeric([[l0,zero],[zero,l1]])
    assert np.allclose(branches[0],(kk+t*ll)/np.sqrt(2))
    assert np.allclose(branches[1],(-kk+t*ll)/np.sqrt(2))
    assert np.allclose(sum(v.conj().T@v for v in branches),(pn+t*t*qn)*np.eye(2))
    delta=t*t*qn/(pn+t*t*qn)
    finite.append({'filter_rounds':n,'success_probability':pn+t*t*qn,
                   'conditional_wrong_unitary_weight':delta,'diamond_error_upper_bound':2*delta})
out={
 'status':'CONDITIONAL CANDIDATE, NOT A PHYSICAL GATE CERTIFICATE',
 'gaussian_B6_exact_checks':'local and all shifted cross Artin relations in Q(zeta_5)',
 'space':'simultaneous-inversion even subspace of C^5 tensor C^5, dimension 13',
 'code':'span(e_1,e_2) on data; ancilla (e_1+e_2)/sqrt(2); e_j=(|j>+|-j>)/sqrt(2)',
 'cross_generator':'G3 |x,y> = zeta_5^((x+y)^2) |x,y>',
 'ideal_return_effect':'ancilla |+><+| for algebraic witness; |minus><minus| gives the same branch with minus input',
 'kraus':'diag(sqrt(5)-zeta_5, sqrt(5)-zeta_5^-1)/4',
 'success_probability':'(7+sqrt(5))/32',
 'success_probability_numeric':float(((7+np.sqrt(5))/32)),
 'eigenvalue_ratio':'(sqrt(5)-zeta_5^-1)/(sqrt(5)-zeta_5)',
 'infinite_projective_order':'exact nonmembership in the 10 roots of unity of Q(zeta_5)',
 'finite_filter_protocol':'ancilla |minus>; G3; project triple totals s,s; apply A^n to ancilla; measure/consume ancilla and average its two phi outcomes',
 'finite_filter_accepted_channel':'(p U rho U^dagger + 25^-n q V rho V^dagger)/(p+25^-n q)',
 'p':'(7+sqrt(5))/32','q':'(5-sqrt(5))/32',
 'finite_protocol_checks':finite,
 'remaining':['Identify this Gaussian module and code with the selected physical six-spinor fusion tree and gauge.',
              'Certify the s,s triple-charge projector and A-filter embedding in that same physical gauge. The finite-filter construction removes the need to assume an exact resource-line projector.',
              'Record and handle every other measurement branch; postselection here is not deterministic feedforward.']}
Path(__file__).with_name('coupled_candidate.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

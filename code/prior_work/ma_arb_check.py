"""CHECK OF PRIOR WORK (not part of our proof).  An AI referee's independent rigorous (Arb ball arithmetic)
re-implementation of the middle-range criterion of Y. Ma, Zenodo doi:10.5281/zenodo.23191831 (Proposition 2 with
delta = Delta(beta_-)), written from scratch from the formulas in that preprint; J, G, T, N are computed by exact
divided differences of exp over the area-one equilateral triangle (no series, no quadrature).
Run by ma_run_boxes.py."""
import sys, json
from flint import arb, ctx
ctx.prec = 256
ONE = arb(1); H = arb(3)**(arb(1)/4); R = H/3; SQ3 = arb(3).sqrt(); ELL = 2/H
# vertex directions d_i (V_i = 2r d_i)
D = [(arb(0), arb(1)), (SQ3/2, -ONE/2), (-SQ3/2, -ONE/2)]

def dd_exp(nodes):
    """Divided difference [w_0..w_n]exp. nodes: list of (key, value_arb); equal keys = exactly equal nodes.
    Different keys must be separated (we use exact integer-coefficient keys)."""
    groups = {}
    for key, val in nodes:
        if key in groups: groups[key][1] += 1
        else: groups[key] = [val, 1]
    keys = list(groups)
    total = arb(0)
    for a in keys:
        wa, ma = groups[a]
        n = ma - 1
        # series of e^{wa+eps} * prod_b (wa-wb+eps)^{-mb}, coefficient of eps^n
        ser = [wa.exp() / arb(_fact(j)) for j in range(n+1)]
        for b in keys:
            if b == a: continue
            wb, mb = groups[b]
            d = diffval(a, b, groups)
            fac = [arb(_binom(-mb, j)) * d**(-mb - j) for j in range(n+1)]
            ser = [sum((ser[i]*fac[j-i] for i in range(j+1)), arb(0)) for j in range(n+1)]
        total += ser[n]
    return total
def _fact(j):
    f = 1
    for i in range(2, j+1): f *= i
    return f
def _binom(m, j):  # generalized binomial (m choose j), m may be negative int
    num = 1
    for i in range(j): num *= (m - i)
    return num // _fact(j) if num % _fact(j) == 0 else None
CUR = {}
def diffval(a, b, groups):
    # keys are (cq, cM) integer pairs; value = cq*q + cM*M ; difference computed from integer keys
    return (a[0]-b[0])*CUR['q'] + (a[1]-b[1])*CUR['M']

def setM(M):
    t = M.tanh(); L = -(t/2).atanh(); K = M - L; q = 2*K; k = 2*K/H; beta = t*K/H
    CUR.update(M=M, t=t, L=L, K=K, q=q, k=k, beta=beta)

def funcs():
    """six summands of the scaled ground state: (sign eps, vertex i, node-key vector)"""
    out = []
    for i in range(3):
        out.append((+1, i, [(-1 + (1 if j == i else 0), 0) for j in range(3)]))   # f_i = 1/2 e^{-q(1-mu_i)}
        out.append((-1, i, [(-(1 if j == i else 0), -2) for j in range(3)]))      # g_i = 1/2 e^{-2M-q mu_i}
    return out
def val(key): return key[0]*CUR['q'] + key[1]*CUR['M']
def I(z):   # int_E exp(z.mu) dA, area 1
    return 2*dd_exp([(kk, val(kk)) for kk in z])
def I0(z):  # int_E mu_0 exp(z.mu) dA = 2 [z0,z0,z1,z2]exp
    return 2*dd_exp([(z[0], val(z[0]))] + [(kk, val(kk)) for kk in z])
def Iside(z, j):  # int over side opposite vertex j (mu_j = 0) of exp(z.mu) ds = ell*[z_a,z_b]exp
    rest = [z[m] for m in range(3) if m != j]
    return ELL*dd_exp([(kk, val(kk)) for kk in rest])

def quantities(M):
    setM(M)
    F = funcs(); k = CUR['k']; beta = CUR['beta']
    N = arb(0); JN = arb(0); GN = arb(0); BN = arb(0)
    for (ea, ia, za) in F:
        for (eb, ib, zb) in F:
            z = [(za[m][0]+zb[m][0], za[m][1]+zb[m][1]) for m in range(3)]
            Iz = I(z) / 4
            N += Iz
            dot = D[ia][0]*D[ib][0] + D[ia][1]*D[ib][1]
            cij = D[ia][0]*D[ib][0] - D[ia][1]*D[ib][1]
            GN += ea*eb*dot*Iz
            # y = r(3 mu_0 - 1)
            JN += ea*eb*cij*(3*I0(z)/4 - Iz)
            BN += sum((Iside(z, j) for j in range(3)), arb(0)) / 4
    G = k**2*GN/N; J = k**2*R*JN/N; T = beta*BN/N
    return dict(N=N, G=G, T=T, J=J, k=k, beta=beta, t=CUR['t'], K=CUR['K'])

def ma_GT(M):
    t = M.tanh(); L = -(t/2).atanh(); K = M - L; k = 2*K/H
    a = t*(1/(1-t**2) + 2/(4-t**2)); kap = a/(K+a); T = 2*kap*k**2
    return T - k**2, T

def Delta(t):
    return (SQ3*(2-t**2)/(t*(4-t**2).sqrt())).log()/2

def criterion(Mlo, Mhi, delta=None):
    Mb = arb((Mlo+Mhi)/2, (Mhi-Mlo)/2)  # ball covering [Mlo,Mhi] (Mlo,Mhi exact dyadics/rationals)
    Mb = Mb.union(arb(Mlo)).union(arb(Mhi))
    qn = quantities(Mb)
    tlo = arb(Mlo).tanh()
    dl = Delta(tlo) if delta is None else delta
    dl = arb(dl.upper())  # outward upper endpoint, as in the preprint
    G, T, J = qn['G'], qn['T'], qn['J']
    sh = dl.sinh()/dl
    Fc = 2*G*sh**2 - T*(arb(3)/4 - dl.exp()*dl/6)
    E = 1 + ((2*dl).exp()-1)*(1+4*R**2*T) + (1-(-dl).exp())*4*R**2*T
    bst = -2*J + R*T/2
    Hc = bst - R*T/2*(1-(-dl).exp())
    ok1 = Fc < 0
    ok2 = (Hc > 0) and ((Fc*E - Hc**2) < 0)
    return ok1, ok2, dict(F=Fc, E=E, H=Hc, FEH=Fc*E-Hc**2, bstar=bst, J=J, G=G, T=T, delta=dl, wbox=qn['beta'])

if __name__ == '__main__':
    from fractions import Fraction
    # --- spot cross-check of direct integrals vs Ma's formula (4) and vs floats ---
    for Mv in ['1.2', '2', '4', '8']:
        M = arb(Mv); qn = quantities(M); G4, T4 = ma_GT(M)
        print('M', Mv, 'beta', qn['beta'].str(12), 'G', qn['G'].str(12), 'G(4)', G4.str(12), 'T', qn['T'].str(12), 'T(4)', T4.str(12), 'J', qn['J'].str(12))
    sys.stdout.flush()

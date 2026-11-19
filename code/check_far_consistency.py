"""Floating-point consistency check (not part of the proof): for random certified far cells, the
stored trial's Rayleigh quotient at a random point of the cell must be >= an accurate eigenvalue
(48-wave Ritz on the actual triangle) and < lambda0(beta)."""
import json, random, sys
import mpmath as mp
from flint import arb
from rig import Q
import sixwave as sw
from nums import ARB, MP
from far_cert import Interval
from ritz_mp import matrices, lowest, plane_basis
from equi_float import lam0
mp.mp.dps = 30
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
d = json.load(open(sys.argv[1]))
cells = random.sample(d['cells'], min(20, len(d['cells'])))
H = mp.root(3, 4); l = 2 / H; r = H / 3
Ve = [(-l / 2, -r), (l / 2, -r), (mp.mpf(0), 2 * r)]
worst = 1e9; bad = 0
def fm(x):
    q = Q(x); return mp.mpf(int(q.p)) / int(q.q)

for b1, b2, a1, a2, c1, c2, C in cells:
    iv = Interval(b1, b2)
    u = random.random(); v = random.random(); w = random.random()
    a = fm(a1) + u * (fm(a2) - fm(a1))
    b = fm(c1) + v * (fm(c2) - fm(c1))
    beta = fm(b1) + w * (fm(b2) - fm(b1))
    ac = (fm(a1) + fm(a2)) / 2
    bc = (fm(c1) + fm(c2)) / 2
    c = [mp.mpf(C[0][j]) + (a - ac) * mp.mpf(C[1][j]) + (b - bc) * mp.mpf(C[2][j]) for j in range(6)]
    T = sw.tensors(MP, iv.Wm, iv.Nm, iv.Sm, [c])
    Num, Den = sw.num_den(MP, T, -beta, a, b, [1])
    R = Num / Den
    VT = [(a * P[0] + b * P[1], P[1] / a) for P in Ve]
    k = mp.sqrt(-R) if R < 0 else mp.mpf(1)
    A, Mm = matrices(VT, plane_basis(k, 16, (0.85, 1.0, 1.15), phase=mp.mpf('0.1')), -beta)
    lam, _ = lowest(A, Mm)
    l0 = lam0(float(beta))
    ok = (R >= lam - 1e-6 * abs(lam)) and (R < l0)
    bad += (not ok)
    print(f'a={float(a):.4f} b={float(b):.4f} beta={float(beta):.4f}  R={float(R):.6f}  ritz48={float(lam):.6f}  lam0={l0:.6f}  {"ok" if ok else "BAD"}', flush=True)
print('bad', bad)

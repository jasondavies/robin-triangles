"""Floating-point illustrations (NOT proof steps) of the numbers quoted in Remarks and in the text of Part II.

 (a) R1(alpha) by conforming P2 finite elements on E (the dual norm of D_{Theta_0}(psi_0, .) for N, by one
     saddle-point solve), compared with a(alpha) = 2 lambda_0 - (5/4) alpha lambda_0' and with the certified upper
     bounds of Lemma "ring": at the grid points (bound with delta_pm = lambda_0 - mu_pm and the certified lower
     bound for lambda_2 that the proof uses) and on the grid intervals (the bound used in Lemma "Ccheck");
     the Hessian 2 (a - R1)/lambda_0 of lambda_1(T, alpha)/lambda_0(alpha) in t at E;
 (b) the true gap lambda_1(T_w, alpha)/lambda_0(alpha) - 1 (P2 value, an upper bound up to rounding) at the inner
     vertices t = 0.057 of the cells next to the local disc;
 (c) the weak-coupling threshold (P - P_E) r^2/(A R_I^2) of Theorem "small" for tall isosceles triangles at t = 1.24.
Writes OUT/float_local.json (OUT = RP_OUT or the certificate directory).  Usage: python3 float_local.py
"""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, math, time
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from fractions import Fraction as Fr
import config, shapes
from local_ring import load, dec
from p2_reference import RefFEM, Lshape

T0 = time.time()
F = RefFEM(28)
CJ = np.array([1.0, -0.5, -0.5])                                     # c_j(0) = cos(2 theta_j)


def true_R1(al):
    K0 = (F.Kxx + F.Kyy + al * (F.B[0] + F.B[1] + F.B[2])).tocsc()
    vals, vecs = spla.eigsh(K0, k=3, M=F.M, sigma=-0.5, which='LM'); o = np.argsort(vals)
    lam0, lam2 = vals[o][0], vals[o][1]; psi = vecs[:, o[0]]; psi /= math.sqrt(psi @ (F.M @ psi))
    Dm = -2 * F.Kxx + 2 * F.Kyy + al * sum(CJ[j] * F.B[j] for j in range(3))
    b = Dm @ psi; Mpsi = F.M @ psi; A = (K0 - lam0 * F.M).tocsc()
    lu = spla.splu(sp.bmat([[A, sp.csc_matrix(Mpsi[:, None])], [sp.csc_matrix(Mpsi[None, :]), None]]).tocsc())
    return lam0, lam2, float(b @ lu.solve(np.append(b - (psi @ b) * Mpsi, 0.0))[:-1])


ring = load(os.path.join(config.CERT, 'ring_nodes.jsonl')); l2n = load(os.path.join(config.CERT, 'lam2_nodes.jsonl'))
lc = json.load(open(os.path.join(config.CERT, 'local_cert.json')))
ivr = {d['k']: d for d in lc['intervals']}
s = float(config.S_RING); rows = []
for k in range(0, config.KTOP + 1):
    al = float(config.ALPHA(k)); lam0f, lam2f, R = true_R1(al)
    l0, l0d = shapes.lam0_d(config.ALPHA(k)); l0 = float(l0.mid()); l0d = float(l0d.mid())
    a = 2 * l0 - 1.25 * al * l0d
    mu = [max(float(dec(v['lb'])) for key, v in ring.items() if key[0] == k and key[1] == sg) for sg in (1, -1)]
    L2 = max(float(dec(v['lb'])) for key, v in l2n.items() if key[0] == 4 * (k // 4))
    d = [l0 - m for m in mu]
    Rb = (2 + (d[0] + d[1]) / (L2 - l0)) / (s * s * (1 / d[0] + 1 / d[1]))
    row = dict(k=k, alpha=al, R1_over_a=R / a, hessian=2 * (a - R) / l0, node_bound_over_true=Rb / R)
    if k in ivr: row['interval_bound_over_true'] = ivr[k]['R1_over_lam'] * l0 / R
    rows.append(row)
ra = [r['R1_over_a'] for r in rows]; kmax = int(np.argmax(ra))
near = lambda lo, hi: [r for r in rows if lo <= r['alpha'] <= hi]
out = dict(note='floating point, not a proof step', fem='conforming P2, n=28',
           R1_over_a_at_0_1=[r for r in rows if abs(r['alpha'] - 0.1) < 0.003][0]['R1_over_a'],
           R1_over_a_max=ra[kmax], R1_over_a_max_alpha=rows[kmax]['alpha'], R1_over_a_last=ra[-1], alpha_last=rows[-1]['alpha'],
           spare_factor=1 / ra[kmax],
           node_bound_over_true_min=min(r['node_bound_over_true'] for r in rows),
           node_bound_over_true_max=max(r['node_bound_over_true'] for r in rows),
           interval_bound_over_true_near_2=max(r['interval_bound_over_true'] for r in near(1.5, 3.0)),
           interval_bound_over_true_max=max(r['interval_bound_over_true'] for r in rows if 'interval_bound_over_true' in r),
           interval_bound_over_true_max_alpha=max((r for r in rows if 'interval_bound_over_true' in r),
                                                  key=lambda r: r['interval_bound_over_true'])['alpha'],
           hessian_min=min(r['hessian'] for r in rows), hessian_min_alpha=min(rows, key=lambda r: r['hessian'])['alpha'],
           hessian_max=max(r['hessian'] for r in rows))
# (b) true gap at the inner vertices t = 0.057 of the cells next to the local disc
Fg = RefFEM(24); gaps = []
for j in range(5):
    psi = (math.pi / 3 + 1e-3) * j / 4; phi = psi / 2; t = 0.057
    L = Lshape(t, phi)                                               # T = L(E) = T_w with |w| = tanh 2t, arg w = psi
    for k in range(0, 401, 8):
        al = float(config.ALPHA(k)); lam = Fg.eig(L, al)[0]
        gaps.append(lam / float(shapes.lam0(config.ALPHA(k)).mid()) - 1)
out.update(gap_inner_min=min(gaps), gap_inner_max=max(gaps), gap_samples=len(gaps))
# (c) threshold of Theorem "small" for tall isosceles triangles (two equal long sides) at t = 1.24
t = 1.24; c2, s2 = math.cosh(2 * t), math.sinh(2 * t); SE = 2 / 3 ** 0.25; PE = 3 * SE
ell = [math.sqrt(c2 + s2 * c) for c in (0.5, 0.5, -1.0)]; a_ = [SE * x for x in ell]; P = sum(a_); r = 2 / P
RI2 = max(r * r + (P / 2 - x) ** 2 for x in a_)
out['small_threshold_tall_isosceles_t_1_24'] = (P - PE) * r * r / RI2
out['seconds'] = round(time.time() - T0, 1)
print(json.dumps(out, indent=1))
json.dump(dict(out, rows=rows), open(os.path.join(config.OUT, 'float_local.json'), 'w'), indent=0)

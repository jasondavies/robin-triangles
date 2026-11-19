"""Region C (neighbourhood of the equilateral triangle), step 1: certified data on the alpha-grid.

 (a) ring nodes:  for alpha_k = ALPHA(k), k = 0..KTOP, and sign = +1, -1, a certified lower bound
        mu_pm(alpha_k) <= Lambda(p_E +- s Theta_0) = lowest eigenvalue of the form
        int_E grad v . diag(1 -+ 2s, 1 +- 2s) grad v + alpha sum_j (1 +- s cos 2theta_j) int_{Gamma_j} v^2 ;
 (b) lambda_2 nodes: for every 4th k a certified lower bound for the SECOND Robin eigenvalue of E.

Output: JSON lines (resumable), certs/ring_nodes.jsonl and certs/lam2_nodes.jsonl.
Usage: python3 local_ring.py ring [n] [kmod kres]   |   python3 local_ring.py lam2 [n]
"""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, time, math
import numpy as np, scipy.linalg as sla
from fractions import Fraction as Fr
from flint import arb, arb_mat, fmpz, ctx
import rcr, shapes

from config import ALPHA, KTOP, S_RING, SHRINK, LB_SHRINK, CERT, N_RING, N_LAM2

def enc(x):
    """exact 'man*2^exp' encoding of the lower endpoint of the arb x, shrunk by the factor (1 - LB_SHRINK)"""
    m, e = (x * (1 - shapes.A_(LB_SHRINK))).lower().man_exp(); return "%d*2^%d" % (int(m), int(e))
def dec(s):
    m, e = s.split("*2^"); m = int(m); e = int(e)
    return Fr(m * 2 ** e) if e >= 0 else Fr(m, 2 ** (-e))

def load(fn):
    out = {}
    if os.path.exists(fn):
        for line in open(fn):
            line = line.strip()
            if line:
                d = json.loads(line); out[tuple(d['key'])] = d
    return out

def ring_node(k, sign, n):
    al = ALPHA(k)
    a2, A, beta = shapes.ring_data(S_RING, sign, al)
    t0 = time.time()
    r = rcr.lower_bound(a2, A, beta, n, kind='U')
    sig = Fr(int(r['sigma'] * 2 ** rcr.BITS * (1 - SHRINK)), 2 ** rcr.BITS)       # shrunk, still a valid bound
    assert 0 < sig <= r['sigma']                                                  # exact comparison of rationals
    lb = rcr.cr_bound(sig, r['h'])
    return dict(key=[k, sign, n], alpha=str(al), s=str(S_RING), kind='U', n=n, N=r['N'], sigma=str(sig), lam_h=r['lam_h'],
                lb=enc(lb), lb_float=float(lb.lower()), xhash=r['xhash'], sec=round(time.time() - t0, 2))

# ------------------------------------------------------------------ second eigenvalue of E
def arb_K_M(M, beta):
    N = M.N
    K = [[arb(0)] * N for _ in range(N)]; md = [arb(0)] * N
    for e, p in zip(M.elems.tolist(), M.epatch.tolist()):
        for m in range(3):
            k, l = (m + 1) % 3, (m + 2) % 3; c = 2 * M.cot[p][m]
            K[e[k]][e[k]] += c; K[e[l]][e[l]] += c; K[e[k]][e[l]] -= c; K[e[l]][e[k]] -= c
        for k in range(3): md[e[k]] += M.area[p] / 3
    for ids, j, ln in M.bdry:
        for i in ids.tolist(): K[i][i] += beta[j] * ln
    return K, md

def lam2_node(k, n, sig2=None, g=None):
    """certified lower bound for lambda_2(E, alpha_k): the CR matrix pencil (K, M) has at most one eigenvalue
    below sigma2, because B = K - sigma2 M + g (M x)(M x)^T is positive definite (x = any vector); B > 0 is shown
    by a congruence C = S^T B S (S = float approximation of an inverse Cholesky factor, entered exactly) and
    Gershgorin's theorem.  Then lambda_2 >= sigma2/(1 + (C_Liu h)^2 sigma2)."""
    al = ALPHA(k); t0 = time.time()
    a2, A = shapes.tri_data((Fr(0), Fr(0))); beta = [shapes.A_(al)] * 3
    M = rcr.build_mesh(a2, A, n, 'U'); N = M.N
    Kf, mdf = rcr.float_mats(M, beta); Kf = Kf.toarray()
    lam, vec = sla.eigh(Kf, np.diag(mdf), subset_by_index=[0, 2])
    x = vec[:, 0] / math.sqrt(vec[:, 0] @ (mdf * vec[:, 0]))
    if sig2 is None: sig2 = Fr(lam[1] * (1 - 1e-6)).limit_denominator(2 ** 40)
    if g is None: g = Fr(2 * (lam[1] - lam[0]) + 1).limit_denominator(2 ** 20)
    mx = mdf * x
    Bf = Kf - float(sig2) * np.diag(mdf) + float(g) * np.outer(mx, mx)
    L = np.linalg.cholesky(Bf); S = np.linalg.inv(L).T
    Ka, mda = arb_K_M(M, beta)
    s2a, ga = shapes.A_(sig2), shapes.A_(g)
    xa = [arb(float(v)) for v in x]; mxa = [mda[i] * xa[i] for i in range(N)]
    B = arb_mat(N, N)
    for i in range(N):
        for j in range(N):
            v = Ka[i][j] + ga * mxa[i] * mxa[j]
            if i == j: v -= s2a * mda[i]
            B[i, j] = v
    Sa = arb_mat(N, N, [arb(float(v)) for v in S.ravel()])
    C = Sa.transpose() * B * Sa
    ok = True; worst = None
    for i in range(N):
        off = arb(0)
        for j in range(N):
            if j != i: off += abs(C[i, j])
        gap = C[i, i] - off
        if not bool(gap > 0): ok = False
        worst = gap if worst is None else worst.min(gap)
    if not ok: raise ValueError('lambda_2 inertia certificate failed')
    lb = rcr.cr_bound(sig2, M.h)
    return dict(key=[k, n], alpha=str(al), n=n, N=N, sigma2=str(sig2), g=str(g), lam1_h=float(lam[0]), lam2_h=float(lam[1]),
                gersh_min=float(worst.lower()), lb=enc(lb), lb_float=float(lb.lower()), sec=round(time.time() - t0, 2))

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'ring':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else N_RING
        kmod, kres = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1, 0)
        fn = os.path.join(CERT, 'ring_nodes.jsonl'); done = load(fn)
        with open(fn, 'a') as f:
            for k in range(KTOP + 1):
                if k % kmod != kres: continue
                for sign in (1, -1):
                    if (k, sign, n) in done: continue
                    d = ring_node(k, sign, n); f.write(json.dumps(d) + "\n"); f.flush()
                    if k % 16 == 0: print(d['key'], d['alpha'], d['lb_float'], d['sec'], flush=True)
    elif mode == 'lam2':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else N_LAM2
        fn = os.path.join(CERT, 'lam2_nodes.jsonl'); done = load(fn)
        with open(fn, 'a') as f:
            for k in range(0, KTOP + 1, 4):
                if (k, n) in done: continue
                d = lam2_node(k, n); f.write(json.dumps(d) + "\n"); f.flush()
                print(d['key'], d['alpha'], d['lam2_h'], d['lb_float'], d['gersh_min'], d['sec'], flush=True)

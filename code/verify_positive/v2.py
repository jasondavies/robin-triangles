"""V2: second eigenvalue of the CR pencil of the equilateral triangle E (mesh 'U').

Method (different from the paper's Lemma 'inertia'): interval LDL^T factorisation of
B = K - sigma2*M in Arb ball arithmetic (banded, after a bandwidth-reducing symmetric
permutation proposed in floating point).  If every pivot ball excludes 0, the exact
factorisation B = L D L^T exists with the pivots in the computed balls, and by Sylvester's
law of inertia the number of negative eigenvalues of B equals the number of negative
pivots.  At most one negative pivot  =>  lambda_{2,h} > sigma2.

K and M are assembled here in Arb directly from Arb vertex coordinates of every mesh
vertex (edge vectors = differences of vertex coordinates), independently of the float
interval assembly used in V1 (the two are compared as a cross-check).
"""
import time
from fractions import Fraction as Fr
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
from flint import arb, ctx, fmpq

import geom
import crmesh

ARB_PREC = 320


def _arb_fr(q):
    q = Fr(q)
    return arb(fmpq(q.numerator, q.denominator))


def assemble_arb_E(n, alpha):
    """dict-of-dict K, list M for the mesh 'U' of E (area 1, side^2 = 4/sqrt 3), beta = alpha"""
    ctx.prec = ARB_PREC
    tri = geom.tri_equilateral(alpha)
    m = crmesh.build_mesh(tri, "U", n)            # topology (integer data) only
    a2 = arb(4) / arb(3).sqrt()
    a = a2.sqrt()
    P1 = (a, arb(0))
    x2 = a / 2
    y2 = (a2 - x2 * x2).sqrt()
    P2 = (x2, y2)
    # explicit coordinates of all lattice vertices
    coords = {}
    for i in range(n + 1):
        for j in range(n + 1 - i):
            coords[(i, j)] = ((P1[0] * i + P2[0] * j) / n, (P1[1] * i + P2[1] * j) / n)
    N = m.N
    K = [dict() for _ in range(N)]
    M = [arb(0) for _ in range(N)]
    al = _arb_fr(alpha)
    min_c = None
    h2 = arb(0)
    area = arb(0)
    for e in range(len(m.elem_lat)):
        pts = [coords[(int(m.elem_ij[e, v, 0]), int(m.elem_ij[e, v, 1]))] for v in range(3)]
        d = []
        for k in range(3):
            pa = pts[(k + 1) % 3]
            pb = pts[(k + 2) % 3]
            d.append((pb[0] - pa[0], pb[1] - pa[1]))
        X = d[0][0] * d[1][1] - d[0][1] * d[1][0]        # 2|K|
        if not (X > 0):
            raise ArithmeticError("element area not certified positive")
        area += X / 2
        ge = [int(v) for v in m.elem_e[e]]
        for k in range(3):
            l2 = d[k][0] * d[k][0] + d[k][1] * d[k][1]
            h2 = h2.max(l2)
            M[ge[k]] += X / 6
            a_, b_ = (k + 1) % 3, (k + 2) % 3
            c = -2 * (d[a_][0] * d[b_][0] + d[a_][1] * d[b_][1]) / X
            if not (c >= 0):
                raise ArithmeticError("coupling sign not certified")
            ia, ib = ge[a_], ge[b_]
            K[ia][ib] = K[ia].get(ib, arb(0)) - c
            K[ib][ia] = K[ib].get(ia, arb(0)) - c
            K[ia][ia] = K[ia].get(ia, arb(0)) + c
            K[ib][ib] = K[ib].get(ib, arb(0)) + c
            if m.loc_side[e, k] >= 0:
                K[ge[k]][ge[k]] = K[ge[k]].get(ge[k], arb(0)) + al * l2.sqrt()
    return m, K, M, h2, area


def ldl_negative_pivots(K, M, sigma, perm=None):
    """number of negative pivots of the LDL^T factorisation of K - sigma M (Arb), or None"""
    N = len(M)
    if perm is None:
        rows = []
        cols = []
        for i in range(N):
            for j in K[i]:
                rows.append(i)
                cols.append(j)
        pat = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
        perm = np.asarray(reverse_cuthill_mckee(pat, symmetric_mode=True))
    pos = np.empty(N, int)
    pos[perm] = np.arange(N)
    assert sorted(perm.tolist()) == list(range(N))
    bw = 0
    for i in range(N):
        for j in K[i]:
            bw = max(bw, abs(int(pos[i]) - int(pos[j])))
    sg = _arb_fr(sigma)
    zero = arb(0)
    # banded lower storage: A[i][j - i + bw], j in [i-bw, i]
    A = [[zero] * (bw + 1) for _ in range(N)]
    for i in range(N):
        pi = int(pos[i])
        for j, v in K[i].items():
            pj = int(pos[j])
            if pj <= pi:
                val = v - sg * M[i] if i == j else v
                A[pi][pj - pi + bw] = val
    neg = 0
    min_abs_pivot = None
    for k in range(N):
        d = A[k][bw]
        if d > 0:
            pass
        elif d < 0:
            neg += 1
        else:
            return None, bw, None
        ad = abs(d).lower() if hasattr(abs(d), "lower") else abs(d)
        min_abs_pivot = ad if min_abs_pivot is None else min(min_abs_pivot, ad, key=float)
        hi = min(N - 1, k + bw)
        colk = [A[i][k - i + bw] for i in range(k + 1, hi + 1)]
        nzidx = [t for t, v in enumerate(colk) if not v.is_zero()]
        for ii in nzidx:
            i = k + 1 + ii
            l = colk[ii] / d
            row = A[i]
            off = bw - i + k + 1
            for jj in nzidx:
                if jj > ii:
                    break
                row[off + jj] = row[off + jj] - l * colk[jj]
    return neg, bw, float(min_abs_pivot)


def verify_lam2(rec, kappa0):
    t0 = time.time()
    out = {"ok": False}
    try:
        ctx.prec = ARB_PREC
        k, n = rec["key"]
        alpha = geom.parse_rat(rec["alpha"])
        out["alpha_matches_grid"] = bool(alpha == geom.alpha_grid(k) and n == rec["n"])
        sigma2 = geom.parse_rat(rec["sigma2"])
        lb = geom.parse_rat(rec["lb"])
        m, K, M, h2, area = assemble_arb_E(n, alpha)
        out["N"] = len(M)
        out["N_matches"] = bool(len(M) == rec["N"])
        out["area_is_one"] = bool(area.contains(1) and float(area.rad()) < 1e-60)
        neg, bw, minpiv = ldl_negative_pivots(K, M, sigma2)
        out["bandwidth"] = bw
        out["neg_pivots"] = neg
        out["min_abs_pivot"] = minpiv
        out["inertia_ok"] = bool(neg is not None and neg <= 1)
        # lb <= sigma2/(1+(kappa0 h)^2 sigma2); exact rationals with an exact upper bound for h^2:
        # the high precision mpmath-interval value from the lattice edge types, and the Arb value from
        # the explicit vertex coordinates must agree
        h2_hi = m.h2_hi
        h2q = arb(fmpq(h2_hi.numerator, h2_hi.denominator))
        out["h2_cross_check"] = bool(h2 <= h2q and (h2q - h2) < arb(2) ** (-150))
        bound = sigma2 / (1 + kappa0 * kappa0 * h2_hi * sigma2)
        out["lb_ok"] = bool(0 < lb <= bound)
        out["lb_rel_slack"] = float((bound - lb) / bound)
        out["ok"] = bool(out["inertia_ok"] and out["lb_ok"] and out["N_matches"] and out["alpha_matches_grid"]
                         and out["area_is_one"] and out["h2_cross_check"] and m.topo_ok)
    except Exception as ex:
        out["reason"] = "exception: %r" % (ex,)
    out["sec"] = round(time.time() - t0, 2)
    return out

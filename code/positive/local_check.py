"""Region C (neighbourhood of E), step 2: the one-variable check of the local lemma (Proposition "local").

For 0 < t <= t_loc, every phi, and every alpha >= 1/100 it verifies (in Arb) the sufficient condition

   [A1 lam0 - (A1 - c_l) alpha lam0'] (1 - (eta + eps_A) gamma)  >  ( sqrt(R1) s1 + rho t_loc sqrt(alpha lam0' gamma) )^2

where, on each interval [alpha_k, alpha_{k+1}] of the grid (and on [alpha_KTOP, infinity)):
   lam0, lam0' : explicit (equilateral triangle), monotone in alpha;
   gamma  >= lam2/(lam2 - lam0)  from the certified lower bounds for lam2(E, .) (monotone in alpha);
   R1     <= [2 + (d+ + d-)/(lam2 - lam0)] / [s^2 (1/d+ + 1/d-)],  d+- >= lam0(alpha) - mu_+-(alpha) on the interval,
            using the chord of the certified ring bounds mu_+-(alpha_k) (concavity in alpha) and two tangents of lam0;
   A1 = 2 <= (cosh 2t - 1)/t^2,  c_l <= (mean_j ell_j - 1)/t^2,  s1 >= sinh(2t)/(2t),  rho >= max_j |rho_j|/t^2,
   eta = 1 - exp(-2 t_loc),  eps_A = cosh(2 t_loc) - 1     (shape constants, by interval arithmetic in (t, phi)).

Usage: python3 local_check.py [t_loc as fraction | -] [n_ring | -] [out.json]
"""
import os, sys, json, hashlib
from fractions import Fraction as Fr
from flint import arb, ctx
import shapes
from shapes import A_
from local_ring import load, dec
from config import ALPHA, KTOP, S_RING, CERT, T_LOC, N_RING, C_ELL, RHO_HAT

ctx.prec = 100
def up(x): return arb(x.upper())
def lo(x): return arb(x.lower())
def hull(a, b): return A_(a).union(A_(b))

def shape_constants(t_loc, NT=40, NP=60):
    """rigorous c_l (lower) and rho (upper) over 0 < t <= t_loc, all phi (phi in [0, pi/3] suffices by symmetry)."""
    PI = arb.pi()
    th2 = [arb(0), 4 * PI / 3, 8 * PI / 3]                      # 2*theta_j
    def C2(t): return arb(2) if t == 0 else ((2 * A_(t)).cosh() - 1) / A_(t) ** 2
    def S1(t): return arb(2) if t == 0 else (2 * A_(t)).sinh() / A_(t)
    cl = None; rho = None
    for i in range(NT):
        t0, t1 = t_loc * i / NT, t_loc * (i + 1) / NT
        tb = hull(t0, t1)
        c2 = hull(lo(C2(t0)), up(C2(t1))); s1 = hull(lo(S1(t0)), up(S1(t1)))
        ch, sh = (2 * tb).cosh(), (2 * tb).sinh()
        for j in range(NP):
            ph = hull(PI / 3 * j / NP, PI / 3 * (j + 1) / NP)
            c = [(2 * ph - a).cos() for a in th2]
            ell = [(ch + sh * cj).sqrt() for cj in c]
            v = [(c2 * tb + s1 * cj) ** 2 / (2 * (1 + lj) ** 2) for cj, lj in zip(c, ell)]
            vbar = (v[0] + v[1] + v[2]) / 3
            a = lo(c2 / 2 - vbar)                                # (lbar - 1)/t^2
            r = up(max([up(abs(vj - vbar)) for vj in v], key=lambda z: float(z.mid())))
            for vj in v: r = r.max(up(abs(vj - vbar)))
            cl = a if cl is None else cl.min(a)
            rho = r if rho is None else rho.max(r)
    return lo(cl), up(rho)

def main(t_loc=T_LOC, n_ring=None, out=None, quiet=False):
    ring = load(os.path.join(CERT, 'ring_nodes.jsonl')); l2n = load(os.path.join(CERT, 'lam2_nodes.jsonl'))
    def mu(k, sign):
        c = [dec(d['lb']) for key, d in ring.items() if key[0] == k and key[1] == sign and (n_ring is None or key[2] == n_ring)]
        if not c: raise KeyError((k, sign))
        return A_(max(c))
    def L2(k):
        k4 = 4 * (k // 4)
        c = [dec(d['lb']) for key, d in l2n.items() if key[0] == k4]
        return A_(max(c))
    s = A_(S_RING); tl = A_(t_loc)
    cl, rho = shape_constants(t_loc)
    cl_computed, rho_computed = cl, rho
    if t_loc == T_LOC:
        # Lemma "shapeconst": the rounded constants printed in the paper are what inequality (C) uses
        if not (bool(cl >= A_(C_ELL)) and bool(rho <= A_(RHO_HAT))): raise ValueError('shape constants of the paper not confirmed')
        cl, rho = A_(C_ELL), A_(RHO_HAT)
    A1 = arb(2); s1 = up((2 * tl).sinh() / (2 * tl)); eta = up(1 - (-2 * tl).exp()); epsA = up((2 * tl).cosh() - 1)
    res = dict(t_loc=str(t_loc), s=str(S_RING), KTOP=KTOP, c_l=float(cl.lower()), rho=float(rho.upper()), s1=float(s1.upper()),
               eta=float(eta.upper()), eps_A=float(epsA.upper()), prec=ctx.prec,
               c_l_computed_lower=float(cl_computed.lower()), rho_computed_upper=float(rho_computed.upper()), intervals=[])
    l0 = {}
    def L0(k):
        if k not in l0: l0[k] = shapes.lam0_d(ALPHA(k))
        return l0[k]
    def check(lam_lo, lam_hi, alam_hi, dp, dm, l2, tag, a_lo):
        gam = up(l2 / (l2 - lam_hi))
        if not bool(l2 - lam_hi > 0): return False, None
        R1 = up((2 + (dp + dm) / (l2 - lam_hi)) / (s * s * (1 / dp + 1 / dm)))
        lhs = (A1 * lam_lo - (A1 - cl) * alam_hi) * (1 - (eta + epsA) * gam)
        rhs = (R1.sqrt() * s1 + rho * tl * (alam_hi * gam).sqrt()) ** 2
        ok = bool(lhs > rhs) and bool(1 - (eta + epsA) * gam > 0)
        res['intervals'].append(dict(k=tag, alpha_lo=a_lo, ok=ok, R1_over_lam=float((R1 / lam_lo).upper()), gamma=float(gam.upper()),
                                     dplus=float(dp.upper()), dminus=float(dm.upper()), lhs=float(lhs.lower()), rhs=float(rhs.upper()),
                                     ratio=float((lhs / rhs).lower())))
        return ok, float((lhs / rhs).lower())
    allok = True; worst = (9e9, None)
    for k in range(KTOP):
        a0, a1 = A_(ALPHA(k)), A_(ALPHA(k + 1))
        (f0, g0), (f1, g1) = L0(k), L0(k + 1)                    # lam0 and lam0' at the two nodes
        p = (f1 - f0 + g0 * a0 - g1 * a1) / (g0 - g1)            # kink of min(tangent_k, tangent_{k+1})
        if not (bool(g0 > g1) and bool(p >= a0) and bool(p <= a1)): raise ArithmeticError('kink not enclosed in its grid interval')
        Up = f0 + g0 * (p - a0)
        d = []
        for sign in (1, -1):
            m0, m1 = mu(k, sign), mu(k + 1, sign)
            chord_p = m0 + (m1 - m0) * (p - a0) / (a1 - a0)
            dd = up(f0 - m0).max(up(f1 - m1)).max(up(Up - chord_p))
            if not bool(dd > 0): raise ValueError('nonpositive delta')
            d.append(dd)
        ok, ratio = check(lo(f0), up(f1), up(a1 * g0), d[0], d[1], L2(k), k, float(ALPHA(k)))
        allok &= ok
        if ratio is not None and ratio < worst[0]: worst = (ratio, k)
    # top interval [alpha_KTOP, infinity)
    fT, gT = L0(KTOP); LD = shapes.LAM_D
    half = shapes.lam0(ALPHA(KTOP) / 2)
    d = [up(LD - mu(KTOP, sign)) for sign in (1, -1)]
    ok, ratio = check(lo(fT), up(LD), up(2 * (LD - half)), d[0], d[1], L2(KTOP), 'top', float(ALPHA(KTOP)))
    allok &= ok
    if ratio is not None and ratio < worst[0]: worst = (ratio, 'top')
    res['ok'] = bool(allok); res['worst_ratio'] = worst[0]; res['worst_k'] = worst[1]
    h = hashlib.sha256()
    for fn in ('ring_nodes.jsonl', 'lam2_nodes.jsonl'): h.update(open(os.path.join(CERT, fn), 'rb').read())
    res['inputs_sha256'] = h.hexdigest()
    if out: json.dump(res, open(out, 'w'), indent=0)
    if not quiet:
        print(f"t_loc={t_loc}: c_l>={float(cl.lower()):.5f} rho<={float(rho.upper()):.5f} ok={allok} worst lhs/rhs={worst[0]:.4f} at k={worst[1]}")
    return res

if __name__ == '__main__':
    t_loc = Fr(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != '-' else T_LOC
    n_ring = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != '-' else None
    out = sys.argv[3] if len(sys.argv) > 3 else None
    if out == 'auto': out = os.path.join(CERT, 'local_cert.json')
    r = main(t_loc, n_ring, out)
    sys.exit(0 if r['ok'] else 1)

"""Certified check (ball arithmetic, python-flint/Arb) for Theorem "uniform weak coupling" (Part II), area 1:
for alpha1 = 1/100 and every shape  T = exp(t S_phi)(E),  0 < t <= T_W = 1.2413,  phi in [0, pi],
at least one of the following holds on the whole box containing (t, phi):
  (W) some ell_j >= 3                                   [Theorem "thin"]
  (S) alpha1 * R_I^2 <= (P - P_E) r^2                   [Theorem "weak coupling"]
  (L) the explicit sufficient form of the Schur criterion, inequality (L) of the paper.
Separately: (T) cosh(2 T_W) + sinh(2 T_W)/2 >= 9, so that t >= T_W implies (W);
            (0) the scaled form of (L) on 0 < t <= t0.
Only elementary functions of two variables are involved.  Run time: well under a minute, one process.
Usage: python3 weak_cert.py [out.json]"""
import sys, json, time
from flint import arb, ctx
ctx.prec = 80
def I(lo, hi): return arb(lo).union(arb(hi))
def Q(p, q=1): return arb(p) / arb(q)
ALPHA1 = Q(1, 100); CC = Q(118, 100); T_W = Q(12413, 10000); T0 = Q(1, 100)
H = arb(3).sqrt().sqrt(); PE = 6 / H; SE = 2 / H
MU2 = 4 * arb.pi() ** 2 / (3 * arb(3).sqrt())           # second Neumann eigenvalue of E (area 1)
G2 = (CC * CC / (1 - 4 * CC * ALPHA1 / PE) - 1 + Q(22, 10) * ALPHA1 / PE) / (CC - 1)
G1 = (G2 - 1 + Q(22, 10) * ALPHA1 / PE).sqrt()
E1 = ALPHA1 * PE / MU2; E2 = ALPHA1 * PE / (MU2 - ALPHA1 * PE)
TH2 = [arb(0), 4 * arb.pi() / 3, 2 * arb.pi() / 3]      # 2*theta_j mod 2 pi
def pos(x): return bool(x > 0)                           # certainly positive
def lower(x): return x.mid() - x.rad()
def rhs_bracket(e2t_minus1, zeta):
    return ALPHA1 * ((G1 * e2t_minus1 + zeta) / (1 - E1).sqrt() + G1 * ALPHA1 * zeta / (MU2 - ALPHA1 * PE).sqrt()) ** 2
def check_box(t, ph):
    c2, s2 = (2 * t).cosh(), (2 * t).sinh()
    ell = [(c2 + s2 * (2 * ph - a).cos()).sqrt() for a in TH2]
    if any(not x.is_finite() for x in ell): return None
    if any(pos(x - 3) or bool(x >= 3) for x in ell): return 'W'
    a = [SE * x for x in ell]; P = a[0] + a[1] + a[2]; r = 2 / P
    if all(pos((P - PE) * r * r - ALPHA1 * (r * r + (P / 2 - ak) ** 2)) for ak in a): return 'S'
    lbar = (ell[0] + ell[1] + ell[2]) / 3; phat = lbar - 1
    z0 = (Q(2, 3) * sum((x - lbar) ** 2 for x in ell)).sqrt()
    zeta = 2 * phat + z0
    lhs = phat * (PE - G2 * ALPHA1) * ((-2 * t).exp() * (1 + E2) - E2)
    if pos(lhs - rhs_bracket((2 * t).exp() - 1, zeta)): return 'L'
    return None
def run():
    out = {'alpha1': '1/100', 'c': '1.18', 'T_W': '1.2413', 't0': '0.01', 'prec': ctx.prec}
    out['g1'] = str(G1); out['g2'] = str(G2); out['mu2'] = str(MU2); out['P_E'] = str(PE)
    # (T) thin threshold
    out['check_T'] = pos((2 * T_W).cosh() + (2 * T_W).sinh() / 2 - 9)
    # (0) scaled check on (0, t0]
    Cc = ((2 * T0).cosh() - 1) / (2 * T0 * T0); Cs = (2 * T0).sinh() / (2 * T0); Ce = ((2 * T0).exp() - 1) / (2 * T0)
    C1 = (T0.exp() - 1) / T0; u0 = (2 * T0).exp() - 1
    m2 = 4 * T0 * T0 * Cc * Cc + 2 * Cs * Cs; m3 = 8 * T0 * Ce ** 3
    K = Q(1, 16) / ((1 - u0) ** 2 * (1 - u0).sqrt())
    cp = 1 - m2 / 8 - K * m3; cz = 2 * Cc * T0 + arb(2).sqrt() * C1
    lhs0 = cp * (PE - G2 * ALPHA1) * ((-2 * T0).exp() * (1 + E2) - E2)
    rhs0 = rhs_bracket(2 * Ce, cz)
    out['check_0'] = pos(lhs0 - rhs0); out['c_p'] = str(cp); out['c_zeta'] = str(cz)
    out['lhs0'] = str(lhs0); out['rhs0'] = str(rhs0)
    # boxes on [t0, T_W] x [0, pi]
    PI = arb.pi(); NT, NP = 64, 96
    stack = [(i, i + 1, j, j + 1, 0) for i in range(NT) for j in range(NP)]
    cnt = {'W': 0, 'S': 0, 'L': 0}; fails = []; nbox = 0; maxdepth = 0
    while stack:
        i0, i1, j0, j1, d = stack.pop(); nbox += 1; maxdepth = max(maxdepth, d)
        den = NT * 2 ** d; denp = NP * 2 ** d
        t = I(T0 + (T_W - T0) * Q(i0, den), T0 + (T_W - T0) * Q(i1, den))
        ph = I(PI * Q(j0, denp), PI * Q(j1, denp))
        res = check_box(t, ph)
        if res: cnt[res] += 1; continue
        if d >= 12: fails.append((str(t), str(ph))); continue
        for (a, b) in ((2 * i0, 2 * i0 + 1), (2 * i0 + 1, 2 * i1)):
            for (c, e) in ((2 * j0, 2 * j0 + 1), (2 * j0 + 1, 2 * j1)):
                stack.append((a, b, c, e, d + 1))
    out['boxes'] = nbox; out['leaves'] = cnt; out['max_depth'] = maxdepth; out['fails'] = fails[:20]; out['n_fails'] = len(fails)
    out['ok'] = bool(out['check_T'] and out['check_0'] and not fails)
    return out
if __name__ == '__main__':
    t0 = time.time(); out = run(); out['seconds'] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1: json.dump(out, open(sys.argv[1], 'w'), indent=1)
    sys.exit(0 if out['ok'] else 1)

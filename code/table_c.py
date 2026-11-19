"""Floating-point numbers quoted in the paper that are illustrations, not part of any proof:
 - Table 1 (Lemma certC in floating point at selected beta);
 - the floating-point threshold of the inequality (certC) (the certified statement starts at M = 13/10);
 - the remark after Theorem A: lim_{T->E} (p-1)/(tau-2) = 3/16 and the coupling beta at which g(beta) = 3/16.
Writes ../verification/table_c.json."""
from math import tanh, atanh, sqrt, asin, degrees, sinh, cos, sin, pi
from large_beta import float_margin
from three_float import Mof
from out import write

H = 3 ** 0.25
rows = []
for beta in [1.2, 1.5, 2.0, 3.0, 4.0, 6.0]:
    M = Mof(beta); f = float_margin(M); t = tanh(M)
    rows.append(dict(beta=beta, M=M, theta_star_deg=degrees(2 * asin(t / 2)), gamma=f['gamma'], Lam=f['Lam'], rhs=f['rhs']))

def margin(M):
    f = float_margin(M)
    return f['rhs'] - f['Lam']

lo, hi = 1.0, 1.3                      # margin(lo) < 0 < margin(hi)
assert margin(lo) < 0 < margin(hi)
for _ in range(60):
    mid = (lo + hi) / 2
    if margin(mid) > 0: hi = mid
    else: lo = mid
Mthr = hi; bthr = tanh(Mthr) * (Mthr + atanh(tanh(Mthr) / 2)) / H

def g_of(M):
    t = tanh(M); K = M + atanh(t / 2); KM = 1 + 2 * (1 - t * t) / (4 - t * t)
    return 0.25 - K / (2 * sinh(2 * M) * KM)

lo, hi = 0.5, 3.0
for _ in range(80):
    mid = (lo + hi) / 2
    if g_of(mid) < 3 / 16: lo = mid
    else: hi = mid
Mg = lo; bg = tanh(Mg) * (Mg + atanh(tanh(Mg) / 2)) / H

def p_tau(x, y):                       # apex (x, y) over the base [0, 1], rescaled to area 1
    s = sqrt(2 / y); P = (1 + sqrt(x * x + y * y) + sqrt((1 - x) ** 2 + y * y)) * s
    tau = (1 + x * x + y * y + (1 - x) ** 2 + y * y) / (sqrt(3) * y)
    return P / (6 / H), tau

ratios = []
for ang in [0.0, 0.7, 1.9, 3.0]:
    eps = 1e-4
    p, tau = p_tau(0.5 + eps * cos(ang), sqrt(3) / 2 + eps * sin(ang))
    ratios.append((p - 1) / (tau - 2))

write('table_c.json', dict(rows=rows, threshold_M=Mthr, threshold_beta=bthr, g_equals_3_16_beta=bg, g_equals_3_16_M=Mg,
                           limit_ratios=ratios, ok=bool(max(abs(r - 3 / 16) for r in ratios) < 1e-4)))
print(rows, Mthr, bthr, bg, ratios)

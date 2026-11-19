"""float cross-check: normalised formulas in large_beta.quantities vs direct closed forms (three_float.scal)"""
from math import *
from three_float import scal, H
from large_beta import quantities, float_margin
from out import write
worst = 0.0
for M in [0.8, 1.3, 2.0, 3.4, 6.0]:
    d = scal(M); X = exp(2 * d['U']) / 16 * 2 / H ** 2
    D3, G1, lN1, bB1, bkq = quantities(exp(-2 * M), d['K'] * exp(-2 * M), d['t'])
    ref = [d['D3'], d['Gd'] + d['Go'] / 2, -d['lam0'] * (d['Nd'] - d['No']),
           d['beta'] * (d['s1'] + 2 * d['s2'] - 2 * d['s3'] - d['s4']), d['beta'] * d['kq']]
    worst = max(worst, max(abs(a * X / b - 1) for a, b in zip([D3, G1, lN1, bB1, bkq], ref)))
    print(M, [f"{a * X / b - 1:.1e}" for a, b in zip([D3, G1, lN1, bB1, bkq], ref)],
          'gamma', (G1 - bB1 + lN1) * X / d['gamma'] - 1)
write('check_normalised.json', dict(max_rel_err=worst, ok=bool(worst < 1e-13)))

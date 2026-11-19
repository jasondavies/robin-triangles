import numpy as np, sys
from out import write

from direct_check import *

rng = np.random.default_rng(12345)
bad = []; worst = {}
nnear = 0
for trial in range(3000):
    M = rng.uniform(1.3, 6.5)
    t = np.tanh(M); ths = 2*np.arcsin(t/2)
    # sample in near region: angles >= ths, with a fraction at corners / edges
    mode = trial % 4
    if mode == 0:   # corner type 1
        eps = rng.uniform(0, 1e-3)*(np.pi/3 - ths)
        th = [ths + eps, ths + eps, np.pi - 2*ths - 2*eps]
    elif mode == 1:  # corner type 2
        eps = rng.uniform(0, 1e-3)*(np.pi/3 - ths)
        a = ths + eps; th = [a, (np.pi - a)/2, (np.pi - a)/2]
    elif mode == 2:  # extremely close to equilateral
        d = rng.normal(size=3); d -= d.mean(); d *= 10**rng.uniform(-7, -2)*(np.pi/3 - ths)
        th = list(np.pi/3 + d)
    else:            # uniform in near region
        while True:
            x = rng.dirichlet([1, 1, 1])*(np.pi - 3*ths) + ths
            th = list(x); break
    th = rng.permutation(th)
    if min(th) <= ths: continue
    r = chain(th, M)
    nnear += 1
    for k in ['ok_1P1', 'ok_qperp', 'ok_LamT', 'ok_Lamhi']:
        if r[k] is False: bad.append((k, M, th, r))
    if not r['cPc'] < 0: bad.append(('cPc', M, th, r))
    if not r['ritz_gap'] < 0: bad.append(('ritz', M, th, r))
    worst['q_err'] = max(worst.get('q_err', 0), r['q_err'])
    worst['LamT_true/Lamhi'] = max(worst.get('LamT_true/Lamhi', -9), r['LamT_true']/r['Lamhi'])
    worst['rho_lo_slack'] = min(worst.get('rho_lo_slack', 9), r['rho'].min() - r['rlo'])
    worst['rho_hi_slack'] = min(worst.get('rho_hi_slack', 9), r['rhi'] - r['rho'].max())
print('near-region triangles tested', nnear, 'failures', len(bad))
for b in bad[:10]: print(b[0], b[1], np.degrees(b[2]), {k: b[3][k] for k in ['cPc', 'ritz_gap', 'LamT_true', 'LamT', 'Lamhi', 'W']})
print(worst)
from collections import Counter
print(Counter(b[0] for b in bad))
print('max W among failures', max(b[3]['W'] for b in bad))
Wbad = max([b[3]['W'] for b in bad], default=0.0)
write('run_chain.json', dict(tested=nnear, failures=len(bad), max_W_among_failures=float(Wbad),
      q_err=float(worst['q_err']), LamT_ratio=float(worst['LamT_true/Lamhi']),
      ok=bool(Wbad < 1e-15 and worst['LamT_true/Lamhi'] < 1)))

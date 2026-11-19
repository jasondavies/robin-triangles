import sys
from fem2 import lam1, lam0_of_beta
import fem2
import direct_check as dc
import numpy as np
from scipy.linalg import eigh
from out import write
rows = []
for beta in [1.16, 1.5, 2.0, 3.0]:
    l0, M = lam0_of_beta(beta); t = np.tanh(M); ths = 2*np.arcsin(t/2)
    E = fem2.tri_from_angles([np.pi/3]*3); fE = lam1(E, beta, 120)
    print(f"beta={beta} M={M:.4f} theta*={np.degrees(ths):.4f}deg  FEM(E)-lam0 = {fE-l0:.2e}")
    gap = np.pi/3 - ths
    for name, th in [('cornerA', [ths*1.0001+0*gap, ths*1.0001, np.pi-2*ths*1.0001]),
                     ('cornerB', [ths*1.0001, (np.pi-ths*1.0001)/2, (np.pi-ths*1.0001)/2]),
                     ('mid', [np.pi/3 - gap/2, np.pi/3, np.pi/3 + gap/2]),
                     ('small', [np.pi/3 - gap/10, np.pi/3+gap/20, np.pi/3 + gap/20])]:
        V = fem2.tri_from_angles(th)
        f = lam1(V, beta, 120)
        d = dc.pencil(dc.tri_from_angles(th), M)
        A = d['G'] - beta*d['Bd']; ritz = eigh(A, d['N'], eigvals_only=True)[0]
        rows.append(dict(beta=beta, shape=name, fem_minus_lam0=float(f - l0), ritz3_minus_lam0=float(ritz - l0), fem_err_E=float(abs(fE - l0)),
                         ok=bool(ritz < l0 and f <= ritz + abs(fE - l0))))
        print(f"  {name:8s} angles {np.round(np.degrees(th),4)}: FEM-lam0 {(f-l0):.4e}  Ritz3-lam0 {(ritz-l0):.4e}  FEM<=Ritz {f <= ritz+1e-9}", flush=True)
write('t_near_fem.json', dict(samples=len(rows), rows=rows, ok=all(r['ok'] for r in rows)))

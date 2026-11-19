"""Float cross-checks (not part of the proof):
 (1) g = D/(2 beta B) from the eight integrals of this section vs the formula of Lemma 2.3 (old paper);
     B/N vs 8 K K_M / (H[(1-t^2)K + t K_M]).
 (2) Lemma geomA (old paper) by random sampling: (p-1) - 0.1425 (tau-2) > 0 whenever p <= 1.384.
 (3) asymptotic constants of Remark (size of the margin)."""
import numpy as np
from math import *
from three_float import scal, Mof, H
from large_beta import float_margin, geom
from out import write
REC = dict(identities_rel_max=0.0, asymptotics={})
for beta in [0.3, 0.8, 1.2, 2.0, 4.0]:
    M = Mof(beta); d = scal(M); t, K = d['t'], d['K']
    Dn = 3 * d['D3']; B = 3 * (d['s1'] + 2 * d['s2'] + 4 * d['s3'] + 2 * d['s4']); N = 3 * d['Nd'] + 6 * d['No']
    KM = 1 + 2 * (1 - t * t) / (4 - t * t)
    REC['identities_rel_max'] = max(REC['identities_rel_max'], abs(Dn/(2*beta*B) / (0.25 - K/(2*sinh(2*M)*KM)) - 1),
                                    abs(B/N / (8*K*KM/(H*((1-t*t)*K+t*KM))) - 1), abs((Dn-beta*B)/N / d['lam0'] - 1))
    print(f"beta={beta}: g closed forms {Dn/(2*beta*B):.12f} formula {0.25 - K/(2*sinh(2*M)*KM):.12f};  "
          f"B/N {B/N:.10f} formula {8*K*KM/(H*((1-t*t)*K+t*KM)):.10f};  (D-beta B)/N {(Dn-beta*B)/N:.10f} lam0 {d['lam0']:.10f}")
rng = np.random.default_rng(7); worst = 1e9; cnt = 0
for _ in range(400000):
    th = rng.dirichlet([1, 1, 1]) * pi
    s = np.sin(th); a = np.sqrt(2 / s.prod()) * s
    p = a.sum() / (6 / H); tau = (a * a).sum() / (2 * sqrt(3))
    if p <= 1.384 and tau - 2 > 1e-9:
        cnt += 1; worst = min(worst, (p - 1) / (tau - 2))
REC.update(geomA_samples=cnt, geomA_min_ratio=float(worst))
print("geomA sampling:", cnt, "triangles with p<=1.384; min (p-1)/(tau-2) =", worst, "(needs > 0.1425)")
for M in [4.0, 6.0, 8.0]:
    f = float_margin(M); t = tanh(M)
    rlo2, rhi2, tau, c = geom(t, sqrt); mu = (tau + sqrt(tau * tau - 4)) / 2
    REC['asymptotics'][str(M)] = dict(gamma=f['gamma']/(1-t), mu=(mu-1)/(1-t), rho=(1-sqrt(rlo2))/(1-t), Lam=f['Lam']/(1-t), rhs=f['rhs'],
                                      one_minus_t=(1-t)/(6*exp(-2*H*f['beta'])))
    print(f"M={M}: gamma/G0/(1-t)={f['gamma']/(1-t):.4f} (3), (mu_hi-1)/(1-t)={(mu-1)/(1-t):.4f} (8/3), "
          f"(1-rho_lo)/(1-t)={(1-sqrt(rlo2))/(1-t):.4f} (2/3), Lam/G0/(1-t)={f['Lam']/(1-t):.4f} (7), rhs={f['rhs']:.5f} (9/4), "
          f"(1-t)/(6 e^(-2 H beta))={(1-t)/(6*exp(-2*H*f['beta'])):.5f}")
a6 = REC['asymptotics']['6.0']
REC['asymptotics_ok'] = bool(abs(a6['gamma'] - 3) < 1e-3 and abs(a6['mu'] - 8 / 3) < 1e-3 and abs(a6['rho'] - 2 / 3) < 1e-3
                             and abs(a6['Lam'] - 7) < 1e-3 and abs(a6['rhs'] - 9 / 4) < 1e-3 and abs(a6['one_minus_t'] - 1) < 1e-3)
REC['ok'] = bool(REC['identities_rel_max'] < 1e-9 and worst > 0.1425 and REC['asymptotics_ok'])
write('check_misc.json', REC)

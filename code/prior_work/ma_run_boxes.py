"""CHECK OF PRIOR WORK (not part of our proof).  An AI referee's independent re-implementation, in Arb ball
arithmetic, of the middle-range criterion of Y. Ma, Zenodo doi:10.5281/zenodo.23191831 (6 October 2026),
Proposition 2 with delta = Delta(beta_-), on 1 <= beta <= 10.  The quantities J, G, T, N are computed by exact
divided differences of exp over the area-one equilateral triangle (ma_arb_check.py), written from the formulas
of that preprint; Ma's own code and certificate files are not part of the Zenodo deposit.
Adaptive subdivision of M in [116/100, 1262/100] (beta(116/100) < 1 < 10 < beta(1262/100)); each box must satisfy
the F-case (F < 0) or the H-case (H > 0 and F E - H^2 < 0).  Writes ../../verification/ma_check.json."""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
from fractions import Fraction as Fr
from flint import arb
from ma_arb_check import setM, CUR, quantities, Delta, R
from out import write, enc


def beta_of(Mfr):
    setM(arb(Mfr.numerator) / Mfr.denominator); return CUR['beta']


def arbf(x): return arb(x.numerator) / x.denominator


Mlo = Fr(116, 100); Mhi = Fr(1262, 100)
blo, bhi = beta_of(Mlo), beta_of(Mhi)
ends_ok = bool(blo < 1) and bool(bhi > 10)
tl = arbf(Mlo).tanh()
angle_ok = bool(tl ** 2 > 2 - arb(2).sqrt())          # theta_c > pi/4 at Mlo (t increasing in M)
stack = []; w = Fr(1, 100); a = Mlo
while a < Mhi:
    b = min(a + w, Mhi); stack.append((a, b)); a = b
nF = nH = 0; evals = 0; ok = True; t0 = time.time(); covered = []
while stack:
    a, b = stack.pop(0); evals += 1
    Mb = arb((arbf(a) + arbf(b)) / 2, float((b - a) / 2) * 1.0000001).union(arbf(a)).union(arbf(b))
    qn = quantities(Mb)
    dl = Delta(arbf(a).tanh()); dl = arb(dl.upper())
    G, T, J = qn['G'], qn['T'], qn['J']
    sh = dl.sinh() / dl
    wgt = arb(3) / 4 - dl.exp() * dl / 6
    if not bool(wgt > 0): ok = False; break
    Fc = 2 * G * sh ** 2 - T * wgt
    E = 1 + ((2 * dl).exp() - 1) * (1 + 4 * R ** 2 * T) + (1 - (-dl).exp()) * 4 * R ** 2 * T
    Hc = -2 * J + R * T / 2 - R * T / 2 * (1 - (-dl).exp())
    if Fc < 0:
        nF += 1; covered.append((a, b))
    elif Hc > 0 and (Fc * E - Hc ** 2) < 0:
        nH += 1; covered.append((a, b))
    else:
        if b - a < Fr(1, 10 ** 7): ok = False; break
        m = (a + b) / 2; stack.insert(0, (m, b)); stack.insert(0, (a, m))
contiguous = bool(covered) and covered[0][0] == Mlo and covered[-1][1] == Mhi and all(
    covered[i][1] == covered[i + 1][0] for i in range(len(covered) - 1))
res = dict(ok=bool(ok and ends_ok and angle_ok and contiguous), prior_work='Ma, doi:10.5281/zenodo.23191831',
           M_lo=str(Mlo), M_hi=str(Mhi), beta_Mlo_upper=enc(blo.upper()), beta_Mhi_lower=enc(bhi.lower()),
           theta_gt_pi_4=angle_ok, boxes=len(covered), F_case=nF, H_case=nH, evaluations=evals,
           boxes_contiguous=contiguous, seconds=round(time.time() - t0, 1))
print(res)
write('ma_check.json', res)
sys.exit(0 if res['ok'] else 1)

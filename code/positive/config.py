"""Global parameters of the computer-assisted part.

Profiles (environment variable RP_PROFILE) differ only in the Crouzeix-Raviart interpolation constant C, where
      || u - Pi_h u ||_{L2(K)} <= C diam(K) || grad(u - Pi_h u) ||_{L2(K)}      for every triangle K, u in H^1(K):
   'ls'  (default)  C = 0.2983 >= sqrt(1/48 + 1/j_{1,1}^2) = 0.298236...   proved by hand in the paper from the
                    Laugesen-Siudeja Poincare inequality (argument of Carstensen-Gallistl);  data/positive/certs/
   'pw'             C = 0.3496 >= sqrt(1/48 + 1/pi^2) = 0.349505...   same proof with the Payne-Weinberger
                    inequality;                                                     data/positive/certs_pw/
   'liu'            C = 0.1893  (Liu 2015, obtained there by verified computation);      data/positive/certs_liu/
Each profile alone proves Theorem 1.1(1).
Environment: RP_CERTS overrides the certificate directory (used for a regeneration into a scratch directory);
RP_OUT is the directory for the outputs of the check scripts (default: the certificate directory)."""
import os
from fractions import Fraction as Fr
PROFILE = os.environ.get('RP_PROFILE', 'ls')
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'positive'))
_P = {'ls': (Fr(2983, 10000), 'certs'), 'pw': (Fr(3496, 10000), 'certs_pw'), 'liu': (Fr(1893, 10000), 'certs_liu')}
C_INTERP = _P[PROFILE][0]; CERT = os.environ.get('RP_CERTS') or os.path.join(ROOT, _P[PROFILE][1])
NSCALE = float(C_INTERP) / 0.1893
OUT = os.environ.get('RP_OUT') or CERT
os.makedirs(CERT, exist_ok=True); os.makedirs(OUT, exist_ok=True)
_pf = os.path.join(CERT, 'profile.json')
if not os.path.exists(_pf):
    import json as _json
    _json.dump({'profile': PROFILE, 'kappa0': str(C_INTERP)}, open(_pf, 'w'))

ALPHA_MIN = Fr(1, 100)                 # alpha <= 1/100 is Theorem "uniform weak coupling" (weak_cert.py)
T_LOC = Fr(3, 50)                      # local region: t <= T_LOC
R_LOC = Fr(1194, 10000)                # <= tanh(2 T_LOC) = 0.119427...; the local disc in Klein coordinates is |w| <= R_LOC
C_ELL = Fr(7361, 10000)               # Lemma "shapeconst": (mean ell - 1) >= C_ELL t^2 for 0 < t <= T_LOC
RHO_HAT = Fr(2938, 10000)             # Lemma "shapeconst": max |rho_j| <= RHO_HAT t^2
S_RING = Fr(1, 10)                     # radius of the ring in the local lemma
N_RING = int(round(160 * NSCALE))      # CR mesh parameter for the ring nodes
N_LAM2 = int(round(16 * NSCALE))       # CR mesh parameter for the lambda_2 nodes
KTOP = 16 * 21                         # ring grid: alpha_0 .. alpha_KTOP = 0.01 * 2^21
KMAX = 16 * 25                         # cover grid: up to 0.01 * 2^25
SHRINK = 1e-7                          # relative shrink of the certified sigma that is stored (room for independent re-check)
LB_SHRINK = Fr(1, 2 ** 60)             # relative shrink of the stored lower bounds

_RHO = [1000000, 1044274, 1090508, 1138789, 1189207, 1241858, 1296840, 1354256,
        1414214, 1476826, 1542211, 1610490, 1681793, 1756252, 1834008, 1915207]      # ~ 2^(j/16) * 10^6
def ALPHA(k):
    """the alpha-grid (exact rationals): alpha_k = (1/100) * 2^(k div 16) * RHO[k mod 16]/10^6 ~ 0.01 * 2^(k/16)"""
    return ALPHA_MIN * 2 ** (k // 16) * Fr(_RHO[k % 16], 10 ** 6)

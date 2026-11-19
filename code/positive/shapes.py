"""Shape space and the equilateral triangle (area 1), in Arb.

Shapes: w = (w1, w2) in the open unit disc (Klein model of the hyperbolic plane),
   G(w) = (I - S(w))/sqrt(1-|w|^2),  S(w) = [[w1, w2],[w2, -w1]],   det G = 1,
   T_w = G(w)^{-1/2}(E) (area 1),  with side lengths  a_j = s_E * ell_j,
   ell_j^2 = tau_j^T G^{-1} tau_j = (1 + w.e_j)/sqrt(1-|w|^2),  e_j = (cos 2 theta_j, sin 2 theta_j),
   theta_j = 0, 2pi/3, 4pi/3  (side directions of E).
In the coordinates of step 1 (T = exp(t S_phi) E):  w = tanh(2t) (cos 2phi, sin 2phi).
<G(w), G(w')> := tr(G(w) G(w')^{-1})/2 = (1 - w.w')/sqrt((1-|w|^2)(1-|w'|^2)) = cosh(2 d(w,w')).
"""
from fractions import Fraction
from flint import arb, fmpz, ctx
from rcr import frac_to_arb

def A_(x):
    return x if isinstance(x, arb) else frac_to_arb(x)

SQ3 = arb(3).sqrt()
H_E = SQ3.sqrt()                 # altitude of E
S_E2 = 4 / SQ3                   # squared side of E
P_E = 6 / H_E                    # perimeter of E
LAM_D = 4 * arb.pi() ** 2 / SQ3  # first Dirichlet eigenvalue of E (area 1) = 4 pi^2/H^2
MU2_N = 4 * arb.pi() ** 2 / (3 * SQ3)   # first nonzero Neumann eigenvalue of E
E_VEC = [(arb(1), arb(0)), (-arb(1) / 2, -SQ3 / 2), (-arb(1) / 2, SQ3 / 2)]

def ell2(w):
    """[ell_j^2] for the shape w (pair of Fractions or arbs)"""
    w1, w2 = A_(w[0]), A_(w[1])
    den = (1 - w1 * w1 - w2 * w2).sqrt()
    return [(1 + w1 * e[0] + w2 * e[1]) / den for e in E_VEC]

def tri_data(w):
    """(a2, A): squared side lengths and area of T_w"""
    return [S_E2 * x for x in ell2(w)], arb(1)

def cosh2d(w, v):
    w1, w2, v1, v2 = A_(w[0]), A_(w[1]), A_(v[0]), A_(v[1])
    return (1 - w1 * v1 - w2 * v2) / ((1 - w1 * w1 - w2 * w2) * (1 - v1 * v1 - v2 * v2)).sqrt()

def ring_data(s, sign, alpha):
    """Triangle and Robin coefficients equivalent to Lambda(G, m) with
       G = diag(1 - 2 sign s, 1 + 2 sign s),  m_j = alpha (1 + sign s cos 2theta_j)   (= p_E + sign*s*Theta_0).
    Lambda(G,m) is the lowest eigenvalue of T' = G^{-1/2} E with Robin coefficient
       beta_j = m_j / (sqrt(det G) |G^{-1/2} tau_j|) on side j;  area(T') = det(G)^{-1/2}."""
    s = A_(s) * sign; alpha = A_(alpha)
    g1, g2 = 1 - 2 * s, 1 + 2 * s; dt = (g1 * g2).sqrt()
    cs = [(arb(1), arb(0)), (arb(1) / 4, arb(3) / 4), (arb(1) / 4, arb(3) / 4)]       # (cos^2, sin^2) of theta_j
    cos2 = [arb(1), -arb(1) / 2, -arb(1) / 2]
    l2 = [c / g1 + s_ / g2 for c, s_ in cs]
    a2 = [S_E2 * x for x in l2]; A = 1 / dt
    beta = [alpha * (1 + s * cos2[j]) / (dt * l2[j].sqrt()) for j in range(3)]
    return a2, A, beta

# ------------------------------------------------------------------ equilateral ground state
def _F(k, al):
    return k * H_E / 2 - (al / k).atan() - (2 * al / k).atan()

def kappa(alpha):
    """enclosure of the root kappa in (0, 2pi/H) of  kappa H/2 = atan(alpha/kappa) + atan(2 alpha/kappa);
    alpha: exact Fraction/arb point (radius 0 expected; for a ball the monotone dependence is used by callers)."""
    al = A_(alpha)
    k = arb(float(min(2 * 3.14159 / 1.3161, (6 / 1.3161 * float(al.mid())) ** 0.5)))
    for _ in range(80):
        km = arb(k.mid())
        Fk = _F(km, al)
        dF = H_E / 2 + al / (km * km + al * al) + 2 * al / (km * km + 4 * al * al)
        k = arb((km - Fk / dF).mid())
    eps = arb(2) ** (-(ctx.prec - 24))
    lo, hi = arb(k.mid()) * (1 - eps), arb(k.mid()) * (1 + eps)
    if not (bool(_F(lo, al) < 0) and bool(_F(hi, al) > 0)):
        raise ValueError('kappa enclosure failed')
    return lo.union(hi)

def lam0(alpha):
    k = kappa(alpha); return k * k

def lam0_d(alpha):
    """(lam0, lam0') at the exact point alpha; lam0' = d lam0/d alpha = int_{bd E} psi0^2 / int_E psi0^2"""
    al = A_(alpha); k = kappa(al)
    Fa = -(k / (k * k + al * al) + 2 * k / (k * k + 4 * al * al))
    Fk = H_E / 2 + al / (k * k + al * al) + 2 * al / (k * k + 4 * al * al)
    return k * k, 2 * k * (-Fa / Fk)

"""Parsing of certificate numbers, the alpha grid, and triangles (mpmath interval arithmetic).

Everything here is derived from the paper (Section "Shapes, concavity and cells",
Lemma "ring") and certs/FORMAT.txt; nothing is imported from the original code.
"""
from fractions import Fraction as Fr
import numpy as np
from mpmath import iv, mpf, mp

PREC = 200
iv.prec = PREC
mp.prec = PREC

RHO = [1000000, 1044274, 1090508, 1138789, 1189207, 1241858, 1296840, 1354256, 1414214,
       1476826, 1542211, 1610490, 1681793, 1756252, 1834008, 1915207]


def alpha_grid(k):
    """alpha_k = (1/100) * 2^(k div 16) * RHO[k mod 16]/10^6, exact."""
    return Fr(1, 100) * (2 ** (k // 16)) * Fr(RHO[k % 16], 10 ** 6)


def parse_rat(s):
    """'p/q' or 'm*2^e' or decimal string -> exact Fraction."""
    s = s.strip()
    if "*2^" in s:
        m, e = s.split("*2^")
        m = int(m)
        e = int(e)
        return Fr(m) * (Fr(2) ** e)
    return Fr(s)


def parse_w(s):
    a, b = s.split(",")
    return (Fr(a), Fr(b))


def ivq(q):
    """interval enclosure of a Fraction"""
    q = Fr(q)
    return iv.mpf(q.numerator) / iv.mpf(q.denominator)


def _raw_to_fr(raw):
    sign, man, exp, bc = raw
    if man == 0 and exp != 0:
        raise ValueError("non-finite mpf")
    return Fr((-1) ** sign * int(man)) * (Fr(2) ** int(exp))


def iv_lo_fr(x):
    """exact Fraction equal to the lower end point of an mpmath interval"""
    return _raw_to_fr(x._mpi_[0])


def iv_hi_fr(x):
    return _raw_to_fr(x._mpi_[1])


def iv_to_float_bounds(x):
    """floats (lo, hi) with lo <= x.a and x.b <= hi, verified by exact comparison."""
    a = iv_lo_fr(x)
    b = iv_hi_fr(x)
    lo = float(a)
    while Fr(lo) > a:
        lo = float(np.nextafter(lo, -np.inf))
    hi = float(b)
    while Fr(hi) < b:
        hi = float(np.nextafter(hi, np.inf))
    return lo, hi


SQRT3 = iv.sqrt(iv.mpf(3))
SE2 = 4 / SQRT3                      # s_E^2 = 4/sqrt(3)  (side of E of area 1, squared)
H_E = iv.sqrt(SQRT3)                 # H = 3^(1/4)
LAMBDA1_E = 4 * iv.pi ** 2 / SQRT3   # first Dirichlet eigenvalue of E = 4 pi^2 / H^2
# e_j = (cos 2theta_j, sin 2theta_j), theta_j = 0, 2pi/3, 4pi/3
E_VEC = [(iv.mpf(1), iv.mpf(0)), (-iv.mpf(1) / 2, -SQRT3 / 2), (-iv.mpf(1) / 2, SQRT3 / 2)]
# cos^2(theta_j), sin^2(theta_j), cos(2 theta_j)  (exact rationals)
COS2 = [Fr(1), Fr(1, 4), Fr(1, 4)]
SIN2 = [Fr(0), Fr(3, 4), Fr(3, 4)]
COS2TH = [Fr(1), Fr(-1, 2), Fr(-1, 2)]


class Triangle:
    """squared sides a2[j], Robin coefficients beta[j] on side j, stated area (all mpmath intervals)"""

    def __init__(self, a2, beta, area_stated, desc):
        self.a2 = a2
        self.beta = beta
        self.area_stated = area_stated
        self.desc = desc

    def heron_16A2(self):
        a, b, c = self.a2
        return 4 * a * b - (a + b - c) ** 2

    def area_consistent(self):
        """the interval for 16|T|^2 from Heron's formula overlaps the stated one and is tiny"""
        x = self.heron_16A2()
        y = 16 * self.area_stated ** 2
        overlap = not (x.b < y.a or y.b < x.a)
        return overlap and (x.b - x.a) < mpf(2) ** (-150) and (y.b - y.a) < mpf(2) ** (-150)


def tri_cover(w, alpha):
    """T_w of the paper: a_j^2 = s_E^2 (1 + w.e_j)/sqrt(1-|w|^2), area 1, beta_j = alpha."""
    w1, w2 = ivq(w[0]), ivq(w[1])
    r2 = w[0] * w[0] + w[1] * w[1]
    if not r2 < 1:
        raise ValueError("w not in the unit disc")
    den = iv.sqrt(1 - ivq(r2))
    a2 = [SE2 * (1 + w1 * e[0] + w2 * e[1]) / den for e in E_VEC]
    al = ivq(alpha)
    return Triangle(a2, [al, al, al], iv.mpf(1), "cover")


def tri_ring(sign, s, alpha):
    """Lemma 'ring' / FORMAT.txt: G = diag(g1, g2), g1 = 1 - 2 sign s, g2 = 1 + 2 sign s."""
    assert sign in (1, -1)
    g1 = 1 - 2 * sign * Fr(s)
    g2 = 1 + 2 * sign * Fr(s)
    assert g1 > 0 and g2 > 0
    l2 = [COS2[j] / g1 + SIN2[j] / g2 for j in range(3)]          # exact rationals
    a2 = [SE2 * ivq(l2[j]) for j in range(3)]
    sg = iv.sqrt(ivq(g1 * g2))
    beta = [ivq(alpha) * ivq(1 + sign * Fr(s) * COS2TH[j]) / (sg * iv.sqrt(ivq(l2[j]))) for j in range(3)]
    return Triangle(a2, beta, 1 / sg, "ring")


def tri_equilateral(alpha):
    al = ivq(alpha)
    return Triangle([SE2, SE2, SE2], [al, al, al], iv.mpf(1), "E")


def certified_argmax(vals):
    """index of the certified largest interval, or None if not decidable"""
    for j in range(len(vals)):
        if all(k == j or vals[j].a > vals[k].b for k in range(len(vals))):
            return j
    return None


def place(tri, base):
    """Vertex coordinates from the side lengths.  Side p=base is P0P1 on the x axis,
    side q = P0P2, side r = P1P2 (q, r the other two sides in cyclic order).
    Returns (p,q,r), a_p, x2, y2 as mpmath intervals."""
    p = base
    q = (base + 1) % 3
    r = (base + 2) % 3
    ap = iv.sqrt(tri.a2[p])
    x2 = (tri.a2[p] + tri.a2[q] - tri.a2[r]) / (2 * ap)
    y2sq = tri.a2[q] - x2 ** 2
    if not y2sq.a > 0:
        raise ValueError("degenerate triangle")
    y2 = iv.sqrt(y2sq)
    return (p, q, r), ap, x2, y2

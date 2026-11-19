"""Rigorous enclosures of lambda_0(alpha) = lambda_1(E, alpha) and lambda_0'(alpha)
(Proposition 'equi', Lemma 'lam0'), in mpmath interval arithmetic, without arctan.

kappa(alpha) is the unique root in (0, 2 pi/H) of
     g(kappa) = kappa H/2 - arctan(alpha/kappa) - arctan(2 alpha/kappa),  g increasing.
With d1 = arctan(alpha/kappa), d2 = arctan(2alpha/kappa) in (0, pi/2) and kappa H/2 in (0, pi),
g lies in (-pi, pi), so sign g = sign sin g, and
     sin g * sqrt(k^2+a^2) sqrt(k^2+4a^2) = sin(kH/2) (k^2 - 2 a^2) - 3 a k cos(kH/2) =: S(k).
A bracket [k_lo, k_hi] inside (0, 2pi/H) with S(k_lo) < 0 < S(k_hi) therefore encloses kappa(alpha).
"""
from fractions import Fraction as Fr
from mpmath import iv, mp, mpf
import mpmath

import geom

_cache = {}


def _S(k, a, H):
    return iv.sin(k * H / 2) * (k * k - 2 * a * a) - 3 * a * k * iv.cos(k * H / 2)


def kappa_enclosure(alpha, scale=1):
    """alpha exact Fraction.  Returns an mpmath interval containing kappa(alpha)."""
    H = geom.H_E
    a = geom.ivq(alpha)
    # float/mp proposal
    mp.prec = geom.PREC
    Hm = mpf(3) ** (mpf(1) / 4)
    am = mpf(alpha.numerator) / mpf(alpha.denominator)
    g = lambda k: k * Hm / 2 - mpmath.atan(am / k) - mpmath.atan(2 * am / k)
    lo, hi = mpf(0), 2 * mpmath.pi / Hm
    for _ in range(geom.PREC + 20):          # bisection proposal (not trusted)
        mid = (lo + hi) / 2
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    k0 = (lo + hi) / 2
    eps = mpf(2) ** (-(geom.PREC - 40))
    klo = iv.mpf(k0 * (1 - eps))
    khi = iv.mpf(k0 * (1 + eps))
    ok = (klo.a > 0) and (khi * H < 2 * iv.pi) and (_S(klo, a, H).b < 0) and (_S(khi, a, H).a > 0)
    if not ok:
        raise ArithmeticError("kappa bracket not certified")
    return iv.mpf([klo.a, khi.b])


def lam0(alpha):
    """(lambda_0(alpha), lambda_0'(alpha)) as mpmath intervals; alpha exact Fraction > 0"""
    alpha = Fr(alpha)
    if alpha in _cache:
        return _cache[alpha]
    H = geom.H_E
    a = geom.ivq(alpha)
    k = kappa_enclosure(alpha)
    l0 = k * k
    num = k / (k * k + a * a) + 2 * k / (k * k + 4 * a * a)
    den = H / 2 + a / (k * k + a * a) + 2 * a / (k * k + 4 * a * a)
    l0p = 2 * k * num / den
    _cache[alpha] = (l0, l0p)
    return l0, l0p

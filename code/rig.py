"""Rigorous helpers (Arb ball arithmetic via python-flint).
 - arb interval constructors
 - Jet2: second-order forward-mode automatic differentiation in 2 variables with arb entries,
   used for interval Hessians over boxes."""
from flint import arb, ctx, fmpq

ctx.prec = 128

def I(lo, hi):
    """arb ball containing [lo, hi] (lo, hi exact: int/fmpq/str/arb)"""
    lo = arb(lo); hi = arb(hi)
    return lo.union(hi)

def Q(x):
    """exact rational from a python float, int, fmpq or decimal/fraction string"""
    from fractions import Fraction
    if isinstance(x, fmpq):
        return x
    f = Fraction(x) if not isinstance(x, float) else Fraction(*x.as_integer_ratio())
    return fmpq(f.numerator, f.denominator)

def lower(a):
    return a.lower()

def upper(a):
    return a.upper()

def pos(a):
    """certified a > 0"""
    return a > 0

class Jet2:
    """f, grad f (2), Hessian (2x2 symmetric: hxx, hxy, hyy)"""
    __slots__ = ('v', 'gx', 'gy', 'hxx', 'hxy', 'hyy')
    def __init__(self, v, gx=0, gy=0, hxx=0, hxy=0, hyy=0):
        self.v = arb(v); self.gx = arb(gx); self.gy = arb(gy)
        self.hxx = arb(hxx); self.hxy = arb(hxy); self.hyy = arb(hyy)
    @staticmethod
    def var(val, which):
        return Jet2(val, 1, 0) if which == 0 else Jet2(val, 0, 1)
    def _c(o):
        return o if isinstance(o, Jet2) else Jet2(o)
    def __add__(s, o):
        o = Jet2._c(o)
        return Jet2(s.v + o.v, s.gx + o.gx, s.gy + o.gy, s.hxx + o.hxx, s.hxy + o.hxy, s.hyy + o.hyy)
    __radd__ = __add__
    def __neg__(s):
        return Jet2(-s.v, -s.gx, -s.gy, -s.hxx, -s.hxy, -s.hyy)
    def __sub__(s, o):
        return s + (-Jet2._c(o))
    def __rsub__(s, o):
        return Jet2._c(o) - s
    def __mul__(s, o):
        o = Jet2._c(o)
        return Jet2(s.v * o.v,
                    s.gx * o.v + s.v * o.gx,
                    s.gy * o.v + s.v * o.gy,
                    s.hxx * o.v + 2 * s.gx * o.gx + s.v * o.hxx,
                    s.hxy * o.v + s.gx * o.gy + s.gy * o.gx + s.v * o.hxy,
                    s.hyy * o.v + 2 * s.gy * o.gy + s.v * o.hyy)
    __rmul__ = __mul__
    def apply(s, f0, f1, f2):
        """compose with scalar function phi: values phi(v), phi'(v), phi''(v)"""
        return Jet2(f0, f1 * s.gx, f1 * s.gy,
                    f2 * s.gx * s.gx + f1 * s.hxx,
                    f2 * s.gx * s.gy + f1 * s.hxy,
                    f2 * s.gy * s.gy + f1 * s.hyy)
    def inv(s):
        r = 1 / s.v
        return s.apply(r, -r * r, 2 * r * r * r)
    def __truediv__(s, o):
        return s * Jet2._c(o).inv()
    def __rtruediv__(s, o):
        return Jet2._c(o) * s.inv()
    def sqrt(s):
        r = s.v.sqrt()
        return s.apply(r, 1 / (2 * r), -1 / (4 * r * s.v))

def sqr(x):
    """x^2 for an arb ball, tight when the ball contains 0 (plain x*x would give a negative lower end)"""
    lo = arb(x.lower()); hi = arb(x.upper())
    a = lo * lo; b = hi * hi
    r = a.union(b)
    if lo < 0 and hi > 0:
        r = r.union(arb(0))
    elif not (lo > 0 or hi < 0):
        r = r.union(arb(0))
    return r

def _jet2_sq(s):
    return Jet2(sqr(s.v), 2 * s.v * s.gx, 2 * s.v * s.gy,
                2 * s.gx * s.gx + 2 * s.v * s.hxx,
                2 * s.gx * s.gy + 2 * s.v * s.hxy,
                2 * s.gy * s.gy + 2 * s.v * s.hyy)
Jet2.sq = _jet2_sq

def square(t):
    if isinstance(t, Jet2):
        return t.sq()
    if isinstance(t, arb):
        return sqr(t)
    return t * t

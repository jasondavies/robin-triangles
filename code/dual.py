"""Dual numbers (value, derivative) over arb, for mean-value forms in the parameter M.
All operations are rigorous: if x ranges over a ball X and v(x), d(x) are enclosed by the
arb parts, results enclose value and derivative of the composed function."""
from flint import arb

class Dual:
    __slots__ = ('v', 'd')
    def __init__(self, v, d=0):
        self.v = v if isinstance(v, arb) else arb(v)
        self.d = d if isinstance(d, arb) else arb(d)
    @staticmethod
    def c(o):
        return o if isinstance(o, Dual) else Dual(o, 0)
    def __add__(s, o):
        o = Dual.c(o); return Dual(s.v + o.v, s.d + o.d)
    __radd__ = __add__
    def __neg__(s):
        return Dual(-s.v, -s.d)
    def __sub__(s, o):
        o = Dual.c(o); return Dual(s.v - o.v, s.d - o.d)
    def __rsub__(s, o):
        return Dual.c(o) - s
    def __mul__(s, o):
        if not isinstance(o, Dual):
            o = arb(o); return Dual(s.v * o, s.d * o)
        return Dual(s.v * o.v, s.d * o.v + s.v * o.d)
    __rmul__ = __mul__
    def __truediv__(s, o):
        if not isinstance(o, Dual):
            o = arb(o); return Dual(s.v / o, s.d / o)
        return Dual(s.v / o.v, (s.d * o.v - s.v * o.d) / (o.v * o.v))
    def __rtruediv__(s, o):
        return Dual.c(o) / s
    def exp(s):
        e = s.v.exp(); return Dual(e, e * s.d)
    def sqrt(s):
        r = s.v.sqrt(); return Dual(r, s.d / (2 * r))
    def tanh(s):
        t = s.v.tanh(); return Dual(t, (1 - t * t) * s.d)
    def atanh(s):
        return Dual(s.v.atanh(), s.d / (1 - s.v * s.v))
    def __abs__(s):
        # only used for branch decisions on the value
        return abs(s.v)
    def mid(s):
        return s.v.mid()
    def __lt__(s, o): return s.v < (o.v if isinstance(o, Dual) else o)
    def __gt__(s, o): return s.v > (o.v if isinstance(o, Dual) else o)
    def __eq__(s, o):
        return False
    def __repr__(s):
        return f"Dual({s.v}, {s.d})"

class DUALNUM:
    """namespace for sixwave with Dual numbers"""
    @staticmethod
    def root4(x): return Dual(arb(x).root(4))
    @staticmethod
    def sqrt(x): return x.sqrt() if isinstance(x, Dual) else Dual(arb(x).sqrt())
    @staticmethod
    def one(): return Dual(1)
    @staticmethod
    def zero(): return Dual(0)
    @staticmethod
    def tanh(x): return x.tanh()
    @staticmethod
    def atanh(x): return x.atanh()
    @staticmethod
    def exp(x): return x.exp()

"""Vectorised interval arithmetic on numpy float64 arrays.

Rigour: every IEEE-754 binary64 operation (+, -, *, /, sqrt) performed by numpy is
correctly rounded to nearest, hence the exact result lies in [pred(r), succ(r)] where
r is the computed result (also when r is subnormal or zero).  We step one float
outwards after each operation, except when the operation is trivially exact
(an operand is exactly zero), which is needed so that exact zeros (right angles
built along coordinate axes) stay exact.
No BLAS, no fused operations, no summation routines are used here.
"""
import numpy as np

_NINF = -np.inf
_PINF = np.inf


def _dn(r, exact):
    return np.where(exact, r, np.nextafter(r, _NINF))


def _up(r, exact):
    return np.where(exact, r, np.nextafter(r, _PINF))


class IV:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = self.lo if hi is None else np.asarray(hi, dtype=np.float64)

    # ---- helpers
    @staticmethod
    def lift(a):
        return a if isinstance(a, IV) else IV(a)

    def __getitem__(self, idx):
        return IV(self.lo[idx], self.hi[idx])

    @property
    def shape(self):
        return np.broadcast(self.lo, self.hi).shape

    def mid(self):
        return 0.5 * (self.lo + self.hi)

    def ok(self):
        return bool(np.all(self.lo <= self.hi) and np.all(np.isfinite(self.lo)) and np.all(np.isfinite(self.hi)))

    # ---- arithmetic
    def __neg__(self):
        return IV(-self.hi, -self.lo)

    def __add__(self, o):
        o = IV.lift(o)
        lo = self.lo + o.lo
        hi = self.hi + o.hi
        return IV(_dn(lo, (self.lo == 0) | (o.lo == 0)), _up(hi, (self.hi == 0) | (o.hi == 0)))

    __radd__ = __add__

    def __sub__(self, o):
        o = IV.lift(o)
        lo = self.lo - o.hi
        hi = self.hi - o.lo
        return IV(_dn(lo, (self.lo == 0) | (o.hi == 0)), _up(hi, (self.hi == 0) | (o.lo == 0)))

    def __rsub__(self, o):
        return IV.lift(o) - self

    def __mul__(self, o):
        o = IV.lift(o)
        los = []
        his = []
        for a in (self.lo, self.hi):
            for b in (o.lo, o.hi):
                p = a * b
                ex = (a == 0) | (b == 0)
                los.append(_dn(p, ex))
                his.append(_up(p, ex))
        lo = np.minimum(np.minimum(los[0], los[1]), np.minimum(los[2], los[3]))
        hi = np.maximum(np.maximum(his[0], his[1]), np.maximum(his[2], his[3]))
        return IV(lo, hi)

    __rmul__ = __mul__

    def div_pos(self, o):
        """self / o, where o is certified positive (o.lo > 0 is checked)."""
        o = IV.lift(o)
        if not np.all(o.lo > 0):
            raise ArithmeticError("division by interval not certified positive")
        los = []
        his = []
        for a in (self.lo, self.hi):
            for b in (o.lo, o.hi):
                p = a / b
                ex = (a == 0)
                los.append(_dn(p, ex))
                his.append(_up(p, ex))
        lo = np.minimum(np.minimum(los[0], los[1]), np.minimum(los[2], los[3]))
        hi = np.maximum(np.maximum(his[0], his[1]), np.maximum(his[2], his[3]))
        return IV(lo, hi)

    def sqrt(self):
        if not np.all(self.lo >= 0):
            raise ArithmeticError("sqrt of interval not certified nonnegative")
        lo = np.sqrt(self.lo)
        hi = np.sqrt(self.hi)
        lo = np.maximum(_dn(lo, self.lo == 0), 0.0)
        return IV(lo, _up(hi, self.hi == 0))


def where(cond, a, b):
    a = IV.lift(a)
    b = IV.lift(b)
    return IV(np.where(cond, a.lo, b.lo), np.where(cond, a.hi, b.hi))


def selftest(seed=1, n=4000):
    """Randomised containment test against exact rational arithmetic."""
    from fractions import Fraction as Fr
    rng = np.random.default_rng(seed)

    def rnd(m):
        v = rng.standard_normal(m) * 10.0 ** rng.integers(-8, 8, m)
        v[rng.random(m) < 0.1] = 0.0
        return v

    a0, a1, b0, b1 = rnd(n), rnd(n), rnd(n), rnd(n)
    A = IV(np.minimum(a0, a1), np.maximum(a0, a1))
    B = IV(np.minimum(b0, b1), np.maximum(b0, b1))
    Bp = IV(np.abs(B.lo) + 1e-300, np.abs(B.lo) + np.abs(B.hi) + 1e-300)
    res = {"add": A + B, "sub": A - B, "mul": A * B, "div": A.div_pos(Bp), "sqrt": Bp.sqrt()}
    for i in range(n):
        for ta in (0.0, 0.3, 1.0):
            for tb in (0.0, 0.7, 1.0):
                xa = Fr(float(A.lo[i])) + (Fr(float(A.hi[i])) - Fr(float(A.lo[i]))) * Fr(ta)
                xb = Fr(float(B.lo[i])) + (Fr(float(B.hi[i])) - Fr(float(B.lo[i]))) * Fr(tb)
                xp = Fr(float(Bp.lo[i])) + (Fr(float(Bp.hi[i])) - Fr(float(Bp.lo[i]))) * Fr(tb)
                for name, val in (("add", xa + xb), ("sub", xa - xb), ("mul", xa * xb), ("div", xa / xp)):
                    r = res[name]
                    if not (Fr(float(r.lo[i])) <= val <= Fr(float(r.hi[i]))):
                        raise AssertionError("IV selftest failed: " + name)
                r = res["sqrt"]
                lo, hi = Fr(float(r.lo[i])), Fr(float(r.hi[i]))
                if not (lo >= 0 and lo * lo <= xp <= hi * hi):
                    raise AssertionError("IV selftest failed: sqrt")
    # exact-zero propagation
    z = IV(np.zeros(3)) * IV(np.array([1.0, -2.0, 3.0]), np.array([1.5, -1.0, 4.0])) + IV(np.zeros(3))
    if not (np.all(z.lo == 0) and np.all(z.hi == 0)):
        raise AssertionError("IV selftest failed: zero propagation")
    return True


if __name__ == "__main__":
    print("selftest", selftest())

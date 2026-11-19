"""Rigorous (Arb) integrals of exponentials of linear functions over segments and triangles.
  seg:  int_{[A,B]} e^{v.x} ds = |B-A| * dd1(v.B, v.A)
  tri:  int_T e^{v.x} dx = 2|T| * dd2(v.P0, v.P1, v.P2)
dd1, dd2 = first/second divided differences of exp, computed by power series when the nodes are
close (with explicit tail bound) and by difference quotients otherwise."""
import rig  # sets working precision
from flint import arb, fmpq
from dual import Dual

NSER = 40
_FACT = [arb(1)]
for _n in range(1, 2 * NSER + 5):
    _FACT.append(_FACT[-1] * _n)

def _tail(m, N, shift):
    """bound for sum_{n>N} (n+1) m^n / (n+shift)!   with 0<=m<=1/2"""
    # (n+1) m^n/(n+shift)! <= (N+2) m^(N+1)/(N+1+shift)! * 2  (ratio <= m*(n+2)/(n+1)/(n+1+shift) < 1/2)
    mp_ = arb(1)
    for _ in range(N + 1):
        mp_ = mp_ * m
    return 2 * (N + 2) * mp_ / _FACT[N + 1 + shift]

def _pow(m, k):
    r = arb(1)
    for _ in range(k):
        r = r * m
    return r

def _add_tail(s, vt, dt):
    """add [-vt,vt] to value and [-dt,dt] to derivative (if Dual)"""
    if isinstance(s, Dual):
        return Dual(s.v + arb(0, vt.upper()), s.d + arb(0, dt.upper()))
    return s + arb(0, vt.upper())

def _dabs(x):
    return abs(x.d).upper() if isinstance(x, Dual) else arb(0)

def dd1(x, y):
    d = x - y
    m = arb(abs(d).upper())
    if m <= arb(1) / 2:
        s = 0 * d; p = 1 + 0 * d
        for n in range(NSER + 1):
            s = s + p / _FACT[n + 1]; p = p * d
        vt = _tail(m, NSER, 1)
        dt = arb(_dabs(d)) * 2 * (NSER + 1) * _pow(m, NSER) / _FACT[NSER + 2]
        s = _add_tail(s, vt, dt)
        return y.exp() * s
    return (x.exp() - y.exp()) / d

def dd2(a, b, c):
    """second divided difference of exp at nodes a,b,c (balls)"""
    nodes = [a, b, c]
    # pick base node = the one with smallest midpoint
    nodes.sort(key=lambda z: float(z.mid()))  # any order is valid; this one keeps u,v small
    lo, mi, hi = nodes
    u = mi - lo; v = hi - lo
    m = arb(abs(u).upper()).max(arb(abs(v).upper()))
    if m <= arb(1) / 2:
        s = 0 * u
        # h_n(u,v) = sum_{i+j=n} u^i v^j
        upow = [1 + 0 * u]; vpow = [1 + 0 * v]
        for n in range(1, NSER + 1):
            upow.append(upow[-1] * u); vpow.append(vpow[-1] * v)
        for n in range(NSER + 1):
            h = 0 * u
            for i in range(n + 1):
                h = h + upow[i] * vpow[n - i]
            s = s + h / _FACT[n + 2]
        vt = _tail(m, NSER, 2)
        dt = (arb(_dabs(u)) + arb(_dabs(v))) * (NSER + 1) * (NSER + 2) * _pow(m, NSER) / _FACT[NSER + 3]
        s = _add_tail(s, vt, dt)
        return lo.exp() * s
    return (dd1(hi, mi) - dd1(mi, lo)) / (hi - lo)

def dot(v, P):
    return v[0] * P[0] + v[1] * P[1]

def seg_exp(A, B, v):
    dx = B[0] - A[0]; dy = B[1] - A[1]
    L = (dx * dx + dy * dy).sqrt()
    return L * dd1(dot(v, B), dot(v, A))

def tri_exp(V, v):
    S = abs((V[1][0] - V[0][0]) * (V[2][1] - V[0][1]) - (V[1][1] - V[0][1]) * (V[2][0] - V[0][0])) / 2
    return 2 * S * dd2(dot(v, V[0]), dot(v, V[1]), dot(v, V[2]))

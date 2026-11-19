"""Triangle geometry in apex coordinates: base (0,0)-(1,0), apex (x,y), rescaled to area 1.
Works with arb balls or Jet2 jets."""
from flint import arb
from equi import PSTAR, SQRT3

def _sqrt(u):
    return u.sqrt()

def sides(x, y):
    d1 = _sqrt(x * x + y * y)
    d2 = _sqrt((1 - x) * (1 - x) + y * y)
    return d1, d2

def p_tau(x, y):
    """p = P/P*, tau = (a^2+b^2+c^2)/(2 sqrt3) for the area-1 rescaling."""
    d1, d2 = sides(x, y)
    s = _sqrt(2 / y)
    p = (1 + d1 + d2) * s / PSTAR
    tau = (1 + x * x + y * y + (1 - x) * (1 - x) + y * y) / (SQRT3 * y)
    return p, tau

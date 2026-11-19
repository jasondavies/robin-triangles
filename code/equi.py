"""Rigorous (Arb) data for the equilateral triangle E of area 1 with Robin parameter alpha=-beta<0.

Ground state (see paper, Prop. 2.1):  psi0 = sum_{i=1}^3 cosh(L + 2K d_i/H),  d_i = distance to side i,
H = 3^{1/4} (height of E), M>0 free parameter, t = tanh M, L = -artanh(t/2), K = M - L,
beta = t K / H,  lambda0 = -4 K^2 / H^2.
Monotonicity in M: K, t, beta increasing; lambda0 decreasing; K_M = dK/dM decreasing."""
from flint import arb, ctx
from rig import I

H = arb(3).root(4)
SQRT3 = arb(3).sqrt()
PSTAR = 6 / H                 # perimeter of E (area 1) = sqrt(12 sqrt3)

def K_of(M):
    return M + (M.tanh() / 2).atanh()

def KM_of(M):
    t = M.tanh()
    return 1 + 2 * (1 - t * t) / (4 - t * t)

def beta_of(M):
    return M.tanh() * K_of(M) / H

def lam0_of(M):
    K = K_of(M)
    return -4 * K * K / (H * H)

def dlam_dalpha_of(M):
    """B/N = dlambda0/dalpha  (Hellmann-Feynman / supergradient)"""
    t = M.tanh(); K = K_of(M); KM = KM_of(M)
    D = (1 - t * t) * K + t * KM
    return 8 * K * KM / (H * D)

def EN_of(M):
    """E/N = ||grad psi0||^2/||psi0||^2 = lambda0 - alpha dlambda0/dalpha"""
    return lam0_of(M) + beta_of(M) * dlam_dalpha_of(M)

def g_of(M):
    """g = (E/N)/(2 beta B/N) = 1/4 - K/(2 sinh(2M) K_M)"""
    return arb(1) / 4 - K_of(M) / (2 * (2 * M).sinh() * KM_of(M))

def M_encl(beta, iters=200):
    """rigorous enclosure [Mlo, Mhi] (exact arb endpoints) of the unique M with beta_of(M)=beta"""
    beta = arb(beta)
    lo, hi = arb(0), arb(1)
    while not (beta_of(hi) > beta):
        hi = hi * 2
    for _ in range(iters):
        mid = (lo + hi) / 2
        mid = arb(mid.mid())
        b = beta_of(mid)
        if b < beta:
            lo = mid
        elif b > beta:
            hi = mid
        else:
            break
        if (hi - lo) < arb(2) ** (-ctx.prec + 8):
            break
    return lo, hi

def lam0_encl(beta):
    """enclosure of lambda0(beta) (ball)"""
    lo, hi = M_encl(beta)
    return lam0_of(hi).union(lam0_of(lo)) if True else None

def lam0_lower(beta):
    """certified lower bound of lambda0(beta) (lambda0 decreasing in M)"""
    lo, hi = M_encl(beta)
    return lam0_of(hi).lower()

if __name__ == '__main__':
    for b in ['0.5', '1', '2', '5']:
        lo, hi = M_encl(arb(b))
        print(b, lo.mid(), hi.mid(), lam0_of(I(lo, hi)), EN_of(I(lo, hi)), dlam_dalpha_of(I(lo, hi)), g_of(I(lo, hi)))

"""Certified one-variable inequality behind Theorem C (three-function space, all M >= M_C).

Notation (paper/large-beta.tex): E equilateral, area 1, height H = 3^{1/4}; M > 0, t = tanh M,
L = -artanh(t/2), K = M - L, U = 2M - L, beta = tK/H, lambda0 = -4K^2/H^2 = -4 beta^2/t^2.
phi_i = cosh(L + 2K b_i), b_i barycentric coordinates;  psi0 = phi_0 + phi_1 + phi_2.

Closed forms (y = 2K b):
  I1  = int_0^{2K} (2K-y) cosh^2(L+y) dy = K^2 - (K/2) sinh 2L + (cosh 2U - cosh 2L)/8
  I1s = int_0^{2K} (2K-y) sinh^2(L+y) dy = -K^2 - (K/2) sinh 2L + (cosh 2U - cosh 2L)/8
  I2  = iint_{y0+y1<=2K} cosh(L+y0) cosh(L+y1) = K sinh 2M - sinh L sinh U + sinh^2 L
  I2s = iint sinh(L+y0) sinh(L+y1)             = K sinh 2M - cosh L cosh U + cosh^2 L
  J2  = int_0^{2K} cosh^2(L+y) dy = K + (sinh 2U - sinh 2L)/4
  cJ3 = cosh L int_0^{2K} cosh(L+y) dy = cosh L sinh U - cosh L sinh L
  J4  = int_0^{2K} cosh(L+y) cosh(U-y) dy = K cosh 2M + sinh(2K)/2
and, with the common factor c = 2/H^2:
  D3      = Gamma_d - Gamma_o             = c (I1s - I2s)
  G1      = Gamma_d + Gamma_o/2           = c (I1s + I2s/2)
  |lam0| N1 = |lam0| (N_d - N_o)          = c (I1 - I2)
  beta B1 = beta (s1 + 2 s2 - 2 s3 - s4)  = c (t/2) (2K cosh^2 L + 2 J2 - 2 cJ3 - J4)
  beta kq = beta (s2 + s4 - s1 - s3)      = c (t/2) (J2 + J4 - cJ3 - 2K cosh^2 L)
All quantities are evaluated after division by c X, X = e^{2U}/16, as polynomials in
  s = e^{-2M},  u = K s,  t,  omega = e^{2L} = (2-t)/(2+t),
so that M = infinity (s = 0) is included.  `num` is a namespace with the arithmetic (arb or float).

The inequality certified for all M >= M_C:
  (I2)  D3 > 0,  kq >= 0
  (I3)  Lam_hi := mu_hi G1 - rho_lo beta B1 + |lam0| N1  <  D3 (rho_lo + beta kq/(2 D3))^2 / rho_hi^2
where rho_lo, rho_hi, mu_hi bound the side ratios and the top eigenvalue of the transplantation matrix on
the near region  {sin(theta_min/2) >= t_lo/2}  (t_lo = lower bound of t on the M-interval)."""
import sys, json
from out import write, enc

def normalised(s, u, t):
    """(I1, I1s, I2, I2s, J2, cJ3, J4, Kc2, K) / X   with X = e^{2U}/16;  Kc2 = K cosh^2 L"""
    om = (2 - t) / (2 + t)
    q = 4 - t * t
    sh2L = -4 * t / q; ch2L = (4 + t * t) / q
    c2 = 4 / q; s2 = t * t / q; cs = -2 * t / q          # cosh^2 L, sinh^2 L, cosh L sinh L
    ss = s * s
    os2 = om * om * ss                                   # omega^2 s^2
    nK2 = 16 * om * u * u                                # K^2
    nK = 16 * om * s * u                                 # K
    nch2U = 8 * (1 + os2 * ss); nsh2U = 8 * (1 - os2 * ss)
    nKsh2M = 8 * om * u * (1 - ss); nKch2M = 8 * om * u * (1 + ss)
    ncLcU = 4 * s * (om * (1 + ss) + (1 + os2))          # cosh L cosh U
    nsLsU = 4 * s * (om * (1 + ss) - (1 + os2))          # sinh L sinh U
    ncLsU = 4 * s * (om * (1 - ss) + (1 - os2))          # cosh L sinh U
    nsh2K = 8 * s * (1 - os2)                            # sinh 2K
    f = 16 * om * ss                                     # 1/X
    I1 = nK2 - nK * sh2L / 2 + (nch2U - f * ch2L) / 8
    I1s = -nK2 - nK * sh2L / 2 + (nch2U - f * ch2L) / 8
    I2 = nKsh2M - nsLsU + f * s2
    I2s = nKsh2M - ncLcU + f * c2
    J2 = nK + (nsh2U - f * sh2L) / 4
    cJ3 = ncLsU - f * cs
    J4 = nKch2M + nsh2K / 2
    Kc2 = nK * c2
    return I1, I1s, I2, I2s, J2, cJ3, J4, Kc2

def quantities(s, u, t):
    """D3, G1, |lam0|N1, beta B1, beta kq  -- all divided by (2/H^2) X"""
    I1, I1s, I2, I2s, J2, cJ3, J4, Kc2 = normalised(s, u, t)
    D3 = I1s - I2s
    G1 = I1s + I2s / 2
    lN1 = I1 - I2
    bB1 = t / 2 * (2 * Kc2 + 2 * J2 - 2 * cJ3 - J4)
    bkq = t / 2 * (J2 + J4 - cJ3 - 2 * Kc2)
    return D3, G1, lN1, bB1, bkq

def geom(tlo, sqrt):
    """bounds on the near region {theta_i >= theta* = 2 arcsin(tlo/2)}:
       rho_i^2 >= sqrt3 tan(theta*/2), rho_i^2 <= sqrt3 cot(theta*), tr(A) <= (3c^2+1)/(sqrt3 c), c = cot theta*.
       Returns rho_lo^2, rho_hi^2, tau_hi, c  (valid when c <= 1, i.e. theta* >= pi/4)."""
    r = sqrt(4 - tlo * tlo)
    s3 = sqrt(3 + 0 * tlo)
    rlo2 = s3 * tlo / r
    c = (2 - tlo * tlo) / (tlo * r)
    rhi2 = s3 * c
    tau = (3 * c * c + 1) / (s3 * c)
    return rlo2, rhi2, tau, c

# ---------------------------------------------------------------- float version (exploration / cross-check)
def float_margin(M):
    from math import exp, tanh, atanh, sqrt
    t = tanh(M); K = M + atanh(t / 2); s = exp(-2 * M)
    D3, G1, lN1, bB1, bkq = quantities(s, K * s, t)
    rlo2, rhi2, tau, c = geom(t, sqrt)
    mu = (tau + sqrt(max(tau * tau - 4, 0.0))) / 2
    rlo = sqrt(rlo2)
    Lam = mu * G1 - rlo * bB1 + lN1
    rhs = D3 * (rlo + bkq / (2 * D3)) ** 2 / rhi2
    return dict(M=M, beta=t * K / 3 ** 0.25, D3=D3, bkq=bkq / D3, Lam=Lam / D3, rhs=rhs / D3, c=c,
                gamma=(G1 - bB1 + lN1) / D3)

# ---------------------------------------------------------------- certified version
def certify(MC, Mtail, dM, verbose=True):
    from flint import arb, ctx, fmpq
    ctx.prec = 128
    H = arb(3).root(4)
    def I(lo, hi):
        return arb(lo).union(arb(hi))
    def check(s, u, t, tlo, tag):
        """tlo: exact lower bound for t on the cell. Returns (ok, worst margin ratio Lam/rhs upper)"""
        D3, G1, lN1, bB1, bkq = quantities(s, u, t)
        rlo2, rhi2, tau, c = geom(tlo, lambda x: x.sqrt())
        if not (c <= 1):                       # theta* >= pi/4 (cot convex on the near region)
            return False, None, 'c'
        if not (D3 > 0 and bkq >= 0):
            return False, None, 'sign'
        mu = (tau + (tau * tau - 4).sqrt()) / 2 if tau > 2 else None
        if mu is None:
            return False, None, 'tau'
        rlo = rlo2.sqrt()
        # Lam_hi is increasing in mu and decreasing in rho_lo because G1, beta B1 >= 0 (quadratic forms);
        # we still evaluate with enclosures of the bounds, which is all that is used.
        Lam = mu * G1 - rlo * bB1 + lN1
        rhs = D3 * (rlo + bkq / (2 * D3)) * (rlo + bkq / (2 * D3)) / rhi2
        # Lam_hi >= gamma > 0 holds analytically (mu_hi >= 1 >= rho_lo, G1, B1 >= 0); only Lam_hi < rhs is checked.
        if not (rhs > 0):
            return False, None, 'rhs'
        ok = bool(Lam < rhs)
        return ok, arb(Lam.upper()) / arb(rhs.lower()), ''
    MC = fmpq(*MC) if isinstance(MC, tuple) else fmpq(MC)
    Mtail = fmpq(Mtail); dM = fmpq(*dM)
    # (I1) beta(M_C) <= 6/5 and theta*(M_C) >= pi/4
    a = arb(MC); tC = a.tanh(); KC = a + (tC / 2).atanh(); betaC = tC * KC / H
    assert betaC < arb(6) / 5, betaC
    res = dict(M_C=str(MC), beta_C_lower=enc(betaC.lower()), beta_C_upper=enc(betaC.upper()), cells=0)
    worst = arb(0); n = 0
    Ma = MC
    while Ma < Mtail:
        Mb = min(Ma + dM, Mtail)
        M = I(Ma, Mb)
        s = (-2 * M).exp(); t = M.tanh(); K = M + (t / 2).atanh(); u = K * s
        tlo = arb(Ma).tanh().lower()
        ok, ratio, why = check(s, u, t, arb(tlo), (Ma, Mb))
        if not ok:
            print("FAIL on", Ma, Mb, why, ratio); return None
        if ratio.upper() > worst: worst = arb(ratio.upper())
        n += 1; Ma = Mb
    # tail  M >= Mtail :  s in (0, s_b],  u = K s in (0, u_b]  (K e^{-2M} is decreasing for M >= 3/4),  t in [t_b, 1)
    b = arb(Mtail); sb = (-2 * b).exp(); tb = b.tanh(); Kb = b + (tb / 2).atanh(); ub = Kb * sb
    assert b > arb(3) / 4
    s = I(0, sb.upper()); u = I(0, ub.upper()); t = I(tb.lower(), 1)
    ok, ratio, why = check(s, u, t, arb(tb.lower()), 'tail')
    if not ok:
        print("FAIL on tail", why, ratio); return None
    res.update(cells=n, worst_ratio_upper=enc(worst.upper()),
               tail_ratio_upper=enc(ratio.upper()), M_tail=str(Mtail), dM=str(dM),
               THEOREM_C_CERTIFIED=True, python_flint=__import__('flint').__version__, prec_bits=ctx.prec)
    if verbose:
        print("OK", res)
    return res

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'float':
        for M in [1.0, 1.1, 1.2, 1.25, 1.3, 1.5, 2, 3, 5, 10, 20]:
            print({k: round(v, 6) for k, v in float_margin(M).items()})
    else:
        r = certify((13, 10), 8, (1, 1000))
        if r is None: sys.exit(1)
        write('thmC.json', r)

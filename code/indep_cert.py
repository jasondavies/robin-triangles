"""Referee's independent certificate of Lemma certC (mpmath interval arithmetic, separate derivation).

Body  [13/10, MBIG]: un-normalised closed forms, evaluated in mpmath.iv from M directly
      (hyperbolics via exp), via the sigma / Gamma / N definitions; adaptive cells.
Tail  [MBIG, inf): sympy-derived normalisation (multiply by 16 e^{-2U}), written in
      s = e^{-2M}, u = K s, w = e^{2L} (omega), t; evaluated on a box containing (s,u,w,t)(M) for all M >= MBIG.
Geometry: theta* from sin(theta*/2) = t_lo/2, via half/double angle formulas (different from the paper's)."""
from mpmath import iv, mp
import sympy as sp, sys, time
from out import write, enc_mpf

iv.dps = 150
ONE = iv.mpf(1); H = iv.sqrt(iv.sqrt(iv.mpf(3))); S3 = iv.sqrt(iv.mpf(3))

def ch(x): e = iv.exp(x); return (e + 1 / e) / 2
def sh(x): e = iv.exp(x); return (e - 1 / e) / 2

def geom(tlo):
    """tlo: iv point (or thin interval). Returns rho_lo, rho_hi^2, mu_hi, cot(theta*)."""
    sn = tlo / 2; cs = iv.sqrt(1 - sn * sn)
    s1 = 2 * sn * cs; c1 = cs * cs - sn * sn            # sin, cos theta*
    s2 = 2 * s1 * c1; c2 = c1 * c1 - s1 * s1            # sin, cos 2 theta*
    cot1 = c1 / s1
    rlo2 = S3 * sn / cs
    rhi2 = S3 * cot1
    tau = 2 / S3 * (2 * cot1 - c2 / s2)                  # (2/sqrt3) * sum cot at vertex (th*,th*,pi-2th*)
    mu = (tau + iv.sqrt(tau * tau - 4)) / 2
    return iv.sqrt(rlo2), rhi2, mu, cot1

def body_quantities(M):
    """M: iv interval. Returns G0, G1, |lam0|N1, beta*B1, beta*kappa (un-normalised)."""
    t = 1 - 2 / (iv.exp(2 * M) + 1)
    L = -iv.log(-1 + 4 / (2 - t)) / 2
    K = M - L; U = 2 * M - L; H2 = H * H
    beta = t * K / H; lam0abs = 4 * K * K / H2
    I1 = K**2 - K / 2 * sh(2 * L) + (ch(2 * U) - ch(2 * L)) / 8
    I1s = -K**2 - K / 2 * sh(2 * L) + (ch(2 * U) - ch(2 * L)) / 8
    I2 = K * sh(2 * M) - sh(L) * (sh(U) - sh(L))
    I2s = K * sh(2 * M) - ch(L) * (ch(U) - ch(L))
    J2 = K + (sh(2 * U) - sh(2 * L)) / 4
    J3 = sh(U) - sh(L)
    J4 = K * ch(2 * M) + sh(2 * K) / 2
    Nd = I1 / (2 * K**2); No = I2 / (2 * K**2)
    Gd = 2 * I1s / H2; Go = 2 * I2s / H2
    s1 = 2 * ch(L)**2 / H; s2 = J2 / (H * K); s3 = ch(L) * J3 / (H * K); s4 = J4 / (H * K)
    G0 = Gd - Go; G1 = Gd + Go / 2
    B1 = s1 + 2 * s2 - 2 * s3 - s4
    kap = (s2 + s3 + s4) - (s1 + 2 * s3)
    N1 = Nd - No
    return G0, G1, lam0abs * N1, beta * B1, beta * kap, beta

def verdict(G0, G1, lN1, bB1, bk, tlo):
    rlo, rhi2, mu, cot1 = geom(tlo)
    if not (cot1 <= 1): return False, 'cot'
    if not (G0 > 0): return False, 'G0'
    if not (bk >= 0): return False, 'kappa'
    Lam = mu * G1 - rlo * bB1 + lN1
    rhs = (2 * G0 * rlo + bk) ** 2 / (4 * G0 * rhi2)
    if not (Lam < rhs): return False, 'ineq'
    return True, float(mp.mpf((iv.mpf(Lam.b) / iv.mpf(rhs.a)).b))

def run_body(MC, MBIG, dM0):
    Ma = mp.mpf(MC); n = 0; worst = 0; dM = mp.mpf(dM0); nfail = 0
    while Ma < MBIG:
        Mb = min(Ma + dM, mp.mpf(MBIG))
        M = iv.mpf([Ma, Mb])
        tlo = 1 - 2 / (iv.exp(2 * iv.mpf(Ma)) + 1)
        tlo = iv.mpf(tlo.a)                       # exact lower endpoint as point
        q = body_quantities(M)
        ok, r = verdict(*q[:5], tlo)
        if ok:
            n += 1; worst = max(worst, float(r)); Ma = Mb; dM = min(dM * 2, mp.mpf(1)) if float(r) < 0.3 else dM
        else:
            nfail += 1
            if dM < mp.mpf('1e-7'): print('BODY FAIL at', Ma, r); return None
            dM = dM / 2
    return n, worst, nfail

# ---------------- tail: sympy normalisation
def tail_exprs():
    s, u, w, t, K = sp.symbols('s u w t K', positive=True)
    m, l = sp.symbols('m l', positive=True)                 # m = e^M, l = e^L ; s = m^-2, w = l^2
    def Ch(e): return (e + 1 / e) / 2
    def Sh(e): return (e - 1 / e) / 2
    eM, eL = m, l; eU = m**2 / l; eK = m / l
    I1 = K**2 - K / 2 * Sh(eL**2) + (Ch(eU**2) - Ch(eL**2)) / 8
    I1s = -K**2 - K / 2 * Sh(eL**2) + (Ch(eU**2) - Ch(eL**2)) / 8
    I2 = K * Sh(eM**2) - Sh(eL) * (Sh(eU) - Sh(eL))
    I2s = K * Sh(eM**2) - Ch(eL) * (Ch(eU) - Ch(eL))
    J2 = K + (Sh(eU**2) - Sh(eL**2)) / 4
    cJ3 = Ch(eL) * (Sh(eU) - Sh(eL))
    J4 = K * Ch(eM**2) + Sh(eK**2) / 2
    KcL2 = K * Ch(eL)**2
    X = eU**2 / 16
    # with common factor 2/H^2 (beta = tK/H, |lam0| = 4K^2/H^2):
    G0 = I1s - I2s; G1 = I1s + I2s / 2; lN1 = I1 - I2
    bB1 = t / 2 * (2 * KcL2 + 2 * J2 - 2 * cJ3 - J4)
    bk = t / 2 * (J2 + J4 - cJ3 - 2 * KcL2)
    out = []
    for e in (G0, G1, lN1, bB1, bk):
        e = sp.expand(e / X)
        e = sp.expand(e.subs(K, u * m**2))                # K = u / s = u m^2
        e = sp.expand(e.subs(l, sp.sqrt(w)).subs(m, 1 / sp.sqrt(s)))
        out.append(e)
    return out, (s, u, w, t)

def run_tail(MBIG):
    exprs, syms = tail_exprs()
    for e in exprs:
        # must be polynomial in s, u, t and Laurent in w (w in [1/3,1]); check no negative powers of s
        P = sp.Poly(e * syms[2]**4, *syms)
        assert all(mon[0] >= 0 and mon[1] >= 0 for mon in P.monoms()), e
    print('tail expressions (normalised):')
    for e in exprs: print('   ', sp.factor_terms(sp.collect(e, syms[1])))
    b = iv.mpf(MBIG)
    sb = iv.exp(-2 * b); tb = 1 - 2 / (iv.exp(2 * b) + 1)
    Kb = b - (-iv.log(-1 + 4 / (2 - tb)) / 2)
    ub = Kb * sb
    box = dict(s=iv.mpf([0, sb.b]), u=iv.mpf([0, ub.b]), t=iv.mpf([tb.a, 1]),
               w=iv.mpf([iv.mpf(1) / 3, ((2 - tb) / (2 + tb)).b]))
    # validity: for M >= MBIG: s<=sb (s decreasing), t>=tb, w=(2-t)/(2+t) in [1/3, w(tb)], u = K e^{-2M} decreasing (K' <= 3/2 <= 2K)
    ns = {'s': box['s'], 'u': box['u'], 't': box['t'], 'w': box['w'], 'sqrt': iv.sqrt}
    vals = [eval(sp.sstr(e).replace('^', '**'), {}, ns) for e in exprs]
    tlo = iv.mpf(tb.a)
    return verdict(*vals, tlo), [str(v) for v in vals]

if __name__ == '__main__':
    MBIG = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    t0 = time.time()
    # beta(13/10)
    M = iv.mpf(13) / 10; t = 1 - 2 / (iv.exp(2 * M) + 1); L = -iv.log(-1 + 4 / (2 - t)) / 2
    print('beta(13/10) in', (t * (M - L) / H), ' (< 6/5:', bool(t * (M - L) / H < iv.mpf(6) / 5), ')')
    r = run_body('1.3', MBIG, '1e-3')
    print('body [1.3, %d]: cells, worst ratio Lam/rhs, rejected-cell count =' % MBIG, r, '%.1fs' % (time.time() - t0))
    (ok, ratio), vals = run_tail(MBIG)
    print('tail M >=', MBIG, ': ok =', ok, 'ratio <=', ratio)
    print('   enclosures G0,G1,lN1,bB1,bk:', vals)
    bC = t * (M - L) / H
    write('thmC_independent.json', dict(
        M_C='13/10', M_big=str(MBIG), beta_C_lower=enc_mpf(bC.a), beta_C_upper=enc_mpf(bC.b),
        beta_C_below_6_5=bool(bC < iv.mpf(6) / 5), body_ok=r is not None,
        body_cells=r[0] if r else None, body_worst_ratio=repr(r[1]) if r else None,
        tail_ok=bool(ok), tail_ratio=repr(ratio) if ok else None, iv_dps=iv.dps,
        ok=bool(r is not None and ok and bC < iv.mpf(6) / 5)))

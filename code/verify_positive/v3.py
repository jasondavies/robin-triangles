"""V3: the local lemma (Lemma 'shapeconst', Lemma 'Ccheck', inequality (C)) in mpmath intervals."""
import time
from fractions import Fraction as Fr
from mpmath import iv, mpf

import geom
import lam0 as L0

KTOP = 336


def _cosh(x):
    e = iv.exp(x)
    return (e + 1 / e) / 2


def _sinh(x):
    e = iv.exp(x)
    return (e - 1 / e) / 2


def _hull(lo, hi):
    return iv.mpf([lo.a, hi.b])


def shape_constants(tstar=Fr(3, 50), nt=20, nphi=420):
    """Rigorous c_l, rho with  lbar-1 >= c_l t^2  and  max_j |rho_j| <= rho t^2  for 0<t<=tstar, all phi.

    Uses the exact identities (paper, eq. (lbar)), re-derived:
        l_j - 1 = u_j/2 - u_j^2/(2(1+l_j)^2),  u_j = l_j^2 - 1 = (cosh 2t - 1) + sinh 2t * c_j,
        (lbar-1)/t^2 = A(t)/2 - vbar,   rho_j/t^2 = -(v_j - vbar),
        v_j = (u_j/t)^2/(2(1+l_j)^2),   u_j/t = t A(t) + c_j B(t),
        A(t) = (cosh 2t - 1)/t^2,  B(t) = sinh(2t)/t   (power series with positive coefficients:
        increasing on t>0, limits 2 at t=0).
    The set {c_j(phi)} = {cos(2phi), cos(2phi-4pi/3), cos(2phi-2pi/3)} is invariant under
    phi -> phi + pi/3, so phi in [0, pi/3] suffices."""
    tstar = Fr(tstar)
    two = iv.mpf(2)
    A_of = lambda t: (_cosh(2 * t) - 1) / (t * t)
    B_of = lambda t: _sinh(2 * t) / t
    cl = None
    rho = None
    third = iv.pi / 3
    shifts = [iv.mpf(0), 4 * iv.pi / 3, 2 * iv.pi / 3]       # 2*theta_j mod 2pi
    for it in range(nt):
        t0 = geom.ivq(tstar * it / nt)
        t1 = geom.ivq(tstar * (it + 1) / nt)
        t = _hull(t0, t1)
        A = _hull(two if it == 0 else A_of(t0), A_of(t1))
        B = _hull(two if it == 0 else B_of(t0), B_of(t1))
        ch = _hull(_cosh(2 * t0), _cosh(2 * t1))
        sh = _hull(_sinh(2 * t0), _sinh(2 * t1))
        for ip in range(nphi):
            phi = _hull(third * ip / nphi, third * (ip + 1) / nphi)
            v = []
            for j in range(3):
                c = iv.cos(2 * phi - shifts[j])
                l = iv.sqrt(ch + sh * c)
                ut = t * A + c * B
                v.append(ut ** 2 / (2 * (1 + l) ** 2))
            vbar = (v[0] + v[1] + v[2]) / 3
            lo = (A / 2 - vbar).a
            cl = lo if cl is None or lo < cl else cl
            for j in range(3):
                dj = (2 * v[j] - v[(j + 1) % 3] - v[(j + 2) % 3]) / 3
                up = max(abs(dj.a), abs(dj.b))
                rho = up if rho is None or up > rho else rho
    return cl, rho          # mpmath intervals (points): certified lower / upper bounds


def check_C(mu, lam2, c_l, rho_hat, tstar=Fr(3, 50), s=Fr(1, 10), K=KTOP, lam0_scale=None):
    """mu[(k,sign)] = exact Fraction lower bounds for Lambda_sign(alpha_k), k=0..K;
    lam2[k] = lower bounds for lambda_2(E, alpha_k), k = 0,4,...,K.
    c_l, rho_hat: exact Fractions (constants of Lemma 'shapeconst').
    Returns per-interval results: K intervals [alpha_k, alpha_{k+1}] and the tail [alpha_K, infinity)."""
    ts = geom.ivq(tstar)
    eta = 1 - iv.exp(-2 * ts)
    eps = _cosh(2 * ts) - 1
    sst = _sinh(2 * ts) / (2 * ts)
    cl = geom.ivq(c_l)
    rh = geom.ivq(rho_hat)
    s2 = geom.ivq(s) ** 2
    res = []
    al = [geom.alpha_grid(k) for k in range(K + 1)]
    l0 = {}
    for k in range(K + 1):
        l0[k] = L0.lam0(al[k])
    L1E = geom.LAMBDA1_E

    def finish(k, first, alam0p, gamma_den_l0, lam2u, dplus, dminus, tail):
        # first: lower bound for lambda_0 in the first factor; alam0p: upper bound for alpha*lambda_0'
        # gamma_den_l0: upper bound for lambda_0 used in gamma and in (ring)
        r = {"k": k, "tail": tail, "ok": False}
        if not (lam2u.a > gamma_den_l0.b):
            r["reason"] = "lambda_2 bound does not exceed lambda_0 upper bound"
            return r
        if not (dplus.a > 0 and dminus.a > 0):
            r["reason"] = "delta not certified positive"
            return r
        gamma = lam2u / (lam2u - gamma_den_l0)
        R1 = (2 + (dplus + dminus) / (lam2u - gamma_den_l0)) / (s2 * (1 / dplus + 1 / dminus))
        f1 = 2 * first - (2 - cl) * alam0p
        f2 = 1 - (eta + eps) * gamma
        if not (f1.a > 0 and f2.a > 0):
            r["reason"] = "a factor of the left side is not certified positive"
            return r
        lhs = f1 * f2
        rhs = (sst * iv.sqrt(R1) + rh * ts * iv.sqrt(alam0p * gamma)) ** 2
        r["lhs"] = float(lhs.a)
        r["rhs"] = float(rhs.b)
        r["ratio"] = float((lhs.a / rhs.b).a)
        r["gamma"] = float(gamma.b)
        r["dplus"] = float(dplus.b)
        r["dminus"] = float(dminus.b)
        r["R1_over_lam"] = float((R1 / gamma_den_l0).b)
        r["ok"] = bool(lhs.a > rhs.b)
        return r

    for k in range(K):
        a0 = geom.ivq(al[k])
        a1 = geom.ivq(al[k + 1])
        (lA, sA), (lB, sB) = l0[k], l0[k + 1]
        if lam0_scale is not None:
            pass
        # tangents T_A(a) = lA + sA (a - a0), T_B(a) = lB + sB (a - a1); kink where they cross
        if not (sA.a > sB.b):
            res.append({"k": k, "tail": False, "ok": False, "reason": "slopes not certified decreasing"})
            continue
        across = (lB - sB * a1 - lA + sA * a0) / (sA - sB)
        deltas = []
        for sign in (1, -1):
            m0 = geom.ivq(mu[(k, sign)])
            m1 = geom.ivq(mu[(k + 1, sign)])
            chord = lambda a: m0 + (m1 - m0) * (a - a0) / (a1 - a0)
            TA = lambda a: lA + sA * (a - a0)
            TB = lambda a: lB + sB * (a - a1)
            # upper bounds for min(TA,TB) - chord at the two end points and at the kink
            c0 = min((TA(a0) - chord(a0)).b, (TB(a0) - chord(a0)).b)
            c1 = min((TA(a1) - chord(a1)).b, (TB(a1) - chord(a1)).b)
            cx = (TA(across) - chord(across)).b
            d = max(c0, c1, cx)
            deltas.append(iv.mpf(d))
        lam2u = geom.ivq(lam2[4 * (k // 4)])
        res.append(finish(k, lA, a1 * sA, lB, lam2u, deltas[0], deltas[1], False))
    # tail [alpha_K, infinity)
    k = K
    dplus = L1E - geom.ivq(mu[(K, 1)])
    dminus = L1E - geom.ivq(mu[(K, -1)])
    lam2u = geom.ivq(lam2[K])
    lhalf = L0.lam0(al[K] / 2)[0]
    alam0p = 2 * (L1E - lhalf)
    res.append(finish(K, l0[K][0], alam0p, L1E, lam2u, dplus, dminus, True))
    consts = {"eta": float(eta.b), "eps": float(eps.b), "s_star": float(sst.b)}
    return res, consts


def tanh_check(rstar=Fr(1194, 10000), tstar=Fr(3, 50)):
    """r_* < tanh(2 t_*)"""
    e = iv.exp(4 * geom.ivq(tstar))
    th = (e - 1) / (e + 1)
    return bool(th.a > geom.ivq(rstar).b), float(th.a)

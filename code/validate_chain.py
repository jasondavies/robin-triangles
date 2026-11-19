"""Float validation (not part of the proof) of every step in the proof of Theorem C on actual triangles.
Matrices are computed by Gauss quadrature on the real triangle T (no closed forms, no transplantation
formulae), so this is independent of large_beta.py / three_float.py.
Also checks the corner bound of Lemma (corner) on arbitrary triangles."""
import numpy as np
from math import *
from numpy.polynomial.legendre import leggauss
from out import write
H = 3 ** 0.25
REC = {}
rng = np.random.default_rng(12345)
xg, wg = leggauss(160); xg = (xg + 1) / 2; wg = wg / 2

def triangle(th):
    s = [sin(a) for a in th]
    twoR = sqrt(2 / (s[0] * s[1] * s[2]))
    a = [twoR * si for si in s]                       # a[i] opposite vertex i, area 1
    V = [np.array([0.0, 0.0]), np.array([a[2], 0.0]), a[1] * np.array([cos(th[0]), sin(th[0])])]
    return V, a

def bary_quad(V):
    """quadrature nodes (barycentric b0,b1,b2), weights (sum = area), grad b_i"""
    Tm = np.array([V[1] - V[0], V[2] - V[0]]).T
    Ti = np.linalg.inv(Tm); gb = [-(Ti[0] + Ti[1]), Ti[0], Ti[1]]
    area = abs(np.linalg.det(Tm)) / 2
    U, Vv = np.meshgrid(xg, xg, indexing='ij'); WU, WV = np.meshgrid(wg, wg, indexing='ij')
    b1 = U; b2 = Vv * (1 - U); b0 = 1 - b1 - b2
    w = WU * WV * (1 - U) * 2 * area
    return [b0, b1, b2], w, gb, area

def three_mats(M, th):
    t = tanh(M); L = -atanh(t / 2); K = M - L; beta = t * K / H; lam0 = -4 * K * K / H ** 2
    V, a = triangle(th)
    b, w, gb, area = bary_quad(V)
    f = [np.cosh(L + 2 * K * bi) for bi in b]; g = [2 * K * np.sinh(L + 2 * K * bi) for bi in b]
    A = np.zeros((3, 3)); N = np.zeros((3, 3)); G = np.zeros((3, 3)); Bd = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            N[i, j] = np.sum(w * f[i] * f[j]); G[i, j] = np.sum(w * g[i] * g[j]) * (gb[i] @ gb[j])
    for i in range(3):
        j, l = (i + 1) % 3, (i + 2) % 3
        bb = [None] * 3; bb[i] = 0 * xg; bb[j] = xg; bb[l] = 1 - xg
        fs = [np.cosh(L + 2 * K * x) for x in bb]
        for p in range(3):
            for q in range(3):
                Bd[p, q] += a[i] * np.sum(wg * fs[p] * fs[q])
    return G - beta * Bd, N, G, Bd, a, dict(t=t, L=L, K=K, beta=beta, lam0=lam0)

def corner_R(th, beta, vertex):
    """Rayleigh quotient of exp(-kappa e.(x-V)) (kappa = beta/sin(theta/2), e bisector at V) by quadrature"""
    V, a = triangle(th)
    i = vertex; j, l = (i + 1) % 3, (i + 2) % 3
    e = (V[j] - V[i]) / np.linalg.norm(V[j] - V[i]) + (V[l] - V[i]) / np.linalg.norm(V[l] - V[i]); e /= np.linalg.norm(e)
    kap = beta / sin(th[i] / 2)
    b, w, gb, area = bary_quad(V)
    X = sum(b[m][..., None] * V[m] for m in range(3))
    u2 = np.exp(-2 * kap * ((X - V[i]) @ e))
    Nn = np.sum(w * u2); bd = 0
    for s in range(3):
        p, q = V[(s + 1) % 3], V[(s + 2) % 3]
        pts = p[None, :] + xg[:, None] * (q - p)[None, :]
        bd += np.linalg.norm(q - p) * np.sum(wg * np.exp(-2 * kap * ((pts - V[i]) @ e)))
    return kap ** 2 - beta * bd / Nn, -beta ** 2 / sin(th[i] / 2) ** 2

def run(nsamp=400):
    Qm = np.array([[1, -1, 0], [1, 1, -2]]).T / np.array([sqrt(2), sqrt(6)])
    worst = dict(rho=1e9, mu=1e9, Lam=1e9, p=1e9, q=1e9, Q=-1e9, crit=1e9)
    # reference (equilateral) scalars from quadrature too
    for it in range(nsamp):
        M = rng.uniform(1.3, 5.0) if it % 4 else 1.3 + 1e-9
        t = tanh(M); ths = 2 * asin(t / 2)
        # random point of the near region {theta_i >= theta*}, biased to vertices/edges sometimes
        wts = rng.dirichlet([0.5, 0.5, 0.5]) if it % 3 else np.eye(3)[it % 3 == 0 and rng.integers(3)]
        th = ths + (pi - 3 * ths) * np.array(wts)
        if max(abs(th - pi / 3)) < 1e-7: continue
        A, N, G, Bd, a, d = three_mats(M, list(th)); beta, lam0 = d['beta'], d['lam0']
        A0, N0, G0, Bd0, a0, _ = three_mats(M, [pi / 3] * 3)
        rho = np.array(a) / (2 / H)
        P = A - lam0 * N; P0 = A0 - lam0 * N0
        assert np.abs(P0 @ np.ones(3)).max() < 1e-9 * np.abs(P0).max() + 1e-9 * np.abs(A0).max()
        # scalars of the equilateral
        Gd = G0[0, 0] / 1.0; Go = -2 * G0[0, 1]              # G0_ij = n_i.n_j Gamma_ij
        D3 = Gd - Go; G1 = Gd + Go / 2
        B1 = Bd0[0, 0] - Bd0[0, 1]; N1 = N0[0, 0] - N0[0, 1]
        # sigma's via side integrals on E:  kq = b_adj - b_own
        # b_own = int_{side 0} phi_0 psi0 ; recompute directly
        L, K = d['L'], d['K']; l0 = 2 / H
        own = l0 * np.sum(wg * cosh(L) * (cosh(L) + np.cosh(L + 2 * K * xg) + np.cosh(L + 2 * K * (1 - xg))))
        adj = l0 * np.sum(wg * np.cosh(L + 2 * K * xg) * (np.cosh(L + 2 * K * xg) + cosh(L) + np.cosh(L + 2 * K * (1 - xg))))
        kq = adj - own
        # geometry bounds
        c = (2 - t * t) / (t * sqrt(4 - t * t)); rlo = sqrt(sqrt(3) * t / sqrt(4 - t * t)); rhi = sqrt(sqrt(3) * c)
        tauhi = (3 * c * c + 1) / (sqrt(3) * c); muhi = (tauhi + sqrt(tauhi ** 2 - 4)) / 2
        S = np.sum(rho ** 2); tau = 2 * S / 3; mu = (tau + sqrt(max(tau * tau - 4, 0))) / 2
        worst['rho'] = min(worst['rho'], rho.min() - rlo, rhi - rho.max()); worst['mu'] = min(worst['mu'], muhi - mu)
        # Lambda
        Pvv = Qm.T @ P @ Qm; lmax = np.linalg.eigvalsh(Pvv)[-1]
        Lam = mu * G1 - beta * rho.min() * B1 - lam0 * N1
        Lamhi = muhi * G1 - beta * rlo * B1 - lam0 * N1
        worst['Lam'] = min(worst['Lam'], (Lam - lmax) / D3, (Lamhi - Lam) / D3)
        # p' and q
        W = np.sum((rho - rho.mean()) ** 2); x = rho ** 2; Vx = np.sum((x - x.mean()) ** 2)
        pp = np.ones(3) @ P @ np.ones(3)
        assert abs(S * S - 2 * np.sum(x * x) - 3) < 1e-9 and abs((S - 3) - 6 * Vx / (S + 3)) < 1e-9
        worst['p'] = min(worst['p'], (D3 * (S - 3) - pp) / D3 / W, (4 * rho.max() ** 2 * W - (S - 3)) / W)
        qv = P @ np.ones(3); Pq = qv - qv.mean()
        hpred = (x - 1) * D3 + beta * kq * (rho - 1); hpred -= hpred.mean()
        assert np.abs(Pq - hpred).max() < 1e-7 * D3 * (np.abs(rho - 1).max() + 1e-3), (Pq, hpred)
        worst['q'] = min(worst['q'], (Pq @ Pq - (2 * D3 * rho.min() + beta * kq) ** 2 * W) / D3 ** 2 / W)
        cvec = np.ones(3) - Pq / Lam
        Qc = cvec @ P @ cvec
        worst['Q'] = max(worst['Q'], Qc / D3 / W, (Qc - (pp - Pq @ Pq / Lam)) / D3 / W)
        worst['crit'] = min(worst['crit'], (rlo + beta * kq / (2 * D3)) ** 2 / rhi ** 2 - Lamhi / D3)
    REC.update(chain_samples=nsamp, worst={k: float(v) for k, v in worst.items()},
               chain_ok=bool(min(v for k, v in worst.items() if k != 'Q') > -1e-10 and worst['Q'] < 0))
    print("worst slacks (must be >= 0 up to rounding): ", {k: float(f"{v:.3e}") for k, v in worst.items() if k != 'Q'})
    print("worst Q(c)/(D3 W) (must be < 0) and Q - (p' - |Pq|^2/Lam) (must be <= 0):", worst['Q'])

def run_corner(n=300):
    """(a) sign check where quadrature resolves the (possibly exponentially small) gap;
       (b) the divergence identity 2 kappa^2 int_T u^2 = sum_s (w.nu_s) int_s u^2 behind the lemma."""
    w = -1e9; widen = 0; cnt = 0
    for _ in range(n):
        th = rng.dirichlet([3, 3, 3]) * pi
        if min(th) < 0.5: continue
        beta = rng.uniform(0.2, 1.0)
        for v in range(3):
            R, bound = corner_R(list(th), beta, v)
            w = max(w, (R - bound) / beta ** 2); cnt += 1
            # identity
            V, a = triangle(list(th)); i = v; j, l = (i + 1) % 3, (i + 2) % 3
            e = (V[j] - V[i]) / np.linalg.norm(V[j] - V[i]) + (V[l] - V[i]) / np.linalg.norm(V[l] - V[i]); e /= np.linalg.norm(e)
            kap = beta / sin(th[i] / 2); wv = -kap * e
            b, wq, gb, area = bary_quad(V)
            X = sum(b[m][..., None] * V[m] for m in range(3))
            Nn = np.sum(wq * np.exp(2 * ((X - V[i]) @ wv))); rhs = 0
            cen = (V[0] + V[1] + V[2]) / 3
            for s_ in range(3):
                p_, q_ = V[(s_ + 1) % 3], V[(s_ + 2) % 3]
                tng = (q_ - p_) / np.linalg.norm(q_ - p_); nu = np.array([tng[1], -tng[0]])
                if nu @ (cen - p_) > 0: nu = -nu
                pts = p_[None, :] + xg[:, None] * (q_ - p_)[None, :]
                rhs += (wv @ nu) * np.linalg.norm(q_ - p_) * np.sum(wg * np.exp(2 * ((pts - V[i]) @ wv)))
            widen = max(widen, abs(2 * kap ** 2 * Nn - rhs) / abs(rhs))
    REC.update(corner_samples=cnt, corner_worst=float(w), divergence_rel_err=float(widen), corner_ok=bool(w < 0 and widen < 1e-12))
    print(f"corner lemma on {cnt} (triangle, vertex, beta) samples: max (R[u] + beta^2/sin^2(theta/2))/beta^2 = {w:.3e} (must be < 0);"
          f" divergence identity rel. error {widen:.1e}")

if __name__ == '__main__':
    run_corner(); run()
    REC['ok'] = REC['chain_ok'] and REC['corner_ok']
    write('validate_chain.json', REC)

"""Independent floating-point cross-checks (not part of the proof).
1. lambda0(beta) formula vs P1-FEM (Richardson) on the equilateral triangle.
2. psi0 = sum cosh(L + 2K d_i/H) satisfies -Lap psi = lambda0 psi and the Robin condition (finite differences).
3. B/N = dlambda0/dalpha and E/N = lambda0 + beta B/N by direct Gauss quadrature of psi0.
4. Transplanted six-wave Rayleigh quotient via the tensor formula vs direct quadrature on T itself."""
import numpy as np, mpmath as mp
from numpy.polynomial.legendre import leggauss
from equi_float import lam0, lam0_and_deriv, M_of_beta, K_of_M
from fem import lam1
import sixwave as sw
from nums import MP
from sw_float import Setup
from out import write
H = 3 ** 0.25; l = 2 / H; r = H / 3
Ve = np.array([(-l / 2, -r), (l / 2, -r), (0, 2 * r)])

def tri_quad(V, f, n=40):
    x, w = leggauss(n)
    u = (x + 1) / 2; wu = w / 2
    tot = 0.0
    A = V[0]; B = V[1]; C = V[2]
    area = abs(np.cross(np.r_[B - A, 0], np.r_[C - A, 0])[2]) / 2
    for i in range(n):
        for j in range(n):
            s, t = u[i], u[j] * (1 - u[i])
            P = A + s * (B - A) + t * (C - A)
            tot += wu[i] * wu[j] * (1 - u[i]) * f(P)
    return 2 * area * tot

def seg_quad(A, B, f, n=60):
    x, w = leggauss(n); L = np.linalg.norm(B - A)
    return sum(wi * f(A + (xi + 1) / 2 * (B - A)) for xi, wi in zip(x, w)) * L / 2

def psi0_fun(beta):
    M = M_of_beta(beta); t = np.tanh(M); L = -np.arctanh(t / 2); K = M - L
    nins = []
    for s in range(3):
        e = Ve[(s + 1) % 3] - Ve[s]; e /= np.linalg.norm(e); nins.append(np.array([-e[1], e[0]]))
    def f(P):
        return sum(np.cosh(L + 2 * K * (n @ P + r) / H) for n in nins)
    return f, nins

ok = True
rec = dict(lam0_vs_fem=[], bc_residual_max=0.0, hf=[], sixwave=[])
print('1. lambda0 vs FEM')
for beta in (0.5, 2.0, 4.0):
    f1 = lam1(Ve, -beta, 60)[0]; f2 = lam1(Ve, -beta, 120)[0]
    print('   beta', beta, 'formula', lam0(beta), 'FEM-Richardson', (4 * f2 - f1) / 3)
    rec['lam0_vs_fem'].append(dict(beta=beta, formula=float(lam0(beta)), fem_richardson=float((4 * f2 - f1) / 3)))
    if abs((4 * f2 - f1) / 3 - lam0(beta)) > 1e-4 * abs(lam0(beta)): ok = False
print('2. PDE and boundary condition of psi0')
for beta in (0.7, 3.0):
    f, nins = psi0_fun(beta); lam = lam0(beta); h = 1e-4
    P = np.array([0.1, 0.05])
    lap = (f(P + [h, 0]) + f(P - [h, 0]) + f(P + [0, h]) + f(P - [0, h]) - 4 * f(P)) / h ** 2
    print('   beta', beta, 'interior residual (-lap - lam psi)/psi', (-lap - lam * f(P)) / f(P))
    for s in range(3):
        A, B = Ve[s], Ve[(s + 1) % 3]
        for tpar in (0.2, 0.5, 0.9):
            Q = A + tpar * (B - A); nout = -nins[s]
            dn = (f(Q + h * nout) - f(Q - h * nout)) / (2 * h)
            res = (dn - beta * f(Q)) / f(Q)
            if abs(res) > 1e-6: ok = False
            rec['bc_residual_max'] = max(rec['bc_residual_max'], abs(float(res)))
        print('   side', s, 'BC residual (last)', res)
print('3. Hellmann-Feynman check by quadrature')
for beta in (0.7, 3.0):
    f, nins = psi0_fun(beta)
    N = tri_quad(Ve, lambda P: f(P) ** 2)
    B = sum(seg_quad(Ve[s], Ve[(s + 1) % 3], lambda P: f(P) ** 2) for s in range(3))
    h = 1e-5
    E = tri_quad(Ve, lambda P: ((f(P + [h, 0]) - f(P - [h, 0])) / (2 * h)) ** 2 + ((f(P + [0, h]) - f(P - [0, h])) / (2 * h)) ** 2)
    l0, d0 = lam0_and_deriv(beta)
    rec['hf'].append(dict(beta=beta, BN_rel=float(abs(B / N / d0 - 1)), EN_rel=float(abs(E / N / (l0 + beta * d0) - 1))))
    if rec['hf'][-1]['BN_rel'] > 1e-9 or rec['hf'][-1]['EN_rel'] > 1e-7: ok = False
    print('   beta', beta, 'B/N quad', B / N, 'dlam/dalpha', d0, '| E/N quad', E / N, 'lam0+beta B/N', l0 + beta * d0, '| Rayleigh', (E - beta * B) / N, l0)
print('4. six-wave transplant Rayleigh quotient: tensor formula vs direct quadrature on T')
st = Setup(M_of_beta(2.0)); rng = np.random.default_rng(1)
for (a, b) in [(1.1, -0.2), (1.7, -0.5), (1.02, -0.01)]:
    c = rng.normal(size=6) * np.array([float(x) for x in st.c0])
    C = [[mp.mpf(x) for x in c]]
    T = sw.tensors(MP, st.W, st.N, st.S, C)
    Num, Den = sw.num_den(MP, T, -st.beta, mp.mpf(a), mp.mpf(b), [1])
    D = np.array([[a, b], [0, 1 / a]]); Dinv = np.linalg.inv(D)
    VT = Ve @ D.T
    W = [np.array([float(w[0]), float(w[1])]) for w in st.W]
    u = lambda X: sum(cj * np.exp(w @ (Dinv @ X)) for cj, w in zip(c, W))
    gu = lambda X: sum(cj * np.exp(w @ (Dinv @ X)) * (Dinv.T @ w) for cj, w in zip(c, W))
    Nq = tri_quad(VT, lambda X: u(X) ** 2); Gq = tri_quad(VT, lambda X: gu(X) @ gu(X))
    Bq = sum(seg_quad(VT[s], VT[(s + 1) % 3], lambda X: u(X) ** 2) for s in range(3))
    Rq = (Gq - float(st.beta) * Bq) / Nq
    print('   (a,b)', (a, b), 'formula', float(Num / Den), 'quadrature', Rq)
    if abs(float(Num / Den) - Rq) > 1e-8 * abs(Rq): ok = False
    rec['sixwave'].append(dict(a=a, b=b, formula=float(Num / Den), quadrature=float(Rq)))
print('ALL CHECKS OK' if ok else 'SOME CHECK FAILED')
rec['sixwave_rel_max'] = max(abs(r['formula'] / r['quadrature'] - 1) for r in rec['sixwave'])
rec['ok'] = bool(ok)
write('test_independent.json', rec)

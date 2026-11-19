"""Exploration (mpmath, not rigorous): Rayleigh-Ritz for the lowest Robin eigenvalue on a triangle
with an exponential (plane-wave) basis e^{w.x}.  All Galerkin integrals are closed form."""
import mpmath as mp
import numpy as np

def dd1(x, y):
    d = x - y
    if abs(d) < mp.mpf(10) ** (-mp.mp.dps // 2):
        return mp.exp(y) * (1 + d / 2 + d * d / 6 + d ** 3 / 24)
    return mp.exp(y) * mp.expm1(d) / d

def dd2(a, b, c):
    """second divided difference of exp at a,b,c (= int over std simplex weights)"""
    pts = sorted([a, b, c])
    a, b, c = pts
    if c - a < mp.mpf(10) ** (-mp.mp.dps // 3):
        m = (a + b + c) / 3
        da, db, dc = a - m, b - m, c - m
        # h_2 = sum of squares and products / 4! ; first order term vanishes (h_1 = 0)
        h2 = da * da + db * db + dc * dc + da * db + db * dc + da * dc
        return mp.exp(m) * (mp.mpf(1) / 2 + h2 / 24)
    return (dd1(c, b) - dd1(b, a)) / (c - a)

def tri_exp(V, v):
    S = abs((V[1][0] - V[0][0]) * (V[2][1] - V[0][1]) - (V[1][1] - V[0][1]) * (V[2][0] - V[0][0])) / 2
    z = [v[0] * P[0] + v[1] * P[1] for P in V]
    return 2 * S * dd2(*z)

def seg_exp(A, B, v):
    L = mp.sqrt((B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2)
    return L * dd1(v[0] * B[0] + v[1] * B[1], v[0] * A[0] + v[1] * A[1])

def matrices(V, W, alpha):
    V = [(mp.mpf(P[0]), mp.mpf(P[1])) for P in V]
    n = len(W)
    A = mp.matrix(n, n); M = mp.matrix(n, n)
    for i in range(n):
        for j in range(i, n):
            v = (W[i][0] + W[j][0], W[i][1] + W[j][1])
            N = tri_exp(V, v)
            B = sum(seg_exp(V[k], V[(k + 1) % 3], v) for k in range(3))
            A[i, j] = A[j, i] = (W[i][0] * W[j][0] + W[i][1] * W[j][1]) * N + alpha * B
            M[i, j] = M[j, i] = N
    return A, M

def lowest(A, M, tol=None):
    """lowest generalized eigenvalue; drop near-null directions of M (relative tol)"""
    n = A.rows
    E, Q = mp.eigsy(M)
    emax = max(E)
    tol = tol if tol is not None else mp.mpf(10) ** (-(mp.mp.dps * 2) // 3)
    keep = [i for i in range(n) if E[i] > tol * emax]
    Z = mp.matrix(n, len(keep))
    for c, i in enumerate(keep):
        for r in range(n):
            Z[r, c] = Q[r, i] / mp.sqrt(E[i])
    Ar = Z.T * A * Z
    w, U = mp.eigsy(Ar)
    i = min(range(len(w)), key=lambda k: w[k])
    c = Z * U[:, i]
    return w[i], c

def plane_basis(k, nth, radii=(1.0,), phase=0.0):
    W = []
    for r in radii:
        for m in range(nth):
            th = 2 * mp.pi * m / nth + phase
            W.append((k * r * mp.cos(th), k * r * mp.sin(th)))
    return W

def lam_ritz(V, beta, nth=24, radii=(0.9, 1.0, 1.1), k=None, dps=50):
    mp.mp.dps = dps
    if k is None:
        k = mp.mpf(2) * beta + 1
    W = plane_basis(mp.mpf(k), nth, radii)
    A, M = matrices(V, W, -mp.mpf(beta))
    return lowest(A, M)

if __name__ == '__main__':
    import sys, time
    from equi_float import lam0
    s = 3 ** 0.25; l = 2 / s
    V = [(0, 0), (l, 0), (l / 2, l * np.sqrt(3) / 2)]
    for beta in (1.0, 3.0):
        k = np.sqrt(-lam0(beta))
        for nth in (6, 12, 24):
            t = time.time()
            lam, c = lam_ritz(V, beta, nth=nth, radii=(1.0,), k=k)
            print(beta, nth, mp.nstr(lam, 20), lam0(beta), time.time() - t)

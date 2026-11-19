"""Independent float check: build an actual triangle T (area 1), trial functions
u_i(x) = cosh(L + 2K b_i(x)), b_i = barycentric coordinate vanishing on side i,
compute the 3x3 Rayleigh matrices by Gauss quadrature on T, compare with closed forms in three_float.mats."""
import numpy as np, sys
from math import *
from three_float import scal, mats, Mof, H
from numpy.polynomial.legendre import leggauss
from out import write

def tri_from_angles(th):
    # vertices V0,V1,V2 with angle th[i] at V_i ; side i is opposite V_i
    s = [sin(a) for a in th]
    twoR = sqrt(2 / (s[0] * s[1] * s[2]))
    a = [twoR * si for si in s]            # side lengths, a[i] opposite V_i
    V0 = np.array([0.0, 0.0]); V1 = np.array([a[2], 0.0])
    V2 = a[1] * np.array([cos(th[0]), sin(th[0])])
    return [V0, V1, V2], a

def quad_mats(M, th, n=200):
    d = scal(M); L, K, beta = d['L'], d['K'], d['beta']
    V, a = tri_from_angles(th)
    xg, wg = leggauss(n); xg = (xg + 1) / 2; wg = wg / 2
    # gradients of barycentric coords
    Tm = np.array([V[1] - V[0], V[2] - V[0]]).T
    Ti = np.linalg.inv(Tm)         # rows: grad b1, grad b2
    gb = [-(Ti[0] + Ti[1]), Ti[0], Ti[1]]
    area = abs(np.linalg.det(Tm)) / 2
    A = np.zeros((3, 3)); N = np.zeros((3, 3))
    # interior: Duffy
    for u, wu in zip(xg, wg):
        for v, wv in zip(xg, wg):
            b1 = u; b2 = v * (1 - u); b0 = 1 - b1 - b2
            w = wu * wv * (1 - u) * 2 * area
            b = [b0, b1, b2]
            f = [cosh(L + 2 * K * bi) for bi in b]
            g = [2 * K * sinh(L + 2 * K * b[i]) * gb[i] for i in range(3)]
            for i in range(3):
                for j in range(3):
                    N[i, j] += w * f[i] * f[j]
                    A[i, j] += w * (g[i] @ g[j])
    # boundary: side i opposite V_i: b_i = 0
    for i in range(3):
        j, l = (i + 1) % 3, (i + 2) % 3
        for u, wu in zip(xg, wg):
            b = [0, 0, 0]; b[j] = u; b[l] = 1 - u
            f = [cosh(L + 2 * K * bi) for bi in b]
            for p in range(3):
                for q in range(3):
                    A[p, q] -= beta * a[i] * wu * f[p] * f[q]
    l0 = 2 / H
    return A, N, [ai / l0 for ai in a], area

if __name__ == '__main__':
    worst = 0.0
    for beta, th in [(2.0, [1.0, 1.1, pi - 2.1]), (3.0, [pi/3 - .01, pi/3 + .03, pi/3 - .02]), (1.3, [0.9, 1.2, pi - 2.1])]:
        M = Mof(beta); d = scal(M)
        A, N, rho, area = quad_mats(M, th)
        A2, N2 = mats(d, rho)
        worst = max(worst, np.abs(A - A2).max() / np.abs(A).max(), np.abs(N - N2).max() / np.abs(N).max())
        print(beta, 'area', area, 'relerr A', np.abs(A - A2).max() / np.abs(A).max(), 'relerr N', np.abs(N - N2).max() / np.abs(N).max())
    write('check_quad.json', dict(max_rel_err=float(worst), ok=bool(worst < 1e-13)))

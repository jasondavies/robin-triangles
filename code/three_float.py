"""Float exploration of the three-function ("corner + reflection") transplant space.
phi_i = cosh(L + k d_i) on E (area 1), i=0,1,2; psi0 = sum phi_i.  All matrix entries in closed form.
Shape enters only through the side-length ratios rho_i = a_i / l."""
import numpy as np
from math import *
H = 3 ** 0.25

def scal(M):
    t = tanh(M); L = -atanh(t / 2); K = M - L; U = 2 * M - L
    beta = t * K / H; lam0 = -4 * K * K / H ** 2
    I1 = K * K - K / 2 * sinh(2 * L) + (cosh(2 * U) - cosh(2 * L)) / 8
    I1s = -K * K - K / 2 * sinh(2 * L) + (cosh(2 * U) - cosh(2 * L)) / 8
    I2 = K * sinh(2 * M) - sinh(L) * (sinh(U) - sinh(L))
    I2s = K * sinh(2 * M) - cosh(L) * (cosh(U) - cosh(L))
    J2 = K + (sinh(2 * U) - sinh(2 * L)) / 4
    J3 = sinh(U) - sinh(L)
    J4 = K * cosh(2 * M) + sinh(2 * K) / 2
    d = dict(M=M, t=t, L=L, K=K, U=U, beta=beta, lam0=lam0)
    d['Nd'] = I1 / (2 * K * K); d['No'] = I2 / (2 * K * K)
    d['Gd'] = 2 / H ** 2 * I1s; d['Go'] = 2 / H ** 2 * I2s
    d['s1'] = 2 / H * cosh(L) ** 2; d['s2'] = J2 / (H * K)
    d['s3'] = cosh(L) * J3 / (H * K); d['s4'] = J4 / (H * K)
    d['D3'] = d['Gd'] - d['Go']
    d['gamma'] = 1.5 * (d['Gd'] - beta * (d['s1'] + 2 * d['s2']) - lam0 * d['Nd'])
    d['kq'] = d['s2'] + d['s4'] - d['s1'] - d['s3']
    return d

def mats(d, rho):
    """A(delta), N (3x3) for side ratios rho (area-1 triangle)"""
    x = [r * r for r in rho]; beta = d['beta']
    A = np.zeros((3, 3)); N = np.zeros((3, 3))
    for i in range(3):
        j, l = (i + 1) % 3, (i + 2) % 3
        A[i, i] = x[i] * d['Gd'] - beta * (rho[i] * d['s1'] + (rho[j] + rho[l]) * d['s2'])
        N[i, i] = d['Nd']
        for (p, q, r) in ((i, j, l),):
            nAn = (x[r] - x[p] - x[q]) / 2
            A[p, q] = A[q, p] = nAn * d['Go'] - beta * ((rho[p] + rho[q]) * d['s3'] + rho[r] * d['s4'])
            N[p, q] = N[q, p] = d['No']
    return A, N

def rho_of_angles(th):
    """side ratios of area-1 triangle with angles th"""
    s = [sin(a) for a in th]
    twoR = sqrt(2 / (s[0] * s[1] * s[2]))
    l = 2 / H
    return [twoR * si / l for si in s]

def Mof(beta):
    lo, hi = 1e-9, 60.0
    for _ in range(200):
        mid = (lo + hi) / 2
        t = tanh(mid); K = mid + atanh(t / 2)
        if t * K / H < beta: lo = mid
        else: hi = mid
    return lo

if __name__ == '__main__':
    import scipy.linalg as sl
    print("beta   M      1-t       gamma/D3   kq/..   nullcheck")
    for beta in [1.2, 1.5, 2, 2.5, 3, 4, 6, 8]:
        M = Mof(beta); d = scal(M)
        A, N = mats(d, [1, 1, 1])
        P0 = A - d['lam0'] * N
        print(f"{beta:4.1f} {M:7.4f} {1-d['t']:.3e} {d['gamma']/d['D3']:.4e} {d['kq']*beta/d['D3']:.4f} "
              f"{np.abs(P0@np.ones(3)).max()/d['D3']:.1e}  eigP0/D3={np.linalg.eigvalsh(P0)/d['D3']}")

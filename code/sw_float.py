"""mpmath prototype of the six-wave Ritz bound Lambda6(a,b,beta) and derivative trial vectors."""
import mpmath as mp
from nums import MP
import sixwave as sw
from ritz_mp import tri_exp, seg_exp, lowest
mp.mp.dps = 40

def Kof(M):
    return M + mp.atanh(mp.tanh(M) / 2)

def M_of_beta(beta):
    H = mp.root(3, 4)
    return mp.findroot(lambda M: mp.tanh(M) * Kof(M) / H - beta, mp.mpf(beta) + 0.5)

class Setup:
    def __init__(self, M):
        self.M = mp.mpf(M)
        self.K, self.c0 = sw.psi0_coeffs(MP, self.M)
        self.W = sw.waves(MP, self.K)
        self.N, self.S = sw.base_matrices(MP, self.W, tri_exp, seg_exp)
        self.H = mp.root(3, 4)
        self.beta = mp.tanh(self.M) * self.K / self.H
        self.lam0 = -4 * self.K ** 2 / self.H ** 2
    def A(self, a, b, alpha):
        Axx, Axy, Ayy, rho = sw.shape_terms(MP, mp.mpf(a), mp.mpf(b))
        n = 6; W = self.W
        A = mp.matrix(n, n); Nm = mp.matrix(n, n)
        for j in range(n):
            for k in range(n):
                g = W[j][0] * W[k][0] * Axx + (W[j][0] * W[k][1] + W[j][1] * W[k][0]) * Axy + W[j][1] * W[k][1] * Ayy
                A[j, k] = g * self.N[j][k] + alpha * sum(rho[s] * self.S[s][j][k] for s in range(3))
                Nm[j, k] = self.N[j][k]
        return A, Nm
    def ritz(self, a, b, beta=None):
        beta = self.beta if beta is None else mp.mpf(beta)
        A, Nm = self.A(a, b, -beta)
        lam, c = lowest(A, Nm)
        return lam, c

if __name__ == '__main__':
    import numpy as np
    for beta in [1, 2, 4]:
        st = Setup(M_of_beta(beta))
        lam, c = st.ritz(1, 0)
        print(beta, 'eq check', lam - st.lam0, [mp.nstr(c[j] / c[0] * st.c0[0] - st.c0[j], 5) for j in range(6)])
        for (x, y) in [(0.5, 0.86), (0.3, 0.7), (0.1, 0.2)]:
            H = mp.root(3, 4); l = 2 / H
            a = mp.sqrt(2 / mp.mpf(y)) / l; b = mp.sqrt(2 / mp.mpf(y)) * (x - 0.5) / H
            lam, c = st.ritz(a, b)
            print('  ', (x, y), (float(a), float(b)), float(lam - st.lam0))

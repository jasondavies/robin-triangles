"""Float-only exploration tools (NOT rigorous).
P2 conforming FEM on the reference equilateral triangle E (area 1); a general area-1 triangle is
T = L(E), det L = 1, and its Robin form is pulled back to E:
   q[v] = int_E grad v . G grad v + alpha * sum_j ell_j int_{side j} v^2,   G = L^{-1} L^{-T},
   ell_j = |L tau_j| (ratio of side lengths).
Assemble Kxx,Kxy,Kyy,M,B_j once; each (shape,alpha) is then one sparse eigen-solve."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla

S_E = 2 / 3 ** 0.25            # side of the area-1 equilateral triangle
H_E = 3 ** 0.25                # its altitude
VE = np.array([[-S_E / 2, -H_E / 3], [S_E / 2, -H_E / 3], [0.0, 2 * H_E / 3]])   # centroid at 0

def p2mesh(n):
    """structured P2 mesh of E: nodes on the lattice of step 1/(2n)."""
    idx = {}; pts = []
    m = 2 * n
    for i in range(m + 1):
        for j in range(m + 1 - i):
            idx[(i, j)] = len(pts)
            pts.append(VE[0] + (VE[1] - VE[0]) * i / m + (VE[2] - VE[0]) * j / m)
    tris = []
    def el(a, b, c):
        mid = lambda u, v: ((u[0] + v[0]) // 2, (u[1] + v[1]) // 2)
        return [idx[a], idx[b], idx[c], idx[mid(a, b)], idx[mid(b, c)], idx[mid(c, a)]]
    for i in range(n):
        for j in range(n - i):
            tris.append(el((2 * i, 2 * j), (2 * i + 2, 2 * j), (2 * i, 2 * j + 2)))
            if i + j < n - 1:
                tris.append(el((2 * i + 2, 2 * j), (2 * i + 2, 2 * j + 2), (2 * i, 2 * j + 2)))
    # boundary edges per side: side 0: V0V1 (j=0), side 1: V1V2 (i+j=m), side 2: V2V0 (i=0)
    bed = [[], [], []]
    for i in range(n):
        bed[0].append([idx[(2 * i, 0)], idx[(2 * i + 2, 0)], idx[(2 * i + 1, 0)]])
        bed[1].append([idx[(m - 2 * i, 2 * i)], idx[(m - 2 * i - 2, 2 * i + 2)], idx[(m - 2 * i - 1, 2 * i + 1)]])
        bed[2].append([idx[(0, 2 * i)], idx[(0, 2 * i + 2)], idx[(0, 2 * i + 1)]])
    return np.array(pts), np.array(tris), [np.array(b) for b in bed]

def _p2ref():
    # 6-point degree-4 quadrature on reference triangle
    a1, b1, w1 = 0.445948490915965, 0.108103018168070, 0.223381589678011
    a2, b2, w2 = 0.091576213509771, 0.816847572980459, 0.109951743655322
    q = np.array([[a1, a1], [a1, b1], [b1, a1], [a2, a2], [a2, b2], [b2, a2]])
    w = np.array([w1, w1, w1, w2, w2, w2]) / 2
    x, y = q[:, 0], q[:, 1]; l0 = 1 - x - y
    N = np.stack([l0 * (2 * l0 - 1), x * (2 * x - 1), y * (2 * y - 1), 4 * l0 * x, 4 * x * y, 4 * y * l0], 1)
    dNx = np.stack([-(4 * l0 - 1), 4 * x - 1, 0 * x, 4 * (l0 - x), 4 * y, -4 * y], 1)
    dNy = np.stack([-(4 * l0 - 1), 0 * x, 4 * y - 1, -4 * x, 4 * x, 4 * (l0 - y)], 1)
    return w, N, dNx, dNy

class RefFEM:
    def __init__(self, n=24):
        p, t, bed = p2mesh(n)
        self.p, self.t, self.bed = p, t, bed
        N_ = len(p)
        w, N, dNx, dNy = _p2ref()
        P0, P1, P2 = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
        J = np.stack([P1 - P0, P2 - P0], 2)            # (ne,2,2) columns
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        inv = np.empty_like(J)
        inv[:, 0, 0] = J[:, 1, 1] / det; inv[:, 0, 1] = -J[:, 0, 1] / det
        inv[:, 1, 0] = -J[:, 1, 0] / det; inv[:, 1, 1] = J[:, 0, 0] / det
        # physical gradients: d/dx = inv[0,0] d/dxi + inv[1,0] d/deta ; d/dy = inv[0,1] d/dxi + inv[1,1] d/deta
        gx = inv[:, 0, 0][:, None, None] * dNx[None] + inv[:, 1, 0][:, None, None] * dNy[None]
        gy = inv[:, 0, 1][:, None, None] * dNx[None] + inv[:, 1, 1][:, None, None] * dNy[None]
        ad = np.abs(det)
        def asm(A, Bq):
            loc = np.einsum('q,eqa,eqb,e->eab', w, A, Bq, ad)
            I = np.repeat(t[:, :, None], 6, 2); Jx = np.repeat(t[:, None, :], 6, 1)
            return sp.csr_matrix((loc.ravel(), (I.ravel(), Jx.ravel())), shape=(N_, N_))
        Nq = np.broadcast_to(N[None], gx.shape)
        self.Kxx = asm(gx, gx); self.Kyy = asm(gy, gy)
        Kxy = asm(gx, gy); self.Kxy = Kxy + Kxy.T
        self.M = asm(Nq, Nq)
        # boundary mass (exact for P2: 3-node Simpson-type matrix)
        Bl = np.array([[4, -1, 2], [-1, 4, 2], [2, 2, 16]]) / 30.0
        self.B = []
        for b in bed:
            L = np.linalg.norm(p[b[:, 1]] - p[b[:, 0]], axis=1)
            loc = L[:, None, None] * Bl[None]
            I = np.repeat(b[:, :, None], 3, 2); Jx = np.repeat(b[:, None, :], 3, 1)
            self.B.append(sp.csr_matrix((loc.ravel(), (I.ravel(), Jx.ravel())), shape=(N_, N_)))
        self.tau = [(VE[1] - VE[0]) / S_E, (VE[2] - VE[1]) / S_E, (VE[0] - VE[2]) / S_E]

    def forms(self, L):
        Li = np.linalg.inv(L); G = Li @ Li.T
        ell = [np.linalg.norm(L @ tj) for tj in self.tau]
        return G, ell

    def eig(self, L, alpha, k=1, vec=False):
        """lowest k Robin eigenvalues of T = L(E), det L = 1."""
        G, ell = self.forms(L)
        K = G[0, 0] * self.Kxx + G[0, 1] * self.Kxy + G[1, 1] * self.Kyy
        for j in range(3):
            K = K + alpha * ell[j] * self.B[j]
        if alpha == np.inf:
            raise ValueError
        vals, vecs = spla.eigsh(K, k=k, M=self.M, sigma=-1e-3, which='LM')
        o = np.argsort(vals)
        return (vals[o], vecs[:, o]) if vec else vals[o]

def Lshape(t, phi):
    """L = exp(t * [[cos 2phi, sin 2phi],[sin 2phi, -cos 2phi]]) (symmetric, det 1)."""
    c, s = np.cosh(t), np.sinh(t)
    return np.array([[c + s * np.cos(2 * phi), s * np.sin(2 * phi)], [s * np.sin(2 * phi), c - s * np.cos(2 * phi)]])

def lam_equi(alpha):
    """exact lambda_1 of the area-1 equilateral triangle: kappa H/2 = atan(alpha/kappa)+atan(2 alpha/kappa)."""
    from scipy.optimize import brentq
    f = lambda k: k * H_E / 2 - np.arctan(alpha / k) - np.arctan(2 * alpha / k)
    k = brentq(f, 1e-12, 2 * np.pi / H_E)
    return k * k

if __name__ == '__main__':
    for n in (8, 16, 32):
        F = RefFEM(n)
        for al in (0.1, 1.0, 10.0, 100.0):
            print(n, al, F.eig(np.eye(2), al, k=3), lam_equi(al))

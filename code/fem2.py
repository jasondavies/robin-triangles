"""Referee's own P2 Lagrange FEM for the Robin Laplacian on a triangle:
   -Lap u = lam u in T, d_nu u + alpha u = 0  (alpha = -beta).
   Conforming => the computed lowest eigenvalue is a Rayleigh-Ritz upper bound for lambda_1(T).
   Mesh: uniform n x n subdivision of T, optionally graded towards all vertices (still conforming)."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla

def mesh(V, n, grade=1.0):
    """uniform barycentric grid with optional grading: nodes at barycentric (i/n, j/n) warped."""
    V = np.asarray(V, float)
    idx = {}; P = []
    def g(x):  # grading map on [0,1] towards both ends
        return x if grade == 1.0 else 0.5 * (1 + np.sign(2*x - 1) * np.abs(2*x - 1) ** grade) if False else x
    for i in range(n + 1):
        for j in range(n + 1 - i):
            idx[(i, j)] = len(P)
            P.append((i, j))
    lam = np.array([[i / n, j / n, 1 - i / n - j / n] for i, j in P])
    if grade != 1.0:
        # grading towards each vertex: w_k = lam_k^grade, renormalised (keeps edges on edges)
        lam = np.clip(lam, 0, None) ** grade; lam /= lam.sum(1, keepdims=True)
    X = lam @ V
    tris = []
    for i in range(n):
        for j in range(n - i):
            a, b, c = idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]
            tris.append((a, b, c))
            if i + j < n - 1:
                d = idx[(i + 1, j + 1)]
                tris.append((b, d, c))
    return X, np.array(tris)

def p2(X, tris):
    edges = {}; nodes = list(map(tuple, X)); T6 = []
    for t in tris:
        mids = []
        for a, b in ((t[1], t[2]), (t[2], t[0]), (t[0], t[1])):
            key = (min(a, b), max(a, b))
            if key not in edges:
                edges[key] = len(nodes); nodes.append(tuple((X[a] + X[b]) / 2))
            mids.append(edges[key])
        T6.append(list(t) + mids)
    # boundary edges: those used by one triangle
    cnt = {}
    for t in tris:
        for a, b in ((t[1], t[2]), (t[2], t[0]), (t[0], t[1])):
            key = (min(a, b), max(a, b)); cnt[key] = cnt.get(key, 0) + 1
    bnd = [(a, edges[(a, b)], b) for (a, b), c in cnt.items() if c == 1]
    return np.array(nodes), np.array(T6), bnd

# quadrature on reference triangle (degree 5, 7 points)
_a1, _b1 = 0.059715871789770, 0.470142064105115
_a2, _b2 = 0.797426985353087, 0.101286507323456
QP = np.array([[1/3, 1/3], [_b1, _b1], [_a1, _b1], [_b1, _a1], [_b2, _b2], [_a2, _b2], [_b2, _a2]])
QW = np.array([0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506,
               0.125939180544827, 0.125939180544827, 0.125939180544827]) / 2

def shape(x, y):
    l1, l2 = x, y; l0 = 1 - x - y
    N = np.array([l0*(2*l0-1), l1*(2*l1-1), l2*(2*l2-1), 4*l1*l2, 4*l2*l0, 4*l0*l1])
    dl = np.array([[-1, -1], [1, 0], [0, 1]], float)
    L = [l0, l1, l2]
    dN = np.array([(4*L[0]-1)*dl[0], (4*L[1]-1)*dl[1], (4*L[2]-1)*dl[2],
                   4*(L[1]*dl[2] + L[2]*dl[1]), 4*(L[2]*dl[0] + L[0]*dl[2]), 4*(L[0]*dl[1] + L[1]*dl[0])])
    return N, dN

SH = [shape(*q) for q in QP]

def assemble(nodes, T6, bnd):
    n = len(nodes); I = []; J = []; Kv = []; Mv = []
    for t in T6:
        p = nodes[t[:3]]
        Jm = np.array([p[1] - p[0], p[2] - p[0]]).T; det = abs(np.linalg.det(Jm)); Ji = np.linalg.inv(Jm)
        Ke = np.zeros((6, 6)); Me = np.zeros((6, 6))
        for (N, dN), w in zip(SH, QW):
            G = dN @ Ji
            Ke += w * det * G @ G.T; Me += w * det * np.outer(N, N)
        I.append(np.repeat(t, 6)); J.append(np.tile(t, 6)); Kv.append(Ke.ravel()); Mv.append(Me.ravel())
    I = np.concatenate(I); J = np.concatenate(J)
    K = sp.csr_matrix((np.concatenate(Kv), (I, J)), shape=(n, n))
    M = sp.csr_matrix((np.concatenate(Mv), (I, J)), shape=(n, n))
    # boundary mass, exact for quadratics squared (deg 4) with 3-pt Gauss
    gx, gw = np.polynomial.legendre.leggauss(3); gx = (gx + 1) / 2; gw = gw / 2
    bI = []; bJ = []; bV = []
    for a, m, b in bnd:
        ln = np.linalg.norm(nodes[a] - nodes[b])
        Be = np.zeros((3, 3))
        for x, w in zip(gx, gw):
            N = np.array([(1-x)*(1-2*x), 4*x*(1-x), x*(2*x-1)])
            Be += w * ln * np.outer(N, N)
        e = [a, m, b]; bI.append(np.repeat(e, 3)); bJ.append(np.tile(e, 3)); bV.append(Be.ravel())
    Bm = sp.csr_matrix((np.concatenate(bV), (np.concatenate(bI), np.concatenate(bJ))), shape=(n, n))
    return K, M, Bm

def lam1(V, beta, n=40, grade=1.0, sigma=None):
    X, tris = mesh(V, n, grade)
    nodes, T6, bnd = p2(X, tris)
    K, M, Bm = assemble(nodes, T6, bnd)
    A = (K - beta * Bm).tocsc()
    if sigma is None:
        # lower bound for the shift: lambda >= -beta^2/sin^2(theta_min/2) * const ; use Gershgorin-ish safe guess
        th = angles(V); sigma = -3 * beta**2 / np.sin(th.min()/2)**2 - 10
    vals = spla.eigsh(A, k=1, M=M.tocsc(), sigma=sigma, which='LM', return_eigenvectors=False)
    return vals.min()

def angles(V):
    V = np.asarray(V, float); th = []
    for i in range(3):
        a = V[(i + 1) % 3] - V[i]; b = V[(i + 2) % 3] - V[i]
        th.append(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
    return np.array(th)

def tri_from_angles(th):
    th = np.asarray(th, float); a = np.sin(th); c = a[2]; b = a[1]
    V = np.array([[0, 0], [c, 0], [b * np.cos(th[0]), b * np.sin(th[0])]])
    area = 0.5 * abs((V[1,0]-V[0,0])*(V[2,1]-V[0,1]) - (V[1,1]-V[0,1])*(V[2,0]-V[0,0]))
    return V / np.sqrt(area)

H = 3 ** 0.25
def lam0_of_beta(beta):
    from scipy.optimize import brentq
    f = lambda M: np.tanh(M) * (M + np.arctanh(np.tanh(M) / 2)) / H - beta
    M = brentq(f, 1e-9, 60); t = np.tanh(M)
    return -4 * beta**2 / t**2, M

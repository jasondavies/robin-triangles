"""Float-only exploration: P1 / P2 conforming FEM for the lowest Robin eigenvalue on a triangle.
Conforming Galerkin => each discrete value is an upper bound (up to float error); NOT rigorous.
Structured mesh: n^2 congruent subtriangles of the triangle with vertices V0,V1,V2."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla

def mesh(V, n):
    V = np.asarray(V, float)
    idx = {}
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            idx[(i, j)] = len(pts)
            pts.append(V[0] + (V[1] - V[0]) * i / n + (V[2] - V[0]) * j / n)
    tris = []
    for i in range(n):
        for j in range(n - i):
            tris.append((idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]))
            if i + j < n - 1:
                tris.append((idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
    bedges = []
    for i in range(n):
        bedges.append((idx[(i, 0)], idx[(i + 1, 0)]))
        bedges.append((idx[(0, i)], idx[(0, i + 1)]))
        bedges.append((idx[(n - i, i)], idx[(n - i - 1, i + 1)]))
    return np.array(pts), np.array(tris), np.array(bedges)

def lam1(V, alpha, n=120, k=1):
    p, t, be = mesh(V, n)
    N = len(p)
    rows, cols, Kv, Mv = [], [], [], []
    for tri in t:
        P = p[tri]
        Bm = np.array([P[1] - P[0], P[2] - P[0]]).T
        area = abs(np.linalg.det(Bm)) / 2
        G = np.linalg.inv(Bm).T @ np.array([[-1, 1, 0], [-1, 0, 1]], float)
        Ke = area * G.T @ G
        Me = area / 12 * (np.ones((3, 3)) + np.eye(3))
        for a in range(3):
            for b in range(3):
                rows.append(tri[a]); cols.append(tri[b]); Kv.append(Ke[a, b]); Mv.append(Me[a, b])
    for e in be:
        L = np.linalg.norm(p[e[1]] - p[e[0]])
        Be = alpha * L / 6 * np.array([[2, 1], [1, 2]])
        for a in range(2):
            for b in range(2):
                rows.append(e[a]); cols.append(e[b]); Kv.append(Be[a, b]); Mv.append(0.0)
    K = sp.csr_matrix((Kv, (rows, cols)), shape=(N, N))
    M = sp.csr_matrix((Mv, (rows, cols)), shape=(N, N))
    # lowest eigenvalue: shift-invert around a guess below
    sigma = -10 * (abs(alpha) ** 2 * 400 + 1)
    w = spla.eigsh(K, k=k, M=M, sigma=sigma, which='LM', return_eigenvectors=False)
    return np.sort(w)

if __name__ == '__main__':
    import sys
    s = 3 ** 0.25
    # equilateral of area 1: side l with sqrt3/4 l^2 = 1
    l = 2 / s
    V = [(0, 0), (l, 0), (l / 2, l * np.sqrt(3) / 2)]
    for n in (40, 80, 160):
        print(n, lam1(V, -1.0, n))

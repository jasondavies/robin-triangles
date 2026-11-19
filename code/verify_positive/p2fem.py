"""Sanity cross-check only (floating point, NOT a proof step): conforming P2 finite elements
on a uniform mesh give a Rayleigh-Ritz upper bound for lambda_1 of a Robin triangle."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import geom

_MASS = np.array([[6, -1, -1, 0, -4, 0], [-1, 6, -1, 0, 0, -4], [-1, -1, 6, -4, 0, 0],
                  [0, 0, -4, 32, 16, 16], [-4, 0, 0, 16, 32, 16], [0, -4, 0, 16, 16, 32]], float) / 180.0
_EDGE = np.array([[4, 2, -1], [2, 16, 2], [-1, 2, 4]], float) / 30.0


def p2_lambda1(tri, m=40):
    (p, q, r), ap, x2, y2 = geom.place(tri, 0)
    P1 = np.array([float(ap.mid), 0.0])
    P2 = np.array([float(x2.mid), float(y2.mid)])
    beta = {"j0": float(tri.beta[p].mid), "i0": float(tri.beta[q].mid), "hyp": float(tri.beta[r].mid)}
    M2 = 2 * m

    def nid(I, J):
        return J * (M2 + 1) - (J * (J - 1)) // 2 + I

    nn = (M2 + 1) * (M2 + 2) // 2
    I, J = np.meshgrid(np.arange(m + 1), np.arange(m + 1), indexing="ij")
    up = (I + J) <= m - 1
    iu, ju = 2 * I[up], 2 * J[up]
    V_up = [(iu, ju), (iu + 2, ju), (iu, ju + 2)]
    dn = (I + J) <= m - 2
    idn, jdn = 2 * I[dn], 2 * J[dn]
    V_dn = [(idn + 2, jdn), (idn + 2, jdn + 2), (idn, jdn + 2)]
    VI = [np.concatenate([V_up[k][0], V_dn[k][0]]) for k in range(3)]
    VJ = [np.concatenate([V_up[k][1], V_dn[k][1]]) for k in range(3)]
    ne = len(VI[0])
    # local nodes: v0, v1, v2, m01, m12, m20
    LI = VI + [(VI[0] + VI[1]) // 2, (VI[1] + VI[2]) // 2, (VI[2] + VI[0]) // 2]
    LJ = VJ + [(VJ[0] + VJ[1]) // 2, (VJ[1] + VJ[2]) // 2, (VJ[2] + VJ[0]) // 2]
    G = np.stack([nid(LI[k], LJ[k]) for k in range(6)], 1)              # (ne,6)
    X = np.stack([np.outer(VI[k], P1) / M2 + np.outer(VJ[k], P2) / M2 for k in range(3)], 1)   # (ne,3,2)
    d1 = X[:, 1] - X[:, 0]
    d2 = X[:, 2] - X[:, 0]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    area = 0.5 * det
    # gradients of barycentric coordinates
    g = np.empty((ne, 3, 2))
    g[:, 1, 0] = d2[:, 1] / det
    g[:, 1, 1] = -d2[:, 0] / det
    g[:, 2, 0] = -d1[:, 1] / det
    g[:, 2, 1] = d1[:, 0] / det
    g[:, 0] = -g[:, 1] - g[:, 2]
    Ke = np.zeros((ne, 6, 6))
    pairs = [(0, 1), (1, 2), (2, 0)]
    for qp in ([0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]):
        mu = np.array(qp)
        B = np.empty((ne, 6, 2))
        for k in range(3):
            B[:, k] = (4 * mu[k] - 1) * g[:, k]
        for t, (a, b) in enumerate(pairs):
            B[:, 3 + t] = 4 * (mu[a] * g[:, b] + mu[b] * g[:, a])
        Ke += np.einsum("eia,eja->eij", B, B) * (area / 3.0)[:, None, None]
    Me = _MASS[None] * area[:, None, None]
    rows = np.repeat(G, 6, axis=1).ravel()
    cols = np.tile(G, (1, 6)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(nn, nn)).tocsr()
    Mm = sp.coo_matrix((Me.ravel(), (rows, cols)), shape=(nn, nn)).tocsr()
    # boundary edges
    br, bc, bv = [], [], []
    for t, (a, b) in enumerate(pairs):
        Ia, Ja, Ib, Jb = LI[a], LJ[a], LI[b], LJ[b]
        for name, cond in (("j0", (Ja == 0) & (Jb == 0)), ("i0", (Ia == 0) & (Ib == 0)),
                           ("hyp", (Ia + Ja == M2) & (Ib + Jb == M2))):
            sel = np.where(cond)[0]
            if len(sel) == 0:
                continue
            ln = np.linalg.norm(X[sel, b] - X[sel, a], axis=1)
            loc = G[sel][:, [a, 3 + t, b]]
            E = _EDGE[None] * (beta[name] * ln)[:, None, None]
            br.append(np.repeat(loc, 3, axis=1).ravel())
            bc.append(np.tile(loc, (1, 3)).ravel())
            bv.append(E.ravel())
    K = K + sp.coo_matrix((np.concatenate(bv), (np.concatenate(br), np.concatenate(bc))), shape=(nn, nn)).tocsr()
    vals = spla.eigsh(K.tocsc(), k=1, M=Mm.tocsc(), sigma=0.0, which="LM", return_eigenvectors=False)
    return float(vals[0])

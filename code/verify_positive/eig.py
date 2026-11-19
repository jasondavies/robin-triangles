"""Floating-point (untrusted) computation of the discrete ground state."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla

import crmesh


def coarse_estimate(tri, kind, n, nc=6):
    """lowest CR eigenvalue on a coarse mesh of the same kind (dense, float); only a shift proposal"""
    nc = min(n, nc)
    m = crmesh.build_mesh(tri, kind, nc)
    c, X, bl, _ = crmesh.element_geometry(m)
    W, diag, mass = crmesh.assemble_float(m, c, X, bl)
    K = np.diag(diag) - W.toarray()
    s = 1.0 / np.sqrt(mass)
    ev = sla.eigvalsh(K * s[:, None] * s[None, :])
    return float(ev[0])


def ground_state(W, diag, mass, lam_est, sweeps=300, maxit=200, info=None):
    N = len(diag)
    s = 0.97 * lam_est
    A = (sp.diags(diag - s * mass) - W).tocsc()
    lu = spla.splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    x = np.ones(N)
    nit = 0
    extra = 0
    for it in range(maxit):
        y = lu.solve(mass * x)
        y = y / np.max(np.abs(y))
        if y[np.argmax(np.abs(y))] < 0:
            y = -y
        chg = np.max(np.abs(y - x) / np.maximum(np.abs(y), 1e-300))
        x = y
        nit += 1
        if chg < 1e-11:
            extra += 1
            if extra >= 2:
                break
    x = np.abs(x)
    Kx = diag * x - W @ x
    lam = float(x @ Kx) / float(x @ (mass * x))
    D = diag - lam * mass
    if np.all(D > 0):
        for _ in range(sweeps):
            x = 0.5 * x + 0.5 * (W @ x) / D
    if info is not None:
        info["inv_iters"] = nit
        info["lam_float"] = lam
        info["shift"] = s
    return x, lam

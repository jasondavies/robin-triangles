"""V1: independent re-verification of one Crouzeix-Raviart lower bound record."""
import time
from fractions import Fraction as Fr
import numpy as np

import geom
import crmesh
import eig


def verify_cr(tri, kind, n, sigma, lb, kappa0, N_expected=None, sweeps=300, x_override=None, want_x=False):
    """sigma, lb, kappa0 exact Fractions.  Returns dict; 'ok' is True only if everything is certified."""
    t0 = time.time()
    out = {"ok": False}
    try:
        out["area_consistent"] = bool(tri.area_consistent())
        m = crmesh.build_mesh(tri, kind, n)
        c, X, bl, _ = crmesh.element_geometry(m)
        W, diag, mass = crmesh.assemble_float(m, c, X, bl)
        out["N"] = int(m.N)
        out["N_matches"] = (N_expected is None) or (int(N_expected) == int(m.N))
        info = {}
        if x_override is None:
            est = eig.coarse_estimate(tri, kind, n)
            x, lam = eig.ground_state(W, diag, mass, est, sweeps=sweeps, info=info)
        else:
            x = x_override
            lam = None
        out.update(info)
        q, flags, area = crmesh.cw_certificate(m, c, X, bl, x)
        out["flags"] = flags
        out["min_cot2"] = float(np.min(c.lo))
        if q is None or not np.isfinite(q):
            out["reason"] = "CW quotient not certified (Kx has a row not certified >= 0)"
            out["cw_ok"] = False
        else:
            out["sigma_cert"] = q
            # exact comparison of the certified quotient with the stored sigma
            out["cw_ok"] = bool(Fr(q) >= sigma and sigma > 0)
            out["cw_rel_margin"] = float((Fr(q) - sigma) / sigma)
        if area is not None:
            a_lo, a_hi = geom.iv_to_float_bounds(tri.area_stated)
            out["area_sum_consistent"] = bool(area[0] <= a_hi and a_lo <= area[1])
        # lower bound: lb <= sigma / (1 + kappa0^2 h^2 sigma), exact rationals, h^2 <= h2_hi
        bound = sigma / (1 + kappa0 * kappa0 * m.h2_hi * sigma)
        out["lb_ok"] = bool(0 < lb <= bound)
        out["lb_rel_slack"] = float((bound - lb) / bound)
        out["h"] = float(m.h2_hi) ** 0.5
        out["ok"] = bool(out.get("cw_ok") and out["lb_ok"] and all(flags.values()) and out["N_matches"]
                         and out["area_consistent"] and out.get("area_sum_consistent", False))
        if want_x:
            out["_x"] = x
            out["_mesh"] = (m, c, X, bl)
    except Exception as ex:  # any failure of a certified step = record not verified
        out["reason"] = "exception: %r" % (ex,)
        out["ok"] = False
    out["sec"] = round(time.time() - t0, 3)
    return out


def tri_from_ring_record(r):
    k, sign, n = r["key"]
    alpha = geom.parse_rat(r["alpha"])
    s = geom.parse_rat(r["s"])
    return geom.tri_ring(int(sign), s, alpha), alpha


def tri_from_node_record(r):
    w = geom.parse_w(r["w"])
    alpha = geom.parse_rat(r["alpha"])
    return geom.tri_cover(w, alpha), alpha

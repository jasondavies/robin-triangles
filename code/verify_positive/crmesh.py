"""Crouzeix-Raviart meshes built from explicit vertex coordinates, and the
Collatz-Wielandt certificate in vectorised interval arithmetic.

Mesh construction (own implementation, from coordinates):
  the triangle is placed with one side on the x axis, P0=(0,0), P1=(a_p,0), P2=(x2,y2);
  'U': vertices P0 + (i/n)(P1-P0) + (j/n)(P2-P0), i+j<=n; elements (i,j),(i+1,j),(i,j+1)
       and (i+1,j),(i+1,j+1),(i,j+1);
  'A': base = longest side, F=(x2,0) foot of the altitude; two right triangles
       (F,P0,P2) and (F,P1,P2), each with the lattice F + i e1 + j e2,
       e1 = (P0-F)/n resp. (P1-F)/n (parallel to the x axis, second coordinate exactly 0),
       e2 = (P2-F)/n (parallel to the y axis, first coordinate exactly 0).
Every element is stored with its three global vertex numbers and the integer lattice
coordinates of its vertices; the edge vectors of an element are the differences of its
vertex coordinates, formed as (di)*e1 + (dj)*e2 with exact small integers di, dj so that
no interval dependency/cancellation occurs.  All element matrices are then computed
element by element from these edge vectors (dot and cross products).
"""
import numpy as np
from fractions import Fraction as Fr
from mpmath import iv
import scipy.sparse as sp

from ivnp import IV, where as ivwhere
import geom


class Mesh:
    pass


def _lattice_elements(n, ccw):
    """elements of the n^2-subdivision of a lattice triangle: integer coordinates (ne,3,2)"""
    I, J = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    up = (I + J) <= n - 1
    iu, ju = I[up], J[up]
    eu = np.stack([np.stack([iu, ju], 1), np.stack([iu + 1, ju], 1), np.stack([iu, ju + 1], 1)], 1)
    dnm = (I + J) <= n - 2
    idn, jdn = I[dnm], J[dnm]
    ed = np.stack([np.stack([idn + 1, jdn], 1), np.stack([idn + 1, jdn + 1], 1), np.stack([idn, jdn + 1], 1)], 1)
    el = np.concatenate([eu, ed], 0)
    if not ccw:
        el = el[:, [0, 2, 1], :]
    return el


def build_mesh(tri, kind, n):
    """Returns a Mesh with topology (integers) and geometry (mpmath intervals + float intervals)."""
    m = Mesh()
    m.kind = kind
    m.n = n
    if kind == "U":
        (p, q, r), ap, x2, y2 = geom.place(tri, 0)
        zero = iv.mpf(0)
        lat = [dict(o=(zero, zero), e1=(ap / n, zero), e2=(x2 / n, y2 / n), ccw=True,
                    side_j0=p, side_i0=q, side_hyp=r)]
        m.P = [(zero, zero), (ap, zero), (x2, y2)]
    elif kind == "A":
        base = geom.certified_argmax(tri.a2)
        if base is None:
            raise ValueError("longest side not decidable")
        (p, q, r), ap, x2, y2 = geom.place(tri, base)
        if not (x2.a > 0 and (ap - x2).a > 0):
            raise ValueError("foot of altitude not certified inside the base")
        zero = iv.mpf(0)
        # lattice 0: F, P0, P2 (contains side q as hypotenuse); e1 points in -x direction: clockwise
        # lattice 1: F, P1, P2 (contains side r as hypotenuse)
        lat = [dict(o=(x2, zero), e1=(-x2 / n, zero), e2=(zero, y2 / n), ccw=False,
                    side_j0=p, side_i0=None, side_hyp=q),
               dict(o=(x2, zero), e1=((ap - x2) / n, zero), e2=(zero, y2 / n), ccw=True,
                    side_j0=p, side_i0=None, side_hyp=r)]
        m.P = [(zero, zero), (ap, zero), (x2, y2)]
    else:
        raise ValueError("unknown mesh kind")
    m.lat = lat
    nl = len(lat)
    els = []
    ell = []
    for li, L in enumerate(lat):
        e = _lattice_elements(n, L["ccw"])
        els.append(e)
        ell.append(np.full(len(e), li))
    m.elem_ij = np.concatenate(els, 0)            # (ne,3,2)
    m.elem_lat = np.concatenate(ell, 0)           # (ne,)
    ne = len(m.elem_lat)
    # global vertex numbers; in 'A' the vertices with i=0 (on the altitude) of both lattices coincide
    latv = np.where(m.elem_ij[:, :, 0] == 0, 0, m.elem_lat[:, None]) if kind == "A" else m.elem_lat[:, None] * np.ones((1, 3), int)
    vkey = (latv * (n + 1) + m.elem_ij[:, :, 0]) * (n + 1) + m.elem_ij[:, :, 1]
    uv, inv = np.unique(vkey.ravel(), return_inverse=True)
    m.elem_v = inv.reshape(ne, 3)
    m.nv = len(uv)
    # global edges: local edge k is opposite to local vertex k, from vertex k+1 to vertex k+2
    a = m.elem_v[:, [1, 2, 0]]
    b = m.elem_v[:, [2, 0, 1]]
    ekey = np.minimum(a, b) * m.nv + np.maximum(a, b)
    ue, einv, cnt = np.unique(ekey.ravel(), return_inverse=True, return_counts=True)
    m.elem_e = einv.reshape(ne, 3)
    m.N = len(ue)
    m.edge_mult = cnt
    # lattice differences of edge vectors d_k = P_{k+2} - P_{k+1}
    m.dij = m.elem_ij[:, [2, 0, 1], :] - m.elem_ij[:, [1, 2, 0], :]       # (ne,3,2)
    # which side of the big triangle a local edge lies on (exact integer tests)
    ia = m.elem_ij[:, [1, 2, 0], 0]
    ja = m.elem_ij[:, [1, 2, 0], 1]
    ib = m.elem_ij[:, [2, 0, 1], 0]
    jb = m.elem_ij[:, [2, 0, 1], 1]
    on_j0 = (ja == 0) & (jb == 0)
    on_i0 = (ia == 0) & (ib == 0)
    on_hyp = (ia + ja == n) & (ib + jb == n)
    side = np.full((ne, 3), -1)
    alt = np.zeros((ne, 3), bool)
    for li, L in enumerate(lat):
        sel = (m.elem_lat == li)[:, None]
        side = np.where(sel & on_j0, L["side_j0"], side)
        side = np.where(sel & on_hyp, L["side_hyp"], side)
        if L["side_i0"] is None:
            alt |= sel & on_i0
        else:
            side = np.where(sel & on_i0, L["side_i0"], side)
    m.loc_side = side                 # -1: not on the boundary of the big triangle
    # conformity / topology checks
    mult = m.edge_mult[m.elem_e]
    topo_ok = bool(np.all((mult == 1) | (mult == 2)))
    topo_ok &= bool(np.all((mult == 1) == (side >= 0)))
    topo_ok &= bool(np.all(mult[alt] == 2))
    # no flag ambiguity: an edge cannot lie on two sides
    topo_ok &= bool(np.all((on_j0.astype(int) + on_i0.astype(int) + on_hyp.astype(int)) <= 1))
    nb_edges = int(np.sum(m.edge_mult == 1))
    topo_ok &= nb_edges == (3 * n if kind == "U" else 4 * n)
    topo_ok &= ne == nl * n * n
    # Euler: V - E + F = 1 for a triangulated disc
    topo_ok &= (m.nv - m.N + ne == 1)
    m.topo_ok = topo_ok
    # edge -> (element, local index) slots
    order = np.argsort(m.elem_e.ravel(), kind="stable")
    flat_e = m.elem_e.ravel()[order]
    first = np.concatenate([[True], flat_e[1:] != flat_e[:-1]])
    slot = np.where(first, 0, 1)
    m.edge_elem = np.full((m.N, 2), -1)
    m.edge_loc = np.zeros((m.N, 2), int)
    m.edge_elem[flat_e, slot] = order // 3
    m.edge_loc[flat_e, slot] = order % 3
    # distinct edge types (lattice, di, dj) -> squared lengths in high precision: element diameters
    types = np.unique(np.concatenate([np.repeat(m.elem_lat, 3)[:, None], m.dij.reshape(-1, 2)], 1), axis=0)
    h2_hi = None
    for li, di, dj in types:
        L = lat[int(li)]
        dx = int(di) * L["e1"][0] + int(dj) * L["e2"][0]
        dy = int(di) * L["e1"][1] + int(dj) * L["e2"][1]
        l2 = geom.iv_hi_fr(dx ** 2 + dy ** 2)
        h2_hi = l2 if (h2_hi is None or l2 > h2_hi) else h2_hi
    m.h2_hi = h2_hi                 # exact rational upper bound for (largest element diameter)^2
    m.n_edge_types = len(types)
    m.beta = tri.beta
    return m


def _ivs(x):
    lo, hi = geom.iv_to_float_bounds(x)
    return lo, hi


def element_geometry(m):
    """Per-element quantities as float intervals, computed from the element's vertex coordinates:
    c[e,k]  = -(d_a . d_b)/|K| for the two edges a,b other than k   (= 2 cot of the angle at vertex k)
    X[e]    = 2|K|  (cross product of consecutive edge vectors)
    blen[e,k] = beta_side * |d_k| for boundary edges, exact 0 otherwise."""
    nl = len(m.lat)
    comp = {}
    for name, idx in (("e1x", ("e1", 0)), ("e1y", ("e1", 1)), ("e2x", ("e2", 0)), ("e2y", ("e2", 1))):
        lo = np.empty(nl)
        hi = np.empty(nl)
        for li, L in enumerate(m.lat):
            lo[li], hi[li] = _ivs(L[idx[0]][idx[1]])
        comp[name] = IV(lo, hi)
    el = m.elem_lat
    di = IV(m.dij[:, :, 0].astype(float))
    dj = IV(m.dij[:, :, 1].astype(float))
    dx = di * comp["e1x"][el][:, None] + dj * comp["e2x"][el][:, None]     # (ne,3)
    dy = di * comp["e1y"][el][:, None] + dj * comp["e2y"][el][:, None]
    # 2|K| = cross(d_0, d_1) for counter-clockwise elements
    X = dx[:, 0] * dy[:, 1] - dy[:, 0] * dx[:, 1]
    if not np.all(X.lo > 0):
        raise ArithmeticError("element orientation/area not certified positive")
    c_lo = np.empty((len(el), 3))
    c_hi = np.empty((len(el), 3))
    for k in range(3):
        a = (k + 1) % 3
        b = (k + 2) % 3
        dot = dx[:, a] * dx[:, b] + dy[:, a] * dy[:, b]
        # stiffness entry S_ab = int grad(phi_a).grad(phi_b) = (d_a.d_b)/|K| = 2 (d_a.d_b)/X ;  c = -S_ab
        ck = (-(dot + dot)).div_pos(X)
        c_lo[:, k] = ck.lo
        c_hi[:, k] = ck.hi
    c = IV(c_lo, c_hi)
    # boundary term beta_side * |edge|
    blo = np.array([_ivs(b)[0] for b in m.beta])
    bhi = np.array([_ivs(b)[1] for b in m.beta])
    sd = m.loc_side
    onb = sd >= 0
    sdc = np.where(onb, sd, 0)
    length = (dx * dx + dy * dy).sqrt()
    bl = length * IV(blo[sdc], bhi[sdc])
    bl = ivwhere(onb, bl, IV(np.zeros(sd.shape)))
    return c, X, bl, length


def assemble_float(m, c, X, bl):
    """float CSR stiffness(+Robin) matrix and diagonal mass from interval midpoints (not trusted)."""
    cm = c.mid()
    N = m.N
    rows = []
    cols = []
    vals = []
    diag = np.zeros(N)
    for k in range(3):
        a = (k + 1) % 3
        b = (k + 2) % 3
        ea = m.elem_e[:, a]
        eb = m.elem_e[:, b]
        rows += [ea, eb]
        cols += [eb, ea]
        vals += [-cm[:, k], -cm[:, k]]
        np.add.at(diag, ea, cm[:, k])
        np.add.at(diag, eb, cm[:, k])
    blm = bl.mid()
    np.add.at(diag, m.elem_e.ravel(), blm.ravel())
    W = sp.csr_matrix((-np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    mass = np.zeros(N)
    Xm = X.mid()
    for k in range(3):
        np.add.at(mass, m.elem_e[:, k], Xm / 6.0)
    return W, diag, mass          # K = diag(diag) - W,  W >= 0 off-diagonal weights


def cw_certificate(m, c, X, bl, x):
    """Rigorous Collatz-Wielandt quotient: returns a float q with
         (K x)_i >= q (M x)_i  for all i   (certified),   plus the certified flags.
    Uses (K x)_i = sum_{elements K containing edge i} sum_{j != i in K} c_ij (x_i - x_j) + beta|e_i| x_i
    (zero row sums of the element stiffness matrices), all in interval arithmetic."""
    N = m.N
    x = np.asarray(x, dtype=np.float64)
    flags = {}
    flags["x_positive"] = bool(np.all(x > 0) and np.all(np.isfinite(x)))
    flags["offdiag_nonpositive"] = bool(np.all(c.lo >= 0))       # all couplings -c <= 0
    flags["areas_positive"] = bool(np.all(X.lo > 0))
    flags["topology"] = bool(m.topo_ok)
    xi = IV(x)
    acc = IV(np.zeros(N))
    massacc = IV(np.zeros(N))
    bacc = IV(np.zeros(N))
    zero = IV(np.zeros(N))
    for s in range(2):
        el = m.edge_elem[:, s]
        valid = el >= 0
        el0 = np.where(valid, el, 0)
        k = m.edge_loc[:, s]
        for t in (1, 2):
            l = (k + t) % 3
            third = 3 - k - l
            coef = IV(c.lo[el0, third], c.hi[el0, third])
            nb = m.elem_e[el0, l]
            term = coef * (xi - IV(x[nb]))
            acc = acc + ivwhere(valid, term, zero)
        massacc = massacc + ivwhere(valid, IV(X.lo[el0], X.hi[el0]), zero)
        bacc = bacc + ivwhere(valid, IV(bl.lo[el0, k], bl.hi[el0, k]), zero)
    Kx = acc + bacc * xi
    Mx = massacc.div_pos(IV(np.full(1, 6.0))) * xi
    flags["mass_positive"] = bool(np.all(Mx.lo > 0))
    if not (flags["mass_positive"] and flags["x_positive"]):
        return None, flags, None
    q = Kx.lo / Mx.hi
    q = np.where(Kx.lo >= 0, np.nextafter(q, -np.inf), -np.inf)
    # total area (interval) for the consistency check
    area = IV(np.zeros(1))
    # summing ne intervals sequentially would be slow in python; bound via min/max instead
    area_lo = float(np.sum(X.lo)) * 0.5 * (1 - 1e-9)
    area_hi = float(np.sum(X.hi)) * 0.5 * (1 + 1e-9)
    return float(np.min(q)), flags, (area_lo, area_hi)

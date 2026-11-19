"""V4: the cover of the middle region (Lemma 'coverage', Proposition 'middle', Corollary 'prisms')."""
import json
import os
import glob
from fractions import Fraction as Fr
from mpmath import iv

import geom
import lam0 as L0

RSTAR = Fr(1194, 10000)        # r_* = 0.1194
KMAX = 400


# ---------------------------------------------------------------- exact planar predicates
def orient(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def F(w):
    return (1 + w[0]) ** 2 - 81 * (1 - w[0] ** 2 - w[1] ** 2)


def ge_sqrt3(y, x):
    """exact test y >= sqrt(3) x for rationals"""
    if x <= 0:
        return y >= 0 or y * y <= 3 * x * x
    return y > 0 and y * y >= 3 * x * x


def lt_sqrt3(y, x):
    return not ge_sqrt3(y, x)


def min_F_on_segment(p, q):
    """exact minimum of the convex quadratic F on the segment [p,q]"""
    # F(p + s(q-p)) = A s^2 + B s + C
    d = (q[0] - p[0], q[1] - p[1])
    A = d[0] ** 2 + 81 * (d[0] ** 2 + d[1] ** 2)
    B = 2 * (1 + p[0]) * d[0] + 162 * (p[0] * d[0] + p[1] * d[1])
    C = F(p)
    assert A * 1 + B * 1 + C == F(q)
    cands = [C, F(q)]
    if A > 0:
        s = -B / (2 * A)
        if 0 < s < 1:
            cands.append(A * s * s + B * s + C)
    return min(cands)


def in_closed_triangle(x, tri):
    o = [orient(tri[i], tri[(i + 1) % 3], x) for i in range(3)]
    return all(v >= 0 for v in o)


def in_open_triangle(x, tri):
    o = [orient(tri[i], tri[(i + 1) % 3], x) for i in range(3)]
    return all(v > 0 for v in o)


def min_F_on_triangle(tri):
    """exact minimum of F over a closed triangle (F strictly convex, minimiser (-1/82, 0))"""
    c = (Fr(-1, 82), Fr(0))
    if in_closed_triangle(c, tri):
        return F(c)
    return min(min_F_on_segment(tri[i], tri[(i + 1) % 3]) for i in range(3))


def boundary_kind(p, q):
    """one of the four closed sets of Proposition 'middle'(a) containing the segment [p,q], or None"""
    r2 = RSTAR * RSTAR
    if p[0] ** 2 + p[1] ** 2 <= r2 and q[0] ** 2 + q[1] ** 2 <= r2:
        return "disc"
    if p[1] <= 0 and q[1] <= 0:
        return "bottom"
    if ge_sqrt3(p[1], p[0]) and ge_sqrt3(q[1], q[0]):
        return "top"
    if min_F_on_segment(p, q) >= 0:
        return "thin"
    return None


def side_matching(tris):
    """hypotheses of Lemma 'coverage' on orientation and sides.  Returns (ok, list of boundary sides)."""
    for t in tris:
        if not orient(*t) > 0:
            return False, []
    cnt = {}
    for t in tris:
        for i in range(3):
            key = (t[i], t[(i + 1) % 3])
            cnt[key] = cnt.get(key, 0) + 1
    bnd = []
    for (p, q), c in cnt.items():
        rev = cnt.get((q, p), 0)
        if c == 1 and rev == 1:
            continue
        if c == 1 and rev == 0:
            bnd.append((p, q))
            continue
        return False, []
    return True, bnd


def in_omega_B(x):
    return (x[1] > 0 and lt_sqrt3(x[1], x[0]) and x[0] ** 2 + x[1] ** 2 > RSTAR ** 2
            and x[0] ** 2 + x[1] ** 2 < 1 and F(x) < 0)


def check_roots(roots, point=(Fr(1, 2), Fr(1, 5))):
    """Proposition 'middle'(a): exact rational check."""
    out = {"ok": False, "roots": len(roots)}
    tris = [tuple(geom.parse_w(v) for v in t) for t in roots]
    out["in_unit_disc"] = all(v[0] ** 2 + v[1] ** 2 < 1 for t in tris for v in t)
    ok, bnd = side_matching(tris)
    out["side_matching"] = ok
    kinds = {}
    bad = []
    for (p, q) in bnd:
        kd = boundary_kind(p, q)
        if kd is None:
            bad.append([str(p), str(q)])
        else:
            kinds[kd] = kinds.get(kd, 0) + 1
    out["boundary_sides"] = len(bnd)
    out["boundary_kinds"] = kinds
    out["bad_boundary_sides"] = len(bad)
    out["point_in_omega_B"] = in_omega_B(point)
    inter = [i for i, t in enumerate(tris) if in_open_triangle(point, t)]
    out["interior_point_root"] = inter
    out["ok"] = bool(ok and not bad and out["in_unit_disc"] and out["point_in_omega_B"] and len(inter) >= 1)
    return out


# ---------------------------------------------------------------- cells
def load_cells(certdir):
    cells = {}
    for f in sorted(glob.glob(os.path.join(certdir, "coverB_cells_*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if line:
                c = json.loads(line)
                cells.setdefault(c["id"], []).append(c)
    return cells


def load_nodes(certdir):
    nodes = {}
    dup = 0
    for f in sorted(glob.glob(os.path.join(certdir, "coverB_nodes_*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if line:
                r = json.loads(line)
                key = (r["w"], int(r["k"]), int(r["n"]))
                if key in nodes:
                    dup += 1
                nodes.setdefault(key, []).append(r)
    return nodes, dup


def on_segment(x, p, q):
    return orient(p, q, x) == 0 and min(p[0], q[0]) <= x[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= x[1] <= max(p[1], q[1])


def check_split(parent, children):
    """children cover the parent: Lemma 'coverage' with Omega = interior of the parent"""
    ok, bnd = side_matching(children)
    if not ok or not children:
        return False
    for (p, q) in bnd:
        if not any(on_segment(p, parent[i], parent[(i + 1) % 3]) and on_segment(q, parent[i], parent[(i + 1) % 3]) for i in range(3)):
            return False
    c = children[0]
    cen = ((c[0][0] + c[1][0] + c[2][0]) / 3, (c[0][1] + c[1][1] + c[2][1]) / 3)
    return in_open_triangle(cen, parent) and in_open_triangle(cen, c)


class LamTab:
    """lambda_0 and its tangents on the alpha grid (mpmath intervals); scale != 1 only for negative controls"""

    def __init__(self, scale=None):
        self.scale = scale
        self.tab = {}
        self.cross = {}

    def get(self, k):
        if k not in self.tab:
            a = geom.alpha_grid(k)
            l, lp = L0.lam0(a)
            if self.scale is not None:
                l = l * geom.ivq(self.scale)
                lp = lp * geom.ivq(self.scale)
            self.tab[k] = (geom.ivq(a), l, lp)
        return self.tab[k]

    def crossing(self, j):
        """intersection of the tangents at alpha_j and alpha_{j+1}, certified inside [alpha_j, alpha_{j+1}]"""
        if j not in self.cross:
            a0, l0, s0 = self.get(j)
            a1, l1, s1 = self.get(j + 1)
            if not (s0.a > s1.b):
                self.cross[j] = None
            else:
                x = (l1 - s1 * a1 - l0 + s0 * a0) / (s0 - s1)
                self.cross[j] = x if (x.a >= a0.b and x.b <= a1.a) else None
        return self.cross[j]


def check_B_cell(cell, nodes, lamtab):
    out = {"ok": False, "id": cell["id"]}
    verts = [geom.parse_w(v) for v in cell["verts"]]
    if not all(v[0] ** 2 + v[1] ** 2 < 1 for v in verts):
        out["reason"] = "vertex outside the unit disc"
        return out
    prisms = cell["prisms"]
    ktop = cell["ktop"]
    cons = (len(prisms) >= 1 and prisms[0][0] == 0 and all(p[0] < p[1] for p in prisms)
            and all(prisms[i][1] == prisms[i + 1][0] for i in range(len(prisms) - 1))
            and prisms[-1][1] == ktop and ktop <= KMAX)
    out["prisms_consecutive"] = bool(cons)
    if not cons:
        out["reason"] = "prisms not consecutive from 0 to ktop"
        return out
    # vertex bounds, matched against node records
    b = {}
    used = set()
    for key, (lbs, n) in cell["nodes"].items():
        i, k = key.split(":")
        i = int(i)
        k = int(k)
        nk = (cell["verts"][i], k, int(n))
        recs = nodes.get(nk)
        val = geom.parse_rat(lbs)
        if not recs or not any(geom.parse_rat(r["lb"]) == val and geom.parse_rat(r["alpha"]) == geom.alpha_grid(k) for r in recs):
            out["reason"] = "vertex bound %s not matched by a node record" % key
            return out
        if not val > 0:
            out["reason"] = "nonpositive bound"
            return out
        b[(i, k)] = geom.ivq(val)
        used.add(nk)
    out["used_nodes"] = used
    need = sorted(set(k for p in prisms for k in p))
    if any((i, k) not in b for i in range(3) for k in need):
        out["reason"] = "missing vertex bound"
        return out
    # sqrt(Gamma_ik)
    sg = {}
    for i in range(3):
        for k in range(i, 3):
            wi, wk = verts[i], verts[k]
            num = 1 - (wi[0] * wk[0] + wi[1] * wk[1])
            den2 = (1 - wi[0] ** 2 - wi[1] ** 2) * (1 - wk[0] ** 2 - wk[1] ** 2)
            G = geom.ivq(num) / iv.sqrt(geom.ivq(den2))
            if not G.a > 0:
                out["reason"] = "Gamma not positive"
                return out
            sg[(i, k)] = iv.sqrt(G)
    minq = None
    npts = 0
    for (k0, k1) in prisms:
        a_lo = lamtab.get(k0)[0]
        a_hi = lamtab.get(k1)[0]
        for (i, k) in sg:
            g0 = iv.sqrt(b[(i, k0)] * b[(k, k0)])
            g1 = iv.sqrt(b[(i, k1)] * b[(k, k1)])
            Lf = lambda a: g0 + (g1 - g0) * (a - a_lo) / (a_hi - a_lo)
            pts = []
            for j in range(k0, k1 + 1):
                aj, lj, sj = lamtab.get(j)
                pts.append((aj, lj))                       # U(alpha_j) = lambda_0(alpha_j)
                if j < k1:
                    x = lamtab.crossing(j)
                    if x is None:
                        out["reason"] = "tangent crossing not certified in the grid interval %d" % j
                        return out
                    pts.append((x, lj + sj * (x - aj)))    # the kink of U between alpha_j and alpha_{j+1}
            for (a, U) in pts:
                lhs = Lf(a)
                rhs = sg[(i, k)] * U
                npts += 1
                if not lhs.a > rhs.b:
                    out["reason"] = "prism inequality fails: prism [%d,%d], pair (%d,%d)" % (k0, k1, i, k)
                    out["fail_ratio"] = float((lhs.a / rhs.b).a)
                    return out
                qv = (lhs.a / rhs.b).a
                minq = qv if minq is None or qv < minq else minq
    out["min_prism_quotient"] = float(minq)
    out["n_points"] = npts
    # closing test (b) at ktop
    minc = None
    for (i, k) in sg:
        lhs = iv.sqrt(b[(i, ktop)] * b[(k, ktop)])
        rhs = sg[(i, k)] * geom.LAMBDA1_E
        if not lhs.a > rhs.b:
            out["reason"] = "closing test fails for pair (%d,%d)" % (i, k)
            out["fail_ratio"] = float((lhs.a / rhs.b).a)
            return out
        qv = (lhs.a / rhs.b).a
        minc = qv if minc is None or qv < minc else minc
    out["min_closing_quotient"] = float(minc)
    out["n_prisms"] = len(prisms)
    out["ok"] = True
    return out


def check_cover(certdir, lam0_scale=None, only_ids=None, drop_root=None):
    """full V4.  Returns report dict and the set of node keys used by 'B' cells."""
    rep = {"ok": False}
    roots = json.load(open(os.path.join(certdir, "coverB_roots.json")))
    if drop_root is not None:
        roots = [t for i, t in enumerate(roots) if i != drop_root]
    rep["a"] = check_roots(roots)
    if only_ids is None and drop_root is not None:
        return rep, set()
    cells = load_cells(certdir)
    nodes, dup = load_nodes(certdir)
    rep["duplicate_node_keys"] = dup
    lamtab = LamTab(lam0_scale)
    leaves = {"B": 0, "W": 0, "L": 0}
    fails = []
    used = set()
    minq = None
    minc = None
    nprisms = 0
    ktopmax = 0
    npts = 0
    worst = None

    def do_leaf(c):
        nonlocal minq, minc, nprisms, ktopmax, npts, worst
        tri = tuple(geom.parse_w(v) for v in c["verts"])
        if c["type"] == "B":
            r = check_B_cell(c, nodes, lamtab)
            if not r["ok"]:
                fails.append((c["id"], r.get("reason")))
                return False
            used.update(r["used_nodes"])
            nprisms += r["n_prisms"]
            npts += r["n_points"]
            ktopmax = max(ktopmax, c["ktop"])
            if minq is None or r["min_prism_quotient"] < minq:
                minq = r["min_prism_quotient"]
                worst = c["id"]
            minc = r["min_closing_quotient"] if minc is None else min(minc, r["min_closing_quotient"])
            return True
        if c["type"] == "W":
            ok = all(v[0] ** 2 + v[1] ** 2 < 1 for v in tri) and min_F_on_triangle(tri) >= 0
            if not ok:
                fails.append((c["id"], "F >= 0 fails"))
            return ok
        if c["type"] == "L":
            ok = all(v[0] ** 2 + v[1] ** 2 <= RSTAR ** 2 for v in tri)
            if not ok:
                fails.append((c["id"], "vertices not in |w| <= r_*"))
            return ok
        fails.append((c["id"], "unknown type"))
        return False

    def resolve(cid, tri_expected):
        cl = cells.get(cid)
        if not cl or len(cl) != 1:
            fails.append((cid, "cell missing or duplicated"))
            return False
        c = cl[0]
        tri = tuple(geom.parse_w(v) for v in c["verts"])
        if tri_expected is not None and sorted(tri) != sorted(tri_expected):
            fails.append((cid, "vertices do not match parent/root"))
            return False
        if c["type"] == "split":
            kids = [k for k in cells if k.startswith(cid + ".") and "." not in k[len(cid) + 1:]]
            ktris = [tuple(geom.parse_w(v) for v in cells[k][0]["verts"]) for k in kids]
            if not check_split(tri, ktris):
                fails.append((cid, "children do not cover the cell"))
                return False
            return all([resolve(k, None) for k in kids])
        if c["type"] in leaves:
            leaves[c["type"]] += 1
        return do_leaf(c)

    ids = range(len(roots)) if only_ids is None else only_ids
    allok = True
    for i in ids:
        tri = tuple(geom.parse_w(v) for v in roots[i])
        allok &= bool(resolve("r%d" % i, tri))
    rep["b"] = {"ok": bool(allok and not fails), "leaves": leaves, "fails": fails[:20], "n_fails": len(fails),
                "n_prisms": nprisms, "n_points_checked": npts, "ktop_max": ktopmax,
                "alpha_ktop_max": float(geom.alpha_grid(ktopmax)),
                "min_prism_quotient": minq, "min_prism_quotient_cell": worst, "min_closing_quotient": minc,
                "used_nodes": len(used), "node_records": sum(len(v) for v in nodes.values())}
    rep["ok"] = bool(rep["a"]["ok"] and rep["b"]["ok"])
    return rep, used

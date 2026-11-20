"""Region B (middle region): certified covering by prisms  (shape triangle in the Klein disc) x [alpha_k0, alpha_k1].

Shape cells are Euclidean triangles with rational vertices w_1, w_2, w_3 in the Klein disc (see shapes.py); every
shape in the cell is G = c * sum_i theta_i G(w_i) with c = (theta^T Gamma theta)^(-1/2), Gamma_ij = <G_i, G_j>
= cosh(2 d_ij), and (Lemma "cell" of the paper)
        lambda_1(T_G, alpha) >= c * sum_i theta_i lambda_1(T_{w_i}, alpha).
With certified vertex bounds b_i(alpha_k) (rcr.py) and concavity in alpha, the prism [alpha_k0, alpha_k1] is certified if
for all pairs i <= j the linear interpolant of sqrt(b_i b_j) between the two alpha-nodes exceeds sqrt(Gamma_ij) * U(alpha)
at the end points and at all kinks of U, where U >= lambda_0 is the minimum of the tangents of lambda_0 at the grid
nodes.  The cell is closed at the top at node k when b_i b_j > Gamma_ij * Lambda_1^D(E)^2 for all pairs.

Leaf types:  'B' certified by prisms for all alpha >= 1/100;  'L' all vertices in the local disc |w| <= R_LOC
(Proposition "local");  'W' the whole triangle is thin (Theorem "thin").  Failed cells are split in four.

Usage: python3 cover.py run P I     (process the root cells with index = I mod P; resumable)
       python3 cover.py roots        (print the root triangulation)
"""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, time, math, hashlib, glob
import numpy as np
from fractions import Fraction as Fr
from flint import arb, ctx
import rcr, shapes
from shapes import A_
from local_ring import enc, dec
from config import ALPHA, CERT, SHRINK, T_LOC, R_LOC, KMAX, NSCALE

MAXDEPTH = 8
T_GRID = [0.057, 0.078, 0.105, 0.155, 0.21, 0.28, 0.38, 0.5, 0.65, 0.82, 1.0, 1.15, 1.285]
M_ANG = 4
DEN = 2 ** 20

def up(x): return arb(x.upper())
def lo(x): return arb(x.lower())

# ------------------------------------------------------------------ root triangulation (rational vertices)
def roots():
    def P(t, j, M):
        r = math.tanh(2 * t); psi = (math.pi / 3 + 1e-3) * j / M
        y = Fr(0) if j == 0 else Fr(round(r * math.sin(psi) * DEN), DEN)
        return (Fr(round(r * math.cos(psi) * DEN), DEN), y)
    tris = []
    m = len(T_GRID)
    for i in range(m - 1):
        for j in range(M_ANG):
            a, b = P(T_GRID[i], j, M_ANG), P(T_GRID[i], j + 1, M_ANG)
            c, d = P(T_GRID[i + 1], j, M_ANG), P(T_GRID[i + 1], j + 1, M_ANG)
            if i < m - 2:
                tris += [(a, c, d), (a, d, b)]
            else:                                  # outermost layer: one more vertex on the outer circle
                e = P(T_GRID[i + 1], 2 * j + 1, 2 * M_ANG)
                tris += [(a, c, e), (a, e, b), (b, e, d)]
    return tris

def children(v):
    mid = lambda p, q: ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    m01, m12, m02 = mid(v[0], v[1]), mid(v[1], v[2]), mid(v[0], v[2])
    return [(v[0], m01, m02), (m01, v[1], m12), (m02, m12, v[2]), (m01, m12, m02)]

def wkey(w): return "%d/%d,%d/%d" % (w[0].numerator, w[0].denominator, w[1].numerator, w[1].denominator)

# ------------------------------------------------------------------ lambda_0 on the alpha grid
class Lam0Grid:
    def __init__(self):
        self.f = {}; self.kk = {}
    def node(self, k):
        if k not in self.f: self.f[k] = shapes.lam0_d(ALPHA(k))
        return self.f[k]
    def kink(self, k):
        """(p, U(p)): intersection of the tangents of lambda_0 at alpha_k and alpha_{k+1}"""
        if k not in self.kk:
            a0, a1 = A_(ALPHA(k)), A_(ALPHA(k + 1)); (f0, g0), (f1, g1) = self.node(k), self.node(k + 1)
            p = (f1 - f0 + g0 * a0 - g1 * a1) / (g0 - g1)
            if not (bool(g0 > g1) and bool(p >= a0) and bool(p <= a1)): raise ArithmeticError('kink not enclosed in its grid interval')
            self.kk[k] = (p, f0 + g0 * (p - a0))
        return self.kk[k]
L0G = Lam0Grid()
LAMD = shapes.LAM_D

# ------------------------------------------------------------------ vertex bounds (cached, persistent)
class Nodes:
    def __init__(self, fn_write, fns_read):
        self.d = {}                                    # (wkey, k) -> list of records
        self.fw = open(fn_write, 'a'); self.nsolve = 0; self.tsolve = 0.0
        for fn in fns_read:
            if os.path.exists(fn):
                for line in open(fn):
                    line = line.strip()
                    if not line: continue
                    try: r = json.loads(line)
                    except Exception: continue
                    self.d.setdefault((r['w'], r['k']), []).append(r)
    def n0(self, w, k):
        r = math.hypot(float(w[0]), float(w[1])); t = math.atanh(r) / 2
        l2 = [float(x.mid()) for x in shapes.ell2(w)]
        amax = math.sqrt(float(shapes.S_E2.mid()) * max(l2))
        lam = float(L0G.node(k)[0].mid()) * (1 + 1.2 * t * t)
        marg = min(0.4 * t * t, 0.4)
        return max(10, int(math.ceil(NSCALE * 0.1893 * amax * math.sqrt(lam / (0.12 * marg)))))
    def get(self, w, k, level):
        """certified lower bound (arb, exact) for lambda_1(T_w, alpha_k); mesh parameter at least n0 * 1.5^level"""
        key = (wkey(w), k); nreq = int(math.ceil(self.n0(w, k) * 1.5 ** level))
        recs = self.d.setdefault(key, [])
        if not any(r['n'] >= nreq for r in recs):
            t0 = time.time()
            a2, A = shapes.tri_data(w); al = A_(ALPHA(k))
            kind = rcr.choose_kind(a2, A)
            res = rcr.lower_bound(a2, A, [al] * 3, nreq, kind=kind)
            sig = Fr(int(res['sigma'] * 2 ** rcr.BITS * (1 - SHRINK)), 2 ** rcr.BITS)
            if not (0 < sig <= res['sigma']): raise ValueError('shrunk sigma not in (0, certified sigma]')   # exact rationals
            lb = rcr.cr_bound(sig, res['h'])
            rec = dict(w=key[0], k=k, alpha=str(ALPHA(k)), n=nreq, kind=kind, N=res['N'], sigma=str(sig), lb=enc(lb), lam_h=res['lam_h'], xhash=res['xhash'])
            recs.append(rec); self.fw.write(json.dumps(rec) + "\n"); self.fw.flush()
            self.nsolve += 1; self.tsolve += time.time() - t0
        best = max(recs, key=lambda r: dec(r['lb']))
        return A_(dec(best['lb'])), best

# ------------------------------------------------------------------ cell certification
def thin_whole(v):
    """True if (certified) every point of the triangle is a thin shape: (1+w1)^2 >= 81 (1-|w|^2) on its bounding box"""
    x = A_(min(p[0] for p in v)).union(A_(max(p[0] for p in v))); y = A_(min(p[1] for p in v)).union(A_(max(p[1] for p in v)))
    return bool((1 + x) ** 2 - 81 * (1 - x * x - y * y) > 0) and bool(1 - x * x - y * y > 0)

def certify_cell(v, nodes, max_level=2):
    """returns a leaf record (dict) or None (split)."""
    if all(p[0] ** 2 + p[1] ** 2 <= R_LOC ** 2 for p in v):
        return dict(type='L')
    if thin_whole(v):
        return dict(type='W')
    if any(p[0] ** 2 + p[1] ** 2 >= Fr(999, 1000) ** 2 for p in v): return None
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    sg = {}
    for (i, j) in pairs:
        sg[(i, j)] = arb(1) if i == j else up(up(shapes.cosh2d(v[i], v[j])).sqrt())
    used = {}
    def b(i, k, level):
        val, rec = nodes.get(v[i], k, level)
        used[(i, k)] = rec
        return val
    def closes(k, level):
        bb = [b(i, k, level) for i in range(3)]
        return all(bool((bb[i] * bb[j]).sqrt() > sg[(i, j)] * LAMD) for (i, j) in pairs)
    def prism_ok(k0, k1, level):
        b0 = [b(i, k0, level) for i in range(3)]; b1 = [b(i, k1, level) for i in range(3)]
        a0, a1 = A_(ALPHA(k0)), A_(ALPHA(k1))
        for (i, j) in pairs:
            s0, s1 = lo((b0[i] * b0[j]).sqrt()), lo((b1[i] * b1[j]).sqrt()); g = sg[(i, j)]
            if not bool(s0 > g * L0G.node(k0)[0]): return False
            if not bool(s1 > g * L0G.node(k1)[0]): return False
            for m in range(k0, k1):
                p, Up = L0G.kink(m)
                if not bool(s0 + (s1 - s0) * (p - a0) / (a1 - a0) > g * Up): return False
        return True
    k = 0; prisms = []; stride = 4; level = 0
    while True:
        try:
            if closes(k, level): break
            if k >= KMAX: return None
            done = False
            s = stride
            while s >= 1:
                k1 = min(k + s, KMAX)
                if prism_ok(k, k1, level):
                    prisms.append([k, k1]); k = k1; stride = min(2 * s, 64); done = True; break
                s //= 2
            if not done:
                if level < max_level: level += 1
                else: return None
        except ValueError as e:                      # a CR certificate could not be produced: treat as failure
            if level < max_level: level += 1
            else: return None
    # keep only the node records that the accepted prisms and the closing test use
    need = set()
    for k0, k1 in prisms: need |= {k0, k1}
    need.add(k)
    nrec = {"%d:%d" % (i, kk): [used[(i, kk)]['lb'], used[(i, kk)]['n']] for (i, kk) in used if kk in need}
    return dict(type='B', prisms=prisms, ktop=k, nodes=nrec, level=level)

def run(P, I):
    assert Fr(R_LOC) ** 2 < 1 and bool(A_(R_LOC) < (2 * A_(T_LOC)).tanh())
    rt = roots()
    fn_roots = os.path.join(CERT, 'coverB_roots.json')
    if not os.path.exists(fn_roots):
        tmp = fn_roots + '.tmp%d' % os.getpid()
        json.dump([[wkey(p) for p in tr] for tr in rt], open(tmp, 'w')); os.replace(tmp, fn_roots)
    else:
        assert json.load(open(fn_roots)) == [[wkey(p) for p in tr] for tr in rt], 'root triangulation changed'
    fn_nodes = os.path.join(CERT, 'coverB_nodes_%d.jsonl' % I); fn_cells = os.path.join(CERT, 'coverB_cells_%d.jsonl' % I)
    nodes = Nodes(fn_nodes, sorted(glob.glob(os.path.join(CERT, 'coverB_nodes_*.jsonl'))))
    done = {}
    if os.path.exists(fn_cells):
        for line in open(fn_cells):
            line = line.strip()
            if line:
                try: r = json.loads(line); done[r['id']] = r
                except Exception: pass
    fc = open(fn_cells, 'a'); t00 = time.time()
    stack = [("r%d" % i, rt[i], 0) for i in range(len(rt)) if i % P == I][::-1]
    nleaf = {'B': 0, 'L': 0, 'W': 0, 'split': 0, 'FAIL': 0}
    while stack:
        cid, v, depth = stack.pop()
        if cid in done:
            r = done[cid]
        else:
            t0 = time.time(); ns0 = nodes.nsolve
            leaf = certify_cell(v, nodes)
            if leaf is None:
                r = dict(id=cid, type='split' if depth < MAXDEPTH else 'FAIL')
            else:
                r = dict(id=cid, **leaf)
            r['verts'] = [wkey(p) for p in v]; r['sec'] = round(time.time() - t0, 2); r['solves'] = nodes.nsolve - ns0
            r['sha'] = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()[:16]
            fc.write(json.dumps(r) + "\n"); fc.flush()
            tt = [math.atanh(math.hypot(float(p[0]), float(p[1]))) / 2 for p in v]
            print(f"{cid:22s} {r['type']:5s} t=[{min(tt):.3f},{max(tt):.3f}] prisms={len(r.get('prisms', []))} ktop={r.get('ktop')} "
                  f"lvl={r.get('level')} solves={r['solves']} {r['sec']}s total={time.time()-t00:.0f}s", flush=True)
        nleaf[r['type']] += 1
        if r['type'] == 'split':
            ch = children(v)
            for q in range(3, -1, -1): stack.append((cid + ".%d" % q, ch[q], depth + 1))
    print("DONE", nleaf, "solves", nodes.nsolve, "solve time %.0f" % nodes.tsolve, "total %.0f" % (time.time() - t00), flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'roots':
        for i, tr in enumerate(roots()): print(i, [(float(p[0]), float(p[1])) for p in tr])
    elif sys.argv[1] == 'run':
        run(int(sys.argv[2]), int(sys.argv[3]))

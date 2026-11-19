"""Negative controls and sanity checks (not proof steps): the certifiers must REJECT false statements.

 N1  Collatz-Wielandt test rejects a sigma that is 1e-6 (relative) above the discrete eigenvalue.
 N2  mesh 'U' is refused for an obtuse triangle (the matrix would not be a Z-matrix).
 N3  the cell certifier fails on the cell nearest to E when lambda_0 (and the Dirichlet value) is replaced by
     1.01*lambda_0, and when the target is the false bound lambda_1(T) >= lambda_0 (1 + 0.6 t^2); it passes unchanged.
 N4  the local check fails for t_loc = 3/25, and fails when the ring bounds are lowered by 1 per cent.
 N5  the coverage check fails if a root triangle is removed.
 S1  sandwich: certified lower bounds lie below conforming P2 finite element values (upper bounds up to rounding)
     and within 1.5 per cent of them, on a random sample of stored vertex bounds.
Usage: python3 negative_controls.py      (about two minutes)
"""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, glob, math, random, tempfile
from fractions import Fraction as Fr
import numpy as np
from flint import arb
import config, rcr, shapes, cover, local_check, coverage_check
from local_ring import dec, load
from shapes import A_

res = {}
def record(name, ok, info=''):
    res[name] = bool(ok); print(('PASS ' if ok else 'FAIL ') + name + (' : ' + info if info else ''), flush=True)

# ---- N1, N2
w = (Fr(1, 5), Fr(1, 10)); a2, A = shapes.tri_data(w); beta = [A_(Fr(3))] * 3
M = rcr.build_mesh(a2, A, 24, 'U'); lam, x, rmin, K, md = rcr.float_eig(M, beta)
sig, _ = rcr.certify(M, beta, x)
try:
    rcr.certify(M, beta, x, sigma=Fr(lam) * (1 + Fr(1, 10 ** 6))); bad = True
except ValueError: bad = False
record('N1 CW rejects sigma above lambda_h', (not bad) and float(sig) <= lam, 'lam_h=%.9f certified sigma=%.9f' % (lam, float(sig)))
wo = (Fr(7, 10), Fr(0)); a2o, Ao = shapes.tri_data(wo)
try:
    rcr.build_mesh(a2o, Ao, 8, 'U'); bad = True
except ValueError: bad = False
record('N2 mesh U refused for obtuse triangle', not bad)

# ---- N3
tmp = tempfile.mkdtemp()
roots = cover.roots()
nodes = cover.Nodes(os.path.join(tmp, 'n.jsonl'), sorted(glob.glob(os.path.join(config.CERT, 'coverB_nodes_*.jsonl'))))
ok_plain = cover.certify_cell(roots[0], nodes) is not None
orig_node, orig_kink, orig_lamd = cover.L0G.node, cover.L0G.kink, cover.LAMD
class Fake(cover.Lam0Grid):
    def __init__(self, fac): super().__init__(); self.fac = fac
    def node(self, k):
        f, g = super().node(k); c = A_(Fr(self.fac(float(config.ALPHA(k))))); return (f * c, g * c)
cover.L0G = Fake(lambda al: 1.01); cover.LAMD = orig_lamd * A_(Fr(101, 100))
fail_101 = cover.certify_cell(roots[0], nodes, max_level=0) is None
tmin = min(math.atanh(math.hypot(float(p[0]), float(p[1]))) / 2 for p in roots[0])
cover.L0G = Fake(lambda al: 1 + 0.6 * tmin ** 2); cover.LAMD = orig_lamd * A_(Fr(1 + 0.6 * tmin ** 2))
fail_false = cover.certify_cell(roots[0], nodes, max_level=0) is None
cover.L0G = cover.Lam0Grid(); cover.LAMD = orig_lamd
record('N3 cell certifier: passes for lambda_0, fails for 1.01*lambda_0 and for lambda_0(1+0.6 t^2)', ok_plain and fail_101 and fail_false,
       'plain=%s fail(1.01)=%s fail(false quadratic bound)=%s' % (ok_plain, fail_101, fail_false))

# ---- N4
r1 = local_check.main(Fr(3, 25), None, None, quiet=True)
orig_dec = local_check.dec
local_check.dec = lambda s: orig_dec(s) * Fr(99, 100)
try:
    r2 = local_check.main(config.T_LOC, None, None, quiet=True); f2 = not r2['ok']
except Exception as e:
    f2 = True
local_check.dec = orig_dec
r3 = local_check.main(config.T_LOC, None, None, quiet=True)
record('N4 local check: fails for t_loc=3/25 and for ring/lambda_2 bounds lowered by 1%; passes unchanged', (not r1['ok']) and f2 and r3['ok'],
       'worst quotient: t_loc=3/25 -> %.3f, unchanged -> %.4f' % (r1['worst_ratio'], r3['worst_ratio']))

# ---- N5
fn = os.path.join(config.CERT, 'coverB_roots.json'); rt = json.load(open(fn))
d2 = tempfile.mkdtemp()
json.dump(rt[:50] + rt[51:], open(os.path.join(d2, 'coverB_roots.json'), 'w'))
for f in glob.glob(os.path.join(config.CERT, 'coverB_cells_*.jsonl')):
    with open(os.path.join(d2, os.path.basename(f)), 'w') as g:
        for line in open(f):
            r = json.loads(line)
            if r['id'] == 'r50': continue
            i = int(r['id'][1:].split('.')[0])
            if i > 50: r['id'] = 'r%d' % (i - 1) + r['id'][len('r%d' % i):]
            g.write(json.dumps(r) + "\n")
saveC, saveO = coverage_check.CERT, coverage_check.OUT; coverage_check.CERT = coverage_check.OUT = d2
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()): okc = coverage_check.main()
coverage_check.CERT, coverage_check.OUT = saveC, saveO
record('N5 coverage check fails with one root triangle removed', not okc)

# ---- S1 sandwich with conforming P2 elements
from p2_reference import RefFEM
Fm = RefFEM(24); random.seed(7)
recs = [json.loads(l) for f in sorted(glob.glob(os.path.join(config.CERT, 'coverB_nodes_*.jsonl'))) for l in open(f)]
worst = (0, 0); okS = True
for r in random.sample(recs, 40):
    a, b = r['w'].split(','); w1, w2 = float(Fr(a)), float(Fr(b)); al = float(Fr(r['alpha']))
    den = math.sqrt(1 - w1 * w1 - w2 * w2); G = np.array([[1 - w1, -w2], [-w2, 1 + w1]]) / den
    ev, U = np.linalg.eigh(G); L = U @ np.diag(ev ** -0.5) @ U.T
    up = Fm.eig(L, al)[0]; lb = float(dec(r['lb'])); gap = (up - lb) / up
    if not (0 < gap < 0.015): okS = False
    worst = (max(worst[0], gap), min(worst[1] if worst[1] else 1, gap))
record('S1 sandwich lb < P2-FEM value, relative gap < 1.5e-2', okS, 'gap in [%.2e, %.2e]' % (worst[1], worst[0]))
res['S1_gap_min'], res['S1_gap_max'] = worst[1], worst[0]
json.dump(res, open(os.path.join(config.OUT, 'negative_controls.json'), 'w'), indent=0)
sys.exit(0 if all(v for v in res.values() if isinstance(v, bool)) else 1)

"""Negative controls and sanity checks (not proof steps): the certifiers must REJECT false statements.

 N1  Collatz-Wielandt test rejects a sigma that is 1e-6 (relative) above the discrete eigenvalue.
 N2  mesh 'U' is refused for an obtuse triangle (the matrix would not be a Z-matrix).
 N3  the cell certifier fails on the cell nearest to E when lambda_0 (and the Dirichlet value) is replaced by
     1.01*lambda_0, and when the target is the false bound lambda_1(T) >= lambda_0 (1 + 0.6 t^2); it passes unchanged.
 N4  the local check fails for t_loc = 3/25, and fails when the ring bounds are lowered by 1 per cent.
 N5  the coverage check fails if a root triangle is removed.
 N6  Lemma CR needs 0 < sigma: a nonpositive sigma (-10^6, 0, and just below the pole -1/(kappa0 h)^2, where
     sigma/(1 + (kappa0 h)^2 sigma) is large and positive) is rejected by rcr.cr_bound, by rcr.certify, by the
     lambda_2 certifier and by the re-checks recert_vertex / recert_ring / recert_lam2 (records tampered in memory);
     a cell with negative vertex bounds (and sigma = -10^6 in their node records) is rejected by recheck.py cells.
 N7  exact sigma: on the one-element mesh 'U' of the triangle with sides 5,5,6 and Robin coefficients 6,6,5
     (x = (1,1,1), CR mass 4I, stiffness 30I + graph Laplacian, discrete eigenvalue exactly 15/2) the supplied
     sigma = 15/2 + 2^-61 is rejected (it used to be rounded down to the 2^-60 grid, verified as 15/2, and the
     unrounded value used by the callers); sigma = 15/2 and a non-dyadic sigma below it are certified and
     returned unchanged.
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
from local_ring import dec, load, enc
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

# ---- N6  hypothesis 0 < sigma of Lemma CR
import recheck, local_ring
def rejects(f):
    try:
        r = f()
    except ValueError as e:
        return 'not positive' in str(e)
    return r is False
hq = rcr.arb_hi_frac(M.h); C0 = Fr(config.C_INTERP)
pole = -1 / (C0 * C0 * hq * hq)
bad_sigmas = [Fr(-10 ** 6), Fr(0), pole * (1 + Fr(1, 10 ** 9))]
n6 = {}
n6['cr_bound'] = all(rejects(lambda s=s_: rcr.cr_bound(s, M.h)) for s_ in bad_sigmas)
n6['formula_would_be_positive_below_pole'] = bool((A_(bad_sigmas[2]) / (1 + A_(C0) ** 2 * A_(hq) ** 2 * A_(bad_sigmas[2]))) > 1000)
n6['certify'] = all(rejects(lambda s=s_: rcr.certify(M, beta, x, sigma=s)) for s_ in bad_sigmas)
vrec = sorted((json.loads(l) for f in sorted(glob.glob(os.path.join(config.CERT, 'coverB_nodes_*.jsonl'))) for l in open(f) if l.strip()),
              key=lambda r: r['N'])[0]
rrec = recheck.read_jsonl('ring_nodes.jsonl')[0]; l2rec = recheck.read_jsonl('lam2_nodes.jsonl')[0]
n6['unmodified_records_pass'] = recheck.recert_vertex(vrec) and recheck.recert_ring(rrec)
def tamper(r, key, sig, poleval):
    t = dict(r); t[key] = str(poleval if sig == 'pole' else sig); t['lb'] = enc(A_(1000)); return t
def pole_of(r, kind):
    if kind == 'lam2':
        a2_, A__ = shapes.tri_data((Fr(0), Fr(0))); Mm = rcr.build_mesh(a2_, A__, r['key'][1], 'U')
    elif kind == 'ring':
        a2_, A__, _ = shapes.ring_data(Fr(r['s']), r['key'][1], Fr(r['alpha'])); Mm = rcr.build_mesh(a2_, A__, r['n'], 'U')
    else:
        a2_, A__ = shapes.tri_data(recheck.pw(r['w'])); Mm = rcr.build_mesh(a2_, A__, r['n'], r['kind'])
    h_ = rcr.arb_hi_frac(Mm.h); return -(1 + Fr(1, 10 ** 9)) / (C0 * C0 * h_ * h_)
for nm, fun, r, key, kind in (('recert_vertex', recheck.recert_vertex, vrec, 'sigma', 'node'),
                              ('recert_ring', recheck.recert_ring, rrec, 'sigma', 'ring'),
                              ('recert_lam2', recheck.recert_lam2, l2rec, 'sigma2', 'lam2')):
    pv = pole_of(r, kind)
    n6[nm] = all(rejects(lambda t=tamper(r, key, s_, pv): fun(t)) for s_ in (Fr(-10 ** 6), Fr(0), 'pole'))
n6['lam2_node'] = all(rejects(lambda s=s_: local_ring.lam2_node(l2rec['key'][0], l2rec['key'][1], sig2=s, g=Fr(l2rec['g'])))
                      for s_ in (Fr(-10 ** 6), Fr(0), pole_of(l2rec, 'lam2')))
# a cell whose three vertex bounds at the closing node are negative (b_i b_j > 0, so sqrt(b_i b_j) is large) and whose
# node records carry sigma = -10^6: the cell re-check and the re-certification must both reject it
d6 = tempfile.mkdtemp()
cl = [json.loads(l) for f in sorted(glob.glob(os.path.join(config.CERT, 'coverB_cells_*.jsonl'))) for l in open(f) if l.strip()]
c0 = [c for c in cl if c['id'] == 'r0'][0]; kt = c0['ktop']; tk = set()
for key in c0['nodes']:
    i, k = map(int, key.split(':'))
    if k == kt: c0['nodes'][key][0] = '-10000*2^0'; tk.add((c0['verts'][i], k, c0['nodes'][key][1]))
open(os.path.join(d6, 'coverB_cells_0.jsonl'), 'w').write(json.dumps(c0) + '\n')
nrecs = [json.loads(l) for f in sorted(glob.glob(os.path.join(config.CERT, 'coverB_nodes_*.jsonl'))) for l in open(f) if l.strip()]
bad = []
with open(os.path.join(d6, 'coverB_nodes_0.jsonl'), 'w') as g:
    for r in nrecs:
        if (r['w'], r['k'], r['n']) in tk: r = dict(r, lb='-10000*2^0', sigma='-1000000'); bad.append(r)
        g.write(json.dumps(r) + '\n')
saveC, saveO = recheck.CERT, recheck.OUT; recheck.CERT = recheck.OUT = d6
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()) as buf6: ok6 = recheck.check_cells()
recheck.CERT, recheck.OUT = saveC, saveO
n6['cell_with_negative_vertex_bounds_rejected'] = (not ok6) and 'nonpositive vertex bound' in buf6.getvalue() and len(bad) == 3
n6['its_node_records_rejected'] = all(rejects(lambda r=r: recheck.recert_vertex(r)) for r in bad)
record('N6 nonpositive sigma (-1e6, 0, below the pole) rejected everywhere Lemma CR is applied', all(n6.values()), json.dumps(n6))
res['N6_detail'] = n6

# ---- N7  the exact supplied sigma is tested and returned (no rounding to the 2^-60 grid)
a2e = [arb(25), arb(25), arb(36)]; Ae = arb(12); be = [A_(6), A_(6), A_(5)]
Me = rcr.build_mesh(a2e, Ae, 1, 'U'); xe = np.ones(Me.N)
n7 = {'N': Me.N}
n7['above_rejected'] = False
try:
    rcr.certify(Me, be, xe, sigma=Fr(15, 2) + Fr(1, 2 ** 61))
except ValueError:
    n7['above_rejected'] = True
s_ex, _ = rcr.certify(Me, be, xe, sigma=Fr(15, 2)); n7['exact_15_2_returned'] = (s_ex == Fr(15, 2))
nd = Fr(15, 2) - Fr(1, 3 * 2 ** 70); s_nd, _ = rcr.certify(Me, be, xe, sigma=nd); n7['non_dyadic_returned_unchanged'] = (s_nd == nd)
s_auto, _ = rcr.certify(Me, be, xe); n7['proposed_le_15_2'] = (0 < s_auto <= Fr(15, 2))
record('N7 CW test on the exact supplied sigma: 15/2 + 2^-61 rejected, 15/2 certified exactly (sides 5,5,6; beta 6,6,5)',
       Me.N == 3 and n7['above_rejected'] and n7['exact_15_2_returned'] and n7['non_dyadic_returned_unchanged'] and n7['proposed_le_15_2'], json.dumps(n7))
res['N7_detail'] = n7

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

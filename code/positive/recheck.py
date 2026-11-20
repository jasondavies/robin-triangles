"""Re-checks of the stored certificates with the original code (see code/verify_positive/ for the independent verifier).

  python3 recheck.py cells      every 'B' cell: the prism inequalities of Corollary "prisms"(a) and the closing test (b)
                                are re-evaluated from the bounds stored in the cell record; every such bound must be the
                                lb of a stored vertex record; prisms must be consecutive from k = 0.
  python3 recheck.py sample N   N random vertex records, N/3 ring records and 3 lambda_2 records are re-certified from
                                scratch: a fresh floating-point eigenvector must pass the exact Collatz-Wielandt test for
                                the STORED sigma, and the stored lb must not exceed sigma/(1+(kappa0 h)^2 sigma).
  python3 recheck.py full       the same for all ring records, all lambda_2 records and all vertex records used by cells.
"""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, glob, random, time
from fractions import Fraction as Fr
from flint import arb
import config, rcr, shapes, cover, local_ring
from local_ring import dec, load, enc
from shapes import A_
from config import ALPHA, CERT, KMAX, OUT

def pw(s):
    a, b = s.split(','); return (Fr(a), Fr(b))
def read_jsonl(pattern):
    out = []
    for fn in sorted(glob.glob(os.path.join(CERT, pattern))):
        for line in open(fn):
            if line.strip(): out.append(json.loads(line))
    return out

def check_cells():
    nodes = {(r['w'], r['k'], r['n']): r for r in read_jsonl('coverB_nodes_*.jsonl')}
    cells = read_jsonl('coverB_cells_*.jsonl'); nB = 0; nprism = 0; ok = True; minmarg = None
    pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    lo = lambda x: arb(x.lower()); up = lambda x: arb(x.upper())
    for c in cells:
        if c['type'] != 'B': continue
        nB += 1; v = [pw(s) for s in c['verts']]
        sg = {(i, j): (arb(1) if i == j else up(up(shapes.cosh2d(v[i], v[j])).sqrt())) for (i, j) in pairs}
        b = {}
        for key, (lb, n) in c['nodes'].items():
            i, k = [int(z) for z in key.split(':')]
            rec = nodes.get((c['verts'][i], k, n))
            if rec is None or rec['lb'] != lb or Fr(rec['alpha']) != ALPHA(k): ok = False; print('node mismatch', c['id'], key)
            if not dec(lb) > 0: ok = False; print('nonpositive vertex bound', c['id'], key)
            b[(i, k)] = A_(dec(lb))
        pr = c['prisms']
        if (pr and pr[0][0] != 0) or any(pr[q][1] != pr[q + 1][0] for q in range(len(pr) - 1)) or (pr[-1][1] if pr else 0) != c['ktop'] \
                or any(not (0 <= k0 < k1) for k0, k1 in pr):
            ok = False; print('prisms not consecutive', c['id'])
        for k0, k1 in pr:
            nprism += 1; a0, a1 = A_(ALPHA(k0)), A_(ALPHA(k1))
            for (i, j) in pairs:
                s0, s1 = lo((b[(i, k0)] * b[(j, k0)]).sqrt()), lo((b[(i, k1)] * b[(j, k1)]).sqrt()); g = sg[(i, j)]
                tests = [s0 - g * cover.L0G.node(k0)[0], s1 - g * cover.L0G.node(k1)[0]]
                for m in range(k0, k1):
                    p, Up = cover.L0G.kink(m)
                    if not (bool(p > a0) and bool(p < a1)): ok = False; print('kink outside', c['id'])
                    tests.append(s0 + (s1 - s0) * (p - a0) / (a1 - a0) - g * Up)
                for t in tests:
                    if not bool(t > 0): ok = False; print('prism inequality fails', c['id'], k0, k1, i, j)
                    rel = float((t / (g * cover.L0G.node(k1)[0])).lower()); minmarg = rel if minmarg is None else min(minmarg, rel)
        kt = c['ktop']
        for (i, j) in pairs:
            if not bool((b[(i, kt)] * b[(j, kt)]).sqrt() > sg[(i, j)] * cover.LAMD): ok = False; print('closing test fails', c['id'])
    res = dict(ok=ok, B_cells=nB, prisms=nprism, min_relative_margin=minmarg)
    print(res); json.dump(res, open(os.path.join(OUT, 'recheck_cells.json'), 'w'))
    return ok

def _lb_ok(lb, sig, h):
    """0 < lb <= sigma/(1 + (kappa0 h)^2 sigma) with sigma the value CERTIFIED by rcr.certify (> 0, enforced there
    and in rcr.cr_bound).  lb > 0 matters downstream: the prism tests use sqrt(b_i b_j)."""
    lb = dec(lb)
    return lb > 0 and bool(A_(lb) <= rcr.cr_bound(sig, h))
def recert_vertex(r):
    a2, A = shapes.tri_data(pw(r['w'])); al = A_(Fr(r['alpha']))
    M = rcr.build_mesh(a2, A, r['n'], r['kind'])
    lam, x, rmin, K, md = rcr.float_eig(M, [al] * 3)
    sig, _ = rcr.certify(M, [al] * 3, x, sigma=Fr(r['sigma']))
    return sig == Fr(r['sigma']) and _lb_ok(r['lb'], sig, M.h) and M.N == r['N'] and Fr(r['alpha']) == ALPHA(r['k'])
def recert_ring(r):
    k, sign, n = r['key']
    if not (sign in (1, -1) and Fr(r['s']) == config.S_RING and r['kind'] == 'U' and r['n'] == n): return False
    a2, A, beta = shapes.ring_data(Fr(r['s']), sign, Fr(r['alpha']))
    M = rcr.build_mesh(a2, A, n, 'U')
    lam, x, rmin, K, md = rcr.float_eig(M, beta)
    sig, _ = rcr.certify(M, beta, x, sigma=Fr(r['sigma']))
    return sig == Fr(r['sigma']) and _lb_ok(r['lb'], sig, M.h) and M.N == r['N'] and Fr(r['alpha']) == ALPHA(k)
def recert_lam2(r):
    k, n = r['key']
    d = local_ring.lam2_node(k, n, sig2=Fr(r['sigma2']), g=Fr(r['g']))     # raises unless sigma2 > 0, g >= 0, B > 0
    return Fr(d['sigma2']) == Fr(r['sigma2']) and 0 < dec(r['lb']) <= dec(d['lb']) and Fr(r['alpha']) == ALPHA(k) and d['N'] == r['N']

def recert(sample=None):
    cells = read_jsonl('coverB_cells_*.jsonl')
    used = set()
    for c in cells:
        if c['type'] == 'B':
            for key, (lb, n) in c['nodes'].items(): used.add((c['verts'][int(key.split(':')[0])], int(key.split(':')[1]), n))
    vrec = [r for r in read_jsonl('coverB_nodes_*.jsonl') if (r['w'], r['k'], r['n']) in used]
    ring = read_jsonl('ring_nodes.jsonl'); l2 = read_jsonl('lam2_nodes.jsonl')
    if sample:
        random.seed(2026); vrec = random.sample(vrec, min(sample, len(vrec))); ring = random.sample(ring, min(sample // 3, len(ring))); l2 = random.sample(l2, 3)
    ok = True; t0 = time.time(); cnt = [0, 0, 0]
    for q, (lst, fun) in enumerate(((vrec, recert_vertex), (ring, recert_ring), (l2, recert_lam2))):
        for r in lst:
            try: good = fun(r)
            except Exception as e: good = False; print('exception', e)
            if not good: ok = False; print('re-certification FAILED', q, r.get('key', (r.get('w'), r.get('k'))))
            cnt[q] += 1
    res = dict(ok=ok, mode='sample' if sample else 'full', vertex_records=cnt[0], vertex_records_used_by_cells=len(used),
               ring_records=cnt[1], lam2_records=cnt[2], seconds=round(time.time() - t0))
    print(res); json.dump(res, open(os.path.join(OUT, 'recheck_%s.json' % res['mode']), 'w'))
    return ok

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'cells': ok = check_cells()
    elif mode == 'sample': ok = recert(int(sys.argv[2]))
    elif mode == 'full': ok = recert(None)
    sys.exit(0 if ok else 1)

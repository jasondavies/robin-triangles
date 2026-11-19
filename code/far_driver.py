"""Adaptive (a,b,beta) certification of the far region for beta in [BLO,BHI].
Uses far_cert.Interval / check_cell; local radii from $CELLS_DIR/near_L*.json.
Writes $CELLS_DIR/far_<tag>.json with every certified cell and its trial data."""
import sys, json, time, glob
import mpmath as mp
from flint import arb, fmpq
from rig import Q, I
from equi import beta_of, PSTAR
import far_cert
from far_cert import Interval, bmin_lower
from out import cells_path

DEBUG = False

def load_local():
    L = []
    for f in sorted(glob.glob(cells_path('near_L*.json'))):
        d = json.load(open(f))
        for M1, M2, r, R, cells in d['intervals']:
            b1 = beta_of(arb(Q(M1))).lower(); b2 = beta_of(arb(Q(M2))).upper()
            L.append((arb(b1), arb(b2), Q(R), Q(M1), Q(M2)))
    L.sort(key=lambda t: float(t[3].p) / float(t[3].q))
    return L

def r_loc_for(L, b1, b2):
    rs = [r for (lb1, lb2, r, M1, M2) in L if not (lb2 < arb(b1) or lb1 > arb(b2))]
    return min(rs) if rs else None

def run(BLO, BHI, tag, w0='0.01', a_max='6'):
    BLO = Q(BLO); BHI = Q(BHI); w0 = Q(w0); a_max = Q(a_max)
    L = load_local()
    cache = {}
    def get_iv(b1, b2):
        k = (b1, b2)
        if k not in cache:
            if len(cache) > 400:
                cache.clear()
            cache[k] = Interval(b1, b2)
        return cache[k]
    cells = []; thin = []; fails = []
    t0 = time.time()
    b = BLO
    tasks = []
    while b < BHI:
        b2 = min(b + w0, BHI)
        tasks.append((b, b2)); b = b2
    for (B1, B2) in reversed(tasks):
        stack = [(B1, B2, fmpq(1), a_max, None, fmpq(0), 0, 0)]
        while stack:
            b1, b2, a1, a2, c1, c2, d, nb = stack.pop()
            iv = get_iv(b1, b2)
            r_loc = r_loc_for(L, b1, b2)
            if r_loc is None:
                fails.append(('no local cover', str(b1), str(b2))); continue
            if c1 is None:
                # initial cell for this beta-interval; thin check
                ok_thin = bool(arb(b1) * 2 * arb(a_max) / 3 * PSTAR > -iv.lam_lo[1])
                thin.append((str(b1), str(b2), str(a_max), ok_thin))
                if not ok_thin:
                    fails.append(('thin', str(b1), str(b2))); continue
                c1 = Q(bmin_lower(a_max).lower().mid().str(20, radius=False)) - fmpq(1, 10 ** 6)
            if c2 < 0 and arb(c2) < bmin_lower(a2):
                continue
            if a1 >= 1 - r_loc and a2 <= 1 + r_loc and c1 >= -r_loc and c2 <= r_loc:
                continue
            ok, lam, res, C = iv.check_cell_data(a1, a2, c1, c2)
            if ok:
                cells.append([str(b1), str(b2), str(a1), str(a2), str(c1), str(c2), C])
                if len(cells) % 500 == 0:
                    print(tag, 'cells', len(cells), 'beta', float(b1.p) / float(b1.q), 'stack', len(stack), round(time.time() - t0), flush=True)
                continue
            if d > 80:
                fails.append(('depth', str(b1), str(b2), str(a1), str(a2), str(c1), str(c2))); continue
            da = max(a1 - 1, 1 - a2, 0); db = max(c1, -c2, 0)
            dist = max(da, db); size = max(a2 - a1, c2 - c1)
            if size * 16 <= dist and nb < 3 and (b2 - b1) > fmpq(1, 10 ** 7):
                if DEBUG and (b2 - b1) < fmpq(1, 10 ** 4):
                    print('BSPLIT', float(b1), float(b2), float(a1), float(a2), float(c1), float(c2), [x.str(5) for x in res], lam, flush=True)
                bm = (b1 + b2) / 2
                stack += [(b1, bm, a1, a2, c1, c2, d + 1, nb + 1), (bm, b2, a1, a2, c1, c2, d + 1, nb + 1)]
            else:
                am = (a1 + a2) / 2; cm = (c1 + c2) / 2
                if (a2 - a1) >= (c2 - c1):
                    stack += [(b1, b2, a1, am, c1, c2, d + 1, 0), (b1, b2, am, a2, c1, c2, d + 1, 0)]
                else:
                    stack += [(b1, b2, a1, a2, c1, cm, d + 1, 0), (b1, b2, a1, a2, cm, c2, d + 1, 0)]
    res = dict(BLO=str(BLO), BHI=str(BHI), a_max=str(a_max), ncells=len(cells), thin=thin,
               fails=fails, ok=not fails, cells=cells, seconds=round(time.time() - t0))
    json.dump(res, open(cells_path(f'far_{tag}.json'), 'w'))
    print('DONE', tag, 'cells', len(cells), 'fails', len(fails), round(time.time() - t0), 's', flush=True)

if __name__ == '__main__':
    run(*sys.argv[1:])

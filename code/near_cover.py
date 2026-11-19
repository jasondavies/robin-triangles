"""For each local M-interval: certify Hessian box |a-1|,|b| <= r, and value-check cells covering
{ r < |(a-1,b)|_inf <= R } (within the fundamental domain), with the exact-psi0 local trial.
Chooses R adaptively (largest of a few multiples of r that succeeds).  Writes $CELLS_DIR/near_<tag>.json"""
import sys, json, time
from flint import arb, fmpq
from rig import Q
from local_cert2 import NearCertifier
from far_cert import bmin_lower
from out import cells_path

def cover_annulus(nc, r, R, maxcells=4000):
    """cells covering [1-R,1+R]x[-R,0] minus the r-box, intersected with domain (a>=1 region suffices
    since a<1 is outside the fundamental domain)."""
    stack = [(fmpq(1), 1 + R, -R, fmpq(0), 0)]
    cells = []
    while stack:
        a1, a2, b1, b2, d = stack.pop()
        if b2 < 0 and arb(b2) < bmin_lower(a2):
            continue
        if a2 <= 1 + r and b1 >= -r:
            continue
        if nc.cell_ok(a1, a2, b1, b2):
            cells.append((str(a1), str(a2), str(b1), str(b2)))
            if len(cells) > maxcells:
                return None
            continue
        if d > 14:
            return None
        am = (a1 + a2) / 2; bm = (b1 + b2) / 2
        stack += [(a1, am, b1, bm, d + 1), (am, a2, b1, bm, d + 1), (a1, am, bm, b2, d + 1), (am, a2, bm, b2, d + 1)]
    return cells

def run(tag, factors=(8, 6, 4, 2), nsplit=10):
    loc = json.load(open(cells_path(f'local_{tag}.json')))
    out = []; t0 = time.time()
    for k, (M1, M2, r) in enumerate(loc['intervals']):
        r = Q(r); M1q = Q(M1); M2q = Q(M2)
        for s in range(nsplit):
            m1 = M1q + (M2q - M1q) * s / nsplit; m2 = M1q + (M2q - M1q) * (s + 1) / nsplit
            nc = NearCertifier(m1, m2)
            best = None
            for f in factors:
                R = r * f
                cells = cover_annulus(nc, r, R)
                if cells is not None:
                    best = (R, cells); break
            if best is None:
                out.append([str(m1), str(m2), str(r), str(r), []])
            else:
                out.append([str(m1), str(m2), str(r), str(best[0]), best[1]])
        if k % 10 == 0:
            print(tag, k, M1, 'r', r, 'R', best[0] if best else None, 'ncells', len(best[1]) if best else 0, round(time.time() - t0), flush=True)
    json.dump(dict(tag=tag, intervals=out), open(cells_path(f'near_{tag}.json'), 'w'))
    print('DONE', tag, round(time.time() - t0))

if __name__ == '__main__':
    run(sys.argv[1])

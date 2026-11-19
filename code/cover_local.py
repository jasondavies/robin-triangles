"""Cover M in [M_lo, M_hi] by intervals on which local_cert2 certifies the near-equilateral box.
Writes $CELLS_DIR/local_<tag>.json : list of [M1, M2, r] (exact rationals as strings)."""
import sys, json, time, math
from flint import arb, fmpq
from rig import Q
from local_cert2 import certify
from out import cells_path

# conservative radius table from rsearch (M -> 1/r)
TABLE = [(1.30, 40), (1.6, 60), (1.9, 120), (2.2, 160), (2.5, 320), (2.8, 640), (3.1, 1000), (3.4, 2000), (3.7, 4000), (4.0, 12000), (4.4, 100000)]
def rinv(M):
    for (m, ri) in TABLE:
        if M <= m:
            return ri
    return TABLE[-1][1]

def run(Mlo, Mhi, tag):
    Mlo = Q(Mlo); Mhi = Q(Mhi)
    out = []; M = Mlo; dM = Q('0.005'); t0 = time.time()
    while M < Mhi:
        M2 = min(M + dM, Mhi)
        ri = rinv(float(M2.p) / float(M2.q))
        r = fmpq(1, ri)
        done = False
        for attempt in range(12):
            if certify(M, M2, r, nsub=4):
                done = True; break
            # first shrink dM, then r
            if attempt % 2 == 0:
                M2 = M + (M2 - M) / 2
            else:
                r = r / 2
        if not done:
            print('FAILED at', M, flush=True); break
        out.append([str(M), str(M2), str(r)])
        M = M2
        if len(out) % 50 == 0:
            print(len(out), float(M.p) / float(M.q), 'r=', r, round(time.time() - t0), flush=True)
    res = dict(Mlo=str(Mlo), Mhi=str(Mhi), complete=(M >= Mhi), intervals=out)
    json.dump(res, open(cells_path(f'local_{tag}.json'), 'w'))
    print('done', tag, 'complete', M >= Mhi, len(out), 'intervals', round(time.time() - t0), 's')

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3])

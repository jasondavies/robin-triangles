"""Re-verify the local (Hessian) and near-annulus certificates with the current code.
Usage: verify_local_near.py L1   (reads $CELLS_DIR/local_L1.json and $CELLS_DIR/near_L1.json;
writes ../verification/cells_localnear_L1.json)"""
import sys, json, time
from rig import Q
from local_cert2 import certify, NearCertifier
from near_cover import cover_annulus
from out import cells_path, write

def main(tag):
    t0 = time.time()
    loc = json.load(open(cells_path(f'local_{tag}.json')))
    near = json.load(open(cells_path(f'near_{tag}.json')))
    bad = []
    # local: contiguous cover of [Mlo, Mhi]
    prev = Q(loc['Mlo'])
    for M1, M2, r in loc['intervals']:
        if Q(M1) != prev: bad.append(('gap', M1))
        prev = Q(M2)
        if not certify(Q(M1), Q(M2), Q(r), nsub=4): bad.append(('local', M1, M2, r))
    if prev < Q(loc['Mhi']): bad.append(('end', str(prev)))
    # near: each local interval split into sub-intervals, each with radius r (same as local) and R
    locmap = [(Q(a), Q(b), Q(r)) for a, b, r in loc['intervals']]
    prev = Q(loc['Mlo']); ncells = 0
    for m1, m2, r, R, cells in near['intervals']:
        m1, m2, r, R = Q(m1), Q(m2), Q(r), Q(R)
        if m1 != prev: bad.append(('near gap', str(m1)))
        prev = m2
        # the Hessian radius r used must be certified by the local interval containing [m1,m2]
        host = [x for x in locmap if x[0] <= m1 and m2 <= x[1]]
        if not host or host[0][2] < r: bad.append(('near host', str(m1)))
        if R > r:
            nc = NearCertifier(m1, m2)
            c = cover_annulus(nc, r, R)
            if c is None: bad.append(('near', str(m1), str(m2)))
            else: ncells += len(c)
    if prev < Q(loc['Mhi']): bad.append(('near end', str(prev)))
    res = dict(tag=tag, local_intervals=len(loc['intervals']), near_intervals=len(near['intervals']),
               near_cells=ncells, bad=bad[:20], ok=not bad, seconds=round(time.time() - t0))
    print(res)
    write(f'cells_localnear_{tag}.json', res)

if __name__ == '__main__':
    main(sys.argv[1])

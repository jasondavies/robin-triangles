"""Independent replay of a far certificate: re-traverses the deterministic subdivision tree,
re-checks every certified cell in Arb with the STORED trial coefficients (no floating-point solver),
and checks that every leaf is either certified, outside the fundamental domain, or inside the
near-equilateral box covered by the near/local certificates.  Usage: verify_far.py far_F1.json"""
import sys, json, time
from flint import arb, fmpq
from rig import Q, I, Jet2
from equi import PSTAR
import sixwave as sw
from nums import ARB
from far_cert import Interval, bmin_lower, to_arb
from far_driver import load_local, r_loc_for
from out import write
import os

def check_with(iv, a1, a2, b1, b2, C):
    Ca = [[arb(v) for v in c] for c in C]
    T = sw.tensors(ARB, iv.Wa, iv.Na, iv.Sa, Ca)
    ac = (a1 + a2) / 2; bc = (b1 + b2) / 2
    for beta, llo in ((iv.b1, iv.lam_lo[0]), (iv.b2, iv.lam_lo[1])):
        alpha = -arb(beta)
        Num, Den = sw.num_den(ARB, T, alpha, arb(ac), arb(bc), [arb(1), arb(0), arb(0)])
        F0 = Num - llo * Den
        A = Jet2.var(I(a1, a2), 0); B = Jet2.var(I(b1, b2), 1)
        NumJ, DenJ = sw.num_den(ARB, T, alpha, A, B, [Jet2(1), A - ac, B - bc])
        FJ = NumJ - llo * DenJ
        Fub = F0 + FJ.gx * I(a1 - ac, a2 - ac) + FJ.gy * I(b1 - bc, b2 - bc)
        if not (Fub < 0):
            return False
    return True

def main(fn):
    d = json.load(open(fn))
    BLO, BHI, a_max = Q(d['BLO']), Q(d['BHI']), Q(d['a_max'])
    cert = {}
    for b1, b2, a1, a2, c1, c2, C in d['cells']:
        cert[(Q(b1), Q(b2), Q(a1), Q(a2), Q(c1), Q(c2))] = C
    L = load_local()
    cache = {}
    nver = 0; bad = []; used = set(); t0 = time.time()
    w0 = Q('0.01')
    tasks = []; b = BLO
    while b < BHI:
        b2 = min(b + w0, BHI); tasks.append((b, b2)); b = b2
    for (B1, B2) in tasks:
        stack = [(B1, B2, fmpq(1), a_max, None, fmpq(0), 0, 0)]
        while stack:
            b1, b2, a1, a2, c1, c2, dd, nb = stack.pop()
            if (b1, b2) not in cache:
                if len(cache) > 400: cache.clear()
                cache[(b1, b2)] = Interval(b1, b2)
            iv = cache[(b1, b2)]
            r_loc = r_loc_for(L, b1, b2)
            if c1 is None:
                if not bool(arb(b1) * 2 * arb(a_max) / 3 * PSTAR > -iv.lam_lo[1]):
                    bad.append(('thin', str(b1))); continue
                c1 = Q(bmin_lower(a_max).lower().mid().str(20, radius=False)) - fmpq(1, 10 ** 6)
            if c2 < 0 and arb(c2) < bmin_lower(a2):
                continue
            if a1 >= 1 - r_loc and a2 <= 1 + r_loc and c1 >= -r_loc and c2 <= r_loc:
                continue
            key = (b1, b2, a1, a2, c1, c2)
            if key in cert:
                used.add(key)
                if check_with(iv, a1, a2, c1, c2, cert[key]):
                    nver += 1
                else:
                    bad.append(('cell', [str(x) for x in key]))
                continue
            if dd > 80:
                bad.append(('uncovered', [str(x) for x in key])); continue
            da = max(a1 - 1, 1 - a2, 0); db = max(c1, -c2, 0)
            dist = max(da, db); size = max(a2 - a1, c2 - c1)
            if size * 16 <= dist and nb < 3 and (b2 - b1) > fmpq(1, 10 ** 7):
                bm = (b1 + b2) / 2
                stack += [(b1, bm, a1, a2, c1, c2, dd + 1, nb + 1), (bm, b2, a1, a2, c1, c2, dd + 1, nb + 1)]
            else:
                am = (a1 + a2) / 2; cm = (c1 + c2) / 2
                if (a2 - a1) >= (c2 - c1):
                    stack += [(b1, b2, a1, am, c1, c2, dd + 1, 0), (b1, b2, am, a2, c1, c2, dd + 1, 0)]
                else:
                    stack += [(b1, b2, a1, a2, c1, cm, dd + 1, 0), (b1, b2, a1, a2, cm, c2, dd + 1, 0)]
    res = dict(file=os.path.basename(fn), verified_cells=nver, stored_cells=len(cert), unused=len(cert) - len(used), bad=bad[:20], nbad=len(bad),
               ok=(not bad), seconds=round(time.time() - t0))
    print(res)
    write('cells_far_%s.json' % os.path.basename(fn)[:-5], res)
    return res

if __name__ == '__main__':
    main(sys.argv[1])

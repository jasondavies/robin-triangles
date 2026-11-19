"""Exact (rational arithmetic) check of Proposition "middle" (a) and of the bookkeeping of the cover:

 1. the leaf cells recorded in coverB_cells_*.jsonl form, for every root triangle, a complete
    subdivision tree (children = the four midpoint sub-triangles), and every leaf has type B, L or W;
 2. the root triangles are positively oriented and every side is matched by exactly one opposite side or is a
    boundary side (hypothesis of Lemma "coverage");
 3. every boundary side lies in  {|w| <= R_LOC}  or  {w2 <= 0}  or  {w2 >= sqrt(3) w1}  or  {F >= 0},
    F(w) = (1+w1)^2 - 81 (1-|w|^2);
 4. the point (1/2, 1/5) of Omega_B is interior to a root triangle and lies on no line carrying a side of a root triangle;
 5. leaves of type L have all vertices in |w| <= R_LOC, leaves of type W have F >= 0 on the whole triangle
    (exact minimum of the convex quadratic F over the triangle), and all vertices are in the unit disc.
Usage: python3 coverage_check.py
"""
import os, sys, json, glob
from fractions import Fraction as Fr
from config import CERT, OUT, R_LOC

def pw(s):
    a, b = s.split(','); return (Fr(a), Fr(b))
def F(p): return (1 + p[0]) ** 2 - 81 * (1 - p[0] ** 2 - p[1] ** 2)
def minF_segment(p, q):
    """exact minimum of F on the segment [p,q]"""
    d = (q[0] - p[0], q[1] - p[1])
    A = d[0] ** 2 + 81 * (d[0] ** 2 + d[1] ** 2)                         # F(p + s d) = A s^2 + B s + C
    B = 2 * (1 + p[0]) * d[0] + 162 * (p[0] * d[0] + p[1] * d[1]); C = F(p)
    vals = [C, A + B + C]
    if A > 0:
        s = -B / (2 * A)
        if 0 < s < 1: vals.append(C - B * B / (4 * A))
    return min(vals)
def orient(a, b, c): return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
def inside_strict(x, tr): return all(orient(tr[i], tr[(i + 1) % 3], x) > 0 for i in range(3))
def minF_triangle(tr):
    """exact minimum of the strictly convex quadratic F over a closed triangle"""
    c = (Fr(-1, 82), Fr(0))                                               # the critical point of F
    if all(orient(tr[i], tr[(i + 1) % 3], c) >= 0 for i in range(3)): return F(c)
    return min(minF_segment(tr[i], tr[(i + 1) % 3]) for i in range(3))
def children(v):
    mid = lambda p, q: ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    m01, m12, m02 = mid(v[0], v[1]), mid(v[1], v[2]), mid(v[0], v[2])
    return [(v[0], m01, m02), (m01, v[1], m12), (m02, m12, v[2]), (m01, m12, m02)]

def main():
    roots = [tuple(pw(s) for s in tr) for tr in json.load(open(os.path.join(CERT, 'coverB_roots.json')))]
    cells = {}
    for fn in sorted(glob.glob(os.path.join(CERT, 'coverB_cells_*.jsonl'))):
        for line in open(fn):
            if line.strip():
                r = json.loads(line); cells[r['id']] = r
    ok = True; cnt = {'B': 0, 'L': 0, 'W': 0}
    # 1 + 5: trees and leaf types
    def walk(cid, v):
        nonlocal ok
        r = cells.get(cid)
        if r is None or tuple(pw(s) for s in r['verts']) != tuple(v):
            print('missing or inconsistent cell', cid); ok = False; return
        if r['type'] == 'split':
            for q, ch in enumerate(children(v)): walk(cid + '.%d' % q, ch)
            return
        if r['type'] not in cnt: print('bad leaf', cid, r['type']); ok = False; return
        cnt[r['type']] += 1
        if not all(p[0] ** 2 + p[1] ** 2 < 1 for p in v): ok = False; print('vertex outside disc', cid)
        if r['type'] == 'L' and not all(p[0] ** 2 + p[1] ** 2 <= R_LOC ** 2 for p in v): ok = False; print('bad L', cid)
        if r['type'] == 'W' and not minF_triangle(v) >= 0: ok = False; print('bad W', cid)
    for i, v in enumerate(roots): walk('r%d' % i, v)
    # 2: orientation and side matching
    sides = {}
    for i, v in enumerate(roots):
        if not orient(*v) > 0: ok = False; print('orientation', i)
        for a in range(3):
            sides.setdefault((v[a], v[(a + 1) % 3]), []).append(i)
    bsides = []
    for (p, q), lst in sides.items():
        if len(lst) != 1: ok = False; print('directed side occurs twice', p, q)
        opp = sides.get((q, p), [])
        if len(opp) == 0: bsides.append((p, q))
        elif len(opp) != 1: ok = False; print('side matched more than once')
    # 3: boundary sides
    kinds = {'disc': 0, 'bottom': 0, 'top': 0, 'thin': 0}
    for p, q in bsides:
        if all(x[0] ** 2 + x[1] ** 2 <= R_LOC ** 2 for x in (p, q)): kinds['disc'] += 1
        elif all(x[1] <= 0 for x in (p, q)): kinds['bottom'] += 1
        elif all(x[1] >= 0 and x[0] >= 0 and x[1] ** 2 >= 3 * x[0] ** 2 for x in (p, q)): kinds['top'] += 1
        elif minF_segment(p, q) >= 0: kinds['thin'] += 1
        else: ok = False; print('boundary side not in an allowed set', p, q)
    # 4: a point of Omega_B in the interior of a root triangle
    x = (Fr(1, 2), Fr(1, 5))
    in_omega = x[0] ** 2 + x[1] ** 2 > R_LOC ** 2 and x[1] > 0 and x[1] ** 2 < 3 * x[0] ** 2 and F(x) < 0
    hit = [i for i, v in enumerate(roots) if inside_strict(x, v)]
    on_no_side = all(orient(v[i], v[(i + 1) % 3], x) != 0 for v in roots for i in range(3))
    if not (in_omega and len(hit) == 1 and on_no_side): ok = False; print('interior point test failed', hit)
    out = dict(ok=ok, roots=len(roots), leaves=cnt, boundary_sides=len(bsides), boundary_kinds=kinds, interior_point_root=hit)
    print(out)
    json.dump(out, open(os.path.join(OUT, 'coverage_check.json'), 'w'))
    return ok

if __name__ == '__main__':
    sys.exit(0 if main() else 1)

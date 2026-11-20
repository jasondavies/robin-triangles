"""Certified Crouzeix-Raviart (CR) lower bounds for the lowest Robin eigenvalue of a triangle.

A triangle is given ONLY by intrinsic data: squared side lengths a2[0..2] (arb), its area A (arb) and one
Robin coefficient beta[j] >= 0 per side (arb).  Nothing else enters (no coordinates).

Meshes (both are conforming triangulations into triangles without obtuse angles):
  'U'  uniform subdivision of T into n^2 congruent copies of T/n            (needs T acute);
  'A'  T is first cut along the altitude on its longest side into two right triangles, each of which
       is subdivided uniformly into n^2 congruent right triangles (the two sub-meshes match on the altitude).

CR basis: one function per mesh edge (value 1 at its midpoint).  With cot(theta) of the element angles,
   K_loc[k,l] = -2 cot(theta_m) (k != l, m the third index),  M = diag(|elements adjacent|/3),
   boundary term = beta_j * |e| on the diagonal for a boundary edge e on side j.
Since all cotangents are >= 0, K is a symmetric Z-matrix and M is diagonal positive, so for any positive
vector x:   lambda_{1,h} >= min_i (Kx)_i/(Mx)_i   (Collatz-Wielandt; Lemma CW of the paper).
Then  lambda_1 >= s/(1 + (C h)^2 s) for every 0 < s <= lambda_{1,h}  (Lemma CR), h = largest element diameter.

Certification (function `certify`): the vector x is rounded to positive INTEGERS, the row sums are formed in
exact integer arithmetic, and the finitely many real coefficients (cotangents, areas, edge lengths, beta) enter
through rigorous rational lower/upper bounds obtained from Arb balls.  The final test  Kx >= sigma M x  is a
sign test on exact integers.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla
from fractions import Fraction
from flint import arb, fmpz, ctx
import config

ctx.prec = 100
C_INTERP = config.C_INTERP           # ||u - Pi_h u||_{L2(K)} <= C_INTERP * diam(K) * ||grad(u - Pi_h u)||_{L2(K)}
BITS = 60                            # coefficient bounds are rounded outward to multiples of 2^-BITS

# ---------------------------------------------------------------- exact conversions
def arb_lo_frac(x):
    """exact Fraction that is a lower bound for the ball x (its lower endpoint rounded down to the working precision)"""
    m, e = x.lower().man_exp(); m = int(m); e = int(e)
    return Fraction(m * 2 ** e) if e >= 0 else Fraction(m, 2 ** (-e))
def arb_hi_frac(x):
    m, e = x.upper().man_exp(); m = int(m); e = int(e)
    return Fraction(m * 2 ** e) if e >= 0 else Fraction(m, 2 ** (-e))
def ilo(x):
    """integer L with L/2^BITS <= every point of the ball x"""
    f = arb_lo_frac(x) * 2 ** BITS; return f.numerator // f.denominator
def ihi(x):
    f = arb_hi_frac(x) * 2 ** BITS; return -((-f.numerator) // f.denominator)
def frac_to_arb(q):
    q = Fraction(q); return arb(fmpz(q.numerator)) / arb(fmpz(q.denominator))

# ---------------------------------------------------------------- combinatorics of one uniform patch
_pc = {}
def patch(n):
    """Uniform patch with lattice vertices (i,j), i+j<=n.  Edge families: 0: (i,j)-(i+1,j); 1: (i,j)-(i,j+1);
    2: (i+1,j)-(i,j+1); each indexed by (i,j), i+j<=n-1, local id = fam*m + idx(i,j), m = n(n+1)/2.
    Returns (elems [ne,3] with column k = edge of family k, bd0, bd1, bd2 = local ids of the boundary edges of
    family 0 (j=0), family 1 (i=0), family 2 (i+j=n-1), each ordered along the side)."""
    if n in _pc: return _pc[n]
    m = n * (n + 1) // 2
    idx = lambda i, j: i * n - i * (i - 1) // 2 + j
    I, J = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    up = (I + J) <= n - 1; iu, ju = I[up], J[up]
    dn = (I + J) <= n - 2; idn, jd = I[dn], J[dn]
    eu = np.stack([idx(iu, ju), m + idx(iu, ju), 2 * m + idx(iu, ju)], 1)
    ed = np.stack([idx(idn, jd + 1), m + idx(idn + 1, jd), 2 * m + idx(idn, jd)], 1)
    k = np.arange(n)
    out = (np.concatenate([eu, ed]).astype(np.int64), idx(k, 0 * k), m + idx(0 * k, k), 2 * m + idx(k, n - 1 - k))
    _pc[n] = out
    return out

class Mesh:
    """Combinatorial mesh + list of real coefficients (as arb) attached to classes."""
    pass

def tri_cots(a2, A):
    """cotangent of the angle opposite side j: (sum a_k^2 - 2 a_j^2)/(4A)"""
    s = a2[0] + a2[1] + a2[2]
    return [(s - 2 * a2[j]) / (4 * A) for j in range(3)]

def choose_kind(a2, A, thresh=0.02):
    c = tri_cots(a2, A)
    return 'U' if all(bool(x > thresh) for x in c) else 'A'

def build_mesh(a2, A, n, kind):
    """a2, A: arb.  Returns Mesh with fields:
       N, elems [ne,3], epatch [ne], npatch, cot[p][k] (arb or exact 0), area[p] (element area, arb),
       bdry: list of (edge ids, side j, edge length arb), h (arb, largest element diameter)."""
    M = Mesh(); M.n = n; M.kind = kind
    if not (isinstance(n, int) and n >= 1): raise ValueError('mesh parameter must be a positive integer')
    if not bool(A > 0): raise ValueError('area not certified positive')      # M must be positive diagonal (Lemma CW)
    el, b0, b1, b2 = patch(n); m3 = 3 * n * (n + 1) // 2
    c = tri_cots(a2, A); a = [x.sqrt() for x in a2]
    if kind == 'U':
        if not all(bool(x > 0) for x in c): raise ValueError('triangle not certified acute')
        M.N = m3; M.elems = el; M.epatch = np.zeros(len(el), dtype=np.int64); M.npatch = 1
        # P0P1 = side 0, P1P2 = side 1, P2P0 = side 2;  fam0 || side0, fam1 || side2, fam2 || side1
        M.cot = [[c[0], c[2], c[1]]]; M.area = [A / (n * n)]
        M.bdry = [(b0, 0, a[0] / n), (b1, 2, a[2] / n), (b2, 1, a[1] / n)]
        M.h = max(a[0].upper(), a[1].upper(), a[2].upper()) / n
        M.sides = (0, 1, 2)
    elif kind == 'A':
        # longest side L (decided on midpoints; any choice with the two other angles acute is valid)
        L = max(range(3), key=lambda j: float(a2[j].mid())); p, q = [(L + 1) % 3, (L + 2) % 3]
        if not (bool(c[p] > 0) and bool(c[q] > 0)): raise ValueError('base angles not certified acute')
        hL = 2 * A / a[L]
        # patch 1 = (F, V_b, V_a): right angle at F, angle B at V_b with cot B = c[q]; hypotenuse = side p
        # patch 2 = (F, V_a, V_c): right angle at F, angle C at V_c with cot C = c[p]; hypotenuse = side q
        remap = np.arange(m3); remap[b0] = b1                       # patch-2 fam0 (j=0) edges == patch-1 fam1 (i=0) edges
        other = np.setdiff1d(np.arange(m3), b0); remap[other] = m3 + np.arange(len(other))
        M.N = m3 + len(other)
        M.elems = np.concatenate([el, remap[el]]); M.epatch = np.concatenate([np.zeros(len(el), np.int64), np.ones(len(el), np.int64)])
        M.npatch = 2
        zero = arb(0)
        M.cot = [[1 / c[q], c[q], zero], [c[p], 1 / c[p], zero]]
        M.area = [hL * hL * c[q] / (2 * n * n), hL * hL * c[p] / (2 * n * n)]
        M.bdry = [(b0, L, hL * c[q] / n), (b2, p, a[p] / n), (remap[b1], L, hL * c[p] / n), (remap[b2], q, a[q] / n)]
        M.h = max(a[p].upper(), a[q].upper()) / n
        M.sides = (L, p, q)
    else:
        raise ValueError(kind)
    return M

# ---------------------------------------------------------------- floating point part (untrusted)
def float_mats(M, beta):
    f = lambda x: float(x.mid()) if isinstance(x, arb) else float(x)
    N = M.N; el = M.elems; ep = M.epatch
    cot = np.array([[f(x) for x in row] for row in M.cot])[ep]          # [ne,3]
    area = np.array([f(x) for x in M.area])[ep]
    rows = []; cols = []; vals = []
    for k in range(3):
        l, m = (k + 1) % 3, (k + 2) % 3
        rows += [el[:, k], el[:, k], el[:, l]]; cols += [el[:, l], el[:, k], el[:, l]]
        vals += [-2 * cot[:, m], 2 * cot[:, m], 2 * cot[:, m]]
        rows.append(el[:, l]); cols.append(el[:, k]); vals.append(-2 * cot[:, m])
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    md = np.zeros(N)
    for k in range(3): np.add.at(md, el[:, k], area / 3)
    bd = np.zeros(N)
    for ids, j, ln in M.bdry: bd[ids] += f(beta[j]) * f(ln)
    return (K + sp.diags(bd)).tocsc(), md

def float_eig(M, beta, x0=None, maxsweep=None):
    """approximate lowest eigenpair (untrusted).  The vector is made componentwise accurate by damped Jacobi
    sweeps on (K - lam M) x = 0, which involve no cancellation and preserve positivity; this matters for the
    tiny components near corners and on the boundary when the Robin coefficient is large."""
    K, md = float_mats(M, beta)
    lam, vec = spla.eigsh(K, k=1, M=sp.diags(md).tocsc(), sigma=0.0, which='LM', v0=x0)
    lam = float(lam[0]); x = vec[:, 0]; x = np.abs(x * np.sign(x.sum())); x[x == 0] = x[x > 0].min()
    d = K.diagonal(); Nn = (sp.diags(d) - K).tocsr(); den = d - lam * md
    if maxsweep is None: maxsweep = max(4000, 400 * M.n)
    rbest = -np.inf; xbest = x; done = 0; stall = 0
    while done < maxsweep:
        for _ in range(50):
            x = 0.5 * x + 0.5 * (Nn @ x) / den
        done += 50
        x = x / x.max()
        r = float(((K @ x) / (md * x)).min())
        if r > rbest + 1e-13 * abs(lam):
            rbest = r; xbest = x.copy(); stall = 0
        else:
            stall += 1
        if rbest >= lam * (1 - 2e-9) or stall >= 20: break
    return lam, xbest, rbest, K, md

# ---------------------------------------------------------------- exact certificate
def exact_ints(x):
    """x: positive float64 vector.  Returns (xi, E): object array of Python ints with xi_k = x_k * 2^E EXACTLY."""
    x = np.asarray(x, dtype=np.float64)
    if not (np.isfinite(x).all() and (x > 0).all()): raise ValueError('vector not positive')
    mant, ex = np.frexp(x)                                   # x = mant * 2^ex, 0.5 <= mant < 1
    m53 = np.ldexp(mant, 53).astype(np.int64)                # exact
    sh = (ex - ex.min()).astype(np.int64)
    xi = np.array([int(m) << int(s) for m, s in zip(m53, sh)], dtype=object)
    return xi, 53 - int(ex.min())

def _split(xi, nparts, w=30):
    mask = (1 << w) - 1
    return [np.array([(int(v) >> (w * q)) & mask for v in xi], dtype=np.int64) for q in range(nparts)]

def certify(M, beta, x, sigma=None, shrink=1e-10):
    """x: positive float64 vector (untrusted).  It is converted EXACTLY to an integer vector xi (a positive
    multiple of x), and  K xi >= sigma M xi  is verified by a sign test on exact integers, the real coefficients
    entering through rational bounds with denominator 2^BITS obtained from Arb balls.
    If sigma is given, exactly that rational is tested (it must be > 0) and returned; otherwise the largest
    multiple of 2^-BITS that the row bounds certify, reduced by a relative `shrink`, is proposed.
    Returns (sigma Fraction, sha256 of x): the certified value; callers must use THIS value.  Hence
    lambda_{1,h} >= sigma > 0 (Lemma CW).  Raises ValueError if the test fails or sigma <= 0."""
    N = M.N; el = M.elems; ep = M.epatch
    xi, _ = exact_ints(x)
    w = 30; nparts = (max(int(v) for v in xi).bit_length() + w - 1) // w
    parts = _split(xi, nparts, w)
    lo = np.zeros(N, dtype=object); cnt = []
    for p in range(M.npatch):
        sel = ep == p; e = el[sel]
        for m in range(3):
            k, l = (m + 1) % 3, (m + 2) % 3
            c = M.cot[p][m]
            if c.is_zero(): continue                            # exact right angle: no coupling
            clo, chi = ilo(2 * c), ihi(2 * c)
            if clo < 0: raise ValueError('cotangent not certified nonnegative')
            S = np.zeros(N, dtype=object)
            for q in range(nparts):                              # exact int64 arithmetic on 30-bit limbs
                Sq = np.zeros(N, dtype=np.int64)
                dkl = parts[q][e[:, k]] - parts[q][e[:, l]]
                np.add.at(Sq, e[:, k], dkl); np.add.at(Sq, e[:, l], -dkl)
                S = S + (Sq.astype(object) << (w * q))
            neg = np.array([v < 0 for v in S], dtype=bool)
            lo = lo + np.where(neg, chi * S, clo * S)            # 2c*S >= (chi if S<0 else clo) * S / 2^BITS
        c_ = np.zeros(N, dtype=np.int64)
        for k in range(3): np.add.at(c_, e[:, k], 1)
        cnt.append(c_)
    for ids, j, ln in M.bdry:
        b = ilo(beta[j] * ln)
        if b < 0: raise ValueError('negative Robin coefficient bound')
        lo[ids] = lo[ids] + b * xi[ids]
    mass = np.zeros(N, dtype=object)
    for p in range(M.npatch):
        mass = mass + ihi(M.area[p] / 3) * cnt[p].astype(object) * xi
    # now  lo/2^BITS <= (K xi)_i  and  (M xi)_i <= mass/2^BITS  for every row i
    lo_l = [int(v) for v in lo]; mass_l = [int(v) for v in mass]
    if min(lo_l) <= 0 or min(mass_l) <= 0: raise ValueError('row sum not positive')
    if sigma is None:
        s = min((a << BITS) // b for a, b in zip(lo_l, mass_l))
        s = int(s * (1 - shrink))                              # (proposal only: the test below decides)
        sig = Fraction(s, 2 ** BITS)
    else:
        sig = Fraction(sigma)                                    # the EXACT supplied rational is tested, no rounding
    if not sig > 0: raise ValueError('sigma not positive: Lemma CR requires 0 < sigma <= lambda_{1,h}')
    p_, q_ = sig.numerator, sig.denominator
    # (K xi)_i >= a/2^BITS >= sigma b/2^BITS >= sigma (M xi)_i  (the last step uses sigma > 0)
    if not all(a * q_ >= p_ * b for a, b in zip(lo_l, mass_l)): raise ValueError('Collatz-Wielandt test failed')
    import hashlib
    hx = hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest()
    return sig, hx

def cr_bound(sigma, h, C=C_INTERP):
    """arb lower bound sigma/(1+(C h)^2 sigma) (exact sigma Fraction, h arb upper bound for element diameters).
    Lemma CR needs 0 < sigma <= lambda_{k,h}; the upper inequality is certified by the caller (Collatz-Wielandt test
    or inertia test), the positivity is enforced here: for sigma < -1/(C h)^2 the formula is positive and large."""
    sigma = Fraction(sigma); C = Fraction(C)
    if not sigma > 0: raise ValueError('sigma not positive: Lemma CR requires 0 < sigma <= lambda_{k,h}')
    if not C > 0: raise ValueError('interpolation constant not positive')
    if not (bool(h > 0) and h.is_finite()): raise ValueError('mesh size not certified positive')
    s = frac_to_arb(sigma); Ch = frac_to_arb(C) * h
    return s / (1 + Ch * Ch * s)

def lower_bound(a2, A, beta, n, kind=None, x0=None):
    """Certified lower bound (arb; use .lower()) for the lowest Robin eigenvalue.  Returns dict."""
    if kind is None: kind = choose_kind(a2, A)
    M = build_mesh(a2, A, n, kind)
    lam, x, rmin, K, md = float_eig(M, beta, x0=x0)
    sigma, hx = certify(M, beta, x)
    lb = cr_bound(sigma, M.h)
    return dict(lb=lb, sigma=sigma, lam_h=lam, kind=kind, n=n, N=M.N, h=M.h, xhash=hx, x=x, mesh=M)

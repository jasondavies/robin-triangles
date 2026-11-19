"""Rigorous far-region certification for a beta-interval [b1,b2].
Shape cells in (a,b) coordinates (equilateral (1,0)); fundamental domain
  D = { a>=1, bmin(a) <= b <= 0 },  bmin(a) = -(a/sqrt3)(sqrt(4-3/a^4)-1).
For each cell: a fixed six-wave trial on T* (waves at M_c, coefficients c(a,b) = c_c + (a-a_c)Ja + (b-b_c)Jb)
transplanted to T; the Rayleigh quotient is affine in alpha, lambda0 concave in alpha, so it suffices
to check  Num(alpha_i) - lambda0_low(beta_i) Den < 0  at both endpoints, uniformly on the cell
(mean-value form with interval gradient)."""
import sys, json, time
import mpmath as mp
from flint import arb, fmpq
import rig
from rig import I, Jet2, Q
from nums import ARB, MP
import sixwave as sw
from expint import tri_exp as tri_arb, seg_exp as seg_arb
from ritz_mp import tri_exp as tri_mp, seg_exp as seg_mp, lowest
from equi import lam0_lower, PSTAR
from sw_float import M_of_beta

mp.mp.dps = 30
SQ3 = arb(3).sqrt()

def fq(x):
    return mp.mpf(int(x.p)) / int(x.q)

def to_arb(x):
    return arb(mp.nstr(x, 32))

class Interval:
    def __init__(self, b1, b2):
        self.b1, self.b2 = Q(b1), Q(b2)
        self.lam_lo = [lam0_lower(arb(self.b1)), lam0_lower(arb(self.b2))]
        bc = (fq(self.b1) + fq(self.b2)) / 2
        self.bc = bc
        Mc = M_of_beta(bc)
        K = Mc + mp.atanh(mp.tanh(Mc) / 2)
        Wmp = sw.waves(MP, K)
        # fixed waves: round to 32 digits, use the SAME numbers in float and arb
        self.Wa = [(to_arb(w[0]), to_arb(w[1])) for w in Wmp]
        self.Wm = [(mp.mpf(w[0].mid().str(40, radius=False)) if False else mp.mpf(mp.nstr(wm[0], 32)),
                    mp.mpf(mp.nstr(wm[1], 32))) for w, wm in zip(self.Wa, Wmp)]
        self.Na, self.Sa = sw.base_matrices(ARB, self.Wa, tri_arb, seg_arb)
        self.Nm, self.Sm = sw.base_matrices(MP, self.Wm, tri_mp, seg_mp)
        self.Nmat = mp.matrix(self.Nm)

    def Amat(self, a, b, alpha):
        Axx, Axy, Ayy, rho = sw.shape_terms(MP, mp.mpf(a), mp.mpf(b))
        W = self.Wm; n = 6
        A = mp.matrix(n, n)
        for j in range(n):
            for k in range(n):
                g = W[j][0] * W[k][0] * Axx + (W[j][0] * W[k][1] + W[j][1] * W[k][0]) * Axy + W[j][1] * W[k][1] * Ayy
                A[j, k] = g * self.Nm[j][k] + alpha * sum(rho[s] * self.Sm[s][j][k] for s in range(3))
        return A

    def trial(self, a, b, h=mp.mpf('1e-7')):
        alpha = -self.bc
        lam, c = lowest(self.Amat(a, b, alpha), self.Nmat)
        nrm = (c.T * self.Nmat * c)[0]
        out = [c]
        for (da, db) in ((h, 0), (0, h)):
            _, cp = lowest(self.Amat(a + da, b + db, alpha), self.Nmat)
            _, cm = lowest(self.Amat(a - da, b - db, alpha), self.Nmat)
            cp = cp * (nrm / (c.T * self.Nmat * cp)[0]); cm = cm * (nrm / (c.T * self.Nmat * cm)[0])
            out.append((cp - cm) / (2 * h))
        return lam, out

    def check_cell(self, a1, a2, b1, b2):
        ac = (a1 + a2) / 2; bc = (b1 + b2) / 2
        lam, C = self.trial(fq(ac), fq(bc))
        self._lastC = C
        Ca = [[to_arb(v) for v in c] for c in C]
        T = sw.tensors(ARB, self.Wa, self.Na, self.Sa, Ca)
        res = []
        for i, (beta, llo) in enumerate(((self.b1, self.lam_lo[0]), (self.b2, self.lam_lo[1]))):
            alpha = -arb(beta)
            # value at centre
            Num, Den = sw.num_den(ARB, T, alpha, arb(ac), arb(bc), [arb(1), arb(0), arb(0)])
            F0 = Num - llo * Den
            # gradient over the box
            A = Jet2.var(I(a1, a2), 0); B = Jet2.var(I(b1, b2), 1)
            NumJ, DenJ = sw.num_den(ARB, T, alpha, A, B, [Jet2(1), A - ac, B - bc])
            FJ = NumJ - llo * DenJ
            Fub = F0 + FJ.gx * I(a1 - ac, a2 - ac) + FJ.gy * I(b1 - bc, b2 - bc)
            res.append(Fub)
        return all(f < 0 for f in res), float(lam - self.bc * 0), res

    def check_cell_data(self, a1, a2, b1, b2):
        ok, lam, res = self.check_cell(a1, a2, b1, b2)
        C = [[mp.nstr(v, 32) for v in c] for c in self._lastC]
        return ok, lam, res, C

def bmin_lower(a):
    """lower bound of bmin on [a1,a2] = bmin(a2) (bmin decreasing)"""
    a = arb(a)
    return -(a / SQ3) * ((4 - 3 / (a * a * a * a)).sqrt() - 1)

def run(b1, b2, r_loc, a_max, maxcells=200000, verbose=False):
    iv = Interval(b1, b2)
    r_loc = Q(r_loc); a_max = Q(a_max)
    # thin region: beta1 * (2 a_max/3) * P* > |lambda0(beta2)|
    thin_ok = bool(arb(iv.b1) * 2 * arb(a_max) / 3 * PSTAR > -iv.lam_lo[1])
    blo = Q(bmin_lower(a_max).lower().mid().str(20, radius=False)) - fmpq(1, 10**6)
    stack = [(fmpq(1), a_max, blo, fmpq(0), 0)]
    ncert = 0; fails = []; nskip = 0
    while stack:
        a1, a2, c1, c2, d = stack.pop()
        # outside domain?
        if c2 < 0 and arb(c2) < bmin_lower(a2):
            nskip += 1; continue
        # inside local box?
        if a1 >= 1 - r_loc and a2 <= 1 + r_loc and c1 >= -r_loc and c2 <= r_loc:
            nskip += 1; continue
        ok, lam, res = iv.check_cell(a1, a2, c1, c2)
        if ok:
            ncert += 1
            if verbose and ncert % 200 == 0: print('  cert', ncert, 'stack', len(stack), flush=True)
            continue
        if d > 30 or ncert + len(fails) > maxcells:
            fails.append((str(a1), str(a2), str(c1), str(c2))); continue
        am = (a1 + a2) / 2; cm = (c1 + c2) / 2
        if (a2 - a1) >= (c2 - c1):
            stack += [(a1, am, c1, c2, d + 1), (am, a2, c1, c2, d + 1)]
        else:
            stack += [(a1, a2, c1, cm, d + 1), (a1, a2, cm, c2, d + 1)]
    return dict(b1=str(iv.b1), b2=str(iv.b2), r_loc=str(r_loc), a_max=str(a_max), thin_ok=thin_ok,
                cells=ncert, skipped=nskip, fails=fails, ok=thin_ok and not fails)

if __name__ == '__main__':
    t = time.time()
    res = run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], verbose=True)
    print({k: v for k, v in res.items() if k != 'fails'}, 'nfails', len(res['fails']), res['fails'][:5], time.time() - t)

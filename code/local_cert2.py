"""Rigorous local argument near the equilateral, with a mean-value form in M.

F(a,b;M) = c^T (A(a,b;M) - lambda0(M) N(M)) c,  c = c0(M) + (a-1) c1 + b c2  (c1,c2 fixed numbers).
F is linear in the 'effective tensors' Eff(M) = (Gxx,Gxy,Gyy, alpha*S_s, lambda0*N)^{pq}.
Hessian_ab F(.;M) in  L_box(Eff(Mc)) + L_box(Eff'([M1,M2])) * [M1-Mc, M2-Mc]   (mean value theorem),
where L_box is the (linear) map Eff -> Hessian enclosure over the (a,b)-box, computed with Jet2."""
import sys, time
import mpmath as mp
from flint import arb, fmpq
import rig
from rig import I, Jet2, Q
from nums import ARB
from dual import Dual, DUALNUM
import sixwave as sw
from expint import tri_exp, seg_exp
from sw_float import Setup
from local_float import deriv_vectors
from equi import K_of, lam0_of, beta_of

def to_arb(x):
    return arb(mp.nstr(x, 40))

def eff_tensors(num, M, c1, c2, part):
    """part: 'v' -> arb values (M arb), 'd' -> derivative parts (M Dual)"""
    K, c0 = sw.psi0_coeffs(num, M)
    W = sw.waves(num, K)
    N, S = sw.base_matrices(num, W, tri_exp, seg_exp)
    Nt, Gxx, Gxy, Gyy, St = sw.tensors(num, W, N, S, [c0, c1, c2])
    H = num.root4(3)
    t = num.tanh(M); L = -num.atanh(t / 2); KK = M - L
    beta = t * KK / H
    lam0 = -4 * KK * KK / (H * H)
    alpha = -beta
    E = {}
    for key in Nt:
        e = (Gxx[key], Gxy[key], Gyy[key], alpha * St[0][key], alpha * St[1][key], alpha * St[2][key], lam0 * Nt[key])
        if part == 'd':
            e = tuple(x.d for x in e)
        E[key] = e
    return E

def F_jet(E, a, b):
    Axx, Axy, Ayy, rho = sw.shape_terms(ARB, a, b)
    d = [Jet2(1), a - 1, b]
    F = Jet2(0)
    for (p, q), e in E.items():
        term = Axx * e[0] + Axy * e[1] + Ayy * e[2] + rho[0] * e[3] + rho[1] * e[4] + rho[2] * e[5] - e[6]
        F = F + d[p] * d[q] * term
    return F

def certify(M1, M2, r, nsub=4, verbose=False):
    M1 = Q(M1); M2 = Q(M2); r = Q(r)
    Mc = (M1 + M2) / 2
    st = Setup(mp.mpf(int(Mc.p)) / int(Mc.q))
    C = deriv_vectors(st)
    c1 = [to_arb(v) for v in C[1]]; c2 = [to_arb(v) for v in C[2]]
    Ev = eff_tensors(ARB, arb(Mc), c1, c2, 'v')
    Ed = eff_tensors(DUALNUM, Dual(I(M1, M2), 1), c1, c2, 'd')
    dM = I(M1 - Mc, M2 - Mc)
    ok = True
    for i in range(nsub):
        for j in range(nsub):
            ab = I(1 - r + 2 * r * fmpq(i, nsub), 1 - r + 2 * r * fmpq(i + 1, nsub))
            bb = I(-r + 2 * r * fmpq(j, nsub), -r + 2 * r * fmpq(j + 1, nsub))
            a = Jet2.var(ab, 0); b = Jet2.var(bb, 1)
            Fv = F_jet(Ev, a, b); Fd = F_jet(Ed, a, b)
            hxx = Fv.hxx + Fd.hxx * dM; hxy = Fv.hxy + Fd.hxy * dM; hyy = Fv.hyy + Fd.hyy * dM
            good = bool(hxx < 0 and (hxx * hyy - hxy * hxy) > 0)
            if not good:
                ok = False
                if verbose: print('  fail sub', i, j, hxx, hxy, hyy)
    return ok

if __name__ == '__main__':
    t = time.time()
    print(certify(sys.argv[1], sys.argv[2], sys.argv[3], nsub=int(sys.argv[4]) if len(sys.argv) > 4 else 4, verbose=True), time.time() - t)


class NearCertifier:
    """Value check F < 0 on (a,b)-cells for all M in [M1,M2] using the local trial
    c = c0(M) + (a-1) c1 + b c2:   F(cell x [M]) in L_cell(Eff(Mc)) + L_cell(Eff'([M])) [M1-Mc, M2-Mc],
    L_cell(E) = range over the cell of the linear functional E -> F, enclosed by the mean-value form."""
    def __init__(self, M1, M2):
        M1 = Q(M1); M2 = Q(M2); self.M1, self.M2 = M1, M2
        Mc = (M1 + M2) / 2
        st = Setup(mp.mpf(int(Mc.p)) / int(Mc.q))
        C = deriv_vectors(st)
        self.c1 = [to_arb(v) for v in C[1]]; self.c2 = [to_arb(v) for v in C[2]]
        self.Ev = eff_tensors(ARB, arb(Mc), self.c1, self.c2, 'v')
        self.Ed = eff_tensors(DUALNUM, Dual(I(M1, M2), 1), self.c1, self.c2, 'd')
        self.dM = I(M1 - Mc, M2 - Mc)

    def hessian_ok(self, r, nsub=4):
        ok = True
        for i in range(nsub):
            for j in range(nsub):
                ab = I(1 - r + 2 * r * fmpq(i, nsub), 1 - r + 2 * r * fmpq(i + 1, nsub))
                bb = I(-r + 2 * r * fmpq(j, nsub), -r + 2 * r * fmpq(j + 1, nsub))
                a = Jet2.var(ab, 0); b = Jet2.var(bb, 1)
                Fv = F_jet(self.Ev, a, b); Fd = F_jet(self.Ed, a, b)
                hxx = Fv.hxx + Fd.hxx * self.dM; hxy = Fv.hxy + Fd.hxy * self.dM; hyy = Fv.hyy + Fd.hyy * self.dM
                if not bool(hxx < 0 and (hxx * hyy - hxy * hxy) > 0):
                    ok = False
        return ok

    def _range(self, E, a1, a2, b1, b2):
        ac = (a1 + a2) / 2; bc = (b1 + b2) / 2
        F0 = F_jet(E, Jet2(arb(ac)), Jet2(arb(bc))).v
        A = Jet2.var(I(a1, a2), 0); B = Jet2.var(I(b1, b2), 1)
        FJ = F_jet(E, A, B)
        return F0 + FJ.gx * I(a1 - ac, a2 - ac) + FJ.gy * I(b1 - bc, b2 - bc)

    def cell_ok(self, a1, a2, b1, b2):
        Fv = self._range(self.Ev, a1, a2, b1, b2)
        Fd = self._range(self.Ed, a1, a2, b1, b2)
        return bool((Fv + Fd * self.dM) < 0)


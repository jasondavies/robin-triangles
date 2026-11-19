"""Float prototype of the local (near-equilateral) argument: c(d) = c0 + da c1 + db c2."""
import mpmath as mp, sys
from sw_float import Setup, M_of_beta
from nums import MP
import sixwave as sw
mp.mp.dps = 40

def deriv_vectors(st, h=mp.mpf('1e-12')):
    Nm = mp.matrix(st.N)
    c0 = mp.matrix(st.c0)
    nrm = (c0.T * Nm * c0)[0]
    out = []
    for (da, db) in ((h, 0), (0, h)):
        _, cp = st.ritz(1 + da, db); _, cm = st.ritz(1 - da, -db)
        cp = cp * (nrm / (c0.T * Nm * cp)[0]); cm = cm * (nrm / (c0.T * Nm * cm)[0])
        out.append((cp - cm) / (2 * h))
    return [list(c0), list(out[0]), list(out[1])]

def F(st, T, a, b):
    Num, Den = sw.num_den(MP, T, -st.beta, mp.mpf(a), mp.mpf(b), [1, a - 1, b])
    return Num - st.lam0 * Den, Den

if __name__ == '__main__':
    for beta in [float(x) for x in sys.argv[1:]]:
        st = Setup(M_of_beta(beta))
        C = deriv_vectors(st)
        T = sw.tensors(MP, st.W, st.N, st.S, C)
        f0, D0 = F(st, T, 1, 0)
        print('beta', beta, 'F(0)/Den', mp.nstr(f0 / D0, 5))
        h = mp.mpf('1e-4')
        def hess(a, b):
            fxx = (F(st, T, a + h, b)[0] - 2 * F(st, T, a, b)[0] + F(st, T, a - h, b)[0]) / h ** 2
            fyy = (F(st, T, a, b + h)[0] - 2 * F(st, T, a, b)[0] + F(st, T, a, b - h)[0]) / h ** 2
            fxy = (F(st, T, a + h, b + h)[0] - F(st, T, a + h, b - h)[0] - F(st, T, a - h, b + h)[0] + F(st, T, a - h, b - h)[0]) / (4 * h * h)
            return fxx / D0, fxy / D0, fyy / D0
        print('  Hessian/Den at 0', [mp.nstr(v, 6) for v in hess(1, 0)])
        for r in ['0.001', '0.003', '0.01', '0.03', '0.1']:
            r = mp.mpf(r); worst = -1e99
            for (sa, sb) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                hx, hxy, hy = hess(1 + sa * r, sb * r)
                # largest eigenvalue
                ev = (hx + hy) / 2 + mp.sqrt(((hx - hy) / 2) ** 2 + hxy ** 2)
                worst = max(worst, ev)
            print('   r', r, 'max eig of Hessian on box corners/edges', mp.nstr(worst, 5))

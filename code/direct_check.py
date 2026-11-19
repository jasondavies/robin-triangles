"""Referee's independent check: build the 3x3 pencil directly by quadrature on actual triangles T
(phi_i = cosh(L + 2K b_i), b_i barycentric coordinate vanishing on side i), with NO use of the
paper's closed forms or transplantation formula, and test every inequality of the chain."""
import numpy as np, sys
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh

H = 3 ** 0.25
ELL = 2 / H
def cr(a, b): return a[0]*b[1] - a[1]*b[0]
def params(M):
    t = np.tanh(M); L = -np.arctanh(t / 2); K = M - L
    return t, L, K, t * K / H, -4 * K ** 2 / H ** 2

NQ = 100
xg, wg = leggauss(NQ); xg = (xg + 1) / 2; wg = wg / 2

def tri_from_angles(th):
    th = np.asarray(th, float)
    a = np.sin(th); c = a[2]; b = a[1]
    V = np.array([[0, 0], [c, 0], [b * np.cos(th[0]), b * np.sin(th[0])]])
    area = 0.5 * abs(cr(V[1] - V[0], V[2] - V[0]))
    return V / np.sqrt(area)

def pencil(V, M):
    t, L, K, beta, lam0 = params(M)
    Tm = np.array([[V[0, 0], V[1, 0], V[2, 0]], [V[0, 1], V[1, 1], V[2, 1]], [1, 1, 1]])
    gradb = np.linalg.inv(Tm)[:, :2]
    area = 0.5 * abs(cr(V[1] - V[0], V[2] - V[0]))
    U, Vv = np.meshgrid(xg, xg, indexing='ij'); W = np.outer(wg, wg) * (1 - U) * 2 * area
    B = np.stack([U, (1 - U) * Vv, (1 - U) * (1 - Vv)])
    ph = np.cosh(L + 2 * K * B); sh = np.sinh(L + 2 * K * B)
    N = np.einsum('ixy,jxy,xy->ij', ph, ph, W)
    G = np.einsum('ixy,jxy,xy->ij', sh, sh, W) * (2 * K) ** 2 * (gradb @ gradb.T)
    Bs = []; sides = []
    for i in range(3):                   # side i joins V_{i+1}, V_{i+2}; b_i = 0 there
        j, l = (i + 1) % 3, (i + 2) % 3
        length = np.linalg.norm(V[j] - V[l]); sides.append(length)
        bb = np.zeros((3, NQ)); bb[j] = xg; bb[l] = 1 - xg
        p = np.cosh(L + 2 * K * bb)
        Bs.append(np.einsum('ix,jx,x->ij', p, p, wg) * length)
    return dict(G=G, N=N, Bs=Bs, Bd=sum(Bs), sides=np.array(sides), area=area)

def angles_of(V):
    th = []
    for i in range(3):
        a = V[(i + 1) % 3] - V[i]; b = V[(i + 2) % 3] - V[i]
        th.append(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
    return np.array(th)

def equi_numbers(M):
    d = pencil(tri_from_angles([np.pi / 3] * 3), M)
    G, N, Bs, Bd = d['G'], d['N'], d['Bs'], d['Bd']
    Gd, Go = G[0, 0], -2 * G[0, 1]
    bown = (Bs[0] @ np.ones(3))[0]; badj = (Bs[1] @ np.ones(3))[0]
    return dict(G0=Gd - Go, G1=Gd + Go / 2, B1=Bd[0, 0] - Bd[0, 1], N1=N[0, 0] - N[0, 1],
                kappa=badj - bown, badj=badj, bown=bown, Bfull=np.ones(3) @ Bd @ np.ones(3))

def paper_closed(M):
    """the paper's closed forms (eq. in proof of Lemma certC), un-normalised"""
    t, L, K, beta, lam0 = params(M); U = 2 * M - L; ch, sh = np.cosh, np.sinh
    I1 = K**2 - K/2*sh(2*L) + (ch(2*U) - ch(2*L))/8
    I1s = -K**2 - K/2*sh(2*L) + (ch(2*U) - ch(2*L))/8
    I2 = K*sh(2*M) - sh(L)*(sh(U) - sh(L)); I2s = K*sh(2*M) - ch(L)*(ch(U) - ch(L))
    J2 = K + (sh(2*U) - sh(2*L))/4; J3 = sh(U) - sh(L); J4 = K*ch(2*M) + sh(2*K)/2
    c = 2 / H**2
    return dict(G0=c*(I1s - I2s), G1=c*(I1s + I2s/2), lN1=c*(I1 - I2),
                bB1=c*t/2*(2*K*ch(L)**2 + 2*J2 - 2*ch(L)*J3 - J4),
                bk=c*t/2*(J2 + J4 - ch(L)*J3 - 2*K*ch(L)**2))

def geom(t):
    r = np.sqrt(4 - t*t); rlo2 = np.sqrt(3)*t/r; c = (2 - t*t)/(t*r); rhi2 = np.sqrt(3)*c
    tau = (3*c*c + 1)/(np.sqrt(3)*c); mu = (tau + np.sqrt(tau*tau - 4))/2
    return np.sqrt(rlo2), np.sqrt(rhi2), mu, c

def chain(th, M, e=None):
    t, L, K, beta, lam0 = params(M)
    if e is None: e = equi_numbers(M)
    V = tri_from_angles(th); d = pencil(V, M)
    A = d['G'] - beta * d['Bd']; P = A - lam0 * d['N']
    rho = d['sides'] / ELL; S = (rho**2).sum(); W = ((rho - rho.mean())**2).sum()
    one = np.ones(3); q = P @ one; qp = q - q.mean()
    G0, G1, B1, N1, kap, badj = e['G0'], e['G1'], e['B1'], e['N1'], e['kappa'], e['badj']
    q_formula = G0*(rho**2 - 1) + beta*kap*(rho - 1) - beta*badj*(rho.sum() - 3)
    basis = np.array([[1, -1, 0], [1, 1, -2]], float); basis /= np.linalg.norm(basis, axis=1)[:, None]
    LamT_true = np.linalg.eigvalsh(basis @ P @ basis.T).max()
    # mu = largest eigenvalue of A-matrix: tau = 2S/3
    tau = 2*S/3; mu = (tau + np.sqrt(max(tau*tau - 4, 0)))/2
    LamT = mu*G1 - beta*rho.min()*B1 - lam0*N1
    rlo, rhi, muhi, cst = geom(t)
    Lamhi = muhi*G1 - beta*rlo*B1 - lam0*N1
    c = one - qp / Lamhi
    ritz = eigh(A, d['N'], eigvals_only=True)[0]
    th_min = np.min(th); near = np.sin(th_min/2) > t/2
    return dict(near=near, q_err=np.abs(q - q_formula).max()/abs(G0),
                ok_1P1=(one@P@one <= G0*(S - 3) + 1e-9*G0) and (G0*(S-3) <= 4*G0*rho.max()**2*W*(1+1e-9)),
                ok_qperp=qp@qp >= (2*G0*rho.min() + beta*kap)**2*W*(1 - 1e-9),
                ok_LamT=LamT_true <= LamT*(1+1e-9) + 1e-12*G0,
                ok_Lamhi=(LamT <= Lamhi*(1+1e-9)) if near else None,
                cPc=c@P@c/G0, ritz_gap=(ritz - lam0)/abs(lam0), LamT_true=LamT_true/G0, LamT=LamT/G0,
                Lamhi=Lamhi/G0, rho=rho, W=W, rlo=rlo, rhi=rhi)

if __name__ == '__main__':
    from out import write
    worst = 0.0; worstP = 0.0
    # 1. closed forms vs quadrature on E
    for M in [0.5, 1.3, 2.0, 3.0, 5.0, 7.0]:
        e = equi_numbers(M); p = paper_closed(M); t, L, K, beta, lam0 = params(M)
        rel = dict(G0=p['G0']/e['G0'] - 1, G1=p['G1']/e['G1'] - 1, lN1=p['lN1']/(-lam0*e['N1']) - 1,
                   bB1=p['bB1']/(beta*e['B1']) - 1, bk=p['bk']/(beta*e['kappa']) - 1)
        # eigen-identity on E: P 1 = 0
        d = pencil(tri_from_angles([np.pi/3]*3), M); P = d['G'] - beta*d['Bd'] - lam0*d['N']
        worst = max(worst, max(abs(v) for v in rel.values())); worstP = max(worstP, np.abs(P@np.ones(3)).max()/e['G0'])
        print(f"M={M}: closed-vs-quad rel err", {k: f"{v:.1e}" for k, v in rel.items()},
              f"|P_E 1|/G0={np.abs(P@np.ones(3)).max()/e['G0']:.1e}")
    write('direct_check.json', dict(max_rel_err=float(worst), max_P1_over_G0=float(worstP), ok=bool(worst < 1e-12 and worstP < 1e-12)))

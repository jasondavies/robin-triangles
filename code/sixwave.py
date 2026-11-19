"""Six-wave transplant framework.  Generic over number types via a small 'num' namespace.

Reference: equilateral T* of area 1, centroid at origin, base horizontal:
  V0=(-l/2,-r), V1=(l/2,-r), V2=(0,2r),  l = 2/H, r = H/3, H = 3^{1/4}.
Sides s0: V0->V1, s1: V1->V2, s2: V2->V0 with unit directions e_s and inward normals n_s.
Waves (6): w = +k n_s and -k n_s (s=0,1,2), k = 2K/H.  psi0 = sum_s cosh(L + 2K d_s/H),
d_s = n_s.x + r, hence psi0 = sum_s [ e^{L+2K/3}/2 e^{k n_s.x} + e^{-L-2K/3}/2 e^{-k n_s.x} ].
Shape coordinates (a,b): T = L(T*), DL = [[a,b],[0,1/a]].
  Acal = DL^{-1} DL^{-T} = [[1/a^2+b^2, -ab], [-ab, a^2]],   rho_s = |DL e_s|  (side-length ratio).
For a trial u = sum_j c_j e^{w_j.x} on T*, transplanted to T:
  R_T = [ sum_jk c_j c_k ( w_j^T Acal w_k N_jk + alpha sum_s rho_s S^s_jk ) ] / sum_jk c_j c_k N_jk
with N_jk = int_{T*} e^{(w_j+w_k).x}, S^s_jk = int_{side s} e^{(w_j+w_k).x}."""
from rig import square

def geometry(num):
    H = num.root4(3)
    s3 = num.sqrt(3)
    l = 2 / H
    r = H / 3
    V = [(-l / 2, -r), (l / 2, -r), (num.zero(), 2 * r)]
    E = [(num.one(), num.zero()), (-num.one() / 2, s3 / 2), (-num.one() / 2, -s3 / 2)]
    Nin = [(-e[1], e[0]) for e in E]          # inward normals (rotate direction by +90deg; CCW orientation)
    return H, l, r, V, E, Nin

def waves(num, K):
    H, l, r, V, E, Nin = geometry(num)
    k = 2 * K / H
    W = []
    for s in range(3):
        W.append((k * Nin[s][0], k * Nin[s][1]))
        W.append((-k * Nin[s][0], -k * Nin[s][1]))
    return W

def psi0_coeffs(num, M):
    t = num.tanh(M)
    L = -num.atanh(t / 2)
    K = M - L
    cp = num.exp(L + 2 * K / 3) / 2
    cm = num.exp(-L - 2 * K / 3) / 2
    return K, [cp, cm, cp, cm, cp, cm]

def base_matrices(num, W, tri_exp, seg_exp):
    """N (6x6), S[s] (6x6) on T*"""
    H, l, r, V, E, Nin = geometry(num)
    n = len(W)
    N = [[None] * n for _ in range(n)]
    S = [[[None] * n for _ in range(n)] for _ in range(3)]
    for j in range(n):
        for k in range(j, n):
            v = (W[j][0] + W[k][0], W[j][1] + W[k][1])
            N[j][k] = N[k][j] = tri_exp(V, v)
            for s in range(3):
                S[s][j][k] = S[s][k][j] = seg_exp(V[s], V[(s + 1) % 3], v)
    return N, S

def tensors(num, W, N, S, C):
    """C: list of coefficient vectors c_p. Returns dicts keyed (p,q):
       Nt[p,q], Gxx, Gxy (= G_xy + G_yx), Gyy, St[s][p,q]"""
    P = len(C); n = len(W)
    Nt = {}; Gxx = {}; Gxy = {}; Gyy = {}; St = [{}, {}, {}]
    for p in range(P):
        for q in range(p, P):
            nt = gxx = gxy = gyy = num.zero(); st = [num.zero()] * 3
            for j in range(n):
                cj = C[p][j]
                for k in range(n):
                    w = cj * C[q][k]
                    if w == 0:
                        continue
                    nn = w * N[j][k]
                    nt = nt + nn
                    gxx = gxx + W[j][0] * W[k][0] * nn
                    gxy = gxy + (W[j][0] * W[k][1] + W[j][1] * W[k][0]) * nn
                    gyy = gyy + W[j][1] * W[k][1] * nn
                    st = [st[s] + w * S[s][j][k] for s in range(3)]
            if p != q:
                # symmetric pair counted once; we store sym sum T[p,q]+T[q,p] at (p,q)
                nt2 = gxx2 = gxy2 = gyy2 = num.zero(); st2 = [num.zero()] * 3
                for j in range(n):
                    cj = C[q][j]
                    for k in range(n):
                        w = cj * C[p][k]
                        if w == 0:
                            continue
                        nn = w * N[j][k]
                        nt2 = nt2 + nn
                        gxx2 = gxx2 + W[j][0] * W[k][0] * nn
                        gxy2 = gxy2 + (W[j][0] * W[k][1] + W[j][1] * W[k][0]) * nn
                        gyy2 = gyy2 + W[j][1] * W[k][1] * nn
                        st2 = [st2[s] + w * S[s][j][k] for s in range(3)]
                nt = nt + nt2; gxx = gxx + gxx2; gxy = gxy + gxy2; gyy = gyy + gyy2
                st = [st[s] + st2[s] for s in range(3)]
            Nt[p, q] = nt; Gxx[p, q] = gxx; Gxy[p, q] = gxy; Gyy[p, q] = gyy
            for s in range(3):
                St[s][p, q] = st[s]
    return Nt, Gxx, Gxy, Gyy, St

def shape_terms(num, a, b):
    """Acal entries and rho_s as functions of (a,b) (works for jets too)"""
    H, l, r, V, E, Nin = geometry(num)
    Axx = 1 / square(a) + square(b)
    Axy = -(a * b)
    Ayy = square(a)
    rho = []
    for e in E:
        u = a * e[0] + b * e[1]
        v = e[1] / a
        q = square(u) + square(v)
        rho.append(q.sqrt() if hasattr(q, 'sqrt') else num.sqrt(q))
    return Axx, Axy, Ayy, rho

def num_den(num, T, alpha, a, b, d):
    """Num, Den for coefficient vector c = sum_p d[p] C[p] (d[0] usually 1); d entries may be jets"""
    Nt, Gxx, Gxy, Gyy, St = T
    Axx, Axy, Ayy, rho = shape_terms(num, a, b)
    Num = None; Den = None
    for (p, q), nt in Nt.items():
        dd = d[p] * d[q]
        term = Axx * Gxx[p, q] + Axy * Gxy[p, q] + Ayy * Gyy[p, q] \
            + alpha * (rho[0] * St[0][p, q] + rho[1] * St[1][p, q] + rho[2] * St[2][p, q])
        tn = dd * term; td = dd * nt
        Num = tn if Num is None else Num + tn
        Den = td if Den is None else Den + td
    return Num, Den

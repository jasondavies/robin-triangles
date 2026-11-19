"""Certificate for Theorem A: for 0 < beta <= BETA_A and every non-equilateral triangle T of area 1,
lambda_1(T,-beta) < lambda_0(beta).   Rigorous (Arb).  Writes ../verification/thmA.json
(exact 'man*2^exp' encodings of the certified endpoints)."""
import json, os, sys, time
from flint import arb, fmpq, ctx
from rig import I, Jet2
from equi import M_encl, K_of, KM_of, H, PSTAR, g_of
from geom import p_tau
from out import write, enc

BETA_A = fmpq(6, 5)
G_A = fmpq(1425, 10000)
NEGATIVE = os.environ.get('THMA_NEGATIVE') == '1'   # negative test: G_A = 0.146 is too large and must fail
if NEGATIVE:
    G_A = fmpq(146, 1000)
Q_A = fmpq(1384, 1000)
out = {'beta_A': str(BETA_A), 'G_A': str(G_A), 'q_A': str(Q_A)}

# ---- Step 1: sup_{0<beta<=beta_A} g(beta) <= G_A -------------------------------------------------
Mlo, Mhi = M_encl(arb(BETA_A))
MA = Mhi
n = 2000
ok1 = True; worst = arb(-1)
for i in range(n):
    m1 = MA * i / n; m2 = MA * (i + 1) / n
    m1 = arb(m1.lower()) if i else arb(0); m2 = arb(m2.upper())
    if i == 0:
        gub = (m2.tanh() * m2.tanh() / 4)           # g <= tanh(M)^2/4  (Lemma)
    else:
        gub = arb(1) / 4 - K_of(m1) / (2 * (2 * m2).sinh() * KM_of(m1))
    if not (gub < G_A):
        ok1 = False; print('step1 fail cell', i, gub)
    if gub.upper() > worst:
        worst = arb(gub.upper())
out['step1_sup_g_ok'] = ok1
out['step1_cells'] = n
out['step1_g_bound_upper'] = enc(worst)
gA = g_of(I(Mlo, Mhi))
out['g_betaA_lower'] = enc(gA.lower()); out['g_betaA_upper'] = enc(gA.upper())
out['M_betaA_lower'] = enc(Mlo); out['M_betaA_upper'] = enc(Mhi)
print('step1', ok1, 'g(beta_A) encl', gA, 'max cell bound', worst)

# ---- Step 2: q(beta_A) = -lambda0(beta_A)/(beta_A P*) <= Q_A ---------------------------------------
qub = 4 * K_of(Mhi) * K_of(Mhi) / (H * H) / (arb(BETA_A) * PSTAR)
ok2 = bool(qub < Q_A)
out['step2_q_upper'] = enc(qub.upper()); out['step2_q_lower'] = enc(qub.lower()); out['step2_ok'] = ok2
print('step2', ok2, qub)

# ---- Step 3: Phi = (p-1) - G_A (tau-2) > 0 on {p <= Q_A} in the fundamental domain, minus equilateral
G = arb(G_A); q = arb(Q_A)
x0 = arb(1) / 2; y0 = arb(3).sqrt() / 2
R0 = fmpq(1, 40)   # half-width of local box around the equilateral apex
# local box: interval Hessian of Phi positive definite
bx = I(fmpq(1, 2) - R0, fmpq(1, 2) + R0); by = (y0 - R0).union(y0 + R0)
NSUB = 16
ok3a = True; minpd = None
for i in range(NSUB):
    for j in range(NSUB):
        sx = I(fmpq(1, 2) - R0 + 2 * R0 * fmpq(i, NSUB), fmpq(1, 2) - R0 + 2 * R0 * fmpq(i + 1, NSUB))
        sy = (y0 - R0 + 2 * R0 * fmpq(j, NSUB)).union(y0 - R0 + 2 * R0 * fmpq(j + 1, NSUB))
        X = Jet2.var(sx, 0); Y = Jet2.var(sy, 1)
        p, tau = p_tau(X, Y)
        Phi = (p - 1) - G * (tau - 2)
        a, b, c = Phi.hxx, Phi.hxy, Phi.hyy
        if not (a > 0 and (a * c - b * b) > 0):
            ok3a = False; print('local subbox fail', i, j, a, b, c)
print('local box Hessian PD on', NSUB, 'x', NSUB, 'sub-boxes:', ok3a)
out['step3_local_box_halfwidth'] = str(R0); out['step3_local_subboxes'] = NSUB * NSUB; out['step3_local_hessian_pd'] = ok3a
# strip y < Y_MIN: p >= 2 sqrt(2/y)/P* > q
Y_MIN = fmpq(19, 100)
ok3b = bool(2 * (2 / arb(Y_MIN)).sqrt() / PSTAR > q)
out['step3_strip_ymin'] = str(Y_MIN); out['step3_strip_ok'] = ok3b
print('strip', ok3b)
# cells over [0,1/2] x [Y_MIN, sqrt3/2 + eps]
YTOP = fmpq(87, 100)
out['step3_ytop'] = str(YTOP)
stack = [(fmpq(0), fmpq(1, 2), Y_MIN, YTOP, 0)]
ncell = 0; fails = []; minphi = None; maxdepth = 0
def inside_local(x1, x2, y1, y2):
    return (arb(x1) >= bx.lower() and arb(x2) <= bx.upper() and arb(y1) >= by.lower() and arb(y2) <= by.upper())
while stack:
    x1, x2, y1, y2, d = stack.pop()
    maxdepth = max(maxdepth, d)
    xb = I(x1, x2); yb = I(y1, y2)
    # outside fundamental domain: (1-x)^2 + y^2 > 1 everywhere on the box
    if ((1 - xb) * (1 - xb) + yb * yb) > 1:
        ncell += 1; continue
    if inside_local(x1, x2, y1, y2):
        ncell += 1; continue
    p, tau = p_tau(xb, yb)
    if p > q:
        ncell += 1; continue
    phi = (p - 1) - G * (tau - 2)
    if phi > 0:
        ncell += 1; continue
    # mean value form: Phi(c) + grad(box).(box - c)
    xm = (x1 + x2) / 2; ym = (y1 + y2) / 2
    J = (lambda X, Y: p_tau(X, Y))(Jet2(xb, 1, 0), Jet2(yb, 0, 1))
    pc, tc = p_tau(arb(xm), arb(ym))
    phic = (pc - 1) - G * (tc - 2)
    gx = J[0].gx - G * J[1].gx; gy = J[0].gy - G * J[1].gy
    phim = phic + gx * (xb - xm) + gy * (yb - ym)
    if phim > 0:
        ncell += 1; continue
    if d > 40 or ncell > (5000 if NEGATIVE else 200000):
        fails.append((str(x1), str(x2), str(y1), str(y2)))
        if NEGATIVE: break                # one uncovered box suffices for the negative test
        continue
    if (x2 - x1) >= (y2 - y1):
        stack.append((x1, xm, y1, y2, d + 1)); stack.append((xm, x2, y1, y2, d + 1))
    else:
        stack.append((x1, x2, y1, ym, d + 1)); stack.append((x1, x2, ym, y2, d + 1))
ok3c = (not fails) and not stack
out['step3_cells'] = ncell; out['step3_cells_ok'] = ok3c; out['step3_maxdepth'] = maxdepth
print('cells', ncell, 'fails', len(fails), 'maxdepth', maxdepth)
out['THEOREM_A_CERTIFIED'] = bool(ok1 and ok2 and ok3a and ok3b and ok3c)
print('THEOREM A certified:', out['THEOREM_A_CERTIFIED'])
write('thmA_negative.json' if NEGATIVE else 'thmA.json', out)

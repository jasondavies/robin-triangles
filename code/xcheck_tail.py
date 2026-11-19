import sys
from indep_cert import *
from out import write
worst = 0
exprs, syms = tail_exprs()
for Mv in [2, 5, 10, 30]:
    M = iv.mpf(Mv); q = body_quantities(M)
    t = 1 - 2/(iv.exp(2*M)+1); L = -iv.log(-1 + 4/(2-t))/2; K = M - L; U = 2*M - L
    X = iv.exp(2*U)/16 * 2/(H*H)
    ns = {'s': iv.exp(-2*M), 'u': K*iv.exp(-2*M), 't': t, 'w': iv.exp(2*L), 'sqrt': iv.sqrt}
    vals = [eval(sp.sstr(e).replace('^', '**'), {}, ns) for e in exprs]
    errs = [max(abs(mp.mpf(((a/X)/b - 1).a)), abs(mp.mpf(((a/X)/b - 1).b))) for a, b in zip(q[:5], vals)]
    worst = max([worst] + errs)
    print(Mv, [mp.nstr(e, 3) for e in errs])
write('xcheck_tail.json', dict(max_rel_diff=mp.nstr(worst, 5), ok=bool(worst < mp.mpf('1e-100'))))

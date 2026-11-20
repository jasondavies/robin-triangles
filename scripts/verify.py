#!/usr/bin/env python3
"""Reproduce every certified inequality, every cross-check and every number in the paper.

  .venv/bin/python scripts/verify.py                       # default: certified, independent, checks, paper
  .venv/bin/python scripts/verify.py --only certified|independent|checks|paper
  .venv/bin/python scripts/verify.py --cells [--cells-dir DIR]   # also replay the six-wave cell certificate (Part I)
  .venv/bin/python scripts/verify.py --positive-full       # also re-certify every stored bound of Part II from
                                                           # scratch, all three runs, with both programs

Each step writes a machine-readable JSON file to verification/.  Each gate below parses that JSON and
compares it with the paper's claims in claims.json, in EXACT rational arithmetic (certified endpoints are
stored as exact 'man*2^exp' numbers; printed decimals are compared with their rounding intervals).  Logs
go to verification/<step>.log and a summary with input and PDF hashes to verification/summary.json.  The
exit status is nonzero if any gate fails, or if the LaTeX log has a warning or a bad box.  If latexmk or
pdflatex is not installed, the paper build and the log check are reported as SKIPPED (the overall status then
reads "PASS (SKIPPED: paper, paper_log_clean)"); every other step still runs and is gated.

The six-wave cell certificate (about 100 MB) is not committed: `scripts/cells.py generate` regenerates it
(several CPU hours) and `--cells` replays it (about half an hour); see README.md.  The certificates of Part II
(alpha > 0) are committed in data/positive/; the default run re-checks them from the stored bounds, runs the
independent verifier (with a from-scratch re-certification of a sample of the default run), and
`--positive-full` re-certifies every stored bound of all three runs from scratch (about 1.5 CPU hours).
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys, time
from decimal import Decimal
from fractions import Fraction as Fr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'verification'
PY = sys.executable
ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
CLAIMS = json.loads((ROOT / 'claims.json').read_text())
CELLS = json.loads((ROOT / 'data' / 'cells_summary.json').read_text())
POS = ROOT / 'data' / 'positive'
POSOUT = LOGS / 'positive'
PROFILES = {'ls': ('certs', '0.2983', ''), 'pw': ('certs_pw', '0.3496', 'pw'), 'liu': ('certs_liu', '0.1893', 'liu')}
WORDS = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six'}


def C(key):
    return Fr(CLAIMS[key])


def X(s):
    """Exact Fraction of an encoded 'man*2^exp' number."""
    m, e = s.split('*2^')
    m, e = int(m), int(e)
    return Fr(m * 2 ** e) if e >= 0 else Fr(m, 2 ** (-e))


def rounds_to(x, printed):
    """x rounds to the printed decimal string: |x - printed| <= half a unit in its last place."""
    d = Decimal(printed)
    return abs(Fr(x) - Fr(d)) <= Fr(1, 2) * Fr(10) ** d.as_tuple().exponent


def J(name):
    return json.loads((LOGS / name).read_text())


def JP(name):
    return json.loads((POSOUT / name).read_text())


def penv(profile='ls', sub=''):
    out = POSOUT / sub if sub else POSOUT
    out.mkdir(parents=True, exist_ok=True)
    return dict(ENV, RP_PROFILE=profile, RP_OUT=str(out))


def sanitize(text):
    return text.replace(str(ROOT), '.').replace(str(Path.home()), '~')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(name, cmd, cwd, gate, env=None):
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env or ENV)
    shown = ['python' if c == PY else sanitize(str(c)) for c in cmd]
    out = sanitize(proc.stdout + proc.stderr)
    msgs = []
    ok = proc.returncode == 0
    if ok:
        try:
            msgs = gate()
            ok = all(m[0] for m in msgs)
        except Exception as exc:          # a missing or malformed JSON output is a failure
            ok = False; msgs = [(False, 'gate error: %r' % exc)]
    (LOGS / (name + '.log')).write_text('$ ' + ' '.join(shown) + '\n' + out +
                                        ''.join('\n[gate] %s %s' % ('ok  ' if g else 'FAIL', t) for g, t in msgs) + '\n')
    print('%-22s %s  (%.0fs)' % (name, 'PASS' if ok else 'FAIL', time.time() - t0), flush=True)
    for g, t in msgs:
        if not g:
            print('    gate failed:', t)
    return {'name': name, 'ok': ok, 'seconds': round(time.time() - t0, 1), 'cmd': shown,
            'gates': [{'ok': g, 'check': t} for g, t in msgs]}


# ------------------------------------------------------------------ gates (one per step)
def g_thmA():
    d = J('thmA.json')
    return [(d['THEOREM_A_CERTIFIED'] is True, 'Theorem A certificate complete'),
            (d['beta_A'] == '6/5' and Fr(d['G_A']) == C('GA') and Fr(d['q_A']) == C('qA'),
             'beta_A = 6/5, G_A = %s, q_A = %s' % (CLAIMS['GA'], CLAIMS['qA'])),
            (d['step1_cells'] == int(CLAIMS['thmAstepOne']) and X(d['step1_g_bound_upper']) < C('GA'),
             '%s intervals, every bound on g below G_A' % CLAIMS['thmAstepOne']),
            (C('gBetaA') <= X(d['g_betaA_lower']) and X(d['g_betaA_upper']) < C('gBetaA') + Fr(1, 10 ** 5),
             'g(beta_A) = %s...' % CLAIMS['gBetaA']),
            (C('qBetaALo') <= X(d['step2_q_lower']) and X(d['step2_q_upper']) <= C('qBetaAHi') < C('qA'),
             '%s <= q(beta_A) <= %s < q_A' % (CLAIMS['qBetaALo'], CLAIMS['qBetaAHi'])),
            (d['step3_local_subboxes'] == int(CLAIMS['thmAsub']) and d['step3_local_box_halfwidth'] == '1/40'
             and d['step3_local_hessian_pd'], '%s Hessian sub-boxes of the box of half-width 1/40' % CLAIMS['thmAsub']),
            (d['step3_strip_ymin'] == '19/100' and d['step3_ytop'] == '87/100' and d['step3_strip_ok'],
             'strip y < 19/100 and region up to y = 87/100'),
            (d['step3_cells'] == int(CLAIMS['thmAboxes']) and d['step3_cells_ok'], '%s boxes' % CLAIMS['thmAboxes'])]


def g_thmA_negative():
    d = J('thmA_negative.json')
    return [(d['THEOREM_A_CERTIFIED'] is False and Fr(d['G_A']) == Fr('0.146'), 'G_A = 0.146 fails, as it must')]


def g_thmC():
    d = J('thmC.json')
    lo, hi = X(d['beta_C_lower']), X(d['beta_C_upper'])
    return [(d['THEOREM_C_CERTIFIED'] is True, 'Lemma certC certificate complete'),
            (d['M_C'] == '13/10' and d['M_tail'] == '8' and d['dM'] == '1/1000'
             and Fr(13, 10) + d['cells'] * Fr(1, 1000) == 8, 'intervals of width 1/1000 tile [13/10, 8], tail M >= 8'),
            (d['cells'] == int(CLAIMS['thmCcells']), '%s intervals' % CLAIMS['thmCcells']),
            (C('betaClo') <= lo <= hi <= C('betaChi') < C('betaCbound') < Fr(6, 5),
             '%s <= beta(13/10) <= %s < %s < 6/5' % (CLAIMS['betaClo'], CLAIMS['betaChi'], CLAIMS['betaCbound'])),
            (X(d['worst_ratio_upper']) <= C('thmCworst') < 1, 'worst ratio <= %s' % CLAIMS['thmCworst']),
            (X(d['tail_ratio_upper']) <= C('thmCtail'), 'tail ratio <= %s' % CLAIMS['thmCtail'])]


def g_negative_c():
    d = J('negative_c.json')
    return [(d['ok'] and d['start_21_20_fails'] and d['start_1_fails'], 'certificate fails when started at M = 21/20 or M = 1')]


def g_indep():
    d = J('thmC_independent.json')
    lo, hi = X(d['beta_C_lower']), X(d['beta_C_upper'])
    return [(d['ok'] and d['body_ok'] and d['tail_ok'] and d['beta_C_below_6_5'], 'independent certificate complete'),
            (d['M_C'] == '13/10' and d['M_big'] == CLAIMS['indepMbig'] and d['iv_dps'] == int(CLAIMS['indepDps']),
             'body [13/10, %s], tail M >= %s, %s digits' % (CLAIMS['indepMbig'], CLAIMS['indepMbig'], CLAIMS['indepDps'])),
            (d['body_cells'] == int(CLAIMS['indepCells']), '%s adaptive cells' % CLAIMS['indepCells']),
            (X(d['body_first_endpoint']) <= Fr(13, 10), 'the first cell starts at an exact dyadic <= 13/10'),
            (Fr(d['body_worst_ratio']) <= C('indepWorst') < 1, 'worst ratio <= %s' % CLAIMS['indepWorst']),
            (Fr(d['tail_ratio']) <= C('indepTail'), 'tail ratio <= %s' % CLAIMS['indepTail']),
            (C('betaClo') <= lo <= hi <= C('betaChi'), 'same enclosure of beta(13/10)')]


def g_table_c():
    d = J('table_c.json')
    msgs = [(d['ok'], '(p-1)/(tau-2) -> 3/16 near E in four directions'),
            (rounds_to(d['threshold_M'], CLAIMS['thrM']) and rounds_to(d['threshold_beta'], CLAIMS['thrBeta']),
             'float threshold M = %s, beta = %s' % (CLAIMS['thrM'], CLAIMS['thrBeta'])),
            (rounds_to(d['g_equals_3_16_beta'], CLAIMS['gLocalBeta']), 'g(beta) = 3/16 at beta = %s' % CLAIMS['gLocalBeta'])]
    for row, r in zip(CLAIMS['tableC'], d['rows']):
        ok = rounds_to(r['beta'], row['beta']) and all(
            rounds_to(r[k2], row[k1]) for k1, k2 in [('M', 'M'), ('theta', 'theta_star_deg'), ('gamma', 'gamma'), ('Lam', 'Lam'), ('rhs', 'rhs')])
        msgs.append((ok, 'Table 1 row beta = %s' % row['beta']))
    return msgs


def g_simple(name, extra):
    def g():
        d = J(name)
        return [(d['ok'], '%s reports ok' % name)] + [(f(d), t) for f, t in extra]
    return g


def g_cells_summary():
    t = CELLS['totals']
    return [(t['local_intervals'] == int(CLAIMS['cellsLocal']) and t['near_intervals'] == int(CLAIMS['cellsNear']),
             '%s local and %s near M-intervals' % (CLAIMS['cellsLocal'], CLAIMS['cellsNear'])),
            (t['far_cells'] == int(CLAIMS['cellsFar']) and t['far_files'] == int(CLAIMS['cellsFarFiles']),
             '%s far cells in %s beta-ranges' % (CLAIMS['cellsFar'], CLAIMS['cellsFarFiles'])),
            (sum(x['cells'] for x in CELLS['far']) == t['far_cells'], 'far cell counts add up'),
            ((Fr(t['r_min']), Fr(t['r_max']), Fr(t['R_min']), Fr(t['R_max'])) ==
             (C('rLocMin'), C('rLocMax'), C('RMin'), C('RMax')), 'radii r and R as printed'),
            (Fr(CELLS['far'][0]['BLO']) == Fr(6, 5) and Fr(CELLS['far'][-1]['BHI']) == 3 and
             all(Fr(a['BHI']) == Fr(b['BLO']) for a, b in zip(CELLS['far'], CELLS['far'][1:])), 'beta-ranges tile [6/5, 3]'),
            (len(CELLS['files']) == 6 + t['far_files'], 'a SHA-256 is recorded for every certificate file')]


def g_cells():
    d = J('cells.json')
    return [(d['ok'], 'every stored cell re-verified; coverage and counts as recorded'),
            (d['local_intervals'] == int(CLAIMS['cellsLocal']) and d['near_intervals'] == int(CLAIMS['cellsNear'])
             and d['far_cells'] == int(CLAIMS['cellsFar']), 'counts %s / %s / %s' % (CLAIMS['cellsLocal'], CLAIMS['cellsNear'], CLAIMS['cellsFar'])),
            (d['M_covers_6_5_to_3'] and d['beta_covers_6_5_to_3'], 'M- and beta-intervals cover 6/5 <= beta <= 3')]


# ------------------------------------------------------------------ gates for Part II (alpha > 0)
def g_pos_weak():
    d = JP('weak_cert.json'); lv = d['leaves']
    up = lambda s: Fr(Decimal(s.strip('[]').split('+/-')[0])) + Fr(Decimal(s.strip('[]').split('+/-')[1]))
    lo = lambda s: Fr(Decimal(s.strip('[]').split('+/-')[0])) - Fr(Decimal(s.strip('[]').split('+/-')[1]))
    return [(d['ok'] and d['n_fails'] == 0 and d['check_T'] and d['check_0'], 'Theorem uniform weak coupling: every box decided'),
            (d['alpha1'] == '1/100' and d['T_W'] == '1.2413' and d['t0'] == '0.01', 'alpha_1 = 1/100, t in [1/100, 1.2413]'),
            (d['boxes'] == int(CLAIMS['pWeakBoxes']) and sum(lv.values()) == int(CLAIMS['pWeakLeaves']), '%s boxes, %s leaves' % (CLAIMS['pWeakBoxes'], CLAIMS['pWeakLeaves'])),
            ((lv['W'], lv['S'], lv['L']) == (int(CLAIMS['pWeakW']), int(CLAIMS['pWeakS']), int(CLAIMS['pWeakL'])) and d['max_depth'] <= int(CLAIMS['pWeakDepth']),
             'leaves W/S/L = %s/%s/%s, depth <= %s' % (CLAIMS['pWeakW'], CLAIMS['pWeakS'], CLAIMS['pWeakL'], CLAIMS['pWeakDepth'])),
            (lo(d['lhs0']) > C('pWeakLhs') and up(d['rhs0']) < C('pWeakRhs') and rounds_to(lo(d['lhs0']) / up(d['rhs0']), CLAIMS['pWeakFactor']),
             'scaled (L) near E: lhs > %s, rhs < %s, factor %s' % (CLAIMS['pWeakLhs'], CLAIMS['pWeakRhs'], CLAIMS['pWeakFactor'])),
            (up(d['g2']) < C('pGtwo') and up(d['g1']) < C('pGone'), 'g_2 < %s, g_1 < %s' % (CLAIMS['pGtwo'], CLAIMS['pGone']))]


def g_pos_local(name='local_cert.json'):
    def g():
        d = JP(name); iv = d['intervals']; ref = json.loads((POS / 'certs' / 'local_cert.json').read_text())
        k = [x for x in iv if x['k'] == int(CLAIMS['pQminK'])][0]
        return [(d['ok'] and all(x['ok'] for x in iv), 'inequality (C) on every interval'),
                (d['t_loc'] == '3/50' and d['s'] == '1/10' and d['KTOP'] == 336 and len(iv) == int(CLAIMS['pIneqC']),
                 't_* = 3/50, s = 1/10, K = 336: %s inequalities' % CLAIMS['pIneqC']),
                (Fr(repr(d['c_l'])) == C('pCell') and Fr(repr(d['rho'])) == C('pRho') and Fr(d['c_l_computed_lower']) >= C('pCellComp')
                 and Fr(d['rho_computed_upper']) <= C('pRhoComp'), 'shape constants: computed %s / %s, used %s / %s' % (
                     CLAIMS['pCellComp'], CLAIMS['pRhoComp'], CLAIMS['pCell'], CLAIMS['pRho'])),
                (Fr(d['worst_ratio']) >= C('pQmin') and rounds_to(d['worst_ratio'], CLAIMS['pQmin']) and d['worst_k'] == int(CLAIMS['pQminK'])
                 and rounds_to(k['alpha_lo'], CLAIMS['pQminAlpha']), 'smallest quotient %s at k = %s (alpha ~ %s)' % (
                     CLAIMS['pQmin'], CLAIMS['pQminK'], CLAIMS['pQminAlpha'])),
                (min(x['ratio'] for x in iv if x['k'] == 'top' or x['alpha_lo'] >= 30) > C('pQgeThirty') and
                 min(x['ratio'] for x in iv if x['k'] != 'top' and x['alpha_lo'] <= 0.6) > C('pQleSix'),
                 'quotient > %s for alpha >= 30 and > %s for alpha <= 0.6' % (CLAIMS['pQgeThirty'], CLAIMS['pQleSix'])),
                ([x['ratio'] for x in iv] == [x['ratio'] for x in ref['intervals']], 'identical to the committed local_cert.json')]
    return g


def g_pos_coverage():
    d = JP('coverage_check.json')
    return [(d['ok'], 'Proposition middle (a): exact coverage check'),
            (d['roots'] == int(CLAIMS['pRoots']) and d['boundary_sides'] == int(CLAIMS['pBsides']) and
             d['leaves'] == {'B': int(CLAIMS['pCellsB']), 'L': 0, 'W': int(CLAIMS['pCellsW'])},
             '%s root triangles (%s B, %s W), %s boundary sides' % (CLAIMS['pRoots'], CLAIMS['pCellsB'], CLAIMS['pCellsW'], CLAIMS['pBsides']))]


def g_pos_cells():
    d = JP('recheck_cells.json')
    return [(d['ok'] and d['B_cells'] == int(CLAIMS['pCellsB']), 'every prism and closing inequality re-evaluated from the stored bounds'),
            (d['prisms'] == int(CLAIMS['pPrisms']), '%s prisms' % CLAIMS['pPrisms']),
            (rounds_to(d['min_relative_margin'], CLAIMS['pMargin'].replace('e', 'E')), 'smallest relative margin %s' % CLAIMS['pMargin'])]


def g_pos_indep(name, p):
    def g():
        d = JP('report_%s.json' % name); b = d['V4']['b']; a = d['V4']['a']; v3 = d['V3']
        sfx = PROFILES[p][2]
        out = [(d['summary']['selftests'] and d['V2']['ok'] and v3['ok'] and d['V4']['ok'], 'independent verifier: selftests, V2, V3, V4 pass'),
               (d['V2']['checked'] == d['V2']['passed'] == int(CLAIMS['pLamRec']) and v3['inequalities'] == v3['passed'] == int(CLAIMS['pIneqC']),
                '%s lambda_2 records, %s inequalities (C)' % (CLAIMS['pLamRec'], CLAIMS['pIneqC'])),
               (rounds_to(v3['min_quotient'], CLAIMS['pQmin' + sfx]) and v3['min_quotient_k'] == int(CLAIMS['pQminK']),
                'smallest quotient %s at k = %s' % (CLAIMS['pQmin' + sfx], CLAIMS['pQminK'])),
               (a['ok'] and a['roots'] == int(CLAIMS['pRoots']) and a['boundary_sides'] == int(CLAIMS['pBsides']), 'coverage'),
               (b['n_prisms'] == int(CLAIMS['pPrisms' + sfx]) and b['used_nodes'] == int(CLAIMS['pNodes' + sfx]) and b['ktop_max'] == int(CLAIMS['pKtop' + sfx])
                and rounds_to(b['alpha_ktop_max'], CLAIMS['pAtop' + sfx]), '%s prisms, %s vertex bounds, k_p <= %s (alpha ~ %s)' % (
                    CLAIMS['pPrisms' + sfx], CLAIMS['pNodes' + sfx], CLAIMS['pKtop' + sfx], CLAIMS['pAtop' + sfx])),
               (d['V4']['used_nodes_max_n'] == int(CLAIMS['pNmax' + sfx]) and d['V4']['used_nodes_max_N'] == int(CLAIMS['pNdof' + sfx]),
                'largest mesh n = %s (%s unknowns)' % (CLAIMS['pNmax' + sfx], CLAIMS['pNdof' + sfx])),
               (b['leaves'] == {'B': int(CLAIMS['pCellsB']), 'W': int(CLAIMS['pCellsW']), 'L': 0} and
                WORDS.get(len(d['V4']['B_cells_with_F_ge_0_on_whole_triangle'])) == CLAIMS['pThinBword'],
                '%s B cells (%s of them inside F >= 0), %s W cell' % (CLAIMS['pCellsB'], CLAIMS['pThinBword'], CLAIMS['pCellsW']))]
        if p == 'ls':
            sc = v3['shape_constants_own']; v1 = d['V1']; v5 = d['V5']
            out += [(v1['ok'] and v1['n_fails'] == 0 and v1['records_computed_in_this_run'] > 0 and v1['ring']['checked'] > 0
                     and v1['cover_nodes']['checked'] > 0, 'V1 (sample): %d ring and %d vertex bounds re-certified from scratch' % (
                         v1['ring']['checked'], v1['cover_nodes']['checked'])),
                    (rounds_to(sc['c_l_lower'], CLAIMS['pCellIndep']) and rounds_to(sc['rho_upper'], CLAIMS['pRhoIndep']),
                     "verifier's own shape constants %s / %s" % (CLAIMS['pCellIndep'], CLAIMS['pRhoIndep'])),
                    (v5['ok'] and v5['v4b_lambda0_times_1.0001_all_cells']['cells_rejected'] == int(CLAIMS['pIndepRej'])
                     and v5['v3_tstar_0.12']['rejected'] and v5['v3_ring_bounds_lowered_0.2pct_k100_139']['rejected'],
                     'negative controls (verifier): 1.0001 lambda_0 rejects %s cells' % CLAIMS['pIndepRej']),
                    (v5['cr_sigma_positive']['ok'] and len(v5['cr_sigma_positive']['v1']) == 3 and len(v5['cr_sigma_positive']['v2']) == 3
                     and all(x['rejected'] for x in v5['cr_sigma_positive']['v1'] + v5['cr_sigma_positive']['v2']),
                     'negative controls (verifier): sigma, sigma_2 = -1e6, 0, just below the pole rejected by V1 and V2'),
                    (v5['exact_sigma_556']['ok'], 'negative control (verifier): sides 5,5,6, sigma = 15/2 + 2^-61 rejected'),
                    (v5['v4b_negative_bounds_rejected']['rejected'], 'negative control (verifier): negative vertex bounds in a cell rejected'),
                    (d['sanity_P2']['ok'] and rounds_to(d['sanity_P2']['min_rel_gap'], CLAIMS['pSandLo'].replace('e', 'E')),
                     'sandwich (float): smallest relative gap %s' % CLAIMS['pSandLo'])]
        return out
    return g


def g_pos_indep_full(name, p):
    def g():
        d = JP('report_%s.json' % name); v1 = d['V1']
        nfile = sum(1 for f in (POS / PROFILES[p][0]).glob('coverB_nodes_*.jsonl') for l in open(f) if l.strip())
        return [(all(v for v in d['summary'].values() if v is not None) and v1['mode'] == 'full', 'independent verifier, V1 full'),
                (v1['ring']['checked'] == v1['ring']['passed'] == int(CLAIMS['pRingRec']) and v1['ring']['all_ring_verified'],
                 'all %s ring bounds re-certified' % CLAIMS['pRingRec']),
                (v1['cover_nodes']['checked'] == v1['cover_nodes']['passed'] == nfile and v1['cover_nodes']['all_referenced_verified'],
                 'all %d stored vertex bounds re-certified' % nfile)]
    return g


def g_pos_recheck_full(p):
    def g():
        d = JP('%s/recheck_full.json' % p)
        return [(d['ok'] and d['mode'] == 'full' and d['vertex_records'] == d['vertex_records_used_by_cells'] == int(CLAIMS['pNodes' + PROFILES[p][2]])
                 and d['ring_records'] == int(CLAIMS['pRingRec']) and d['lam2_records'] == int(CLAIMS['pLamRec']),
                 'original code: all used vertex bounds, ring and lambda_2 bounds re-certified')]
    return g


def g_pos_summary(p):
    def g():
        d = JP('%s/summary.json' % p); sfx = PROFILES[p][2]
        t = '%d, %d, %d' % tuple(round(d[k] / 60) for k in ('ring_sec', 'lam2_sec', 'cover_sec'))
        out = [(d['ring_records'] == int(CLAIMS['pRingRec']) and d['lam2_records'] == int(CLAIMS['pLamRec']), 'record counts'),
               (d['ring_n'] == [int(CLAIMS['pRingN' + sfx])] and d['ring_N'] == int(CLAIMS['pRingNN' + sfx]) and
                d['lam2_n'] == [int(CLAIMS['pLamN' + sfx])] and d['lam2_N'] == int(CLAIMS['pLamNN' + sfx]),
                'mesh parameters: ring %s (%s), lambda_2 %s (%s)' % (CLAIMS['pRingN' + sfx], CLAIMS['pRingNN' + sfx], CLAIMS['pLamN' + sfx], CLAIMS['pLamNN' + sfx])),
               (d['prisms'] == int(CLAIMS['pPrisms' + sfx]) and d['vertex_bounds_used'] == int(CLAIMS['pNodes' + sfx]) and
                d['cover_n_max'] == int(CLAIMS['pNmax' + sfx]) and d['cover_N_max'] == int(CLAIMS['pNdof' + sfx]), 'cover statistics'),
               (rounds_to(d['local_min_quotient'], CLAIMS['pQmin' + sfx]), 'smallest quotient %s' % CLAIMS['pQmin' + sfx]),
               (t == CLAIMS['pTime' + (sfx or 'ls')], 'times %s min' % CLAIMS['pTime' + (sfx or 'ls')])]
        if p == 'ls':
            out.append((d['ring_records'] + d['lam2_records'] + d['vertex_bounds_computed'] == int(CLAIMS['pEigCount'])
                        and d['ring_sec'] + d['lam2_sec'] + d['cover_sec'] < 1800, '%s eigenvalue problems, < 30 min' % CLAIMS['pEigCount']))
        return out
    return g


def g_pos_negative():
    d = JP('negative_controls.json')
    n6 = [k for k in d if k.startswith('N6 ')]; n7 = [k for k in d if k.startswith('N7 ')]
    return [(all(v for k, v in d.items() if isinstance(v, bool)) and sum(isinstance(v, bool) for v in d.values()) == 8,
             'negative controls N1-N7 rejected, sandwich S1 holds'),
            (len(n6) == 1 and d[n6[0]] is True and all(d['N6_detail'].values()),
             'N6: sigma = -1e6, 0 and just below the pole -1/(kappa0 h)^2 rejected by cr_bound, certify, lambda_2 certifier and re-checks; '
             'negative vertex bounds rejected by the cell re-check'),
            (len(n7) == 1 and d[n7[0]] is True and d['N7_detail']['above_rejected'] and d['N7_detail']['exact_15_2_returned'],
             'N7: exact sigma tested (sides 5,5,6: 15/2 + 2^-61 rejected, 15/2 returned exactly)'),
            (rounds_to(d['S1_gap_max'], CLAIMS['pSandHi'].replace('e', 'E')), 'sandwich (float): largest relative gap %s' % CLAIMS['pSandHi'])]


def g_pos_float():
    d = JP('float_local.json')
    pc = lambda x: 100 * (x - 1)
    return [(rounds_to(d['R1_over_a_at_0_1'], CLAIMS['pRaLow']) and rounds_to(d['R1_over_a_max'], CLAIMS['pRaMax'])
             and rounds_to(d['R1_over_a_max_alpha'], CLAIMS['pRaMaxAlpha']) and rounds_to(d['R1_over_a_last'], CLAIMS['pRaLast'])
             and rounds_to(d['spare_factor'], CLAIMS['pSpare']), 'R_1/a: %s, max %s near %s, %s; factor %s' % (
                 CLAIMS['pRaLow'], CLAIMS['pRaMax'], CLAIMS['pRaMaxAlpha'], CLAIMS['pRaLast'], CLAIMS['pSpare'])),
            (rounds_to(d['hessian_min'], CLAIMS['pHessMin']) and rounds_to(d['hessian_min_alpha'], CLAIMS['pHessMinAlpha'])
             and rounds_to(d['hessian_max'], CLAIMS['pHessMax']), 'Hessian between %s and %s' % (CLAIMS['pHessMin'], CLAIMS['pHessMax'])),
            (rounds_to(pc(d['node_bound_over_true_min']), CLAIMS['pNodeLo']) and rounds_to(pc(d['node_bound_over_true_max']), CLAIMS['pNodeHi'])
             and rounds_to(pc(d['interval_bound_over_true_near_2']), CLAIMS['pIvNear']) and rounds_to(pc(d['interval_bound_over_true_max']), CLAIMS['pIvMax'])
             and d['interval_bound_over_true_max_alpha'] < 0.1 and d['node_bound_over_true_min'] > 1,
             'ring bound exceeds R_1 by %s-%s %% (nodes), %s %% near 2, <= %s %%' % (CLAIMS['pNodeLo'], CLAIMS['pNodeHi'], CLAIMS['pIvNear'], CLAIMS['pIvMax'])),
            (rounds_to(100 * d['gap_inner_min'], CLAIMS['pGapLo']) and rounds_to(100 * d['gap_inner_max'], CLAIMS['pGapHi']),
             'true gap at t = 0.057: %s-%s %%' % (CLAIMS['pGapLo'], CLAIMS['pGapHi'])),
            (rounds_to(d['small_threshold_tall_isosceles_t_1_24'], CLAIMS['pSmallThr']), 'weak-coupling threshold %s' % CLAIMS['pSmallThr'])]


def g_ma():
    d = J('ma_check.json')
    return [(d['ok'] and d['boxes_contiguous'] and d['theta_gt_pi_4'], "Ma's middle-range criterion (prior work): every box passes"),
            (X(d['beta_Mlo_upper']) < 1 and X(d['beta_Mhi_lower']) > 10, 'boxes cover 1 <= beta <= 10'),
            (d['boxes'] == int(CLAIMS['maBoxes']) and d['F_case'] + d['H_case'] == d['boxes'], '%s boxes' % CLAIMS['maBoxes'])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', choices=['certified', 'independent', 'checks', 'paper'])
    ap.add_argument('--cells', action='store_true', help='also replay the six-wave cell certificate')
    ap.add_argument('--cells-dir', default=os.environ.get('CELLS_DIR', str(ROOT / 'cells')))
    ap.add_argument('--positive-full', action='store_true',
                    help='also re-certify every stored bound of Part II (three runs) from scratch with both programs')
    a = ap.parse_args()
    LOGS.mkdir(exist_ok=True); POSOUT.mkdir(exist_ok=True)
    code = ROOT / 'code'
    pos = code / 'positive'
    vind = str(code / 'verify_positive' / 'verify_indep.py')
    want = lambda g: a.only in (None, g)
    steps = []
    if want('certified'):
        steps += [('thmA', [PY, 'thmA.py'], code, g_thmA),
                  ('thmC', [PY, 'large_beta.py'], code, g_thmC),
                  ('pos_weak', [PY, 'weak_cert.py', str(POSOUT / 'weak_cert.json')], pos, g_pos_weak, penv()),
                  ('pos_local', [PY, 'local_check.py', '-', '-', str(POSOUT / 'local_cert.json')], pos, g_pos_local(), penv()),
                  ('pos_coverage', [PY, 'coverage_check.py'], pos, g_pos_coverage, penv()),
                  ('pos_cells', [PY, 'recheck.py', 'cells'], pos, g_pos_cells, penv())]
    if want('independent'):
        steps += [('thmC_independent', [PY, 'indep_cert.py', CLAIMS['indepMbig']], code, g_indep),
                  ('xcheck_tail', [PY, 'xcheck_tail.py'], code, g_simple('xcheck_tail.json', []))]
        steps += [('pos_indep', [PY, vind, str(POS / 'certs'), '0.2983', '--v1', 'sample', '--fresh', '--out', str(POSOUT), '--name', 'certs'],
                   ROOT, g_pos_indep('certs', 'ls'), penv())]
        for p in ('pw', 'liu'):
            d, k, _ = PROFILES[p]
            steps.append(('pos_indep_' + p, [PY, vind, str(POS / d), k, '--v1', 'none', '--no-controls', '--no-sanity', '--out', str(POSOUT),
                                             '--name', d], ROOT, g_pos_indep(d, p), penv(p)))
    if want('checks'):
        steps += [
            ('thmA_negative', [PY, 'thmA.py'], code, g_thmA_negative, dict(ENV, THMA_NEGATIVE='1')),
            ('negative_c', [PY, 'negative_c.py'], code, g_negative_c),
            ('table_c', [PY, 'table_c.py'], code, g_table_c),
            ('check_normalised', [PY, 'check_normalised.py'], code, g_simple('check_normalised.json', [])),
            ('check_quad', [PY, 'check_quad.py'], code, g_simple('check_quad.json', [
                (lambda d: d['max_rel_err'] <= C('quadErr'), 'closed forms vs quadrature on T <= %s' % CLAIMS['quadErr'])])),
            ('direct_check', [PY, 'direct_check.py'], code, g_simple('direct_check.json', [
                (lambda d: d['max_rel_err'] <= C('directErr'), 'referee quadrature vs closed forms <= %s' % CLAIMS['directErr'])])),
            ('validate_chain', [PY, 'validate_chain.py'], code, g_simple('validate_chain.json', [
                (lambda d: d['chain_samples'] == int(CLAIMS['chainSamples']), '%s near-region samples' % CLAIMS['chainSamples']),
                (lambda d: d['corner_samples'] == int(CLAIMS['cornerSamples']) and d['corner_worst'] < 0,
                 'corner bound on %s samples' % CLAIMS['cornerSamples']),
                (lambda d: d['divergence_rel_err'] <= C('divErr'), 'divergence identity <= %s' % CLAIMS['divErr'])])),
            ('run_chain', [PY, 'run_chain.py'], code, g_simple('run_chain.json', [
                (lambda d: d['tested'] == int(CLAIMS['runChainTested']), '%s referee samples' % CLAIMS['runChainTested']),
                (lambda d: d['max_W_among_failures'] < C('runChainW'), 'failures only at W < %s' % CLAIMS['runChainW'])])),
            ('t_near_fem', [PY, 't_near_fem.py'], code, g_simple('t_near_fem.json', [
                (lambda d: d['samples'] == int(CLAIMS['nearFemSamples']), '%s FEM samples' % CLAIMS['nearFemSamples'])])),
            ('test_independent', [PY, 'test_independent.py'], code, g_simple('test_independent.json', [
                (lambda d: [r for r in d['lam0_vs_fem'] if r['beta'] == 2.0][0]['fem_richardson'] and
                 rounds_to([r for r in d['lam0_vs_fem'] if r['beta'] == 2.0][0]['fem_richardson'], CLAIMS['lamFEMtwo']) and
                 rounds_to([r for r in d['lam0_vs_fem'] if r['beta'] == 2.0][0]['formula'], CLAIMS['lamZeroTwo']),
                 'beta = 2: FEM %s, lambda_0 %s' % (CLAIMS['lamFEMtwo'], CLAIMS['lamZeroTwo']))])),
            ('check_misc', [PY, 'check_misc.py'], code, g_simple('check_misc.json', [
                (lambda d: d['geomA_samples'] == int(CLAIMS['geomAsamples']) and
                 C('geomAmin') <= Fr(d['geomA_min_ratio']) < C('geomAmin') + Fr(1, 10 ** 4) and C('geomAmin') > C('GA'),
                 'sampled min (p-1)/(tau-2) = %s... on %s triangles' % (CLAIMS['geomAmin'], CLAIMS['geomAsamples']))])),
            ('cells_summary', [PY, '-c', 'pass'], ROOT, g_cells_summary),
            ('pos_negative', [PY, 'negative_controls.py'], pos, g_pos_negative, penv()),
            ('pos_float', [PY, 'float_local.py'], pos, g_pos_float, penv()),
        ]
        steps += [('pos_summary_' + p, [PY, 'summary.py'], pos, g_pos_summary(p), penv(p, p)) for p in PROFILES]
        steps += [('prior_work_ma', [PY, 'ma_run_boxes.py'], code / 'prior_work', g_ma)]
    if a.cells:
        steps += [('cells_replay', [PY, 'scripts/cells.py', 'replay', a.cells_dir], ROOT, g_cells)]
    if a.positive_full:
        for p, (d, k, _) in PROFILES.items():
            steps += [('pos_recheck_full_' + p, [PY, 'recheck.py', 'full'], pos, g_pos_recheck_full(p), penv(p, p)),
                      ('pos_indep_full_' + p, [PY, vind, str(POS / d), k, '--v1', 'full', '--all-nodes', '--fresh', '--no-controls',
                                               '--no-sanity', '--out', str(POSOUT), '--name', 'full_' + d], ROOT, g_pos_indep_full('full_' + d, p), penv(p))]
    tex = all(shutil.which(t) for t in ('latexmk', 'pdflatex'))
    if want('paper'):
        steps += [('claims_tex', [PY, 'scripts/make_claims_tex.py', '--check'], ROOT,
                   lambda: [(True, 'paper/claims.tex is in sync with claims.json')])]
        if tex:
            steps += [('paper', ['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', '-gg',
                                 'robin-triangles.tex'], ROOT / 'paper', lambda: [(True, 'latexmk succeeded')])]
    results = [run(*s) for s in steps]
    skipped = []
    if want('paper') and not tex:
        # no TeX installation: the paper build and its log check are reported as SKIPPED (not as PASS)
        msg = 'latexmk/pdflatex not found on PATH; install TeX Live (latexmk, pdflatex) to build and check the paper'
        for nm in ('paper', 'paper_log_clean'):
            results.append({'name': nm, 'ok': True, 'skipped': True, 'reason': msg})
            print('%-22s SKIPPED (%s)' % (nm, msg))
        skipped = ['paper', 'paper_log_clean']
    elif want('paper'):
        log = ROOT / 'paper' / 'robin-triangles.log'
        txt = log.read_text(errors='replace') if log.exists() else 'missing log: There were undefined'
        bad = re.findall(r'(LaTeX Warning|Package \w+ Warning|Overfull \\[hv]box|Underfull \\[hv]box|There were undefined)', txt)
        results.append({'name': 'paper_log_clean', 'ok': not bad, 'issues': len(bad)})
        print('%-22s %s' % ('paper_log_clean', 'PASS' if not bad else 'FAIL (%d issues)' % len(bad)))
    inputs = sorted(p for p in ROOT.rglob('*') if p.is_file() and p.suffix in {'.py', '.tex', '.md', '.txt', '.cff', '.json', '.jsonl'}
                    and not ({'.git', '.venv', 'verification', 'cells', '__pycache__'} & set(p.relative_to(ROOT).parts)))
    import flint, numpy, scipy, mpmath, sympy
    summary = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'only': a.only, 'cells': a.cells,
               'positive_full': a.positive_full,
               'python': sys.version.split()[0], 'python_flint': flint.__version__, 'numpy': numpy.__version__,
               'scipy': scipy.__version__, 'mpmath': mpmath.__version__, 'sympy': sympy.__version__,
               'results': results, 'inputs_sha256': {str(p.relative_to(ROOT)): sha(p) for p in inputs}}
    pdf = ROOT / 'paper' / 'robin-triangles.pdf'
    if pdf.exists():
        summary['pdf_sha256'] = sha(pdf)
    allok = all(r['ok'] for r in results)
    summary['overall'] = ('PASS' if not skipped else 'PASS (SKIPPED: %s)' % ', '.join(skipped)) if allok else 'FAIL'
    summary['skipped'] = skipped
    extra = '-'.join(x for x, f in (('cells', a.cells), ('positive-full', a.positive_full)) if f)
    (LOGS / ('summary-partial.json' if a.only else 'summary-%s.json' % extra if extra else 'summary.json')).write_text(
        json.dumps(summary, indent=1) + '\n')
    print('OVERALL', summary['overall'])
    return 0 if allok else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Summary statistics of the certificates of the current profile (for the paper's table)."""
import os, json, glob, collections
import config
from local_ring import load, dec
C = config.CERT
def jl(p):
    return [json.loads(l) for f in sorted(glob.glob(os.path.join(C, p))) for l in open(f) if l.strip()]
out = dict(profile=config.PROFILE, kappa0=float(config.C_INTERP))
ring = jl('ring_nodes.jsonl'); l2 = jl('lam2_nodes.jsonl')
if ring: out.update(ring_records=len(ring), ring_n=sorted(set(r['n'] for r in ring)), ring_N=max(r['N'] for r in ring), ring_sec=round(sum(r['sec'] for r in ring)))
if l2: out.update(lam2_records=len(l2), lam2_n=sorted(set(r['n'] for r in l2)), lam2_N=max(r['N'] for r in l2), lam2_sec=round(sum(r['sec'] for r in l2)))
fn = os.path.join(C, 'local_cert.json')
if os.path.exists(fn):
    lc = json.load(open(fn)); out.update(local_ok=lc['ok'], local_min_quotient=lc['worst_ratio'], local_worst_k=lc['worst_k'], c_l=lc['c_l'], rho=lc['rho'])
    iv = lc['intervals']
    out['quotient_min_alpha_le_0.6'] = min(d['ratio'] for d in iv if d['alpha_lo'] <= 0.6)
    out['quotient_min_alpha_ge_30'] = min(d['ratio'] for d in iv if d['alpha_lo'] >= 30)
    out['R1bound_over_lam0_max'] = max(d['R1_over_lam'] for d in iv)
cells = jl('coverB_cells_*.jsonl'); nodes = jl('coverB_nodes_*.jsonl')
if cells:
    used = set(); pr = 0
    for c in cells:
        if c['type'] == 'B':
            pr += len(c['prisms'])
            for key, (lb, n) in c['nodes'].items(): used.add((c['verts'][int(key.split(':')[0])], int(key.split(':')[1]), n))
    nd = {(r['w'], r['k'], r['n']): r for r in nodes}
    out.update(cells=dict(collections.Counter(c['type'] for c in cells)), prisms=pr, vertex_bounds_used=len(used), vertex_bounds_computed=len(nodes),
               cover_n_max=max(n for _, _, n in used), cover_N_max=max(nd[u]['N'] for u in used), ktop_max=max(c['ktop'] for c in cells if c['type'] == 'B'),
               ktop_min=min(c['ktop'] for c in cells if c['type'] == 'B'), alpha_top_max=float(config.ALPHA(max(c['ktop'] for c in cells if c['type'] == 'B'))),
               cover_sec=round(sum(c['sec'] for c in cells)), kinds=dict(collections.Counter(nd[u]['kind'] for u in used)),
               shape_vertices=len(set(u[0] for u in used)))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(config.OUT, 'summary.json'), 'w'), indent=1)

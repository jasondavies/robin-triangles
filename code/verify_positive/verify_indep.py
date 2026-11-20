#!/usr/bin/env python3
"""Independent verifier for the certificates of the Robin-triangle proof.

usage:  nice -n 10 python3 code/verify_positive/verify_indep.py <certdir> <kappa0> [--v1 full|sample|none]
                 [--all-nodes] [--name NAME] [--out DIR] [--no-controls] [--no-sanity] [--sweeps N] [--fresh]

Written by a separate AI instance from the text of the alpha > 0 proof (now Part II of paper/robin-triangles.tex)
and the certificate format (data/positive/FORMAT.txt) only, without access to the certifying code in code/positive/.
Log, report and V1 cache are written to --out (default: this directory).
V1  Crouzeix-Raviart lower bound records (ring_nodes.jsonl, coverB_nodes_*.jsonl)
V2  second eigenvalue records (lam2_nodes.jsonl)
V3  Lemma 'shapeconst' and the 337 inequalities (C) of Lemma 'Ccheck'
V4  the cover: coverage (exact rationals), prism inequalities, closing tests, W/L cells
V5  negative controls;  plus a floating-point sanity cross-check (not a proof step).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from fractions import Fraction as Fr

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
from mpmath import iv  # noqa: E402

import ivnp  # noqa: E402
import geom  # noqa: E402
import crmesh  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v3  # noqa: E402
import v4  # noqa: E402
import lam0 as L0  # noqa: E402

LOG = None


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG is not None:
        LOG.write(s + "\n")
        LOG.flush()


def read_jsonl(path):
    out = []
    bad = 0
    if not os.path.exists(path):
        return out, 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            bad += 1        # a line still being written
    return out, bad


def rec_hash(src, r, kappa0):
    key = json.dumps([src, r.get("key"), r.get("w"), r.get("k"), r.get("kind"), r.get("n"), r.get("alpha"),
                      r.get("sigma"), r.get("lb"), str(kappa0)], sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:24]


# ------------------------------------------------------------------------------------------- selftests
def selftests():
    out = {}
    out["interval_arithmetic_selftest"] = bool(ivnp.selftest())
    # float-interval assembly (V1 path) versus Arb assembly from explicit vertex coordinates (V2 path)
    alpha = Fr(7, 3)
    n = 6
    tri = geom.tri_equilateral(alpha)
    m = crmesh.build_mesh(tri, "U", n)
    c, X, bl, _ = crmesh.element_geometry(m)
    W, diag, mass = crmesh.assemble_float(m, c, X, bl)
    m2, K, M, h2, area = v2.assemble_arb_E(n, alpha)
    Kf = np.diag(diag) - W.toarray()
    err = 0.0
    for i in range(m.N):
        err = max(err, abs(float(M[i].mid()) - mass[i]))
        for j in range(m.N):
            kij = float(K[i][j].mid()) if j in K[i] else 0.0
            err = max(err, abs(kij - Kf[i, j]))
    out["assembly_cross_check_max_abs_diff"] = err
    # explicit float vertex coordinates reproduce the lattice edge vectors (U and A, a scalene triangle)
    w = (Fr(3, 5), Fr(1, 4))
    tri = geom.tri_cover(w, Fr(1))
    worst = 0.0
    for kind in ("U", "A"):
        mm = crmesh.build_mesh(tri, kind, 5)
        cc, XX, bb, ln = crmesh.element_geometry(mm)
        tot = 0.0
        for e in range(len(mm.elem_lat)):
            L = mm.lat[int(mm.elem_lat[e])]
            P = []
            for v in range(3):
                i, j = int(mm.elem_ij[e, v, 0]), int(mm.elem_ij[e, v, 1])
                P.append((float((L["o"][0] + i * L["e1"][0] + j * L["e2"][0]).mid),
                          float((L["o"][1] + i * L["e1"][1] + j * L["e2"][1]).mid)))
            ar = 0.5 * ((P[1][0] - P[0][0]) * (P[2][1] - P[0][1]) - (P[1][1] - P[0][1]) * (P[2][0] - P[0][0]))
            tot += ar
            worst = max(worst, abs(ar - 0.5 * XX.mid()[e]))
            for k in range(3):
                a_, b_ = P[(k + 1) % 3], P[(k + 2) % 3]
                o = P[k]
                u = (a_[0] - o[0], a_[1] - o[1])
                vv = (b_[0] - o[0], b_[1] - o[1])
                cot = (u[0] * vv[0] + u[1] * vv[1]) / (u[0] * vv[1] - u[1] * vv[0])
                worst = max(worst, abs(2 * cot - cc.mid()[e, k]))
        worst = max(worst, abs(tot - 1.0))
    out["explicit_coordinates_cross_check_max_abs_diff"] = worst
    # CR bound for E against the closed formula of Proposition 'equi'
    al = Fr(3, 2)
    tri = geom.tri_equilateral(al)
    mm = crmesh.build_mesh(tri, "U", 40)
    cc, XX, bb, _ = crmesh.element_geometry(mm)
    W, diag, mass = crmesh.assemble_float(mm, cc, XX, bb)
    import eig
    x, lam = eig.ground_state(W, diag, mass, eig.coarse_estimate(tri, "U", 40))
    exact = float(L0.lam0(al)[0].mid)
    out["CR_E_n40_vs_closed_formula"] = [lam, exact]
    # (the CR eigenvalue itself need not be below the exact one; the corrected value must be)
    hh = float(mm.h2_hi) ** 0.5
    corrected = lam / (1 + (0.1893 * hh) ** 2 * lam)
    out["CR_E_n40_corrected_kappa0_0.1893"] = corrected
    out["ok"] = bool(out["interval_arithmetic_selftest"] and err < 1e-12 and worst < 1e-12
                     and abs(exact - lam) < 1e-3 * exact and corrected < exact)
    return out


# ------------------------------------------------------------------------------------------- V1
def v1_items(certdir, mode, used_nodes, all_nodes, seed=12345):
    ring, bad = read_jsonl(os.path.join(certdir, "ring_nodes.jsonl"))
    items = []
    for r in ring:
        k = r["key"][0]
        if mode == "full" or k % 4 == 0:
            items.append(("ring", r))
    nodes, dup = v4.load_nodes(certdir)
    flat = [r for lst in nodes.values() for r in lst]
    if mode == "full":
        sel = [r for r in flat if all_nodes or (r["w"], int(r["k"]), int(r["n"])) in used_nodes]
    else:
        rng = random.Random(seed)
        usedl = [r for r in flat if (r["w"], int(r["k"]), int(r["n"])) in used_nodes]
        big = sorted(usedl, key=lambda r: -r["N"])[:40]
        rest = [r for r in usedl if r not in big]
        sel = big + rng.sample(rest, min(len(rest), 400))
    for r in sel:
        items.append(("node", r))
    return items, ring, bad, flat


def run_v1(certdir, kappa0, mode, used_nodes, all_nodes, cache_path, sweeps):
    rep = {"mode": mode}
    items, ring, bad, flat = v1_items(certdir, mode, used_nodes, all_nodes)
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            try:
                d = json.loads(line)
                cache[d["h"]] = d
            except Exception:
                pass
    cf = open(cache_path, "a")
    t0 = time.time()
    res = {"ring": [], "node": []}
    fails = []
    ncached = 0
    for idx, (src, r) in enumerate(items):
        h = rec_hash(src, r, kappa0)
        if h in cache and cache[h].get("ok"):
            d = cache[h]
            ncached += 1
        else:
            if src == "ring":
                k, sign, n = r["key"]
                tri, alpha = v1.tri_from_ring_record(r)
                meta_ok = (alpha == geom.alpha_grid(k) and r["kind"] == "U" and r["n"] == n
                           and geom.parse_rat(r["s"]) == Fr(1, 10) and sign in (1, -1))
                ident = [k, sign, n]
            else:
                tri, alpha = v1.tri_from_node_record(r)
                meta_ok = alpha == geom.alpha_grid(int(r["k"]))
                ident = [r["w"], r["k"], r["n"]]
            sigma = geom.parse_rat(r["sigma"])
            lb = geom.parse_rat(r["lb"])
            o = v1.verify_cr(tri, r["kind"], r["n"], sigma, lb, kappa0, r["N"], sweeps=sweeps)
            if not o["ok"] and o.get("lb_ok") and not o.get("cw_ok", False):
                # the float eigenvector may need more polishing; retry (does not weaken the certificate)
                o2 = v1.verify_cr(tri, r["kind"], r["n"], sigma, lb, kappa0, r["N"], sweeps=8 * sweeps)
                o2["retried"] = True
                o = o2
            d = {"h": h, "src": src, "id": ident, "kind": r["kind"], "n": r["n"], "N": o.get("N"),
                 "ok": bool(o["ok"] and meta_ok), "meta_ok": bool(meta_ok),
                 "cw_rel_margin": o.get("cw_rel_margin"), "lb_rel_slack": o.get("lb_rel_slack"),
                 "min_cot2": o.get("min_cot2"), "sec": o.get("sec"), "reason": o.get("reason"),
                 "flags": o.get("flags"), "retried": o.get("retried", False), "lb_ok": o.get("lb_ok"),
                 "cw_ok": o.get("cw_ok"), "alpha": float(alpha)}
            cf.write(json.dumps(d) + "\n")
            cf.flush()
        res[src].append(d)
        if not d["ok"]:
            fails.append(d)
            log("  V1 FAIL", src, d["id"], d.get("reason"), d.get("cw_rel_margin"), d.get("flags"))
        if (idx + 1) % 200 == 0:
            log("  V1 progress %d/%d  (%.0f s)" % (idx + 1, len(items), time.time() - t0))
    cf.close()

    def summ(lst):
        if not lst:
            return {"checked": 0}
        okl = [d for d in lst if d["ok"]]
        mg = [d["cw_rel_margin"] for d in lst if d.get("cw_rel_margin") is not None]
        sl = [d["lb_rel_slack"] for d in lst if d.get("lb_rel_slack") is not None]
        s = {"checked": len(lst), "passed": len(okl), "failed": len(lst) - len(okl),
             "min_cw_rel_margin": min(mg) if mg else None, "max_cw_rel_margin": max(mg) if mg else None,
             "min_lb_rel_slack": min(sl) if sl else None, "max_lb_rel_slack": max(sl) if sl else None,
             "max_N": max(d["N"] or 0 for d in lst), "sum_N": sum(d["N"] or 0 for d in lst),
             "retried": sum(1 for d in lst if d.get("retried")),
             "kinds": {kd: sum(1 for d in lst if d["kind"] == kd) for kd in ("U", "A")},
             "min_2cot_U": min([d["min_cot2"] for d in lst if d["kind"] == "U" and d.get("min_cot2") is not None] or [None], key=lambda v: (v is None, v)),
             "solve_seconds": round(sum(d.get("sec") or 0 for d in lst), 1)}
        return s

    rep["ring"] = summ(res["ring"])
    rep["cover_nodes"] = summ(res["node"])
    keys = sorted((r["key"][0], r["key"][1]) for r in ring)
    want = sorted((k, s) for k in range(v3.KTOP + 1) for s in (1, -1))
    rep["ring"]["records_in_file"] = len(ring)
    rep["ring"]["unparsable_lines"] = bad
    rep["ring"]["complete_k0_336_both_signs"] = bool(sorted(set(keys)) == want)
    rep["ring"]["duplicate_keys"] = len(keys) - len(set(keys))
    rep["ring"]["n_values"] = sorted(set(r["n"] for r in ring))
    rep["cover_nodes"]["records_in_files"] = len(flat)
    rep["cover_nodes"]["referenced_by_B_cells"] = len(used_nodes)
    verified_nodes = set((d["id"][0], int(d["id"][1]), int(d["id"][2])) for d in res["node"] if d["ok"])
    rep["cover_nodes"]["referenced_and_verified"] = len(verified_nodes & used_nodes)
    rep["cover_nodes"]["all_referenced_verified"] = bool(used_nodes <= verified_nodes)
    verified_ring = set((d["id"][0], d["id"][1]) for d in res["ring"] if d["ok"])
    rep["ring"]["all_ring_verified"] = bool(set(want) <= verified_ring)
    rep["fails"] = [{"src": d["src"], "id": d["id"], "reason": d.get("reason"), "cw_rel_margin": d.get("cw_rel_margin"),
                     "flags": d.get("flags"), "lb_ok": d.get("lb_ok")} for d in fails[:50]]
    rep["n_fails"] = len(fails)
    rep["ok"] = bool(len(fails) == 0 and len(items) > 0)
    rep["records_taken_from_cache_of_earlier_run_of_this_verifier"] = ncached
    rep["records_computed_in_this_run"] = len(items) - ncached
    rep["seconds"] = round(time.time() - t0, 1)
    return rep, res


# ------------------------------------------------------------------------------------------- V2
def run_v2(certdir, kappa0):
    recs, bad = read_jsonl(os.path.join(certdir, "lam2_nodes.jsonl"))
    t0 = time.time()
    rep = {"records_in_file": len(recs), "checked": 0, "passed": 0, "fails": []}
    minpiv = None
    negs = {}
    slack = []
    for r in recs:
        o = v2.verify_lam2(r, kappa0)
        rep["checked"] += 1
        if o["ok"]:
            rep["passed"] += 1
        else:
            rep["fails"].append({"key": r["key"], "detail": {k: v for k, v in o.items()}})
            log("  V2 FAIL", r["key"], o)
        if o.get("min_abs_pivot") is not None:
            minpiv = o["min_abs_pivot"] if minpiv is None else min(minpiv, o["min_abs_pivot"])
        negs[str(o.get("neg_pivots"))] = negs.get(str(o.get("neg_pivots")), 0) + 1
        if o.get("lb_rel_slack") is not None:
            slack.append(o["lb_rel_slack"])
    ks = sorted(r["key"][0] for r in recs)
    rep["keys_complete_0_4_336"] = bool(ks == list(range(0, v3.KTOP + 1, 4)))
    rep["n_values"] = sorted(set(r["n"] for r in recs))
    rep["N_values"] = sorted(set(r["N"] for r in recs))
    rep["negative_pivot_histogram"] = negs
    rep["min_abs_pivot"] = minpiv
    rep["min_lb_rel_slack"] = min(slack) if slack else None
    rep["max_lb_rel_slack"] = max(slack) if slack else None
    rep["arb_prec_bits"] = v2.ARB_PREC
    rep["ok"] = bool(rep["passed"] == rep["checked"] == len(recs) and len(recs) > 0)
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


# ------------------------------------------------------------------------------------------- V3
PAPER_CL = Fr("0.7361")
PAPER_RHO = Fr("0.2938")


def load_local_inputs(certdir):
    ring, _ = read_jsonl(os.path.join(certdir, "ring_nodes.jsonl"))
    mu = {}
    for r in ring:
        k, sg, n = r["key"]
        val = geom.parse_rat(r["lb"])
        if (k, sg) not in mu or val > mu[(k, sg)]:
            mu[(k, sg)] = val
    lam2r, _ = read_jsonl(os.path.join(certdir, "lam2_nodes.jsonl"))
    lam2 = {}
    for r in lam2r:
        k = r["key"][0]
        val = geom.parse_rat(r["lb"])
        if k not in lam2 or val > lam2[k]:
            lam2[k] = val
    return mu, lam2


def run_v3(certdir):
    t0 = time.time()
    rep = {}
    cl, rho = v3.shape_constants()
    rep["shape_constants_own"] = {"c_l_lower": float(cl.a), "rho_upper": float(rho.b), "grid": "20 x 420 boxes on (0,3/50] x [0,pi/3]"}
    rep["paper_constants"] = {"c_l": float(PAPER_CL), "rho": float(PAPER_RHO)}
    rep["lemma_shapeconst_verified"] = bool(cl.a >= geom.ivq(PAPER_CL).b and rho.b <= geom.ivq(PAPER_RHO).a)
    okt, th = v3.tanh_check()
    rep["rstar_lt_tanh_2tstar"] = okt
    rep["tanh_2tstar"] = th
    mu, lam2 = load_local_inputs(certdir)
    need = [(k, s) for k in range(v3.KTOP + 1) for s in (1, -1)]
    missing = [x for x in need if x not in mu]
    missing2 = [k for k in range(0, v3.KTOP + 1, 4) if k not in lam2]
    rep["missing_ring_bounds"] = len(missing)
    rep["missing_lam2_bounds"] = len(missing2)
    if missing or missing2:
        rep["ok"] = False
        rep["reason"] = "ring or lambda_2 bounds incomplete"
        rep["seconds"] = round(time.time() - t0, 1)
        return rep
    res, consts = v3.check_C(mu, lam2, PAPER_CL, PAPER_RHO)
    rep["constants"] = consts
    rep["inequalities"] = len(res)
    rep["passed"] = sum(1 for r in res if r["ok"])
    worst = min(res, key=lambda r: r.get("ratio", -1))
    rep["min_quotient"] = worst.get("ratio")
    rep["min_quotient_k"] = worst["k"]
    rep["min_quotient_alpha"] = float(geom.alpha_grid(worst["k"]))
    rep["tail_quotient"] = res[-1].get("ratio")
    # the paper's side remarks: quotient > 1.3 for alpha >= 30 and alpha <= 0.6
    hi = [r["ratio"] for r in res if geom.alpha_grid(r["k"]) >= 30 and "ratio" in r]
    lo = [r["ratio"] for r in res[:-1] if geom.alpha_grid(r["k"]) <= Fr(6, 10) and "ratio" in r]   # intervals meeting alpha <= 0.6
    rep["min_quotient_alpha_ge_30"] = min(hi) if hi else None
    rep["min_quotient_alpha_le_0.6"] = min(lo) if lo else None
    # with the verifier's own (sharper) shape constants, rounded safely
    cl_own = Fr(int(float(cl.a) * 10 ** 6), 10 ** 6)
    rho_own = Fr(int(float(rho.b) * 10 ** 6) + 1, 10 ** 6)
    if geom.ivq(cl_own).b <= cl.a and geom.ivq(rho_own).a >= rho.b:
        res2, _ = v3.check_C(mu, lam2, cl_own, rho_own)
        w2 = min(res2, key=lambda r: r.get("ratio", -1))
        rep["with_own_constants"] = {"c_l": float(cl_own), "rho": float(rho_own), "passed": sum(1 for r in res2 if r["ok"]),
                                     "min_quotient": w2.get("ratio"), "min_quotient_k": w2["k"]}
    lc = os.path.join(certdir, "local_cert.json")
    if os.path.exists(lc):
        try:
            c = json.load(open(lc))
            ivs = c.get("intervals", [])
            cmp_ = {"present": True, "ok_flag": c.get("ok"), "c_l": c.get("c_l"), "rho": c.get("rho"),
                    "worst_ratio": c.get("worst_ratio"), "worst_k": c.get("worst_k"), "n_intervals": len(ivs)}
            if len(ivs) == len(res):
                cmp_["max_rel_diff_ratio"] = max(abs(a["ratio"] / b["ratio"] - 1) for a, b in zip(res, ivs))
                cmp_["max_rel_diff_dplus"] = max(abs(a["dplus"] / b["dplus"] - 1) for a, b in zip(res, ivs))
                cmp_["max_rel_diff_dminus"] = max(abs(a["dminus"] / b["dminus"] - 1) for a, b in zip(res, ivs))
                cmp_["max_rel_diff_gamma"] = max(abs(a["gamma"] / b["gamma"] - 1) for a, b in zip(res, ivs))
            cmp_["stored_constants_not_sharper_than_own"] = bool(c.get("c_l", 0) <= float(cl.a) and c.get("rho", 1) >= float(rho.b))
            rep["local_cert_json"] = cmp_
        except Exception as ex:
            rep["local_cert_json"] = {"present": True, "error": repr(ex)}
    else:
        rep["local_cert_json"] = {"present": False}
    rep["ok"] = bool(rep["passed"] == len(res) == v3.KTOP + 1 and rep["lemma_shapeconst_verified"] and okt)
    rep["per_interval"] = [{"k": r["k"], "ratio": r.get("ratio"), "ok": r["ok"]} for r in res]
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


# ------------------------------------------------------------------------------------------- V5
def cr_positivity_controls(flat, kappa0):
    """V1 and V2 must reject a nonpositive sigma (sigma2), with the explicit reason, before anything else."""
    kappa0 = Fr(kappa0)
    out = {}
    # V2 on the small mesh n = 3 of E (alpha = 1/100); the record of the external review: sigma2 = -10^6, lb = 40
    base = {"key": [0, 3], "alpha": "1/100", "n": 3, "N": 18}
    h2 = crmesh.build_mesh(geom.tri_equilateral(Fr(1, 100)), "U", 3).h2_hi
    pole2 = -(1 + Fr(1, 10 ** 9)) / (kappa0 * kappa0 * h2)
    rows = []
    for s, lb in ((Fr(-10 ** 6), Fr(40)), (Fr(0), Fr(1, 1000)), (pole2, Fr(1000))):
        rr = dict(base, sigma2="%d/%d" % (s.numerator, s.denominator), lb="%d/%d" % (lb.numerator, lb.denominator))
        o = v2.verify_lam2(rr, kappa0)
        rows.append({"sigma2": float(s), "lb": float(lb), "rejected": not o["ok"], "reason": o.get("reason"),
                     "formula_value": float(s / (1 + kappa0 * kappa0 * h2 * s)) if s != 0 else 0.0})
    out["v2"] = rows
    # V1 on the smallest stored vertex record (mesh, eigenvector and lb unchanged; only sigma replaced)
    r = sorted(flat, key=lambda q: q["N"])[0]
    tri, alpha = v1.tri_from_node_record(r)
    h2v = crmesh.build_mesh(tri, r["kind"], r["n"]).h2_hi
    polev = -(1 + Fr(1, 10 ** 9)) / (kappa0 * kappa0 * h2v)
    rows = []
    for s, lb in ((Fr(-10 ** 6), Fr(1000)), (Fr(0), geom.parse_rat(r["lb"])), (polev, Fr(1000))):
        o = v1.verify_cr(tri, r["kind"], r["n"], s, lb, kappa0, r["N"])
        rows.append({"sigma": float(s), "lb": float(lb), "rejected": not o["ok"], "reason": o.get("reason")})
    out["v1"] = rows
    out["v1_record"] = [r["w"], r["k"], r["n"]]
    out["ok"] = bool(all(x["rejected"] and "not positive" in (x["reason"] or "") for x in out["v2"] + out["v1"])
                     and out["v2"][2]["formula_value"] > 1000)
    return out


def exact_sigma_control(kappa0):
    tri = geom.Triangle([iv.mpf(25), iv.mpf(25), iv.mpf(36)], [iv.mpf(6), iv.mpf(6), iv.mpf(5)], iv.mpf(12), "5-5-6")
    x = np.ones(3)
    run = lambda s: v1.verify_cr(tri, "U", 1, s, Fr(1, 10), Fr(kappa0), 3, x_override=x)
    above = run(Fr(15, 2) + Fr(1, 2 ** 61))
    below = run(Fr(15, 2) * (1 - Fr(1, 10 ** 9)))
    out = {"above_rejected": not above["ok"] and not above.get("cw_ok", True), "above_reason_cw": above.get("cw_ok") is False,
           "below_accepted": bool(below["ok"]), "sigma_cert_float": below.get("sigma_cert")}
    out["ok"] = bool(out["above_rejected"] and out["below_accepted"] and out["sigma_cert_float"] <= 7.5)
    return out


def run_controls(certdir, kappa0):
    rep = {}
    t0 = time.time()
    nodes, _ = v4.load_nodes(certdir)
    flat = [r for lst in nodes.values() for r in lst]
    ring, _ = read_jsonl(os.path.join(certdir, "ring_nodes.jsonl"))
    # --- V1 controls
    c1 = []
    picks = []
    smallU = [r for r in flat if r["kind"] == "U" and r["N"] < 3000]
    smallA = [r for r in flat if r["kind"] == "A" and r["N"] < 3000]
    midU = [r for r in flat if r["kind"] == "U" and 3000 <= r["N"] < 12000]
    for lst in (smallU[:1], smallU[len(smallU) // 2:len(smallU) // 2 + 1], smallA[:1], smallA[-1:], midU[-1:]):
        picks += [("node", r) for r in lst]
    if ring:
        picks.append(("ring", ring[0]))
        picks.append(("ring", ring[-1]))
    for src, r in picks:
        tri, alpha = (v1.tri_from_ring_record(r) if src == "ring" else v1.tri_from_node_record(r))
        sigma = geom.parse_rat(r["sigma"])
        lb = geom.parse_rat(r["lb"])
        base = v1.verify_cr(tri, r["kind"], r["n"], sigma, lb, kappa0, r["N"])
        bad_sigma = v1.verify_cr(tri, r["kind"], r["n"], sigma * (1 + Fr(1, 10 ** 4)), lb, kappa0, r["N"], sweeps=2400)
        bad_lb = v1.verify_cr(tri, r["kind"], r["n"], sigma, lb * (1 + Fr(1, 10 ** 15)), kappa0, r["N"])
        bad_k = v1.verify_cr(tri, r["kind"], r["n"], sigma, lb, kappa0 * Fr(101, 100), r["N"])
        bad_n = v1.verify_cr(tri, r["kind"], r["n"] - 1, sigma, lb, kappa0, None)
        c1.append({"src": src, "id": r.get("key") or [r["w"], r["k"], r["n"]], "kind": r["kind"],
                   "unmodified_passes": base["ok"],
                   "sigma_times_1.0001_rejected": (not bad_sigma["ok"]) and (not bad_sigma.get("cw_ok", False)),
                   "sigma_ctrl_margin": bad_sigma.get("cw_rel_margin"),
                   "lb_times_1+1e-15_rejected": (not bad_lb["ok"]) and (not bad_lb["lb_ok"]),
                   "kappa0_times_1.01_rejected": (not bad_k["ok"]) and (not bad_k["lb_ok"]),
                   "n_minus_1_rejected": not bad_n["ok"]})
    # mesh 'U' on an obtuse triangle must be rejected (positive off-diagonal entries)
    obt = None
    for r in flat:
        if r["kind"] == "A":
            w = geom.parse_w(r["w"])
            tri = geom.tri_cover(w, geom.parse_rat(r["alpha"]))
            a2 = tri.a2
            j = geom.certified_argmax(a2)
            if j is not None and (a2[j] - a2[(j + 1) % 3] - a2[(j + 2) % 3]).a > 0:
                obt = (r, tri)
                break
    if obt is not None:
        r, tri = obt
        o = v1.verify_cr(tri, "U", r["n"], geom.parse_rat(r["sigma"]) / 2, geom.parse_rat(r["lb"]) / 4, kappa0, None)
        rep["v1_obtuse_U_rejected"] = {"w": r["w"], "rejected": not o["ok"],
                                       "offdiag_flag": (o.get("flags") or {}).get("offdiag_nonpositive")}
    rep["v1"] = c1
    rep["v1_ok"] = bool(c1 and all(c["unmodified_passes"] and c["sigma_times_1.0001_rejected"] and c["lb_times_1+1e-15_rejected"]
                                   and c["kappa0_times_1.01_rejected"] and c["n_minus_1_rejected"] for c in c1)
                        and (obt is None or rep["v1_obtuse_U_rejected"]["rejected"]))
    # --- V2 control: sigma2 above the float second eigenvalue
    lam2r, _ = read_jsonl(os.path.join(certdir, "lam2_nodes.jsonl"))
    c2 = []
    for r in (lam2r[:1] + lam2r[-1:]):
        rr = dict(r)
        s2 = Fr(r["lam2_h"]) * (1 + Fr(1, 10 ** 4))
        rr["sigma2"] = "%d/%d" % (s2.numerator, s2.denominator)
        rr["lb"] = "1/1000"
        o = v2.verify_lam2(rr, kappa0)
        c2.append({"key": r["key"], "rejected": not o["ok"], "neg_pivots": o.get("neg_pivots")})
    rep["v2"] = c2
    rep["v2_ok"] = bool(c2 and all(c["rejected"] and (c["neg_pivots"] is None or c["neg_pivots"] >= 2) for c in c2))
    # --- hypothesis 0 < sigma of Lemma 'CR' (V1 and V2): sigma = -10^6, 0, and just below the pole -1/(kappa0 h)^2,
    # where sigma/(1 + (kappa0 h)^2 sigma) is large and positive and the inertia test (K - sigma M > 0) passes.
    rep["cr_sigma_positive"] = cr_positivity_controls(flat, kappa0)
    # --- exact sigma (V1): one-element mesh 'U' of the triangle with sides 5,5,6, Robin coefficients 6,6,5, x = (1,1,1):
    # mass 4I, stiffness 30I + graph Laplacian, discrete eigenvalue exactly 15/2
    rep["exact_sigma_556"] = exact_sigma_control(kappa0)
    # --- V3 controls
    mu, lam2 = load_local_inputs(certdir)
    if all((k, s) in mu for k in range(v3.KTOP + 1) for s in (1, -1)) and all(k in lam2 for k in range(0, v3.KTOP + 1, 4)):
        res, _ = v3.check_C(mu, lam2, PAPER_CL, PAPER_RHO, tstar=Fr(12, 100))
        nf = sum(1 for r in res if not r["ok"])
        rep["v3_tstar_0.12"] = {"failed_inequalities": nf, "of": len(res), "rejected": nf > 0}
        mu2 = dict(mu)
        for k in range(100, 140):
            for s in (1, -1):
                mu2[(k, s)] = mu[(k, s)] * Fr(998, 1000)
        res, _ = v3.check_C(mu2, lam2, PAPER_CL, PAPER_RHO)
        nf2 = sum(1 for r in res if not r["ok"])
        rep["v3_ring_bounds_lowered_0.2pct_k100_139"] = {"failed_inequalities": nf2, "rejected": nf2 > 0}
        rep["v3_ok"] = bool(nf > 0)
    else:
        rep["v3_ok"] = None
        rep["v3_note"] = "ring bounds incomplete; control skipped"
    # --- V4 controls
    roots = json.load(open(os.path.join(certdir, "coverB_roots.json")))
    cells = v4.load_cells(certdir)
    wcells = [cid for cid, cl in cells.items() if cl[0]["type"] == "W"]
    drops = []
    for i in range(len(roots)):
        r, _ = v4.check_cover(certdir, drop_root=i)
        drops.append(bool(r["a"]["ok"]))
    still_ok = [i for i, d in enumerate(drops) if d]
    rep["v4a_drop_one_root"] = {"tested": len(roots), "rejected": len(roots) - len(still_ok), "still_accepted": still_ok,
                                "note": "removing a root may legitimately pass only if all its sides lie in the four closed sets (e.g. the W cell)",
                                "W_cells": wcells}
    # a root whose removal is accepted must be disjoint from Omega_B; certify this independently
    def _disjoint(i):
        tri = tuple(geom.parse_w(v) for v in roots[i])
        return bool(v4.min_F_on_triangle(tri) >= 0 or all(v[1] <= 0 for v in tri)
                    or all(v4.ge_sqrt3(v[1], v[0]) for v in tri)
                    or all(v[0] ** 2 + v[1] ** 2 <= v4.RSTAR ** 2 for v in tri))
    rep["v4a_drop_one_root"]["still_accepted_certified_disjoint_from_Omega_B"] = [bool(_disjoint(i)) for i in still_ok]
    rep["v4a_drop_one_root"]["still_accepted_types"] = [cells["r%d" % i][0]["type"] for i in still_ok]
    rep["v4a_ok"] = bool(all(_disjoint(i) for i in still_ok) and len(still_ok) < len(roots))
    # flipped orientation of one root
    import copy
    tris = [tuple(geom.parse_w(v) for v in t) for t in roots]
    t0_ = tris[5]
    flipped = tris[:5] + [(t0_[0], t0_[2], t0_[1])] + tris[6:]
    okf, _b = v4.side_matching(flipped)
    rep["v4a_flipped_orientation_rejected"] = not okf
    r101, _ = v4.check_cover(certdir, lam0_scale=Fr(101, 100), only_ids=[0])
    r100, _ = v4.check_cover(certdir, only_ids=[0])
    rep["v4b_lambda0_times_1.01_cell_r0"] = {"unmodified_passes": r100["b"]["ok"], "rejected": not r101["b"]["ok"],
                                              "reason": r101["b"]["fails"][:1]}
    r1001, _ = v4.check_cover(certdir, lam0_scale=Fr(10001, 10000))
    rep["v4b_lambda0_times_1.0001_all_cells"] = {"cells_rejected": r1001["b"]["n_fails"]}
    # altered vertex bound -> no matching node record
    nodes_d, _ = v4.load_nodes(certdir)
    c0 = copy.deepcopy(cells["r0"][0])
    key0 = sorted(c0["nodes"].keys())[0]
    val = geom.parse_rat(c0["nodes"][key0][0]) * Fr(1005, 1000)
    c0["nodes"][key0][0] = "%d/%d" % (val.numerator, val.denominator)
    o = v4.check_B_cell(c0, nodes_d, v4.LamTab())
    rep["v4b_altered_bound_rejected"] = {"rejected": not o["ok"], "reason": o.get("reason")}
    # prism gap
    c0 = copy.deepcopy(cells["r0"][0])
    if len(c0["prisms"]) > 2:
        del c0["prisms"][1]
        o = v4.check_B_cell(c0, nodes_d, v4.LamTab())
        rep["v4b_prism_removed_rejected"] = {"rejected": not o["ok"], "reason": o.get("reason")}
    # negative vertex bounds at the closing node (b_i b_j > 0), with matching node records
    c0 = copy.deepcopy(cells["r0"][0])
    nodes_n = copy.deepcopy(nodes_d)
    for key in c0["nodes"]:
        i, k = map(int, key.split(":"))
        if k == c0["ktop"]:
            c0["nodes"][key][0] = "-10000*2^0"
            for rr in nodes_n[(c0["verts"][i], k, int(c0["nodes"][key][1]))]:
                rr["lb"] = "-10000*2^0"
                rr["sigma"] = "-1000000"
    o = v4.check_B_cell(c0, nodes_n, v4.LamTab())
    rep["v4b_negative_bounds_rejected"] = {"rejected": not o["ok"], "reason": o.get("reason")}
    rep["v4b_ok"] = bool(r100["b"]["ok"] and not r101["b"]["ok"] and rep["v4b_altered_bound_rejected"]["rejected"]
                         and rep["v4b_negative_bounds_rejected"]["rejected"]
                         and rep.get("v4b_prism_removed_rejected", {"rejected": True})["rejected"])
    # synthetic split cell (code path not exercised by the data)
    P = (Fr(0), Fr(0)), (Fr(1), Fr(0)), (Fr(0), Fr(1))
    mid = lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    m01, m12, m20 = mid(P[0], P[1]), mid(P[1], P[2]), mid(P[2], P[0])
    kids = [(P[0], m01, m20), (m01, P[1], m12), (m20, m12, P[2]), (m01, m12, m20)]
    rep["v4_split_selftest"] = {"good_split_accepted": v4.check_split(P, kids), "incomplete_split_rejected": not v4.check_split(P, kids[:3])}
    rep["ok"] = bool(rep["v1_ok"] and rep["v2_ok"] and rep["cr_sigma_positive"]["ok"] and rep["exact_sigma_556"]["ok"]
                     and rep["v3_ok"] in (True, None) and rep["v4a_ok"] and rep["v4b_ok"]
                     and rep["v4a_flipped_orientation_rejected"]
                     and rep["v4_split_selftest"]["good_split_accepted"] and rep["v4_split_selftest"]["incomplete_split_rejected"])
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


# ------------------------------------------------------------------------------------------- sanity
def run_sanity(certdir, n=20, seed=7):
    import p2fem
    t0 = time.time()
    nodes, _ = v4.load_nodes(certdir)
    flat = [r for lst in nodes.values() for r in lst]
    ring, _ = read_jsonl(os.path.join(certdir, "ring_nodes.jsonl"))
    rng = random.Random(seed)
    picks = [("node", r) for r in rng.sample(flat, min(len(flat), n - 6))]
    picks += [("ring", r) for r in rng.sample(ring, min(len(ring), 6))]
    rows = []
    for src, r in picks:
        tri, alpha = (v1.tri_from_ring_record(r) if src == "ring" else v1.tri_from_node_record(r))
        lb = float(geom.parse_rat(r["lb"]))
        ub = p2fem.p2_lambda1(tri, 48)
        ub2 = p2fem.p2_lambda1(tri, 24)
        rows.append({"src": src, "id": r.get("key") or [r["w"], r["k"], r["n"]], "alpha": float(alpha), "kind": r["kind"], "n": r["n"],
                     "lb": lb, "p2_upper_m48": ub, "p2_upper_m24": ub2, "rel_gap": (ub - lb) / ub, "below": lb < ub})
    # the equilateral triangle: P2 against the closed formula
    eq = []
    for a in (Fr(1, 100), Fr(2), Fr(500)):
        eq.append([float(a), p2fem.p2_lambda1(geom.tri_equilateral(a), 48), float(L0.lam0(a)[0].mid)])
    rep = {"rows": rows, "all_below": all(x["below"] for x in rows), "max_rel_gap": max(x["rel_gap"] for x in rows),
           "min_rel_gap": min(x["rel_gap"] for x in rows), "equilateral_p2_vs_formula": eq,
           "note": "floating point, not a proof step", "seconds": round(time.time() - t0, 1)}
    rep["ok"] = bool(rep["all_below"] and rep["max_rel_gap"] < 0.05)
    return rep


# ------------------------------------------------------------------------------------------- main
def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("certdir")
    ap.add_argument("kappa0")
    ap.add_argument("--v1", choices=["full", "sample", "none"], default="full")
    ap.add_argument("--all-nodes", action="store_true", help="V1 full: also node records not referenced by a B cell")
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--no-sanity", action="store_true")
    ap.add_argument("--sweeps", type=int, default=300)
    ap.add_argument("--fresh", action="store_true", help="ignore the V1 cache")
    ap.add_argument("--out", default=HERE, help="directory for the log, the report and the V1 cache")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    certdir = os.path.abspath(args.certdir)
    name = args.name or os.path.basename(certdir.rstrip("/"))
    kappa0 = Fr(args.kappa0)
    LOG = open(os.path.join(args.out, "log_%s.txt" % name), "w")
    cache_path = os.path.join(args.out, "cache_%s_v1.jsonl" % name)
    if args.fresh and os.path.exists(cache_path):
        os.remove(cache_path)
    T0 = time.time()
    log("=" * 100)
    log("verify_indep  certdir=%s  kappa0=%s  v1=%s  %s" % (os.path.basename(certdir), kappa0, args.v1, time.strftime("%Y-%m-%d %H:%M:%S")))
    report = {"certdir": os.path.basename(certdir), "kappa0": str(args.kappa0), "v1_mode": args.v1, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "mpmath_interval_prec_bits": geom.PREC}

    st = selftests()
    report["selftests"] = st
    log("selftests:", json.dumps(st))

    log("V4: cover ...")
    t = time.time()
    r4, used = v4.check_cover(certdir)
    r4["seconds"] = round(time.time() - t, 1)
    roots = json.load(open(os.path.join(certdir, "coverB_roots.json")))
    allv = set(v for tr in roots for v in tr)
    r4["distinct_root_vertices"] = len(allv)
    r4["denominators_divide_2^20"] = all((2 ** 20) % c.denominator == 0 for v in allv for c in geom.parse_w(v))
    cells_all = v4.load_cells(certdir)
    r4["B_cells_with_F_ge_0_on_whole_triangle"] = sorted(
        cid for cid, cl in cells_all.items() if cl[0]["type"] == "B"
        and v4.min_F_on_triangle(tuple(geom.parse_w(v) for v in cl[0]["verts"])) >= 0)
    nodes_all, _ = v4.load_nodes(certdir)
    usedrecs = [nodes_all[kk][0] for kk in used]
    r4["used_nodes_max_n"] = max(r_["n"] for r_ in usedrecs) if usedrecs else None
    r4["used_nodes_max_N"] = max(r_["N"] for r_ in usedrecs) if usedrecs else None
    r4["used_nodes_kinds"] = {kd: sum(1 for r_ in usedrecs if r_["kind"] == kd) for kd in ("U", "A")}
    r4["ktop_min"] = min(cl[0]["ktop"] for cl in cells_all.values() if cl[0]["type"] == "B")
    report["V4"] = r4
    log("V4a coverage:", json.dumps(r4["a"]))
    log("V4b/c cells:", json.dumps(r4["b"]), " [%.1f s]" % r4["seconds"])

    log("V2: second eigenvalue records ...")
    r2 = run_v2(certdir, kappa0)
    report["V2"] = r2
    log("V2:", json.dumps({k: v for k, v in r2.items()}))

    log("V3: local lemma ...")
    r3 = run_v3(certdir)
    report["V3"] = r3
    log("V3:", json.dumps({k: v for k, v in r3.items() if k != "per_interval"}))

    if args.v1 != "none":
        log("V1: Crouzeix-Raviart records (%s) ..." % args.v1)
        r1, _res = run_v1(certdir, kappa0, args.v1, used, args.all_nodes, cache_path, args.sweeps)
        report["V1"] = r1
        log("V1:", json.dumps(r1))
    else:
        report["V1"] = {"mode": "none", "ok": None}

    if not args.no_controls:
        log("V5: negative controls ...")
        r5 = run_controls(certdir, kappa0)
        report["V5"] = r5
        log("V5:", json.dumps(r5))
    if not args.no_sanity:
        log("sanity cross-check (P2 FEM, not a proof step) ...")
        rs = run_sanity(certdir)
        report["sanity_P2"] = rs
        log("sanity:", json.dumps({k: v for k, v in rs.items() if k != "rows"}))
        for x in rs["rows"]:
            log("   ", json.dumps(x))

    report["paper_statements_compared"] = {
        "note": "the paper text describes the profile kappa0 = 0.2983",
        "prisms (paper 1730)": r4["b"]["n_prisms"], "vertex bounds used (paper 1779)": r4["b"]["used_nodes"],
        "max n (paper 184)": r4["used_nodes_max_n"], "max unknowns (paper 51060)": r4["used_nodes_max_N"],
        "largest ktop (paper 286)": r4["b"]["ktop_max"], "alpha at largest ktop (paper ~2400)": r4["b"]["alpha_ktop_max"],
        "root triangles (paper 100; abstract says 91 cells)": r4["a"]["roots"],
        "cells with certificate type W (paper: one)": r4["b"]["leaves"]["W"],
        "boundary sides (paper 36)": r4["a"]["boundary_sides"],
        "min quotient in (C) (paper 1.17 at k=123)": [r3.get("min_quotient"), r3.get("min_quotient_k")],
        "quotient > 1.3 for alpha>=30 and alpha<=0.6": [r3.get("min_quotient_alpha_ge_30"), r3.get("min_quotient_alpha_le_0.6")],
        "shape constants (paper c_l=0.7361, rho=0.2938; true values ~0.742, ~0.288)": r3["shape_constants_own"],
        "ring records / lam2 records (paper 2*337 / 85)": [report["V1"].get("ring", {}).get("records_in_file"), r2["records_in_file"]],
        "distinct vertices of the roots": r4["distinct_root_vertices"]}
    report["summary"] = {"selftests": st["ok"], "V1": report["V1"].get("ok"), "V2": r2["ok"], "V3": r3["ok"], "V4": r4["ok"],
                         "V5": report.get("V5", {}).get("ok"), "sanity": report.get("sanity_P2", {}).get("ok")}
    report["total_seconds"] = round(time.time() - T0, 1)
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = os.path.join(args.out, "report_%s.json" % name)
    json.dump(report, open(out, "w"), indent=1, default=str)
    log("SUMMARY:", json.dumps(report["summary"]), " total %.0f s;  report -> %s" % (report["total_seconds"], out))
    allok = all(v for v in report["summary"].values() if v is not None)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())

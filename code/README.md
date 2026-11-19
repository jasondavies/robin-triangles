# Computational material

Run everything with `.venv/bin/python scripts/verify.py` from the repository root; it writes every output
to `../verification/` and compares it with `../claims.json`.

- `code/` (this directory): Part I of the paper (α < 0).
- `code/positive/`: Part II (α > 0), the certifier; certificates in `../data/positive/`.
- `code/verify_positive/`: the independent verifier of the Part II certificates.
- `code/prior_work/`: a check of Ma's proof of the case α < 0 (prior work, not part of our proof).

# Part I (α < 0)

Individual scripts are run from `code/`, and each writes one machine-readable JSON file to
`../verification/` (`out.py`). In those files, certified endpoints are exact `man*2^exp` numbers, which
`verify.py` compares with `../claims.json`.

Certified steps use Arb ball arithmetic (python-flint 0.9.0, 128 bits, set in `rig.py`). Floating point
is used only to generate trial data and in the cross-checks.

## Certified: together these make up the proof

| Script | What it certifies | Paper |
|---|---|---|
| `rig.py`, `equi.py`, `geom.py` | Library: intervals, second-order jets (`Jet2`); the equilateral data β(M), λ₀(M), g(M) and the enclosure of M(β); p and τ in apex coordinates | §2 |
| `thmA.py` | Theorem A (β ≤ 6/5): Lemma 3.1 (2000 intervals; q(6/5)) and Lemma 3.2 (256 Hessian sub-boxes, 344 boxes). Writes `thmA.json`. With `THMA_NEGATIVE=1` it runs the negative test G_A = 0.146, which must fail, and writes `thmA_negative.json` | §3 |
| `large_beta.py` | Lemma 4.8 (certC) in cell form: 6700 intervals of width 1/1000 on [13/10, 8] and one tail box, with geometry at t′ = tanh M_a. Writes `thmC.json`. `large_beta.py float` prints floating-point margins | §4.3 |

## Independent re-certification of Lemma 4.8 (an AI referee's code; shares no code with the above)

| Script | What it does |
|---|---|
| `indep_cert.py [MBIG]` | mpmath.iv at 150 digits. On the body [13/10, MBIG] (default 40) it uses the un-normalised closed forms from M, on adaptive cells. On the tail M ≥ MBIG it uses a sympy-derived normalisation with integer coefficients. The geometry comes from half- and double-angle formulas. Writes `thmC_independent.json` |
| `xcheck_tail.py` | The two forms of `indep_cert.py` agree to more than 100 digits at M = 2, 5, 10, 30 |

## Floating-point cross-checks (NOT rigorous, NOT part of the proof)

| Script | Check |
|---|---|
| `negative_c.py` | The certificate of Lemma 4.8 fails when started at M = 21/20 (the inequality is false there) and at M = 1 (cot θ > 1) |
| `table_c.py` | Table 1; the floating-point threshold M ≈ 1.103 (β ≈ 0.930); g(β) = 3/16 at β ≈ 1.63; (p−1)/(τ−2) → 3/16 near E |
| `three_float.py` | Library: closed forms of the eight integrals; the 3 × 3 matrices from the side ratios |
| `check_normalised.py` | The normalised polynomial forms used by `large_beta.py` against the closed forms |
| `check_quad.py` | Closed-form matrices against Gauss quadrature on actual scalene triangles |
| `validate_chain.py` | Every inequality of §4 on 400 random near-region triangles, with matrices from quadrature on T. Also the corner bound and its divergence identity |
| `direct_check.py`, `run_chain.py` | Referee's code: the pencil by direct quadrature, and the chain of inequalities on 3000 near-region triangles. Failures occur only at W < 10⁻¹⁵, below rounding level |
| `fem2.py`, `t_near_fem.py` | Referee's P2 FEM: FEM ≤ three-function Ritz value (up to the FEM error on E) < λ₀ on 16 near-region triangles |
| `fem.py`, `test_independent.py` | P1 FEM against λ₀; the PDE and boundary condition of ψ₀; B/N and 𝒢/N by quadrature; the six-wave tensor formula against quadrature on T |
| `check_misc.py` | The eight integrals reproduce g, B/N, λ₀ (Lemma 2.3); sampling of Lemma 3.2; the asymptotic constants of Remark 4.9 |

## Six-wave cell certificate on 6/5 ≤ β ≤ 3 (optional, §6.2)

These scripts are run by `scripts/cells.py` (`generate`, `replay`). The certificate directory is
`$CELLS_DIR` (default `../cells`, git-ignored).

| Script | Role |
|---|---|
| `dual.py`, `expint.py`, `sixwave.py`, `nums.py` | Arb library: dual numbers for mean-value forms in M; divided differences of exp with tail bounds; six-wave transplant tensors |
| `local_cert2.py`, `cover_local.py`, `near_cover.py` | (a) local Hessian boxes and (b) the near annulus: `local_L*.json`, `near_L*.json` |
| `far_cert.py`, `far_driver.py` | (c) far cells and (d) thin triangles: `far_*.json` |
| `verify_local_near.py`, `verify_far.py` | Replay: re-check (a), (b) and every stored far cell from its stored trial coefficients, and the contiguity of all intervals |
| `sw_float.py`, `local_float.py`, `ritz_mp.py`, `equi_float.py` | Floating-point (mpmath) generation of trial data. mpmath is pure Python, so this is deterministic |
| `check_far_consistency.py FILE` | Float check on random far cells: λ(48-wave Ritz) ≤ stored trial value < λ₀ |

## Notation

| Paper | Code |
|---|---|
| G₀ | `D3` |
| G_⊥, B_⊥, N_⊥ | `G1`, `B1`, `N1` |
| β B_⊥, β χ, \|λ₀\| N_⊥ | `bB1`, `bkq`, `lN1` |
| Σ₁, …, Σ₄ | `s1`, …, `s4` |

In the referee's scripts, χ is `kappa` and βχ is `bk`.

# Part II (α > 0)

The scripts in `code/positive/` are run from that directory. `RP_PROFILE=ls|pw|liu` selects the run
(interpolation constant κ₀ = 0.2983, 0.3496, 0.1893; certificates in `../data/positive/certs`,
`certs_pw`, `certs_liu`). `RP_OUT` is the directory for outputs (`scripts/verify.py` uses
`verification/positive/`), and `RP_CERTS` overrides the certificate directory, for a regeneration. The
certificate format is described in `../data/positive/FORMAT.txt`. Certified steps use Arb (100 bits; 80 in
`weak_cert.py`), exact integers and exact rationals; floating point only proposes eigenvectors, factors
and trial values.

## Certifier (`code/positive/`)

| Script | What it does | Paper |
|---|---|---|
| `config.py`, `shapes.py` | Profiles, the exact α-grid, t\_* = 3/50, r\_* = 0.1194; the Klein shape coordinates; λ₀ and λ₀′ of E in Arb | §§8, 11 |
| `rcr.py` | Crouzeix–Raviart meshes (U) and (A), the exact Collatz–Wielandt test on integers, the CR lower bound | §12 |
| `weak_cert.py` | Theorem "uniform weak coupling": 445 704 boxes in (t, φ) | §13 |
| `local_ring.py` | Generation: the 674 ring bounds and 85 bounds for λ₂(E) | §14 |
| `local_check.py` | Lemma "shapeconst" and the 337 inequalities (C) | §14 |
| `cover.py` | Generation: the vertex bounds and prisms of the 100 cells | §15 |
| `coverage_check.py` | Proposition "middle"(a): exact coverage check | §15 |
| `recheck.py cells` | All prism and closing inequalities from the stored bounds | §15 |
| `recheck.py sample N`, `recheck.py full` | From-scratch re-certification of N (or all) stored bounds | §17 |
| `negative_controls.py` | Perturbed statements must be rejected; P₂ sandwich (`p2_reference.py`, floating point) | §17 |
| `float_local.py` | Floating-point values of Remark "local" and the text (P₂ finite elements; not a proof step) | §14 |
| `summary.py` | Statistics of a run (Table 2) | §16 |

Regeneration of a run into a scratch directory `DIR` (about half an hour on one core for the default run):
`RP_CERTS=DIR python local_ring.py lam2; RP_CERTS=DIR python local_ring.py ring;
RP_CERTS=DIR python local_check.py - - DIR/local_cert.json; RP_CERTS=DIR python cover.py run 1 0`.
The floating-point eigenvectors are not stored (only their hashes), so a regeneration gives slightly
different but equally valid certificates.

## Independent verifier (`code/verify_positive/`)

Written by a separate AI instance from the text of Part II and `FORMAT.txt` only, without access to
`code/positive/`. `verify_indep.py CERTDIR KAPPA0 [--v1 full|sample|none] [--all-nodes] [--out DIR]`:
V1 re-certifies Crouzeix–Raviart records with its own meshes (from coordinates), eigenvectors and
outward-rounded interval arithmetic (`crmesh.py`, `eig.py`, `ivnp.py`, `v1.py`); V2 the λ₂ records by an
interval LDLᵀ factorisation and Sylvester's law of inertia (`v2.py`); V3 Lemma "shapeconst" and the
inequalities (C) in mpmath.iv (`v3.py`, `lam0.py`); V4 the coverage in exact rationals and all prism
inequalities (`v4.py`, `geom.py`); V5 negative controls; and a floating-point P₂ sanity check (`p2fem.py`).

# Check of prior work (`code/prior_work/`)

`ma_arb_check.py`, `ma_run_boxes.py`: an AI referee's re-implementation, in Arb, of the middle-range
criterion of Y. Ma (Zenodo, doi:10.5281/zenodo.23191831, Proposition 2), from the formulas of that
preprint; it confirms the criterion on 1 ≤ β ≤ 10 with 1146 boxes. Not part of our proof.

Dependencies are pinned in `../requirements.txt`. The scripts read no external data, apart from the
committed certificates of Part II and the cell certificates in the optional replay.

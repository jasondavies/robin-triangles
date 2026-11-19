# The equilateral triangle is the extremal triangle for the lowest Robin eigenvalue, for every boundary parameter

This repository holds a paper and its verification material. The paper proves both halves of the
conjecture of Lotoreichik and Lu (AIM problem list *Shape optimization with surface interactions*,
Conjecture 1.3).

Let λ₁(T, α) be the lowest eigenvalue of −Δ on a triangle T with ∂_ν u + α u = 0. Among triangles of
given area, the equilateral triangle is the unique minimiser of λ₁(T, α) for every α > 0 and the unique
maximiser for every α < 0. The same holds among triangles of given perimeter. The proof is
computer-assisted.

## Results and credit

- **Theorem 1.1(1), α > 0 (Part II).** This half is new. Before, the inequality was known for all
  triangles only when α√|T| is below an absolute but non-explicit constant (Ferudun, Zenodo 2026).
- **Theorem 1.1(2), α < 0 (Part I).** The first proof is due to Yicen Ma (Zenodo,
  [doi:10.5281/zenodo.23191831](https://doi.org/10.5281/zenodo.23191831), 6 October 2026).
  - Ma's proof is computer-assisted and unrefereed, and its code and certificate files are not in the
    deposit.
  - An AI referee re-implemented Ma's middle-range criterion in Arb and confirmed it on 1 ≤ β ≤ 10. That
    script is in `code/prior_work/`, labelled as a check of prior work.
  - We found Part I independently and learned of Ma's preprint only after it was complete.
  - Shared with Ma: the small-coupling trial functions and the transplanted ground-state identity, the
    exact corner bound λ₁(T, −β) < −β²/sin²(θ/2) (Ma's (7)–(8)), the same near/far threshold and the
    convexity bound for the strain.
  - Different: our near-equilateral step uses three transplanted corner waves, exact side-ratio
    identities and a tunnelling-gap estimate. One certified inequality covers all β ≥ 1.1531, and two
    trial functions cover β ≤ 6/5. Ma uses span{u, xu, yu} and Taylor bounds in the strain, certifies
    1 ≤ β ≤ 10 and argues analytically outside that range.
- **Corollary 1.2.** The same statements hold at fixed perimeter, for both signs of α.

## Part I: α < 0

Write β = −α√|T|.
- **Theorem 1.3** (β ≤ 6/5). The proof uses two trial functions: the transplanted ground state of E and
  the constant. It also needs a two-variable inequality, which is certified in Arb.
- **Theorem 1.4** (β ≥ β(13/10)). Here 1.1530164635 ≤ β(13/10) ≤ 1.1530164636. Two steps:
  - The corner bound λ₁(T, −β) < −β²/sin²(θ/2) holds for every angle θ and every β > 0. Its proof is an
    exact divergence identity. It settles every triangle with a small angle.
  - On the remaining near-equilateral triangles, three transplanted corner waves and a one-step
    perturbation estimate reduce the problem to a one-variable inequality. That inequality is
    certified in Arb for all M ≥ 13/10.
- Two independent checks are included. Neither is needed for the proof:
  - a second certificate of the one-variable inequality, written by an AI referee in mpmath interval
    arithmetic;
  - the earlier six-wave cell certificate on 6/5 ≤ β ≤ 3, which has 414 + 4140 M-intervals and
    109 717 far cells.

## Part II: α > 0

With |T| = 1, a shape is a point w of the Klein disc, and w = 0 is E. The proof covers four regions:
- **thin triangles** (longest side at least the perimeter of E), all α: slicing, by hand;
- **weak coupling** α ≤ 1/100, all shapes: an explicit vector field and a Schur-complement criterion,
  with an Arb box check (445 704 boxes);
- **near E** (|w| ≤ 0.1194), α ≥ 1/100: the Schur criterion needs an upper bound for the second-order
  perturbation constant R₁(α). It comes from certified *lower* bounds for two perturbed eigenvalue
  problems (674 ring bounds and 85 bounds for λ₂(E)). The rest is 337 one-variable inequalities.
- **middle region**, α ≥ 1/100: 100 cells. An exact cell identity in the Klein model and concavity
  reduce each prism (cell × α-interval; 1730 in all) to certified Crouzeix–Raviart lower bounds at 1779
  vertex nodes. Each bound is verified by an exact Collatz–Wielandt test on integers.

The Crouzeix–Raviart bounds need an interpolation constant.
- The default run uses κ₀ = 0.2983, which the paper proves from the Laugesen–Siudeja Poincaré
  inequality.
- The whole computation was also run with κ₀ = 0.3496 (Payne–Weinberger) and with κ₀ = 0.1893 (Liu,
  from a verified computation).
- Each run alone proves the theorem. The certificates of all three runs are committed in
  `data/positive/`.

An **independent verifier** (`code/verify_positive/`) checks the certificates. A separate AI instance
wrote it from the text of Part II and the certificate format only. It has its own meshes built from
coordinates and its own eigenvectors. It uses outward-rounded interval arithmetic, an interval LDLᵀ
inertia test, mpmath and exact rationals.

## Read

- [Manuscript](paper/robin-triangles.pdf) and its [LaTeX source](paper/robin-triangles.tex).
- [code/README.md](code/README.md): the scripts.
- [data/positive/FORMAT.txt](data/positive/FORMAT.txt): the certificate format of Part II.

## Reproduce

You need:
- Python 3.12 with the pins in `requirements.txt`: python-flint 0.9.0, numpy 2.5.5, scipy 1.18.1,
  mpmath 1.3.0, sympy 1.14.0. sympy 1.14 requires mpmath < 1.4.
- `latexmk`/`pdflatex`.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/verify.py                    # default: a few minutes on one core
.venv/bin/python scripts/verify.py --only certified|independent|checks|paper
.venv/bin/python scripts/verify.py --positive-full    # default + every Part II bound re-certified (~1.5 CPU hours)
.venv/bin/python scripts/cells.py generate -j 3       # optional: regenerate the six-wave cells (~4 CPU hours)
.venv/bin/python scripts/verify.py --cells            # default steps + replay of the cells (~25 more minutes)
```

`scripts/verify.py` works as follows:
- Each step writes a JSON file to `verification/` (for Part II, `verification/positive/`).
- Each gate compares that output with the paper's claims in [claims.json](claims.json), in exact
  rational arithmetic. Certified endpoints are exact `man*2^exp` numbers. A printed decimal must be the
  correct rounding of the computed value.
- The paper's numbers come from the same file, through the generated macros `paper/claims.tex`
  (`scripts/make_claims_tex.py`).
- Logs go to `verification/<step>.log`. A summary with input and PDF hashes goes to
  `verification/summary.json`, or to `summary-cells.json` or `summary-positive-full.json`.
- It exits nonzero on any failed gate, LaTeX warning or bad box. It uses one core (BLAS threads = 1).

The default run covers:
- **certified:**
  - Part I: Theorem A (`thmA.py`) and Lemma certC (`large_beta.py`);
  - Part II: `weak_cert.py`, `local_check.py` and `coverage_check.py` from scratch, and every prism and
    closing inequality from the stored bounds (`recheck.py cells`).
- **independent:**
  - Part I: the referee's mpmath.iv certificate of Lemma certC (`indep_cert.py`) and a cross-check of
    its two normalisations (`xcheck_tail.py`);
  - Part II: the independent verifier on the default run, and a replay of the two other runs. On the
    default run it checks the λ₂ records, the inequalities (C), the coverage, the prisms and its
    negative controls, and re-certifies a sample of the stored bounds from scratch.
- **checks:**
  - Part I: negative tests, the floating-point table and thresholds, the floating-point cross-checks,
    and consistency of the recorded cell summary;
  - Part II: negative controls, the floating-point P₂ values quoted in the text, and the statistics of
    the three runs (Table 2);
  - the check of Ma's criterion (prior work).
- **paper:** the claims macros and a clean LaTeX build.

`--positive-full` adds, for each of the three runs, `recheck.py full` and the independent verifier in
mode `--v1 full --all-nodes`. Together they re-certify every stored ring, λ₂ and vertex bound from
scratch with both programs.

**Six-wave cell certificate (Part I).** The cell files, about 100 MB of JSON, are not committed.
[data/cells_summary.json](data/cells_summary.json) records the generation parameters, the counts and
a content hash of every file of the run reported in the paper. The content hash is the SHA-256 of the
canonical JSON without the run-time field `seconds`.
- `scripts/cells.py generate [DIR]` regenerates the files into `DIR`; the default is `$CELLS_DIR` or
  `./cells`, which is git-ignored.
- `scripts/verify.py --cells [--cells-dir DIR]` then re-checks every stored cell in Arb from its stored
  trial coefficients, together with the contiguity of all intervals and the counts.
- The replay also reports whether the content hashes agree with the recorded run. With the pinned
  versions, the trial generation (pure-Python mpmath) is deterministic. A regeneration of the far piece
  G11 reproduced its hash. The replay does not depend on this.

## What is proved, certified, or only numerical

**Proved by hand in the paper:**
- Part I: the equilateral ground state, the Hellmann–Feynman formulas and the bound on g; the
  transplantation formulas; the corner bound (Lemma 4.1); the side-ratio identities, the reduction of
  the 3 × 3 pencil to eight integrals, and their closed forms; the perturbative criterion
  (Proposition 4.6); the near-region geometry; the assembly of Theorems 1.3 and 1.4.
- Part II: the equilateral ground state and λ₀′; the thin-triangle and weak-coupling theorems; the
  Schur-complement criterion; the cell identity in the Klein model and the prism corollary; the
  interpolation constant 0.2983, the Crouzeix–Raviart lower bound with Robin terms, and the
  Collatz–Wielandt and inertia lemmas; the symmetry reduction and the two-sided ring bound; the
  uniformity in α; the coverage lemma; the assembly of Theorem 1.1(1).
- Corollary 1.2 from Theorem 1.1.

**Certified by computer:**
- Part I, in Arb (128 bits):
  - Lemma 3.1, on 2000 intervals: g(β) < 0.1425 for β ≤ 6/5, and q(6/5) < 1.384;
  - Lemma 3.2: 256 Hessian sub-boxes and 344 boxes;
  - Lemma 4.8: 6700 intervals on [13/10, 8] and one tail box for M ≥ 8, with worst ratio ≤ 0.6013.
- Part II, in exact integers, exact rationals and Arb (100 bits):
  - 2660 Crouzeix–Raviart bounds, each by a Collatz–Wielandt test on integers;
  - the 337 inequalities (C), with smallest quotient 1.1758;
  - 1730 prism inequalities, with smallest relative margin 8·10⁻⁶;
  - the coverage, and the weak-coupling box check.

**Independent re-certification:**
- Part I: Lemma 4.8, by an AI referee in mpmath.iv: the body [13/10, 40] on 938 adaptive cells, worst
  ratio ≤ 0.6136, and the tail M ≥ 40 from a separately derived normalisation.
- Part II: every record of all three runs, by the independent verifier and by `recheck.py full`.

**Numerical only (floating point):**
- Part I: Table 1 and the thresholds M ≈ 1.103 and β ≈ 1.63; quadrature and FEM cross-checks;
  random-sample checks of the inequality chain.
- Part II: the P₂ finite element values in Remark 14.6 and in the text (R₁/a, the accuracy of the ring
  bound, the true gap in the middle region), and the P₂ sandwich of the certified bounds.

**Main residual risks:**
- Part I: the by-hand derivations, especially the closed forms of the eight integrals and their
  transcription into code. Two independently written scripts cross-check the closed forms against
  quadrature on actual triangles.
- Part II: next to E, the true gap λ₁(T)/λ₀ − 1 in the middle region is only 0.14–0.34 per cent (at the
  inner cell vertices, in floating point). Individual prism inequalities hold with relative margins
  down to 8·10⁻⁶. That region therefore rests entirely on the exactness of the Collatz–Wielandt test,
  of the constants of the Crouzeix–Raviart bound and of the ball arithmetic.
- Both: the correctness of Arb/python-flint, of mpmath's interval arithmetic and of Python's integers.

No independent human review has been recorded. All checks so far are by AI systems and by computer.

## Authorship and AI disclosure

Author: Jason Davies. AI systems produced the mathematics, the code and the manuscript: Claude Opus 5.5
and Claude Fable 5.2 (Anthropic), via Claude Code, under the author's direction. The author has not
read the manuscript or checked the arguments; see the declaration in the paper.

## License

Apache License 2.0; see [LICENSE](LICENSE). Third-party dependencies keep their own licences.

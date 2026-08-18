# Architecture

How `lgpsf-hessian` is put together, why the API looks the way it does, and how to
wire your own solver into it. Companion math documents: `glr-distributed-design.tex`
(the ScaLAPACK-backend design record), `glr-woodbury-formulas.tex` (the
replicated-backend recipe, step by step), and `deflation-correction-notes.tex`
(the stage-2.5 formulas as implemented).

## The problem shape

The intended user has an inverse problem with:

- a **parameter field** discretized on a mesh: `N` dofs with coordinates,
  a diagonal (lumped) mass matrix `M`, distributed over MPI ranks in contiguous
  global-id ranges;
- a **misfit Hessian** `H_misfit` available only through matvecs (each one a
  forward + adjoint PDE solve — expensive);
- a **prior / regularization** with the standard square-root structure
  `R = Z M⁻¹ Z` (bilaplacian-like: `Z` a shifted Laplacian).

The library approximates `H(c) = H_misfit + c·R` in two stages and operates with
it in a third. One build serves every shift `c`.

## The three objects

| object | stage | holds | needs |
|---|---|---|---|
| `lgh_fit_t` | 1: sparse LG-PSF fit | coords/mass/halo machinery, the fitted rows | MPI + lgpsf (+ Eigen) |
| `lgh_prior_t` | the user's prior, wrapped once | mass scalings, Z applies/solves (callbacks or an internally built CG+AMG solver with blocked Chebyshev tiers) | PETSc |
| `lgh_glr_t` | 2–3: eigendecomposition + operations | the treated spectrum, the (implicit or materialized) basis, references to `B` and the prior | PETSc (+ ScaLAPACK for the distributed backend) |

## Math ↔ function table

With `F = M^{1/2} Z⁻¹ B Z⁻ᵀ M^{1/2} ≈ U Λ Uᵀ` (never formed) and
`H(c) = B + c R = Z M^{-1/2} (F + cI) M^{-1/2} Z`:

| mathematics | function | frame |
|---|---|---|
| `B ≈ H_misfit` (sparse, weighted-symmetrized) | `lgh_fit_hessian` / `lgh_fit_probes` → `lgh_fit_get_mat` | — |
| `F ≈ U Λ Uᵀ` (randomized, hashed sketch) | `lgh_glr_compute` | — |
| grow the sketch, re-select | `lgh_glr_extend` | — |
| `x = H(c)⁻¹ rhs` | `lgh_glr_solve` | full space |
| `y = H(c) v` (exact: `Bv + cRv`) | `lgh_glr_apply` | full space |
| `x = G⁻ᵀ ξ`, `GGᵀ = H(c)` (draw with covariance `H(c)⁻¹`) | `lgh_glr_sample` | full space |
| `G, Gᵀ, G⁻¹, G⁻ᵀ` individually | `lgh_glr_factor` | full space |
| `Σ_kept [log(λ+c) − log c]` | `lgh_glr_logdet` | prior-weighted |
| `(F + cI)^p w`, `p ∈ {−1, ±1/2, +1}` | `lgh_glr_pow` | prior-weighted |
| `f(F + cI) w` (master formula) | `lgh_glr_filter` | prior-weighted |
| read treated `Λ` | `lgh_glr_eigs` | — |
| deflate `(H−H̃)v = d H̃ v`, graft into `(U, Λ)` | `lgh_glr_correct_probes` / `lgh_glr_correct` | — |

**Frame discipline.** Full-space operations act where your parameter lives (they
compose the outer `Z`, `M^{±1/2}` factors internally). Prior-weighted operations
act in the whitened frame of `F`; that is where the log-determinant and matrix
functions are well defined and mesh-independent.

## Stage 2.5: the deflation correction

The GLR object approximates the misfit Hessian twice over — lgpsf row-fit error
plus sketch truncation. `lgh_glr_correct` measures the dominant modes of that
error against the **true** Hessian and folds a low-rank correction into
`(U, Λ)`, after which every operation in the table above serves the corrected
operator unchanged. The mathematics, in the whitened frame:

- the generalized error EVP `(H − H̃)v = d·H̃v` at a reference shift `c0`
  becomes the ordinary symmetric problem for
  `E_w = S⁻(F_true + c0 I)S⁻ − I` with `S^± = (UΛUᵀ + c0 I)^{±1/2}` — and
  `S^±` is one filter apply, so none of the metric contortions of an
  operator-blind implementation are needed;
- the regularization **cancels** in `H − H̃`, so the stored correction
  approximates the c-independent absolute error: one corrected build still
  serves every shift, with the error *metric* (which modes count as dominant)
  anchored at `c0`;
- `eig(H̃⁻¹H) = 1 + d`: deflating the largest `|d|` is exactly the right
  metric for a preconditioner.

**Signs are information.** The correction's negative eigenvalues mean "the fit
came out too big in that direction" and are never flipped (FLIP exists to fix
fit artifacts at the stage where the operator is *supposed* to be SPD). Exact
generalized eigenvalues satisfy `1 + d > 0` automatically; the clamp at
`−1 + clamp_eps` guards only against estimation noise, so a nonzero
`report.clamped` is a budget-too-small diagnostic. After the graft the combined
spectrum may be signed: shift-taking operations require `c > report.floor`
(`= max(0, −min λ')`, with `floor < c0` guaranteed) instead of `c > 0`.

**Where the budget goes** (from the originating deflation study: error
*directions* are nearly free, error *values* are the binding constraint):
`lgh_glr_correct_probes` recombines the fit stage's probe pairs
`(Ω, H_d Ω)` into the whitened error basis with **zero** new Hessian applies,
then spends `opts.applies` true applies pricing the top directions exactly
(one power step past the basis when budget remains). The measured allocation
rule: fit to a moderate probe count, then spend the rest of the matvec budget
here. `lgh_glr_correct` is the probe-free fallback (a fresh hashed sketch of
`E_w`, ~2× the applies for the same rank).

Order matters: `compute → extend (as needed) → correct` (last). After a
correction the sketch no longer represents the operator, so `lgh_glr_extend`
is refused, `lgh_glr_apply` switches from the exact sparse action to the
corrected low-rank action, and ScaLAPACK-built objects materialize their
implicit `U` (the object dispatches by representation from then on).

## Design rationale

- **One API, not four.** The implementation history had a serial and a distributed
  fit bridge, and a replicated and a ScaLAPACK eigensolver engine. Those are not
  user concepts. The fit API is the SPMD one (it runs at every communicator size,
  including 1); the two engines are **backends of one `lgh_glr_t`**
  (`opts.backend`), with an identical operation surface and identical invariants
  (spectra, operator actions, logdets — eigenvector signs are grid-dependent, so
  cross-backend/cross-P comparisons use invariants only).
- **The prior is an object.** The GLR object holds the prior, so full-space
  operations come for free, and the whitening sandwich lives inside the library
  instead of in every consumer. `lgh_prior_create_mat` is the easy on-ramp (give
  it `Z` as a Mat); `lgh_prior_create_callbacks` is for applications that own a
  tuned `Z` solver. **Callbacks are RAW `Z` actions** — the library composes the
  mass scalings itself.
- **Sigma is caller-supplied.** The per-node kernel covariances are the one
  genuinely problem-specific input of the fit (in the originating ice-sheet
  application they came from a glaciological recipe, which stayed with that
  application). `lgh_sigma_isotropic` covers the simplest construction;
  `lgh_fit_smooth_field` provides the generic distributed smoothing mechanism.
- **`c` and the regularization scale stay explicit.** One eigendecomposition
  serves every shift; rescaling the regularization rescales the spectrum exactly
  (see the design record).
- **Determinism.** Hashed probes and hashed sketch columns are pure functions of
  `(seed, global index, column)`: fits and builds are partition-independent, and
  `lgh_glr_extend` is *exactly* the build you would have done with the wider
  sketch. Floating-point bit-identity holds within a build, not across
  compilers/flags (see `platform-notes.md`).

## Layering and dependencies

```
fit.h ── MPI only (PETSc-free by design)
prior.h, glr.h ── PETSc
impl/fit_impl.hpp ── MPI + lgpsf (+ Eigen); self-contained
impl/glr_common_impl.h ── PETSc + LAPACK  (kit + replicated backend + ops)
impl/glr_scalapack_impl.h ── + ScaLAPACK/BLACS   [LGH_WITH_SCALAPACK]
impl/zsolve_impl.h ── PETSc                (the prior's Mat path)
```

The GLR side never includes lgpsf — it consumes an arbitrary symmetric `Mat`
through `MatMult`, so it is usable with any sparse approximation, not only
LG-PSF fits. Consumers compile ONE C++ TU containing
`#include <lgpsf_hessian/impl.hpp>`; every other file includes
`lgpsf_hessian.h` (plain C is fine). The Eigen-heavy fit implementation is
included only when the `lgpsf_hessian::fit` CMake target is linked
(`LGH_HAVE_LGPSF`), so glr-only TUs stay light to compile.

## Integrating your own solver

The integration surface is deliberately small. What your code provides:

1. **A misfit-Hessian matvec** on local arrays,
   `void apply(const double *in, double *out, void *ctx)` — collective; typically
   wraps your forward+adjoint solve. (In the originating application this adapter
   is ~15 lines: copy local array into the application vector, run the Hessian,
   copy back.)
2. **Coordinates, lumped mass, sigma** for your owned dofs (contiguous global
   ids).
3. **The prior**: either `Z` as an assembled `Mat` (`lgh_prior_create_mat`), or
   raw `applyZ`/`solveZ` callbacks over your own solver
   (`lgh_prior_create_callbacks`; blocked variants optional, for BLAS-3 builds).

Then the quickstart in the README is the whole integration. Using
`lgh_glr_solve` as a Newton-CG preconditioner is a 3-line `PCShell`.

## Accuracy control

- **Stage 1**: one knob, `opts.qc_target` — the held-out energy-ratio QC,
  `sqrt(Σ|Bz−Hz|² / Σ|Hz|²)` over whitened probes, a stable estimate of the
  relative Frobenius error `|B−H|_F/|H|_F` in mass-whitened variables. The
  ladder adds probes (folding previous QC probes into the fit pool) until the
  target is met. Probe demand grows with dimension and `wedge_order`; if the fit
  comes back globally empty the library says so (error 6) with the row-level
  reason.
- **Stage 2**: `trunc_abs`/`trunc_rel` set the spectral cut; `report.next_abs`
  (the largest |λ| *below* the cut) is the evidence the rank converged — extend
  until it clears the cut with margin.

## Testing policy

- Engine gates run against **in-test dense-LAPACK oracles** on synthetic
  known-spectrum operators, at communicator sizes 1, 2, 4, for both backends.
- **Invariants only** across process counts, grids, and backends: eigenvalues,
  operator actions, log-determinants. Never compare eigenvector entries or
  sign-dependent quantities.
- Absolute costs and timings are comparable only within a fixed rank count.

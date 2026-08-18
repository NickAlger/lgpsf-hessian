/* glr.h — Stages 2 and 3: global low rank, and everything you do with it.
 *
 * Stage 2 computes a randomized eigendecomposition of the
 * prior-preconditioned approximation
 *
 *     F  =  M^{1/2} Z^{-1} B Z^{-1} M^{1/2}   ~=   U diag(lam) U^T,
 *
 * (F is never formed; only its action through B matvecs and Z solves), with
 * a deterministic, partition-independent hashed random sketch, optional
 * power passes, and eigenvalue treatment (FLIP by default: lam <- |lam|,
 * because large-magnitude negatives are genuine flipped-curvature modes —
 * clipping them to zero leaves outliers in the preconditioned spectrum).
 *
 * Stage 3 operates with H(c) = B + c R through the master formula
 *     f(F + c I)  =  U [ f(diag(lam) + c I) - f(c) I ] U^T  +  f(c) I,
 * where the truncated tail acts exactly like 0 — so log-determinants are
 * truncation-consistent, and sampling and logdet read the same treated
 * spectrum (detailed balance cannot be broken by truncation or FLIP, only
 * acceptance rates).
 *
 * COORDINATE FRAMES.  Every operation states its frame:
 *   full space      — vectors live where your parameter does
 *                     (solve / apply / sample / factor);
 *   prior-weighted  — vectors live in the whitened frame of F
 *                     (logdet / pow / filter).
 *
 * One build serves every shift c: pass c per call, never bake it in.
 *
 * Backends: REPLICATED (LAPACK only; eigensolve redundant on every rank —
 * simple and dependency-light, fine into the low tens of thousands of
 * sketch columns) and SCALAPACK (2D block-cyclic distributed eigensolve,
 * implicit U — the at-scale engine).  Identical API and identical
 * invariants (spectra, operator actions, logdets); eigenvector signs and
 * near-degenerate rotations may differ, so cross-backend and cross-P
 * comparisons must use invariants only.
 */

#ifndef LGPSF_HESSIAN_GLR_H
#define LGPSF_HESSIAN_GLR_H

#include <petscmat.h>
#include <petscvec.h>
#include "lgpsf_hessian/prior.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct lgh_glr lgh_glr_t; /* opaque */

typedef enum
{
  LGH_GLR_DEFAULT    = 0,  /* SCALAPACK if compiled in, else REPLICATED    */
  LGH_GLR_REPLICATED = 1,  /* LAPACK only                                  */
  LGH_GLR_SCALAPACK  = 2   /* requires LGH_WITH_SCALAPACK                  */
}
lgh_glr_backend_t;

typedef enum
{
  LGH_TREAT_FLIP = 0,      /* lam <- |lam|  (default; see header comment)  */
  LGH_TREAT_RELU = 1,      /* lam <- max(lam, 0)                           */
  LGH_TREAT_NONE = 2
}
lgh_treat_t;

typedef struct lgh_glr_opts
{
  int               ell;       /* sketch width (target rank <= ell)        */
  int               q_power;   /* power-iteration passes                   */
  double            trunc_abs; /* keep treated |lam| > trunc_abs           */
  double            trunc_rel; /* ... or > trunc_rel * max|lam| (0 = off)  */
  lgh_treat_t       treat;
  unsigned long     seed;      /* hashed-sketch seed (deterministic,       *
                                * partition-independent)                   */
  lgh_glr_backend_t backend;
  int               nb;        /* SCALAPACK block-cyclic block size        */
  int               panel;     /* orthogonalization panel width            */
  int               grid2d;    /* 0 (default) = 1 x P process grid — safe  *
                                * everywhere; 1 = 2D grid (faster, needs a *
                                * ScaLAPACK free of the zero-byte-type     *
                                * MPICH bug; see docs/platform-notes.md)   */
  int               check;     /* run internal self-check gates on build   */
}
lgh_glr_opts_t;

lgh_glr_opts_t lgh_glr_opts_default (void);

typedef struct lgh_glr_report
{
  int    kept;                  /* modes surviving treatment + truncation  */
  double lam_abs_max;           /* largest treated |lam|                   */
  double lam_raw_min, lam_raw_max;
  double next_abs;              /* smallest sketched |lam| NOT kept — your
                                   evidence the rank converged (compare to
                                   trunc_abs; extend if too close)         */
  int    n_negative_raw;        /* raw negative eigenvalues (FLIP count)   */
  double resid_max;             /* optional residual check (opts.check)    */
  double t_operator, t_dense;   /* seconds: F sweeps vs dense eigensolve   */
}
lgh_glr_report_t;

/* Build.  B: your sparse (or shell) symmetric approximation of the misfit
 * Hessian — the engine only ever calls MatMult on it.  The GLR object keeps
 * references to B and prior, and uses the prior's Z solves during the build
 * and its applies/solves in the full-space operations below.  Communicator
 * and row layout are taken from B.  report may be NULL. */
int lgh_glr_compute (Mat B, lgh_prior_t *prior, const lgh_glr_opts_t *opts,
                     lgh_glr_t **glr, lgh_glr_report_t *report);

/* Incremental rank growth: extend the sketch by k_new deterministic hashed
 * columns (continuing the same sequence — extension is exactly the build
 * you would have done at ell + k_new), re-eigensolve, re-select.  Turns the
 * unknown-rank gamble into a stopping criterion: grow until
 * report->next_abs clears trunc_abs with margin.
 * Not available after lgh_glr_correct (correct last; see below). */
int lgh_glr_extend (lgh_glr_t *glr, int k_new, lgh_glr_report_t *report);

void lgh_glr_destroy (lgh_glr_t *glr);

/* ---- full space: H(c) = B + c R --------------------------------------- */

/* x = H(c)^{-1} rhs — the Newton-CG preconditioner apply.
 * One Z solve on each side plus rank-sized arithmetic. */
int lgh_glr_solve (lgh_glr_t *glr, double c, Vec rhs, Vec x);

/* y = H(c) v, exact (B matvec + prior applies; no truncation error). */
int lgh_glr_apply (lgh_glr_t *glr, double c, Vec v, Vec y);

/* x = G^{-T} xi with G G^T = H(c):  for xi ~ N(0, I) coordinate-iid, x is
 * a draw with covariance H(c)^{-1}.  You supply xi (reproducibility is
 * yours to control). */
int lgh_glr_sample (lgh_glr_t *glr, double c, Vec xi, Vec x);

/* The individual factor actions, G = Z M^{-1/2} (F + c I)^{1/2}:
 * everything MCMC needs beyond lgh_glr_sample. */
typedef enum
{
  LGH_FACTOR_G     = 0,
  LGH_FACTOR_GT    = 1,
  LGH_FACTOR_GINV  = 2,
  LGH_FACTOR_GINVT = 3   /* == lgh_glr_sample */
}
lgh_factor_t;

int lgh_glr_factor (lgh_glr_t *glr, double c, lgh_factor_t which,
                    Vec in, Vec out);

/* ---- prior-weighted frame --------------------------------------------- */

/* sum over kept modes of [ log(lam + c) - log c ]:  the log-determinant of
 * H(c) relative to c R (the state-independent Z and M factors cancel
 * between builds).  Truncation-consistent: truncated modes contribute
 * exactly 0.  Local read after a successful build; returns the value. */
double lgh_glr_logdet (lgh_glr_t *glr, double c);

/* out = (F + c I)^p in,  p in {-1, -1/2, +1/2, +1}. */
int lgh_glr_pow (lgh_glr_t *glr, double c, double p, Vec in, Vec out);

/* General spectral function via the master formula: f_lam[i] = f(lam_i + c)
 * over the kept modes (order matching lgh_glr_eigs), f_c = f(c) for the
 * truncated tail. */
int lgh_glr_filter (lgh_glr_t *glr, const double *f_lam, double f_c,
                    Vec in, Vec out);

/* Read the treated, kept spectrum (|lam|-descending; replicated).  Borrowed
 * pointer, valid until extend/destroy.  Not collective. */
int lgh_glr_eigs (const lgh_glr_t *glr, int *nkept, const double **lam);

/* ---- stage 2.5: deflation correction against the TRUE Hessian --------- */

/* The GLR object approximates the misfit Hessian twice over (lgpsf row-fit
 * error + sketch truncation).  lgh_glr_correct measures the dominant modes
 * of that error against the TRUE misfit Hessian — the generalized EVP
 *
 *     (H - Htilde) v  =  d Htilde v,      Htilde(c0) evaluated at a
 *                                         reference shift c0 —
 *
 * and folds a rank-r correction into (U, lam).  Every downstream operation
 * then serves the corrected operator unchanged, at EVERY shift c: the
 * regularization cancels in H - Htilde, so the stored correction is
 * c-independent (c0 only sets the error metric — which modes count as
 * dominant).
 *
 * The correction's eigenvalues are SIGNED information (d < 0 means the fit
 * came out too big in that direction) and are never flipped.  Exact
 * generalized eigenvalues satisfy 1 + d > 0 automatically (both operators
 * PD at c0); the clamp d >= -1 + clamp_eps guards only against estimation
 * noise in the Rayleigh values — a nonzero report->clamped means the apply
 * budget was too small to price what it found, not that H is indefinite.
 * After correction the combined spectrum may contain (small) negative
 * lam': shift-taking operations then require c > report->floor
 * ( = max(0, -min lam'), with floor < c0 guaranteed) instead of c > 0.
 *
 * lgh_glr_apply switches from the exact sparse action (B v + c R v) to the
 * corrected low-rank action Z M^{-1/2}(U'lam'U'^T + cI)M^{-1/2}Z v — the
 * sparse B alone no longer represents the corrected operator. */

typedef struct lgh_glr_correct_opts
{
  double        c0;        /* reference shift for the error metric (your
                              working shift; default 1.0)                 */
  int           rank;      /* max correction rank; -1 (default) = all     */
  int           applies;   /* TRUE-Hessian apply budget (default 30) —
                              the binding cost: each apply is typically a
                              forward + adjoint PDE solve                 */
  int           n_qc;      /* probes path: the TRAILING n_qc supplied
                              probe pairs are held out for the qc_before/
                              qc_after report fields (default 0)          */
  int           q_power;   /* fresh path: extra power passes (default 0)  */
  double        clamp_eps; /* d clamped to >= -1 + clamp_eps (default .05)*/
  double        drop_tol;  /* relative basis drop tolerance (default 1e-12)*/
  unsigned long seed;      /* fresh-path hashed sketch seed               */
  int           verbose;
}
lgh_glr_correct_opts_t;

lgh_glr_correct_opts_t lgh_glr_correct_opts_default (void);

typedef struct lgh_glr_correct_report
{
  int    kept;             /* correction modes grafted                    */
  int    basis;            /* error-basis dimension available             */
  int    clamped;          /* Rayleigh values clamped at -1 + clamp_eps   */
  int    hd_applies;       /* true-Hessian applies spent                  */
  double d_min, d_max;     /* correction eigenvalues, post-clamp          */
  double floor;            /* shift validity: ops need c > floor          */
  double qc_before, qc_after;  /* whitened energy ratio on held-out
                                  probes (probes path, n_qc > 0); -1
                                  when unavailable                        */
  double t_operator, t_dense;
}
lgh_glr_correct_report_t;

/* The misfit-Hessian callback family shared with the fit stage: y = H x on
 * LOCAL arrays (this rank's owned dofs, ascending global order);
 * collective. */
#ifndef LGH_HESSIAN_FN_DEFINED
#define LGH_HESSIAN_FN_DEFINED
typedef void (*lgh_hessian_fn) (const double *in_local, double *out_local,
                                void *ctx);
#endif

/* Value-pass correction (recommended): reuse the k probe pairs the fit
 * stage already paid for — V and HV column-major as in lgh_fit_probes
 * (V[j*nloc + i] = entry i of probe j, HV the true-Hessian responses).
 * The error basis comes from this data for FREE (no new Hessian applies);
 * opts->applies are then spent pricing the top basis directions exactly
 * (with a power step past the basis when the budget allows).  Collective;
 * mutates glr in place. */
int lgh_glr_correct_probes (lgh_glr_t *glr, int k, const double *V,
                            const double *HV, lgh_hessian_fn hessian_apply,
                            void *ctx, const lgh_glr_correct_opts_t *opts,
                            lgh_glr_correct_report_t *report);

/* Fresh randomized correction (no probe data): a hashed Gaussian sketch of
 * the whitened error operator; costs ~2x the applies of the value pass for
 * the same rank.  Collective; mutates glr in place. */
int lgh_glr_correct (lgh_glr_t *glr, lgh_hessian_fn hessian_apply, void *ctx,
                     const lgh_glr_correct_opts_t *opts,
                     lgh_glr_correct_report_t *report);

/* ---- the seam from stage 1 -------------------------------------------- */

#ifdef LGPSF_HESSIAN_FIT_H
/* B from a successful fit as a symmetric MPIAIJ Mat on the fit's
 * communicator and layout.  The fit owns the Mat (valid until the next
 * fit/destroy; refit replaces it — holders such as lgh_glr_t keep their own
 * PETSc reference).  Declared here rather than fit.h so the fit stage stays
 * PETSc-free; visible when fit.h is included first (the umbrella header
 * does this). */
int lgh_fit_get_mat (lgh_fit_t *fit, Mat *B);
#endif

#ifdef __cplusplus
}
#endif

#endif /* LGPSF_HESSIAN_GLR_H */

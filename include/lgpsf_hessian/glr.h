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
 * report->next_abs clears trunc_abs with margin. */
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

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
  /* The sparse matvec with B inside F (GLR-BALANCE-DESIGN.md).  Both need an
   * AIJ B; a shell B keeps the column loop whatever is set here.          */
  int               balance;   /* 1: B's matvec runs in a contiguous layout *
                                * balanced by row weight nnz_i + row_cost   *
                                * (a view of B built once per build); the   *
                                * vectors move there and back in column     *
                                * tiles.  0 (default): B's own layout.      */
  int               balance_row_cost; /* per-row weight constant (1)        */
  int               matvec_block;     /* 1: one sparse x dense product per   *
                                       * tile (MatMatMult) instead of one    *
                                       * MatMult per column.  0 (default).   */
  int               matvec_tile;      /* columns per tile (256): bounds the  *
                                       * balanced layout's scratch and       *
                                       * PETSc's ghost buffer                */
  /* Incremental extension (lgh_glr_extend; SCALAPACK backend -- the
   * replicated backend rebuilds at the wider ell and ignores both).       */
  int               extend_deflate;   /* 1: the new block's power passes    *
                                       * run on the complement of the       *
                                       * existing basis -- each pass's F X  *
                                       * is projected against Q_old (twice) *
                                       * before its orthonormalization.     *
                                       * 0 (default): internal orth only.   *
                                       * See lgh_glr_extend.                */
  int               check_extend;     /* 1: after an extension, measure     *
                                       * max |Q_old^T Q_new| (report->      *
                                       * ext_orth_max; one more projection  *
                                       * sweep's GEMMs).  0 (default)       */
  int               diag_orth;        /* 1: orthogonality diagnostics, SCA- *
                                       * LAPACK backend, printed ([GLR-DIAG])*
                                       * on the communicator: after every   *
                                       * BCGS2 call the block it made       *
                                       * against itself and against the     *
                                       * columns before it (streamed in     *
                                       * panels) and the CholQR jitter      *
                                       * events; after every pdsyevd the    *
                                       * eigenvector columns' norms and     *
                                       * |V^T V x - x| on random x.  Costs  *
                                       * about one more BCGS2 sweep per     *
                                       * call.  0 (default).  2026-10-16:   *
                                       * the gamma 1e-3 kept-basis defect.  */
}
lgh_glr_opts_t;

lgh_glr_opts_t lgh_glr_opts_default (void);

/* Build-time breakdown, for strong-scaling diagnosis.  Each region is timed
 * on every rank; the report carries the max and the mean over ranks
 * (max / mean = load imbalance).  Collectives have regions of their own, so
 * time spent waiting at one is charged there, not to the local work before
 * it.  The top-level regions (LGH_GT_SETUP .. LGH_GT_CHECK) partition the
 * build; the LGH_GT_ZS_* rows split the two solve rows further (CHOLMOD
 * priors only; zero otherwise).  Each region is also a PETSc log event
 * ("LghGlr..."), so -log_view cross-checks it with flop, message and
 * message-length counts.  The ScaLAPACK backend fills every row; the
 * replicated backend fills the F rows only.                              */
typedef enum
{
  LGH_GT_SETUP = 0,  /* BLACS grid, descriptors, tall-block allocation     */
  LGH_GT_OMEGA,      /* hashed random fill                                 */
  LGH_GT_F_SCALE,    /* copies + M^{1/2} scalings inside F                 */
  LGH_GT_ZT_SOLVE,   /* Z^{-T} block solves                                */
  LGH_GT_MATVEC,     /* column-wise MatMult with B                         */
  LGH_GT_Z_SOLVE,    /* Z^{-1} block solves                                */
  LGH_GT_ORTH_LOCAL, /* BCGS2 local GEMM / POTRF / TRSM                    */
  LGH_GT_ORTH_RED,   /* BCGS2 allreduces                                   */
  LGH_GT_COPY,       /* Y -> Q copies; Q slab growth on extension          */
  LGH_GT_T_GEMM,     /* Q^T Y panels; extension border Q_all^T F Q_new     */
  LGH_GT_T_RED,      /* their allreduces; extension T rebuild allreduces   */
  LGH_GT_T_SCATTER,  /* block-cyclic scatter + symmetrize; column store    */
  LGH_GT_EIG_WORK,   /* pdsyevd workspace query + allocation               */
  LGH_GT_EIG,        /* pdsyevd                                            */
  LGH_GT_SELECT,     /* sort, truncation, treatment                        */
  LGH_GT_BAL_SETUP,  /* balanced / blocked matvec setup: row weights,      *
                      * gather, split, the balanced view of B, scratch     */
  LGH_GT_BAL_EXCH,   /* moving column tiles into the matvec layout and back*
                      * (a local copy when only blocking is on)            */
  LGH_GT_CHECK,      /* opts.check gates (keep off in production)          */
  LGH_GT_ZS_EXCH,    /* of the solve rows: Alltoallv redistribution        */
  LGH_GT_ZS_COPY,    /*   pack / unpack copies                             */
  LGH_GT_ZS_LOCAL,   /*   local permuted triangular solves                 */
  LGH_GT_N
}
lgh_glr_timer_t;
#define LGH_GT_NTOP (LGH_GT_CHECK + 1)   /* the partitioning rows */

/* "setup", "omega", ... (stable keys for logs); NULL out of range. */
const char *lgh_glr_timer_name (int k);

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
  /* breakdown of this build / extension (see lgh_glr_timer_t) */
  double t_max[LGH_GT_N], t_mean[LGH_GT_N];   /* seconds over ranks        */
  double bytes_max[LGH_GT_N];   /* payload this rank put into the region's
                                   collectives, max over ranks             */
  double calls_max[LGH_GT_N];   /* collective calls (MatMults for MATVEC,
                                   columns for OMEGA), max over ranks      */
  double t_total_max, t_total_mean;            /* the whole build / step   */
  double mem_mb_range, mem_mb_T, mem_mb_eig;   /* peak RSS (MB, max over
                                   ranks) after the range finder, after T,
                                   after the eigensolve; 0 if unavailable  */
  double nloc_max, nloc_mean;   /* rows per rank                           */
  double nnzB_max, nnzB_mean;   /* local nnz of B per rank (-1: not AIJ)   */
  /* the matvec's layout (opts.balance / opts.matvec_block) */
  int    balanced, blocked;     /* what ran (0 when not applicable)        */
  double nnzBal_max, nnzBal_mean;/* nnz per rank in the matvec's layout    */
  double rowsBal_max;           /* rows per rank there, max                */
  double bal_predicted;         /* the split's predicted weight max/mean   */
  int    bal_floor;             /* 1: the longest row set the split's cap  */
  /* an extension's orthogonality (opts.check_extend; -1 when not measured):
   * max |Q_old^T Q_new| over the old x new block after the new columns'
   * orthogonalization -- O(eps) when the new columns are well separated
   * from the old span; larger when they were numerically inside it      */
  double ext_orth_max;
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
 * columns (continuing the same sequence), re-eigensolve, re-select.  Turns
 * the unknown-rank gamble into a stopping criterion: grow until
 * report->next_abs clears trunc_abs with margin.
 *
 * The new block's power passes (opts.q_power) and the old basis Q_old:
 *   opts.extend_deflate = 0 (default): internal orthonormalization only; in
 *     exact arithmetic exactly the one-shot build at ell + k_new.
 *   opts.extend_deflate = 1: each pass orthonormalizes (I - Q_old Q_old^T) F X,
 *     so the new columns are drawn from F's action on the complement of
 *     span(Q_old) -- a block-Krylov extension, NOT span-equivalent to the
 *     one-shot build.  Motivated by a laptop model in which the
 *     span-equivalent extension left late-rung columns numerically inside
 *     span(Q_old) (|Q_old^T Q_new| 2.3e-9 vs 3.6e-15 one-shot, CHOLMOD prior
 *     with delta 1e-8, lam_max 1.78e8).  At production scale (albmap_40,
 *     gamma 1e-3, 48k columns, 2026-10-16) the old x new block measured
 *     1e-16 with either setting, and the kept basis's 2e-2 defect seen there
 *     is present with either setting: this option is not its cure, hence
 *     OFF by default (on 2026-10-15 / 6182183 it was briefly the default).
 * opts.check_extend = 1 measures the old x new block (report->ext_orth_max).
 * The replicated backend rebuilds at ell + k_new (both options moot).
 * Not available after lgh_glr_correct (correct last; see below). */
int lgh_glr_extend (lgh_glr_t *glr, int k_new, lgh_glr_report_t *report);

/* Adaptive sketch: build at ell0, then extend until the OVERSAMPLING
 * condition kept <= rho * ell holds (the sketch is at least 1/rho wider
 * than the rank at the cut: with q = 1 the last Ritz values of a randomized
 * sketch are biased low, so "next_abs < trunc_abs" alone can stop early),
 * or ell reaches ell_max (the build is then RANK-CAPPED).  The next width is
 * min(growth * ell, ell_max) -- a FIXED LADDER, data-independent apart from
 * the rung it stops at (fixed_ladder = 1, the default since 2026-10-12: the
 * memory of every rung is then known in advance).  With fixed_ladder = 0 the
 * older rule applies: ell grows by `growth` only while saturated (kept == ell)
 * and otherwise jumps to ceil(1.1 * kept / rho).  Each step is lgh_glr_extend
 * (with opts.extend_deflate = 0, the default, the result is the one-shot build
 * at the final ell, span-equivalent; with 1 each rung is a deflated
 * block-Krylov extension of the previous one).
 * opts->ell is ignored.  on_step (optional) is called after every
 * build/extension on every rank. */
typedef void (*lgh_glr_adapt_step_fn) (int step, int ell,
                                       const lgh_glr_report_t *rep, void *ctx);
typedef struct lgh_glr_adapt
{
  double                rho;      /* stop once kept <= rho * ell (0.7)      */
  int                   ell_max;  /* ceiling; capped beyond it (30000)      */
  int                   ell0;     /* first width (4000) of EVERY build;     *
                                   * warm-starting later builds from the    *
                                   * previous kept rank made them one-shot  *
                                   * builds at the ceiling -- not advised   */
  double                growth;   /* width factor per rung (2.0)            */
  int                   fixed_ladder; /* 1 (default): next = growth * ell;  *
                                   * 0: the pre-2026-10-12 kept-based jump  */
  lgh_glr_adapt_step_fn on_step;  /* optional per-step report               */
  void                 *ctx;
}
lgh_glr_adapt_t;

lgh_glr_adapt_t lgh_glr_adapt_default (void);

typedef struct lgh_glr_adapt_report
{
  int    ell_final;               /* sketch width at the end                */
  int    steps;                   /* 1 build + (steps - 1) extensions       */
  int    capped;                  /* 1: stopped at ell_max, not by rho      */
  double t_operator, t_dense;     /* seconds, summed over the steps         */
  double t_max[LGH_GT_N], t_mean[LGH_GT_N];  /* per-step max / mean over
                                     ranks, summed over the steps           */
  double t_total_max, t_total_mean;          /* likewise                    */
}
lgh_glr_adapt_report_t;

int lgh_glr_compute_adaptive (Mat B, lgh_prior_t *prior,
                              const lgh_glr_opts_t *opts,
                              const lgh_glr_adapt_t *adapt, lgh_glr_t **glr,
                              lgh_glr_report_t *report,
                              lgh_glr_adapt_report_t *areport);

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

/* ---- diagnostics: is the applied operator what the formulas say? -------
 * (2026-10-14: the albmap_40 gamma 1e-3 run's DIVERGED_INDEFINITE_PC exits at
 * the 48,000-column ceiling -- lgpsf-antarctica/H1EXP-HANDOFF.md.)  Three
 * measurements at a shift c, all collective:
 *  (1) orthonormality of the kept basis U: the first ncols columns in
 *      |lam|-descending order (ncols < 0: all) against ALL kept columns, in
 *      panels through the block primitives.  The applied filter f_c I +
 *      U diag(f - f_c) U^T is positive in a kept direction only while
 *      |U^T U - I| stays below c / lam there;
 *  (2) the applied preconditioner H(c)^{-1} = Z^{-T} M^{1/2} (F + cI)^{-1}
 *      M^{1/2} Z^{-1} along the n_top stiffest and the n_low lowest kept
 *      directions: with r_j = Z M^{-1/2} u_j the form r_j^T H(c)^{-1} r_j is
 *      exactly 1/(lam_j + c); dev_full = q (lam_j + c) - 1 through the full
 *      apply (lgh_glr_solve: both Z solves) and dev_white through the
 *      whitened filter alone.  The full probes need the prior's applyZ
 *      callback (n_dir_full = 0 without it);
 *  (3) n_rand hashed-Gaussian vectors r: q1 = r^T H(c)^{-1} r (CG's beta)
 *      against q2 = w^T (F + cI)^{-1} w, w = M^{1/2} Z^{-1} r (the Z-transpose
 *      consistency), the sign of q1, and the symmetry |r_a^T H^{-1} r_b -
 *      r_b^T H^{-1} r_a| over consecutive pairs.
 * Cost: (1) two passes over the basis per 64 columns (every column: ~4 N
 * kept^2 flops, minutes on 24 nodes at kept 44,000); (2) + (3) a few solves. */
typedef struct lgh_glr_basis_report
{
  int    kept;
  double c;
  int    orth_cols;             /* columns checked (the stiffest first)    */
  double orth_diag_max;         /* max_j |u_j^T u_j - 1|                   */
  int    orth_diag_arg;
  double orth_off_max;          /* max_{i != j} |u_i^T u_j|                */
  int    orth_off_arg_i, orth_off_arg_j;
  double orth_top_max;          /* both maxima over the n_top stiffest     */
  int    n_dir, n_dir_full;     /* directions probed; through the full apply */
  double dir_dev_white_max;     /* max |q (lam + c) - 1|, whitened filter  */
  double dir_dev_full_max;      /* the same through the full apply         */
  double dir_q_full_min;        /* min q (lam + c), full apply (< 0: not positive there) */
  int    dir_arg_full;          /* the direction of dir_dev_full_max       */
  int    n_rand;
  double rand_rel_max;          /* max |q1 - q2| / |q2|                    */
  double rand_q1_min;           /* min q1 / (r^T r)  (< 0: not positive)   */
  int    n_sym;
  double sym_max;               /* max relative asymmetry                  */
  double t_orth, t_probe;       /* seconds                                 */
}
lgh_glr_basis_report_t;

int lgh_glr_check_basis (lgh_glr_t *glr, double c, int ncols, int n_top,
                         int n_low, int n_rand, lgh_glr_basis_report_t *rep);

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
  double        d_keep_min;/* drop priced modes with |d| below this
                              (default 0 = keep all).  Rayleigh values at
                              the estimation-noise level add noise rather
                              than information; when report->d_min/d_max
                              hug zero there is nothing worth deflating.
                              If NO priced mode clears the threshold the
                              object is left UNCORRECTED (kept = 0; still
                              extendable/correctable).                    */
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

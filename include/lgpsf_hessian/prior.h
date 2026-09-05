/* prior.h — the user's prior / regularization, wrapped once.
 *
 * The library assumes the standard square-root structure for the prior
 * precision (bilaplacian-like, as in hIPPYlib and the ice-sheet inversion
 * literature):
 *
 *     R = Z M^{-1} Z,
 *
 * where M is the diagonal (lumped) mass matrix and Z is a symmetric
 * "square-root factor" — typically a shifted Laplacian (stiffness + mass)
 * at a reference regularization scale.  Keep any regularization weight
 * OUTSIDE Z: downstream operations take the shift c explicitly
 * (H(c) = B + c R), so one GLR build serves every shift, and rescaling the
 * regularization is free.
 *
 * Two ways in:
 *   lgh_prior_create_mat        — give Z as an assembled Mat; the library
 *                                 builds the solver machinery internally
 *                                 (spectral bounds + blocked Chebyshev
 *                                 around a multigrid V-cycle).  The easy
 *                                 on-ramp, and the right choice unless you
 *                                 already own a tuned Z solver.
 *   lgh_prior_create_callbacks  — bring your own applies/solves.
 */

#ifndef LGPSF_HESSIAN_PRIOR_H
#define LGPSF_HESSIAN_PRIOR_H

#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct lgh_prior lgh_prior_t; /* opaque */

/* y = Op x. */
typedef void (*lgh_vec_fn) (Vec x, Vec y, void *ctx);
/* Y = Op X on MATDENSE column blocks (the engines sweep in blocks; a
 * blocked solve is what keeps the build BLAS-3).  Optional: when absent the
 * library wraps the Vec-wise callback by looping columns. */
typedef void (*lgh_mat_fn) (Mat X, Mat Y, void *ctx);

typedef struct lgh_prior_callbacks
{
  lgh_vec_fn applyZ;           /* required                                 */
  lgh_vec_fn applyZt;          /* NULL => Z treated as symmetric           */
  lgh_vec_fn solveZ;           /* required                                 */
  lgh_vec_fn solveZt;          /* NULL => symmetric                        */
  lgh_mat_fn solveZ_blocked;   /* optional, performance                    */
  lgh_mat_fn solveZt_blocked;  /* optional                                 */
  void      *ctx;
}
lgh_prior_callbacks_t;

/* mass_lumps: diagonal of M, same layout as the operator.  The prior keeps
 * a reference (PETSc refcount).  Communicator is taken from mass_lumps. */
int lgh_prior_create_callbacks (Vec mass_lumps,
                                const lgh_prior_callbacks_t *callbacks,
                                lgh_prior_t **prior);

/* Solver tuning for the Mat path.  The library creates a CG + AMG solver
 * for Z internally (PETSc options prefix "lgh_prior_" for overrides).
 * blocked_mode selects how the engine's BLOCK solves are performed:
 *   0 (default) — per-column KSP solves: robust, no setup cost;
 *   2 — blocked fixed-count Chebyshev with the AMG preconditioner applied
 *       through PCMatApply (BLAS-3 across the block);
 *   3 — blocked Chebyshev around a block V-cycle on the harvested AMG
 *       hierarchy (the at-scale configuration: all smoothing is blocked).
 * Modes 2/3 estimate the preconditioned spectral interval once at create
 * time (deterministic probe) and derive the fixed iteration count from
 * cheb_rtol; single-vector solves always use the KSP path. */
typedef struct lgh_prior_mat_opts
{
  double ksp_rtol;     /* per-column KSP solve tolerance (default 1e-12) */
  int    blocked_mode; /* 0 (default) | 2 | 3, see above                 */
  int    tile;         /* block width for modes 2/3 (default 64)         */
  double cheb_rtol;    /* Chebyshev accuracy target (default 1e-10)      */
  /* mode-3 hierarchy harvest tuning */
  int    smax_krylov;  /* 1 (default): per-level smoother bound via
                          CG-Lanczos; 0: power iteration                 */
  double smoother_top; /* smoothing-interval top factor (default 1.1)    */
  int    nu_force;     /* force smoother degree (0 = take the level's)   */
  /* mode-3 hierarchy SOURCE (2026-09-04).  The blocked V-cycle needs only
   * the level operators and prolongators; where they come from is a
   * choice.  LGH_HIERARCHY_PCMG harvests them from the KSP's PCMG/GAMG
   * preconditioner (the original path).  LGH_HIERARCHY_HYPRE builds a
   * BoomerAMG hierarchy with hypre at setup (classical coarsening +
   * interpolation, no smoother of hypre's is ever applied), reads the
   * levels back, and discards the hypre solver; available when PETSc was
   * configured with hypre (PETSC_HAVE_HYPRE).  LGH_HIERARCHY_AUTO (default)
   * picks hypre when it is available and PCMG otherwise.  Measured on the
   * ice-sheet prior (-Delta + 1e-5 M): GAMG's default aggressive coarsening
   * leaves ~9% of the modes below 0.7 in the preconditioned spectrum,
   * hypre's hierarchy leaves none ([0.64, 1] with our Chebyshev(2)). */
  int    hierarchy;        /* LGH_HIERARCHY_AUTO | _PCMG | _HYPRE           */
  int    hypre_coarsen;    /* BoomerAMG coarsening type (10 = HMIS)         */
  int    hypre_interp;     /* BoomerAMG interpolation (6 = extended+i)      */
  double hypre_strong;     /* strength threshold (0.25)                     */
  int    hypre_agg_nl;     /* aggressive-coarsening levels (0: keep it 0)   */
  int    hypre_max_coarse; /* coarsest-level size cap (200; dense LU below) */
  int    verbose;
}
lgh_prior_mat_opts_t;

#define LGH_HIERARCHY_AUTO  0
#define LGH_HIERARCHY_PCMG  1
#define LGH_HIERARCHY_HYPRE 2

/* 1 when the hypre hierarchy source is compiled in (PETSC_HAVE_HYPRE). */
int lgh_have_hypre (void);

lgh_prior_mat_opts_t lgh_prior_mat_opts_default (void);

/* Z: assembled symmetric Mat (kept by reference).  opts NULL = defaults. */
int lgh_prior_create_mat (Mat Z, Vec mass_lumps,
                          const lgh_prior_mat_opts_t *opts,
                          lgh_prior_t **prior);

/* Bring your own ALREADY-CONFIGURED solver for Z: ksp is borrowed
 * (refcounted) and never mutated — its operator supplies applyZ, its
 * solves supply solveZ, and the blocked tiers of opts wrap it exactly as
 * in the Mat path.  For applications with a production-tuned Z solver
 * that the library should reuse rather than replace. */
int lgh_prior_create_ksp (KSP ksp, Vec mass_lumps,
                          const lgh_prior_mat_opts_t *opts,
                          lgh_prior_t **prior);

void lgh_prior_destroy (lgh_prior_t *prior);

/* y = R x = Z M^{-1} Z x.  (Completeness / testing; cheap.) */
int lgh_prior_apply (lgh_prior_t *prior, Vec x, Vec y);

#ifdef __cplusplus
}
#endif

#endif /* LGPSF_HESSIAN_PRIOR_H */

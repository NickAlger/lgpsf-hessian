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

/* Solver tuning for the Mat path.  Defaults reproduce the validated
 * production configuration.  (Field set to be finalized when the solver
 * machinery lands — kept small deliberately.) */
typedef struct lgh_prior_mat_opts
{
  double cheb_rtol;   /* Chebyshev solve tolerance                        */
  int    verbose;
}
lgh_prior_mat_opts_t;

lgh_prior_mat_opts_t lgh_prior_mat_opts_default (void);

/* Z: assembled symmetric Mat (kept by reference).  opts NULL = defaults. */
int lgh_prior_create_mat (Mat Z, Vec mass_lumps,
                          const lgh_prior_mat_opts_t *opts,
                          lgh_prior_t **prior);

void lgh_prior_destroy (lgh_prior_t *prior);

/* y = R x = Z M^{-1} Z x.  (Completeness / testing; cheap.) */
int lgh_prior_apply (lgh_prior_t *prior, Vec x, Vec y);

#ifdef __cplusplus
}
#endif

#endif /* LGPSF_HESSIAN_PRIOR_H */

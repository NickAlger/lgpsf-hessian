/* glr_common_impl.h — the GLR kit, the prior object, the REPLICATED
 * backend, and every downstream operation.  Compiled inside the single
 * consumer TU (see impl.hpp).
 *
 * Provenance: the randomized-eigensolver kernels (hashed Gaussian sketch,
 * CholQR2, subspace iteration, T projection, treatment/truncation) are the
 * engine validated end-to-end in a large-scale ice-sheet inversion
 * (gate-checked against a replicated dense reference and in production
 * Newton runs); they move here with the operator composition generalized:
 * the prior supplies RAW Z applies/solves and the mass, and THIS file
 * composes F = M^{1/2} Z^{-1} B Z^{-T} M^{1/2} (previously the mass folding
 * lived inside the caller's solve callbacks).
 *
 * Math references (docs/ once migrated): the whitened master formula
 *   f(F + cI) = U [f(diag(lam) + cI) - f(c) I] U^T + f(c) I
 * with the truncated tail acting exactly like 0, and the factor
 *   G = Z M^{-1/2} (F + cI)^{1/2},   G G^T = H(c) = B + c R.
 */

#ifndef LGPSF_HESSIAN_GLR_COMMON_IMPL_H
#define LGPSF_HESSIAN_GLR_COMMON_IMPL_H

#include <petsc.h>

#include "lgpsf_hessian/prior.h"
#include "lgpsf_hessian/glr.h"
#include "lgpsf_hessian/impl/f77_decls.h"

/* ================================================================== */
/* hashed Gaussian kit                                                 */

/* xorshift64* + Box-Muller: deterministic per state.                  */
typedef struct { unsigned long s; int have; double spare; } lgh_rng_t;

static double
lgh_randn (lgh_rng_t *r)
{
  if (r->have) { r->have = 0; return r->spare; }
  for (;;) {
    double u1, u2, s;
    unsigned long x;
    x = r->s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; r->s = x;
    u1 = (double) ((x * 2685821657736338717UL) >> 11) / 9007199254740992.0;
    x = r->s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; r->s = x;
    u2 = (double) ((x * 2685821657736338717UL) >> 11) / 9007199254740992.0;
    u1 = 2. * u1 - 1.; u2 = 2. * u2 - 1.; s = u1 * u1 + u2 * u2;
    if (s > 0. && s < 1.) {
      double f = sqrt (-2. * log (s) / s);
      r->spare = u2 * f; r->have = 1;
      return u1 * f;
    }
  }
}

/* One standard normal as a pure function of (seed, i, j): splitmix64
 * mixes the triple into a nonzero xorshift64* state, then one Box-Muller
 * draw.  The sketch is therefore independent of the rank layout, and
 * extending the sketch just continues j past ell.                      */
static double
lgh_randn_at (unsigned long seed, PetscInt i, PetscInt j)
{
  lgh_rng_t           r;
  unsigned long       z = seed
      + 0x9E3779B97F4A7C15UL * (unsigned long) (i + 1)
      + 0xC2B2AE3D27D4EB4FUL * (unsigned long) (j + 1);

  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
  z = z ^ (z >> 31);
  r.s = z ? z : 0x853C49E6748FEA9BUL;
  r.have = 0;
  return lgh_randn (&r);
}

/* ================================================================== */
/* the prior object                                                    */

struct lgh_zs_ctx;                  /* Z-solve machinery for the Mat path
                                       (impl/zsolve_impl.h)              */

struct lgh_prior
{
  int                   refs;       /* lgh_glr_t holds a reference       */
  Vec                   mass;       /* diagonal of M (referenced)        */
  Vec                   msqrt;      /* sqrt(mass)                        */
  Vec                   minvsqrt;   /* 1/sqrt(mass)                      */
  Vec                   work;       /* scratch, same layout              */
  lgh_prior_callbacks_t cb;         /* the callbacks path                */
  /* Mat path (lgh_prior_create_mat): the callbacks above are wired to
   * internal adapters over these.                                       */
  Mat                   Z;          /* referenced                        */
  KSP                   zksp;       /* owned CG + AMG solver for Z       */
  struct lgh_zs_ctx    *zs;         /* blocked-solve machinery           */
};

/* defined in impl/zsolve_impl.h (same TU, included after) */
static void lgh_zs_prior_teardown (lgh_prior_t *p);

lgh_prior_mat_opts_t
lgh_prior_mat_opts_default (void)
{
  lgh_prior_mat_opts_t o;
  o.ksp_rtol = 1e-12;
  o.blocked_mode = 0;
  o.tile = 64;
  o.cheb_rtol = 1e-10;
  o.smax_krylov = 1;
  o.smoother_top = 1.1;
  o.nu_force = 0;
  o.hierarchy = LGH_HIERARCHY_AUTO;
  o.hypre_coarsen = 10;      /* HMIS */
  o.hypre_interp = 6;        /* extended+i */
  o.hypre_strong = 0.1;      /* 0.25 until 2026-09-05: at equal delivered accuracy
                                0.1 is 20% cheaper per column on the continental
                                ice-sheet prior (high-order stiffness, 43% positive
                                off-diagonals: the negative-connection strength
                                graph is sparse and a lower threshold keeps more
                                of it) and neutral on a sub-mesh and a structured
                                grid; PMIS (8) is -18% alone but the two do not
                                compound, and smoother degree 3 buys nothing */
  o.hypre_agg_nl = 0;
  o.hypre_max_coarse = 200;
  o.verbose = 0;
  return o;
}

static PetscErrorCode
lgh_prior_init_common (Vec mass_lumps, lgh_prior_t **prior)
{
  lgh_prior_t        *p;

  PetscCall (PetscNew (&p));
  p->refs = 1;
  p->mass = mass_lumps;
  PetscCall (PetscObjectReference ((PetscObject) mass_lumps));
  PetscCall (VecDuplicate (mass_lumps, &p->msqrt));
  PetscCall (VecCopy (mass_lumps, p->msqrt));
  PetscCall (VecSqrtAbs (p->msqrt));
  PetscCall (VecDuplicate (p->msqrt, &p->minvsqrt));
  PetscCall (VecCopy (p->msqrt, p->minvsqrt));
  PetscCall (VecReciprocal (p->minvsqrt));
  PetscCall (VecDuplicate (mass_lumps, &p->work));
  *prior = p;
  return PETSC_SUCCESS;
}

int
lgh_prior_create_callbacks (Vec mass_lumps,
                            const lgh_prior_callbacks_t *callbacks,
                            lgh_prior_t **prior)
{
  PetscCheck (callbacks != NULL && callbacks->applyZ != NULL
              && callbacks->solveZ != NULL, PETSC_COMM_SELF,
              PETSC_ERR_ARG_NULL,
              "lgh_prior_create_callbacks: applyZ and solveZ are required");
  PetscCall (lgh_prior_init_common (mass_lumps, prior));
  (*prior)->cb = *callbacks;
  return PETSC_SUCCESS;
}

/* lgh_prior_create_mat is defined in impl/zsolve_impl.h (it owns the
 * Z-solver machinery); same TU.                                          */

static void
lgh_prior_ref (lgh_prior_t *p)
{
  p->refs++;
}

void
lgh_prior_destroy (lgh_prior_t *prior)
{
  if (prior == NULL || --prior->refs > 0) return;
  lgh_zs_prior_teardown (prior);    /* Mat-path machinery, if any */
  (void) VecDestroy (&prior->msqrt);
  (void) VecDestroy (&prior->minvsqrt);
  (void) VecDestroy (&prior->work);
  (void) VecDestroy (&prior->mass);
  (void) MatDestroy (&prior->Z);
  (void) PetscFree (prior);
}

/* Vec-wise applies/solves (transpose falls back to the symmetric case). */
static PetscErrorCode
lgh_prior_applyZ_vec (lgh_prior_t *p, Vec x, Vec y)
{
  p->cb.applyZ (x, y, p->cb.ctx);
  return PETSC_SUCCESS;
}

static PetscErrorCode
lgh_prior_applyZt_vec (lgh_prior_t *p, Vec x, Vec y)
{
  if (p->cb.applyZt != NULL) p->cb.applyZt (x, y, p->cb.ctx);
  else p->cb.applyZ (x, y, p->cb.ctx);
  return PETSC_SUCCESS;
}

static PetscErrorCode
lgh_prior_solveZ_vec (lgh_prior_t *p, Vec x, Vec y)
{
  p->cb.solveZ (x, y, p->cb.ctx);
  return PETSC_SUCCESS;
}

static PetscErrorCode
lgh_prior_solveZt_vec (lgh_prior_t *p, Vec x, Vec y)
{
  if (p->cb.solveZt != NULL) p->cb.solveZt (x, y, p->cb.ctx);
  else p->cb.solveZ (x, y, p->cb.ctx);
  return PETSC_SUCCESS;
}

/* Blocked solves on MATDENSE column blocks: user-provided when available,
 * else a column loop over the Vec-wise callback.                        */
typedef enum { LGH_PRIOR_SOLVEZ, LGH_PRIOR_SOLVEZT } lgh_prior_block_which_t;

static PetscErrorCode
lgh_prior_solve_block (lgh_prior_t *p, lgh_prior_block_which_t which,
                       Mat X, Mat Y)
{
  lgh_mat_fn          blocked = (which == LGH_PRIOR_SOLVEZ)
      ? p->cb.solveZ_blocked
      : (p->cb.solveZt_blocked != NULL ? p->cb.solveZt_blocked
                                       : p->cb.solveZ_blocked);
  PetscInt            j, ncols;

  /* NULL solveZt_blocked with distinct solveZt: fall through to columns. */
  if (which == LGH_PRIOR_SOLVEZT && p->cb.solveZt_blocked == NULL
      && p->cb.solveZt != NULL)
    blocked = NULL;
  if (blocked != NULL) {
    blocked (X, Y, p->cb.ctx);
    return PETSC_SUCCESS;
  }
  PetscCall (MatGetSize (X, NULL, &ncols));
  for (j = 0; j < ncols; j++) {
    Vec                 xj, yj;
    PetscCall (MatDenseGetColumnVecRead (X, j, &xj));
    PetscCall (MatDenseGetColumnVecWrite (Y, j, &yj));
    if (which == LGH_PRIOR_SOLVEZ) PetscCall (lgh_prior_solveZ_vec (p, xj, yj));
    else PetscCall (lgh_prior_solveZt_vec (p, xj, yj));
    PetscCall (MatDenseRestoreColumnVecWrite (Y, j, &yj));
    PetscCall (MatDenseRestoreColumnVecRead (X, j, &xj));
  }
  return PETSC_SUCCESS;
}

int
lgh_prior_apply (lgh_prior_t *prior, Vec x, Vec y)
{
  /* y = R x = Z M^{-1} Z^T x */
  PetscCall (lgh_prior_applyZt_vec (prior, x, prior->work));
  PetscCall (VecPointwiseDivide (prior->work, prior->work, prior->mass));
  PetscCall (lgh_prior_applyZ_vec (prior, prior->work, y));
  return PETSC_SUCCESS;
}

/* ================================================================== */
/* the GLR object                                                      */

struct lgh_glrd_state;              /* ScaLAPACK backend state (impl/
                                       glr_scalapack_impl.h)            */

struct lgh_glr
{
  MPI_Comm            comm;
  PetscInt            nloc, Nglob;
  Mat                 B;            /* referenced                       */
  lgh_prior_t        *prior;        /* referenced                      */
  lgh_glr_opts_t      opts;         /* resolved (backend concrete)     */
  /* build product.  Both backends expose the same primitives:
   * UTmult (w = U^T x, replicated array) / Umult (y (+)= U w).
   * REPLICATED materializes U as an N x kept row slab; SCALAPACK keeps
   * U implicit (Q row slab + block-cyclic V) in `dist`.                */
  Mat                 U;            /* replicated backend only          */
  struct lgh_glrd_state *dist;      /* scalapack backend only           */
  PetscReal          *lam;          /* treated, |lam|-descending        */
  PetscInt            kept;
  /* deflation correction (lgh_glr_correct): after a graft the spectrum is
   * signed, U is always explicit (dispatch is by representation), the
   * sketch state is gone (extend forbidden), and shift-taking ops require
   * c > floor instead of c > 0.                                          */
  int                 corrected;
  double              floor;        /* max(0, -min lam'); 0 uncorrected  */
  /* scratch */
  Vec                 w1, w2;       /* full-space work                  */
  double             *wbuf;         /* kept-sized coefficient work      */
  double             *wbuf2;        /* kept-sized local reduction stage */
  double             *fbuf;         /* kept-sized filter values         */
};

#ifdef LGH_WITH_SCALAPACK
/* defined in impl/glr_scalapack_impl.h (same TU, included after) */
static PetscErrorCode lgh_glrd_build (lgh_glr_t *g, lgh_glr_report_t *rep);
static PetscErrorCode lgh_glrd_extend_incr (lgh_glr_t *g, int k_new,
                                            lgh_glr_report_t *rep);
static void           lgh_glrd_destroy_state (lgh_glr_t *g);
static PetscErrorCode lgh_glrd_UTmult (lgh_glr_t *g, Vec x, double *w);
static PetscErrorCode lgh_glrd_Umult (lgh_glr_t *g, const double *w, Vec y,
                                      PetscBool add);
static PetscErrorCode lgh_glrd_UTmult_block (lgh_glr_t *g, Mat X, double *W);
static PetscErrorCode lgh_glrd_Umult_block (lgh_glr_t *g, const double *W,
                                            Mat Y, PetscBool add);
static PetscErrorCode lgh_glrd_materialize_U (lgh_glr_t *g, Mat *Uout);
#endif

lgh_glr_opts_t
lgh_glr_opts_default (void)
{
  lgh_glr_opts_t      o;

  o.ell = 200;
  o.q_power = 1;
  o.trunc_abs = 0.;
  o.trunc_rel = 1e-3;
  o.treat = LGH_TREAT_FLIP;
  o.seed = 271828UL;
  o.backend = LGH_GLR_DEFAULT;
  o.nb = 64;
  o.panel = 64;
  o.grid2d = 0;
  o.check = 1;
  return o;
}

/* Y = F X = Msqrt . Z^{-1} . B . Z^{-T} . Msqrt . X;  W is scratch of the
 * same shape.  X is not modified.                                       */
static PetscErrorCode
lgh_glr_apply_F_block (lgh_glr_t *g, Mat X, Mat Y, Mat W)
{
  PetscInt            j, ncols;

  PetscCall (MatCopy (X, W, SAME_NONZERO_PATTERN));
  PetscCall (MatDiagonalScale (W, g->prior->msqrt, NULL));
  PetscCall (lgh_prior_solve_block (g->prior, LGH_PRIOR_SOLVEZT, W, Y));
  PetscCall (MatGetSize (X, NULL, &ncols));
  for (j = 0; j < ncols; j++) {
    Vec                 yj, wj;
    PetscCall (MatDenseGetColumnVecRead (Y, j, &yj));
    PetscCall (MatDenseGetColumnVecWrite (W, j, &wj));
    PetscCall (MatMult (g->B, yj, wj));
    PetscCall (MatDenseRestoreColumnVecWrite (W, j, &wj));
    PetscCall (MatDenseRestoreColumnVecRead (Y, j, &yj));
  }
  PetscCall (lgh_prior_solve_block (g->prior, LGH_PRIOR_SOLVEZ, W, Y));
  PetscCall (MatDiagonalScale (Y, g->prior->msqrt, NULL));
  return PETSC_SUCCESS;
}

/* G = X^T X (global): local GEMM + one allreduce; ell x ell column-major,
 * replicated on every rank.                                             */
static PetscErrorCode
lgh_glr_gram (Mat X, PetscReal *G, PetscReal *Gloc)
{
  PetscInt            nloc, ncols, lda, i;
  const PetscScalar  *xa;
  int                 bm, bn, blda;
  double              one = 1.0, zero = 0.0;
  MPI_Comm            comm;

  PetscCall (MatGetLocalSize (X, &nloc, NULL));
  PetscCall (MatGetSize (X, NULL, &ncols));
  PetscCall (MatDenseGetLDA (X, &lda));
  PetscCall (MatDenseGetArrayRead (X, &xa));
  bm = (int) nloc; bn = (int) ncols; blda = (int) lda;
  if (nloc > 0) {
    LGH_BLAS_DGEMM ("T", "N", &bn, &bn, &bm, &one, xa, &blda, xa, &blda,
                    &zero, Gloc, &bn);
  }
  else {
    for (i = 0; i < ncols * ncols; i++) Gloc[i] = 0.;
  }
  PetscCall (MatDenseRestoreArrayRead (X, &xa));
  PetscCall (PetscObjectGetComm ((PetscObject) X, &comm));
  PetscCallMPI (MPI_Allreduce (Gloc, G, (int) (ncols * ncols), MPIU_REAL,
                               MPI_SUM, comm));
  return PETSC_SUCCESS;
}

/* X <- X L^{-T} for the Cholesky factor L of G = X^T X; two passes
 * (CholQR2).  Jitters the Gram diagonal once if potrf fails.            */
static PetscErrorCode
lgh_glr_cholqr2 (Mat X, PetscReal *G, PetscReal *Gloc)
{
  PetscInt            pass, nloc, ncols, lda, i;

  PetscCall (MatGetLocalSize (X, &nloc, NULL));
  PetscCall (MatGetSize (X, NULL, &ncols));
  PetscCall (MatDenseGetLDA (X, &lda));
  for (pass = 0; pass < 2; pass++) {
    PetscScalar        *xa;
    int                 bn = (int) ncols, info = 0;
    int                 sm, sn, slda;
    double              one = 1.0;
    int                 attempt;

    PetscCall (lgh_glr_gram (X, G, Gloc));
    for (attempt = 0; attempt < 2; attempt++) {
      LGH_LAPACK_DPOTRF ("L", &bn, G, &bn, &info);
      if (info == 0) break;
      /* re-form G (potrf destroyed it) and jitter the diagonal */
      PetscCall (lgh_glr_gram (X, G, Gloc));
      {
        PetscReal           tr = 0.;
        for (i = 0; i < ncols; i++) tr += G[i + i * ncols];
        for (i = 0; i < ncols; i++) G[i + i * ncols] += 1e-12 * tr / ncols;
      }
    }
    PetscCheck (info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB,
                "lgh_glr_cholqr2: potrf failed twice");
    PetscCall (MatDenseGetArray (X, &xa));
    sm = (int) nloc; sn = (int) ncols; slda = (int) lda;
    if (nloc > 0) {
      LGH_BLAS_DTRSM ("R", "L", "T", "N", &sm, &sn, &one, G, &sn, xa, &slda);
    }
    PetscCall (MatDenseRestoreArray (X, &xa));
  }
  return PETSC_SUCCESS;
}

/* The replicated build: randomized subspace iteration with q power passes
 * and CholQR2 (BLAS-3 throughout; ONE allreduce per Gram), redundant
 * dsyevd of the projected T on every rank, |lam|-magnitude truncation,
 * eigenvalue treatment, U = Q V_kept materialized.  Fills g->{U,lam,kept}
 * and rep.                                                              */
static PetscErrorCode
lgh_glr_build_replicated (lgh_glr_t *g, lgh_glr_report_t *rep)
{
  const PetscInt      ell = g->opts.ell;
  const PetscInt      nloc = g->nloc, Nglob = g->Nglob;
  MPI_Comm            comm = g->comm;
  Mat                 Y, Q, W;
  PetscReal          *G, *Gloc, *T, *evals;
  PetscInt           *order;
  PetscInt            i, j, p, kept;
  PetscReal           absmax, cut, t0;

  PetscCall (PetscMemzero (rep, sizeof (*rep)));
  PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, ell, NULL, &Y));
  PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, ell, NULL, &Q));
  PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, ell, NULL, &W));
  PetscCall (PetscMalloc4 ((size_t) ell * ell, &G, (size_t) ell * ell, &Gloc,
                           (size_t) ell * ell, &T, ell, &evals));
  PetscCall (PetscMalloc1 (ell, &order));

  /* Omega (into Q), Gaussian, hashed per (seed, global row, column) */
  {
    PetscScalar        *qa;
    PetscInt            lda, rstart;
    PetscCall (MatGetOwnershipRange (Q, &rstart, NULL));
    PetscCall (MatDenseGetLDA (Q, &lda));
    PetscCall (MatDenseGetArrayWrite (Q, &qa));
    for (j = 0; j < ell; j++)
      for (i = 0; i < nloc; i++)
        qa[i + j * lda] = lgh_randn_at (g->opts.seed, rstart + i, j);
    PetscCall (MatDenseRestoreArrayWrite (Q, &qa));
  }

  /* Y = F Omega; power passes with re-orthonormalization */
  t0 = MPI_Wtime ();
  PetscCall (lgh_glr_apply_F_block (g, Q, Y, W));
  rep->t_operator += MPI_Wtime () - t0;
  for (p = 0; p < g->opts.q_power; p++) {
    t0 = MPI_Wtime ();
    PetscCall (lgh_glr_cholqr2 (Y, G, Gloc));
    rep->t_dense += MPI_Wtime () - t0;
    PetscCall (MatCopy (Y, Q, SAME_NONZERO_PATTERN));
    t0 = MPI_Wtime ();
    PetscCall (lgh_glr_apply_F_block (g, Q, Y, W));
    rep->t_operator += MPI_Wtime () - t0;
  }
  t0 = MPI_Wtime ();
  PetscCall (lgh_glr_cholqr2 (Y, G, Gloc));
  rep->t_dense += MPI_Wtime () - t0;
  PetscCall (MatCopy (Y, Q, SAME_NONZERO_PATTERN));   /* Q orthonormal */

  /* T = Q^T F Q, symmetrized; redundant dense eig on every rank */
  t0 = MPI_Wtime ();
  PetscCall (lgh_glr_apply_F_block (g, Q, Y, W));
  rep->t_operator += MPI_Wtime () - t0;
  {
    const PetscScalar  *qa, *ya;
    PetscInt            ldq, ldy;
    int                 bm, bn, bldq, bldy;
    double              one = 1.0, zero = 0.0;

    PetscCall (MatDenseGetLDA (Q, &ldq));
    PetscCall (MatDenseGetLDA (Y, &ldy));
    PetscCall (MatDenseGetArrayRead (Q, &qa));
    PetscCall (MatDenseGetArrayRead (Y, &ya));
    bm = (int) nloc; bn = (int) ell; bldq = (int) ldq; bldy = (int) ldy;
    if (nloc > 0) {
      LGH_BLAS_DGEMM ("T", "N", &bn, &bn, &bm, &one, qa, &bldq, ya, &bldy,
                      &zero, Gloc, &bn);
    }
    else { for (i = 0; i < ell * ell; i++) Gloc[i] = 0.; }
    PetscCall (MatDenseRestoreArrayRead (Q, &qa));
    PetscCall (MatDenseRestoreArrayRead (Y, &ya));
    PetscCallMPI (MPI_Allreduce (Gloc, T, (int) (ell * ell), MPIU_REAL,
                                 MPI_SUM, comm));
  }
  for (j = 0; j < ell; j++)
    for (i = 0; i < j; i++) {
      PetscReal           avg = 0.5 * (T[i + j * ell] + T[j + i * ell]);
      T[i + j * ell] = avg; T[j + i * ell] = avg;
    }
  t0 = MPI_Wtime ();
  {
    int                 bn = (int) ell, lwork = -1, liwork = -1, iwkopt = 0;
    int                 info = 0;
    double              wkopt;
    double             *work;
    int                *iwork;

    LGH_LAPACK_DSYEVD ("V", "U", &bn, T, &bn, evals, &wkopt, &lwork,
                       &iwkopt, &liwork, &info);
    lwork = (int) wkopt; liwork = iwkopt;
    PetscCall (PetscMalloc1 (lwork, &work));
    PetscCall (PetscMalloc1 (liwork, &iwork));
    LGH_LAPACK_DSYEVD ("V", "U", &bn, T, &bn, evals, work, &lwork,
                       iwork, &liwork, &info);
    PetscCall (PetscFree (work));
    PetscCall (PetscFree (iwork));
    PetscCheck (info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB,
                "lgh_glr_build_replicated: dsyevd failed");
  }
  rep->t_dense += MPI_Wtime () - t0;

  /* rank by |lam| descending; truncate; treat */
  for (i = 0; i < ell; i++) order[i] = i;
  for (i = 0; i < ell; i++)
    for (j = i + 1; j < ell; j++)
      if (PetscAbsReal (evals[order[j]]) > PetscAbsReal (evals[order[i]])) {
        PetscInt            t = order[i]; order[i] = order[j]; order[j] = t;
      }
  absmax = PetscAbsReal (evals[order[0]]);
  cut = 0.;
  if (g->opts.trunc_abs > 0.) cut = PetscMax (cut, g->opts.trunc_abs);
  if (g->opts.trunc_rel > 0.) cut = PetscMax (cut, g->opts.trunc_rel * absmax);
  kept = 0;
  while (kept < ell && PetscAbsReal (evals[order[kept]]) > cut) kept++;
  rep->kept = (int) kept;
  rep->lam_abs_max = absmax;
  rep->next_abs = (kept < ell) ? PetscAbsReal (evals[order[kept]]) : 0.;
  rep->lam_raw_max = evals[ell - 1];    /* dsyevd ascending */
  rep->lam_raw_min = evals[0];
  for (i = 0; i < ell; i++)
    if (evals[i] < 0.) rep->n_negative_raw++;

  /* U = Q C_kept (local GEMM), lam treated */
  PetscCall (PetscMalloc1 (PetscMax (kept, 1), &g->lam));
  PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob,
                             PetscMax (kept, 1), NULL, &g->U));
  {
    const PetscScalar  *qa;
    PetscScalar        *ua;
    PetscReal          *C;
    PetscInt            ldq, ldu;
    int                 bm, bn, bk, bldq, bldu;
    double              one = 1.0, zero = 0.0;

    PetscCall (PetscMalloc1 ((size_t) ell * PetscMax (kept, 1), &C));
    for (j = 0; j < kept; j++) {
      PetscReal           l = evals[order[j]];
      for (i = 0; i < ell; i++) C[i + j * ell] = T[i + order[j] * ell];
      switch (g->opts.treat) {
      case LGH_TREAT_FLIP: g->lam[j] = PetscAbsReal (l); break;
      case LGH_TREAT_RELU: g->lam[j] = PetscMax (l, 0.); break;
      default:             g->lam[j] = l; break;
      }
    }
    PetscCall (MatDenseGetLDA (Q, &ldq));
    PetscCall (MatDenseGetLDA (g->U, &ldu));
    PetscCall (MatDenseGetArrayRead (Q, &qa));
    PetscCall (MatDenseGetArrayWrite (g->U, &ua));
    bm = (int) nloc; bk = (int) PetscMax (kept, 1);
    bn = (int) ell; bldq = (int) ldq; bldu = (int) ldu;
    if (nloc > 0 && kept > 0) {
      LGH_BLAS_DGEMM ("N", "N", &bm, &bk, &bn, &one, qa, &bldq, C, &bn,
                      &zero, ua, &bldu);
    }
    PetscCall (MatDenseRestoreArrayRead (Q, &qa));
    PetscCall (MatDenseRestoreArrayWrite (g->U, &ua));
    PetscCall (PetscFree (C));
  }
  g->kept = kept;

  /* optional residual check: ||F u_j - lam_raw_j u_j||_2 per kept column */
  if (g->opts.check && kept > 0) {
    Mat                 FU, WU;
    PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, kept, NULL,
                               &FU));
    PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, kept, NULL,
                               &WU));
    t0 = MPI_Wtime ();
    PetscCall (lgh_glr_apply_F_block (g, g->U, FU, WU));
    rep->t_operator += MPI_Wtime () - t0;
    for (j = 0; j < kept; j++) {
      Vec                 fj, uj;
      PetscReal           nrm;
      PetscReal           lraw = evals[order[j]];
      PetscCall (MatDenseGetColumnVecWrite (FU, j, &fj));
      PetscCall (MatDenseGetColumnVecRead (g->U, j, &uj));
      PetscCall (VecAXPY (fj, -lraw, uj));
      PetscCall (VecNorm (fj, NORM_2, &nrm));
      rep->resid_max = PetscMax (rep->resid_max, nrm);
      PetscCall (MatDenseRestoreColumnVecRead (g->U, j, &uj));
      PetscCall (MatDenseRestoreColumnVecWrite (FU, j, &fj));
    }
    PetscCall (MatDestroy (&FU));
    PetscCall (MatDestroy (&WU));
  }

  PetscCall (PetscFree (order));
  PetscCall (PetscFree4 (G, Gloc, T, evals));
  PetscCall (MatDestroy (&Y));
  PetscCall (MatDestroy (&Q));
  PetscCall (MatDestroy (&W));
  return PETSC_SUCCESS;
}

/* Scratch depends on kept, so it is (re)created after every build.      */
static PetscErrorCode
lgh_glr_setup_scratch (lgh_glr_t *g)
{
  PetscCall (PetscFree (g->wbuf));
  PetscCall (PetscFree (g->wbuf2));
  PetscCall (PetscFree (g->fbuf));
  if (g->w1 == NULL) {
    PetscCall (MatCreateVecs (g->B, &g->w1, NULL));
    PetscCall (MatCreateVecs (g->B, &g->w2, NULL));
  }
  PetscCall (PetscMalloc1 (PetscMax (g->kept, 1), &g->wbuf));
  PetscCall (PetscMalloc1 (PetscMax (g->kept, 1), &g->wbuf2));
  PetscCall (PetscMalloc1 (PetscMax (g->kept, 1), &g->fbuf));
  return PETSC_SUCCESS;
}

static PetscErrorCode
lgh_glr_teardown_build (lgh_glr_t *g)
{
  PetscCall (MatDestroy (&g->U));
  PetscCall (PetscFree (g->lam));
  g->lam = NULL;
  g->kept = 0;
  return PETSC_SUCCESS;
}

int
lgh_glr_compute (Mat B, lgh_prior_t *prior, const lgh_glr_opts_t *opts,
                 lgh_glr_t **glr, lgh_glr_report_t *report)
{
  lgh_glr_t          *g;
  lgh_glr_report_t    local_rep;
  PetscInt            Nglob;

  PetscCall (PetscNew (&g));
  g->opts = (opts != NULL) ? *opts : lgh_glr_opts_default ();
  PetscCall (PetscObjectGetComm ((PetscObject) B, &g->comm));
  PetscCall (MatGetLocalSize (B, &g->nloc, NULL));
  PetscCall (MatGetSize (B, &Nglob, NULL));
  g->Nglob = Nglob;
  PetscCheck (g->opts.ell > 0 && g->opts.ell <= Nglob, g->comm,
              PETSC_ERR_ARG_OUTOFRANGE,
              "lgh_glr_compute: need 0 < ell (%d) <= N (%d)",
              (int) g->opts.ell, (int) Nglob);

#ifdef LGH_WITH_SCALAPACK
  if (g->opts.backend == LGH_GLR_DEFAULT)
    g->opts.backend = LGH_GLR_SCALAPACK;
#else
  if (g->opts.backend == LGH_GLR_DEFAULT)
    g->opts.backend = LGH_GLR_REPLICATED;
  PetscCheck (g->opts.backend != LGH_GLR_SCALAPACK, g->comm, PETSC_ERR_SUP,
              "lgh_glr_compute: this build has no ScaLAPACK backend "
              "(configure with LGH_WITH_SCALAPACK=ON)");
#endif

  g->B = B;
  PetscCall (PetscObjectReference ((PetscObject) B));
  g->prior = prior;
  lgh_prior_ref (prior);

#ifdef LGH_WITH_SCALAPACK
  if (g->opts.backend == LGH_GLR_SCALAPACK)
    PetscCall (lgh_glrd_build (g, &local_rep));
  else
#endif
  PetscCall (lgh_glr_build_replicated (g, &local_rep));
  PetscCall (lgh_glr_setup_scratch (g));
  if (report != NULL) *report = local_rep;
  *glr = g;
  return PETSC_SUCCESS;
}

int
lgh_glr_extend (lgh_glr_t *glr, int k_new, lgh_glr_report_t *report)
{
  lgh_glr_report_t    local_rep;

  /* The hashed sketch makes extension EXACTLY "the build you would have
   * done at ell + k_new" (columns are a pure function of (seed, i, j), so
   * the first ell columns of the wider sketch are the same draws).  The
   * ScaLAPACK backend realizes it incrementally (reuses Q, borders T);
   * the replicated backend recomputes at the wider ell — correct, not
   * incremental-cost.                                                   */
  PetscCheck (!glr->corrected, glr->comm, PETSC_ERR_ORDER,
              "lgh_glr_extend: not available after lgh_glr_correct (the "
              "sketch no longer represents the corrected operator) — "
              "extend first, correct last");
  PetscCheck (k_new > 0, glr->comm, PETSC_ERR_ARG_OUTOFRANGE,
              "lgh_glr_extend: k_new must be positive");
  PetscCheck (glr->opts.ell + k_new <= glr->Nglob, glr->comm,
              PETSC_ERR_ARG_OUTOFRANGE,
              "lgh_glr_extend: extended ell (%d) exceeds N (%d)",
              (int) glr->opts.ell + k_new, (int) glr->Nglob);
#ifdef LGH_WITH_SCALAPACK
  if (glr->opts.backend == LGH_GLR_SCALAPACK) {
    PetscCall (lgh_glrd_extend_incr (glr, k_new, &local_rep));
  }
  else
#endif
  {
    glr->opts.ell += k_new;
    PetscCall (lgh_glr_teardown_build (glr));
    PetscCall (lgh_glr_build_replicated (glr, &local_rep));
  }
  PetscCall (lgh_glr_setup_scratch (glr));
  if (report != NULL) *report = local_rep;
  return PETSC_SUCCESS;
}

void
lgh_glr_destroy (lgh_glr_t *glr)
{
  if (glr == NULL) return;
  (void) MatDestroy (&glr->U);
#ifdef LGH_WITH_SCALAPACK
  lgh_glrd_destroy_state (glr);
#endif
  (void) PetscFree (glr->lam);
  (void) VecDestroy (&glr->w1);
  (void) VecDestroy (&glr->w2);
  (void) PetscFree (glr->wbuf);
  (void) PetscFree (glr->wbuf2);
  (void) PetscFree (glr->fbuf);
  (void) MatDestroy (&glr->B);
  lgh_prior_destroy (glr->prior);
  (void) PetscFree (glr);
}

/* ================================================================== */
/* downstream operations                                               */

/* Implicit-U primitives, replicated backend: U is an N x kept row slab,
 * so U^T x is a local GEMV + one allreduce of kept doubles, and U w is a
 * purely local GEMV.  (Same interface as the ScaLAPACK backend's
 * Q(V w) / V^T(Q^T x) primitives — the op layer dispatches.)            */
static PetscErrorCode
lgh_glre_UTmult (lgh_glr_t *g, Vec x, double *w)
{
  const PetscScalar  *ua, *xa;
  PetscInt            nloc, lda;
  double             *wl = g->wbuf2;
  PetscInt            i;

  PetscCall (MatGetLocalSize (g->U, &nloc, NULL));
  PetscCall (MatDenseGetLDA (g->U, &lda));
  PetscCall (MatDenseGetArrayRead (g->U, &ua));
  PetscCall (VecGetArrayRead (x, &xa));
  if (nloc > 0) {
    int                 bm = (int) nloc, bk = (int) g->kept;
    int                 blda = (int) lda, ione = 1;
    double              one = 1.0, zero = 0.0;
    LGH_BLAS_DGEMV ("T", &bm, &bk, &one, ua, &blda, (const double *) xa,
                    &ione, &zero, wl, &ione);
  }
  else { for (i = 0; i < g->kept; i++) wl[i] = 0.; }
  PetscCall (VecRestoreArrayRead (x, &xa));
  PetscCall (MatDenseRestoreArrayRead (g->U, &ua));
  PetscCallMPI (MPI_Allreduce (wl, w, (int) g->kept, MPI_DOUBLE, MPI_SUM,
                               g->comm));
  return PETSC_SUCCESS;
}

static PetscErrorCode
lgh_glre_Umult (lgh_glr_t *g, const double *w, Vec y, PetscBool add)
{
  const PetscScalar  *ua;
  PetscScalar        *ya;
  PetscInt            nloc, lda;

  PetscCall (MatGetLocalSize (g->U, &nloc, NULL));
  PetscCall (MatDenseGetLDA (g->U, &lda));
  PetscCall (MatDenseGetArrayRead (g->U, &ua));
  PetscCall (VecGetArray (y, &ya));
  if (nloc > 0) {
    int                 bm = (int) nloc, bk = (int) g->kept;
    int                 blda = (int) lda, ione = 1;
    double              one = 1.0, beta = add ? 1.0 : 0.0;
    LGH_BLAS_DGEMV ("N", &bm, &bk, &one, ua, &blda, w, &ione, &beta, ya,
                    &ione);
  }
  PetscCall (VecRestoreArray (y, &ya));
  PetscCall (MatDenseRestoreArrayRead (g->U, &ua));
  return PETSC_SUCCESS;
}

/* Dispatch is by REPRESENTATION, not backend: a graft (lgh_glr_correct)
 * leaves every object with an explicit U, including ScaLAPACK-built ones. */
static PetscErrorCode
lgh_glr_UTmult (lgh_glr_t *g, Vec x, double *w)
{
#ifdef LGH_WITH_SCALAPACK
  if (g->U == NULL)
    return lgh_glrd_UTmult (g, x, w);
#endif
  return lgh_glre_UTmult (g, x, w);
}

static PetscErrorCode
lgh_glr_Umult (lgh_glr_t *g, const double *w, Vec y, PetscBool add)
{
#ifdef LGH_WITH_SCALAPACK
  if (g->U == NULL)
    return lgh_glrd_Umult (g, w, y, add);
#endif
  return lgh_glre_Umult (g, w, y, add);
}

/* ---- block (MATDENSE) forms of the implicit-U primitives -------------- */
/* W is a replicated kept x m coefficient block, column-major.  One GEMM +
 * one allreduce per call (vs one allreduce per column with the Vec forms —
 * and, downstream, per-column Z solves always take the expensive KSP path,
 * so the correction sweeps must be block-based end to end).               */

static PetscErrorCode
lgh_glre_UTmult_block (lgh_glr_t *g, Mat X, double *W)
{
  const PetscScalar  *ua, *xa;
  PetscInt            nloc, ldu, ldx, m, i;
  double             *Wl;
  int                 bm, bk, bmm, bldu, bldx;
  double              one = 1.0, zero = 0.0;

  PetscCall (MatGetSize (X, NULL, &m));
  PetscCall (PetscMalloc1 ((size_t) PetscMax (g->kept * m, 1), &Wl));
  PetscCall (MatGetLocalSize (g->U, &nloc, NULL));
  PetscCall (MatDenseGetLDA (g->U, &ldu));
  PetscCall (MatDenseGetLDA (X, &ldx));
  PetscCall (MatDenseGetArrayRead (g->U, &ua));
  PetscCall (MatDenseGetArrayRead (X, &xa));
  bm = (int) nloc; bk = (int) g->kept; bmm = (int) m;
  bldu = (int) ldu; bldx = (int) ldx;
  if (nloc > 0 && g->kept > 0) {
    LGH_BLAS_DGEMM ("T", "N", &bk, &bmm, &bm, &one, ua, &bldu, xa, &bldx,
                    &zero, Wl, &bk);
  }
  else { for (i = 0; i < g->kept * m; i++) Wl[i] = 0.; }
  PetscCall (MatDenseRestoreArrayRead (X, &xa));
  PetscCall (MatDenseRestoreArrayRead (g->U, &ua));
  if (g->kept > 0)
    PetscCallMPI (MPI_Allreduce (Wl, W, (int) (g->kept * m), MPI_DOUBLE,
                                 MPI_SUM, g->comm));
  PetscCall (PetscFree (Wl));
  return PETSC_SUCCESS;
}

static PetscErrorCode
lgh_glre_Umult_block (lgh_glr_t *g, const double *W, Mat Y, PetscBool add)
{
  const PetscScalar  *ua;
  PetscScalar        *ya;
  PetscInt            nloc, ldu, ldy, m;
  int                 bm, bk, bmm, bldu, bldy;
  double              one = 1.0, beta;

  PetscCall (MatGetSize (Y, NULL, &m));
  PetscCall (MatGetLocalSize (g->U, &nloc, NULL));
  PetscCall (MatDenseGetLDA (g->U, &ldu));
  PetscCall (MatDenseGetLDA (Y, &ldy));
  PetscCall (MatDenseGetArrayRead (g->U, &ua));
  PetscCall (MatDenseGetArray (Y, &ya));
  bm = (int) nloc; bk = (int) g->kept; bmm = (int) m;
  bldu = (int) ldu; bldy = (int) ldy;
  beta = add ? 1.0 : 0.0;
  if (nloc > 0 && g->kept > 0) {
    LGH_BLAS_DGEMM ("N", "N", &bm, &bmm, &bk, &one, ua, &bldu, W, &bk,
                    &beta, ya, &bldy);
  }
  else if (nloc > 0 && !add) {
    PetscInt            j, jc;
    for (jc = 0; jc < m; jc++)
      for (j = 0; j < nloc; j++) ya[j + (size_t) jc * ldy] = 0.;
  }
  PetscCall (MatDenseRestoreArray (Y, &ya));
  PetscCall (MatDenseRestoreArrayRead (g->U, &ua));
  return PETSC_SUCCESS;
}

static PetscErrorCode
lgh_glr_UTmult_block (lgh_glr_t *g, Mat X, double *W)
{
#ifdef LGH_WITH_SCALAPACK
  if (g->U == NULL)
    return lgh_glrd_UTmult_block (g, X, W);
#endif
  return lgh_glre_UTmult_block (g, X, W);
}

static PetscErrorCode
lgh_glr_Umult_block (lgh_glr_t *g, const double *W, Mat Y, PetscBool add)
{
#ifdef LGH_WITH_SCALAPACK
  if (g->U == NULL)
    return lgh_glrd_Umult_block (g, W, Y, add);
#endif
  return lgh_glre_Umult_block (g, W, Y, add);
}

/* Y = f_c X + U diag(f_lam - f_c) U^T X  (block master formula).         */
static PetscErrorCode
lgh_glr_filter_block (lgh_glr_t *g, const double *f_lam, double f_c,
                      Mat X, Mat Y)
{
  PetscInt            m, k, j;
  double             *W = NULL;

  PetscCall (MatGetSize (X, NULL, &m));
  PetscCall (MatCopy (X, Y, SAME_NONZERO_PATTERN));
  PetscCall (MatScale (Y, f_c));
  if (g->kept == 0) return PETSC_SUCCESS;
  PetscCall (PetscMalloc1 ((size_t) g->kept * m, &W));
  PetscCall (lgh_glr_UTmult_block (g, X, W));
  for (j = 0; j < m; j++)
    for (k = 0; k < g->kept; k++)
      W[k + (size_t) j * g->kept] *= f_lam[k] - f_c;
  PetscCall (lgh_glr_Umult_block (g, W, Y, PETSC_TRUE));
  PetscCall (PetscFree (W));
  return PETSC_SUCCESS;
}

/* Y = (F + cI)^p X over the kept modes (block form of lgh_glr_pow).      */
static PetscErrorCode
lgh_glr_pow_block (lgh_glr_t *g, double c, double p, Mat X, Mat Y)
{
  double             *fv = NULL;
  PetscInt            i;
  PetscErrorCode      ierr;

  PetscCall (PetscMalloc1 (PetscMax (g->kept, 1), &fv));
  for (i = 0; i < g->kept; i++)
    fv[i] = pow (g->lam[i] + c, p);
  ierr = lgh_glr_filter_block (g, fv, pow (c, p), X, Y);
  PetscCall (PetscFree (fv));
  return ierr;
}

/* out = f_c in + U diag(f_lam - f_c) U^T in  (the master formula).     */
static PetscErrorCode
lgh_glr_filter_core (lgh_glr_t *g, const double *f_lam, double f_c,
                     Vec in, Vec out)
{
  PetscInt            k;

  if (g->kept == 0) {
    PetscCall (VecCopy (in, out));
    PetscCall (VecScale (out, f_c));
    return PETSC_SUCCESS;
  }
  PetscCall (lgh_glr_UTmult (g, in, g->wbuf));
  for (k = 0; k < g->kept; k++)
    g->wbuf[k] *= f_lam[k] - f_c;
  PetscCall (VecCopy (in, out));
  PetscCall (VecScale (out, f_c));
  PetscCall (lgh_glr_Umult (g, g->wbuf, out, PETSC_TRUE));
  return PETSC_SUCCESS;
}

int
lgh_glr_filter (lgh_glr_t *glr, const double *f_lam, double f_c,
                Vec in, Vec out)
{
  return (int) lgh_glr_filter_core (glr, f_lam, f_c, in, out);
}

int
lgh_glr_pow (lgh_glr_t *glr, double c, double p, Vec in, Vec out)
{
  PetscInt            i;

  PetscCheck (c > glr->floor, glr->comm, PETSC_ERR_ARG_OUTOFRANGE,
              "lgh_glr_pow: shift c must exceed the spectrum floor %g",
              glr->floor);
  for (i = 0; i < glr->kept; i++)
    glr->fbuf[i] = pow (glr->lam[i] + c, p);
  return (int) lgh_glr_filter_core (glr, glr->fbuf, pow (c, p), in, out);
}

double
lgh_glr_logdet (lgh_glr_t *glr, double c)
{
  double              s = 0.;
  PetscInt            i;

  for (i = 0; i < glr->kept; i++)
    s += log (glr->lam[i] + c) - log (c);
  return s;
}

int
lgh_glr_eigs (const lgh_glr_t *glr, int *nkept, const double **lam)
{
  *nkept = (int) glr->kept;
  *lam = glr->lam;
  return 0;
}

int
lgh_glr_solve (lgh_glr_t *glr, double c, Vec rhs, Vec x)
{
  /* H(c)^{-1} = Z^{-T} M^{1/2} (F + cI)^{-1} M^{1/2} Z^{-1} */
  PetscInt            i;

  PetscCheck (c > glr->floor, glr->comm, PETSC_ERR_ARG_OUTOFRANGE,
              "lgh_glr_solve: shift c must exceed the spectrum floor %g",
              glr->floor);
  PetscCall (lgh_prior_solveZ_vec (glr->prior, rhs, glr->w1));
  PetscCall (VecPointwiseMult (glr->w1, glr->w1, glr->prior->msqrt));
  for (i = 0; i < glr->kept; i++)
    glr->fbuf[i] = 1. / (glr->lam[i] + c);
  PetscCall (lgh_glr_filter_core (glr, glr->fbuf, 1. / c, glr->w1, glr->w2));
  PetscCall (VecPointwiseMult (glr->w2, glr->w2, glr->prior->msqrt));
  PetscCall (lgh_prior_solveZt_vec (glr->prior, glr->w2, x));
  return PETSC_SUCCESS;
}

int
lgh_glr_apply (lgh_glr_t *glr, double c, Vec v, Vec y)
{
  if (glr->corrected) {
    /* After a graft the sparse B alone no longer represents the operator:
     * apply the corrected low-rank action
     * y = Z M^{-1/2} (U' lam' U'^T + cI) M^{-1/2} Z^T v.                 */
    PetscInt            i;
    PetscCall (lgh_prior_applyZt_vec (glr->prior, v, glr->w1));
    PetscCall (VecPointwiseMult (glr->w1, glr->w1, glr->prior->minvsqrt));
    for (i = 0; i < glr->kept; i++)
      glr->fbuf[i] = glr->lam[i] + c;
    PetscCall (lgh_glr_filter_core (glr, glr->fbuf, c, glr->w1, glr->w2));
    PetscCall (VecPointwiseMult (glr->w2, glr->w2, glr->prior->minvsqrt));
    PetscCall (lgh_prior_applyZ_vec (glr->prior, glr->w2, y));
    return PETSC_SUCCESS;
  }
  /* exact: y = B v + c R v (no truncation error) */
  PetscCall (MatMult (glr->B, v, y));
  PetscCall (lgh_prior_apply (glr->prior, v, glr->w1));
  PetscCall (VecAXPY (y, c, glr->w1));
  return PETSC_SUCCESS;
}

int
lgh_glr_factor (lgh_glr_t *glr, double c, lgh_factor_t which,
                Vec in, Vec out)
{
  /* G = Z M^{-1/2} S,  S = (F + cI)^{1/2} symmetric.                    */
  PetscInt            i;
  const double        half = 0.5;

  PetscCheck (c > glr->floor, glr->comm, PETSC_ERR_ARG_OUTOFRANGE,
              "lgh_glr_factor: shift c must exceed the spectrum floor %g",
              glr->floor);
  switch (which) {
  case LGH_FACTOR_G:            /* Z M^{-1/2} S in                       */
    for (i = 0; i < glr->kept; i++)
      glr->fbuf[i] = pow (glr->lam[i] + c, half);
    PetscCall (lgh_glr_filter_core (glr, glr->fbuf, sqrt (c), in, glr->w1));
    PetscCall (VecPointwiseMult (glr->w1, glr->w1, glr->prior->minvsqrt));
    PetscCall (lgh_prior_applyZ_vec (glr->prior, glr->w1, out));
    break;
  case LGH_FACTOR_GT:           /* S M^{-1/2} Z^T in                     */
    PetscCall (lgh_prior_applyZt_vec (glr->prior, in, glr->w1));
    PetscCall (VecPointwiseMult (glr->w1, glr->w1, glr->prior->minvsqrt));
    for (i = 0; i < glr->kept; i++)
      glr->fbuf[i] = pow (glr->lam[i] + c, half);
    PetscCall (lgh_glr_filter_core (glr, glr->fbuf, sqrt (c), glr->w1, out));
    break;
  case LGH_FACTOR_GINV:         /* S^{-1} M^{1/2} Z^{-1} in              */
    PetscCall (lgh_prior_solveZ_vec (glr->prior, in, glr->w1));
    PetscCall (VecPointwiseMult (glr->w1, glr->w1, glr->prior->msqrt));
    for (i = 0; i < glr->kept; i++)
      glr->fbuf[i] = 1. / pow (glr->lam[i] + c, half);
    PetscCall (lgh_glr_filter_core (glr, glr->fbuf, 1. / sqrt (c), glr->w1,
                                    out));
    break;
  case LGH_FACTOR_GINVT:        /* Z^{-T} M^{1/2} S^{-1} in              */
    for (i = 0; i < glr->kept; i++)
      glr->fbuf[i] = 1. / pow (glr->lam[i] + c, half);
    PetscCall (lgh_glr_filter_core (glr, glr->fbuf, 1. / sqrt (c), in,
                                    glr->w1));
    PetscCall (VecPointwiseMult (glr->w1, glr->w1, glr->prior->msqrt));
    PetscCall (lgh_prior_solveZt_vec (glr->prior, glr->w1, out));
    break;
  default:
    SETERRQ (glr->comm, PETSC_ERR_ARG_OUTOFRANGE,
             "lgh_glr_factor: unknown factor action");
  }
  return PETSC_SUCCESS;
}

int
lgh_glr_sample (lgh_glr_t *glr, double c, Vec xi, Vec x)
{
  /* x = G^{-T} xi: for xi ~ N(0, I), cov(x) = (G G^T)^{-1} = H(c)^{-1} */
  return lgh_glr_factor (glr, c, LGH_FACTOR_GINVT, xi, x);
}

#endif /* LGPSF_HESSIAN_GLR_COMMON_IMPL_H */

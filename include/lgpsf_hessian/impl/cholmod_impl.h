/* cholmod_impl.h — the Cholesky prior behind lgh_prior_create_cholmod.
 * Compiled inside the single consumer TU, after glr_common_impl.h, and only
 * with LGH_HAVE_CHOLMOD (the CMake target defines it when CHOLMOD is found).
 *
 * Every rank holds the same sparse factor P A P^T = L L^T of the WHOLE
 * matrix (AMD ordering, supernodal LL^T).  Conventions (CHOLMOD's L->Perm):
 * row k of P A P^T is row Perm[k] of A, i.e.
 *     (P x)[k] = x[Perm[k]],        (P^T y)[Perm[k]] = y[k].
 * Operations on full-length column-major arrays (n x k):
 *   A_IS_R:  Z = P^T L P.   Z x = P^T L P x,     Z^T x = P^T L^T P x,
 *                           Z^-1 y = P^T L^-1 P y, Z^-T y = P^T L^-T P y.
 *   A_IS_Z:  Z = A.         Z x = A x (distributed MatMult),
 *                           Z^-1 y = P^T L^-T L^-1 P y   (symmetric).
 * L and L^T are applied by a direct sweep over the supernodes (no second
 * copy of L); the triangular solves are CHOLMOD's supernodal ones
 * (dtrsv/dgemv for one column, dtrsm/dgemm for a block), run in place on
 * our own buffers.
 */

#ifndef LGPSF_HESSIAN_CHOLMOD_IMPL_H
#define LGPSF_HESSIAN_CHOLMOD_IMPL_H

#ifndef LGPSF_HESSIAN_GLR_COMMON_IMPL_H
#error "include impl/glr_common_impl.h first (impl.hpp does this)"
#endif

#include <cholmod.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* BLAS threads: CHOLMOD's BLAS runs single-threaded, so every rank computes
 * a bitwise-identical factor.  The thread count is set to 1 around each
 * CHOLMOD call and restored afterwards; the controls are weak symbols, so
 * whichever of OpenBLAS / MKL is linked is handled and neither is needed.
 * (MKL: the C entry point MKL_Set_Num_Threads_Local, by value; the
 * lower-case mkl_set_num_threads_local symbol is the Fortran one, by
 * reference.) */

#if defined(__GNUC__)
#ifdef __cplusplus
extern "C" {
#endif
extern int  openblas_get_num_threads (void) __attribute__ ((weak));
extern void openblas_set_num_threads (int) __attribute__ ((weak));
extern int  MKL_Set_Num_Threads_Local (int) __attribute__ ((weak));
#ifdef __cplusplus
}
#endif
#define LGH_CHOL_HAVE_WEAK 1
#endif

typedef struct { int openblas, mkl; } lgh_chol_threads_t;

static lgh_chol_threads_t
lgh_chol_threads_enter (void)
{
  lgh_chol_threads_t  t = { -1, -1 };
#ifdef LGH_CHOL_HAVE_WEAK
  if (openblas_get_num_threads != NULL && openblas_set_num_threads != NULL) {
    t.openblas = openblas_get_num_threads ();
    if (t.openblas != 1) openblas_set_num_threads (1);
  }
  if (MKL_Set_Num_Threads_Local != NULL)
    t.mkl = MKL_Set_Num_Threads_Local (1);
#endif
  return t;
}

static void
lgh_chol_threads_leave (lgh_chol_threads_t t)
{
#ifdef LGH_CHOL_HAVE_WEAK
  if (t.openblas > 1 && openblas_set_num_threads != NULL)
    openblas_set_num_threads (t.openblas);
  if (t.mkl >= 0 && MKL_Set_Num_Threads_Local != NULL)
    (void) MKL_Set_Num_Threads_Local (t.mkl);
#else
  (void) t;
#endif
}

/* ------------------------------------------------------------------ */

struct lgh_chol_ctx
{
  MPI_Comm            comm;
  PetscMPIInt         size, rank;
  int                 mode;
  PetscInt            n, nloc, rstart;
  int                *rcounts, *rdispls;   /* row ownership, per rank     */
  cholmod_common      cc;
  int                 cc_started;
  cholmod_factor     *L;
  const int64_t      *perm;                /* L->Perm                     */
  /* single-vector work: two full-length arrays + the solve workspace  */
  double             *xg, *t1;
  double             *ework;
  size_t              ework_len;
  /* blocked work, grown on demand */
  double             *bsend, *brecv, *bfull, *bfull2;
  size_t              bsend_len, brecv_len, bfull_len;
  int                *scnt, *sdsp, *rcnt, *rdsp, *cstart;
  lgh_prior_cholmod_stats_t st;
};

/* ---- full-length kernels (n x k, column-major, leading dimension n) -- */

/* out = P in */
static void
lgh_chol_perm_fwd (const struct lgh_chol_ctx *c, const double *in,
                   double *out, PetscInt k)
{
  const PetscInt      n = c->n;

  for (PetscInt j = 0; j < k; j++) {
    const double       *a = in + (size_t) j * n;
    double             *b = out + (size_t) j * n;
    for (PetscInt r = 0; r < n; r++) b[r] = a[c->perm[r]];
  }
}

/* out = P^T in */
static void
lgh_chol_perm_bwd (const struct lgh_chol_ctx *c, const double *in,
                   double *out, PetscInt k)
{
  const PetscInt      n = c->n;

  for (PetscInt j = 0; j < k; j++) {
    const double       *a = in + (size_t) j * n;
    double             *b = out + (size_t) j * n;
    for (PetscInt r = 0; r < n; r++) b[c->perm[r]] = a[r];
  }
}

/* out = L in (trans = 0) or L^T in (trans = 1).  Supernode s holds columns
 * k1..k2-1 as a dense nsrow x nscol column-major block at Lx + px[s]; its
 * row indices are Ls[pi[s] ..], the first nscol of which are k1..k2-1.
 * Only the lower triangle of the diagonal block is read.               */
static void
lgh_chol_lmul (const struct lgh_chol_ctx *c, int trans, const double *in,
               double *out, PetscInt k)
{
  const cholmod_factor *L = c->L;
  const int64_t      *Super = (const int64_t *) L->super;
  const int64_t      *Lpi = (const int64_t *) L->pi;
  const int64_t      *Lpx = (const int64_t *) L->px;
  const int64_t      *Ls = (const int64_t *) L->s;
  const double       *Lx = (const double *) L->x;
  const PetscInt      n = c->n;
  const int64_t       nsuper = (int64_t) L->nsuper;

  for (PetscInt jc = 0; jc < k; jc++) {
    const double       *x = in + (size_t) jc * n;
    double             *y = out + (size_t) jc * n;

    if (!trans) memset (y, 0, sizeof (double) * (size_t) n);
    for (int64_t s = 0; s < nsuper; s++) {
      const int64_t       k1 = Super[s], k2 = Super[s + 1];
      const int64_t       psi = Lpi[s], psx = Lpx[s];
      const int64_t       nsrow = Lpi[s + 1] - psi, nscol = k2 - k1;

      for (int64_t j = 0; j < nscol; j++) {
        const double       *col = Lx + psx + j * nsrow;
        if (!trans) {
          const double        xj = x[k1 + j];
          for (int64_t i = j; i < nsrow; i++) y[Ls[psi + i]] += col[i] * xj;
        }
        else {
          double              sum = 0.;
          for (int64_t i = j; i < nsrow; i++) sum += col[i] * x[Ls[psi + i]];
          y[k1 + j] = sum;
        }
      }
    }
  }
}

/* in place: X = L^{-1} X (trans = 0) or L^{-T} X (trans = 1) */
static PetscErrorCode
lgh_chol_lsolve (struct lgh_chol_ctx *c, int trans, double *X, PetscInt k)
{
  cholmod_dense       Xd, Ed;
  const size_t        need = (size_t) k * PetscMax ((size_t) c->L->maxesize,
                                                     (size_t) 1);
  lgh_chol_threads_t  th;
  int                 ok;

  if (k == 0 || c->n == 0) return PETSC_SUCCESS;
  if (c->ework_len < need) {
    PetscCall (PetscFree (c->ework));
    PetscCall (PetscMalloc1 (need, &c->ework));
    c->ework_len = need;
  }
  Xd.nrow = (size_t) c->n; Xd.ncol = (size_t) k;
  Xd.nzmax = (size_t) c->n * (size_t) k; Xd.d = (size_t) c->n;
  Xd.x = X; Xd.z = NULL; Xd.xtype = CHOLMOD_REAL; Xd.dtype = CHOLMOD_DOUBLE;
  Ed.nrow = need; Ed.ncol = 1; Ed.nzmax = need; Ed.d = need;
  Ed.x = c->ework; Ed.z = NULL; Ed.xtype = CHOLMOD_REAL;
  Ed.dtype = CHOLMOD_DOUBLE;
  th = lgh_chol_threads_enter ();
  ok = trans ? cholmod_l_super_ltsolve (c->L, &Xd, &Ed, &c->cc)
             : cholmod_l_super_lsolve (c->L, &Xd, &Ed, &c->cc);
  lgh_chol_threads_leave (th);
  PetscCheck (ok && c->cc.status == CHOLMOD_OK, PETSC_COMM_SELF, PETSC_ERR_LIB,
              "lgh_prior_cholmod: CHOLMOD triangular solve failed (status %d)",
              c->cc.status);
  return PETSC_SUCCESS;
}

typedef enum
{
  LGH_CHOL_OP_APPLYZ, LGH_CHOL_OP_APPLYZT, LGH_CHOL_OP_SOLVEZ,
  LGH_CHOL_OP_SOLVEZT
}
lgh_chol_op_t;

/* X (n x k) <- op(X), using W (n x k) as scratch.  Not for A_IS_Z applies
 * (those are distributed MatMults).                                    */
static PetscErrorCode
lgh_chol_full_op (struct lgh_chol_ctx *c, lgh_chol_op_t op, double *X,
                  double *W, PetscInt k)
{
  lgh_chol_perm_fwd (c, X, W, k);                       /* W = P X       */
  switch (op) {
  case LGH_CHOL_OP_APPLYZ:                              /* P^T L P       */
  case LGH_CHOL_OP_APPLYZT:                             /* P^T L^T P     */
    PetscCheck (c->mode == LGH_CHOL_A_IS_R, PETSC_COMM_SELF, PETSC_ERR_PLIB,
                "lgh_prior_cholmod: factor apply outside A_IS_R");
    lgh_chol_lmul (c, op == LGH_CHOL_OP_APPLYZT, W, X, k);
    lgh_chol_perm_bwd (c, X, W, k);
    PetscCall (PetscArraycpy (X, W, (size_t) c->n * (size_t) k));
    return PETSC_SUCCESS;
  case LGH_CHOL_OP_SOLVEZ:
    if (c->mode == LGH_CHOL_A_IS_R) {                   /* P^T L^-1 P    */
      PetscCall (lgh_chol_lsolve (c, 0, W, k));
    }
    else {                                              /* P^T L^-T L^-1 P */
      PetscCall (lgh_chol_lsolve (c, 0, W, k));
      PetscCall (lgh_chol_lsolve (c, 1, W, k));
    }
    break;
  case LGH_CHOL_OP_SOLVEZT:
    if (c->mode == LGH_CHOL_A_IS_R) {                   /* P^T L^-T P    */
      PetscCall (lgh_chol_lsolve (c, 1, W, k));
    }
    else {                                              /* symmetric     */
      PetscCall (lgh_chol_lsolve (c, 0, W, k));
      PetscCall (lgh_chol_lsolve (c, 1, W, k));
    }
    break;
  }
  lgh_chol_perm_bwd (c, W, X, k);                       /* X = P^T W     */
  return PETSC_SUCCESS;
}

/* ---- single-vector path: all-gather, full op, keep the local rows ----- */

static PetscErrorCode
lgh_chol_vec_op (lgh_prior_t *p, lgh_chol_op_t op, Vec x, Vec y)
{
  struct lgh_chol_ctx *c = p->chol;
  const double        t0 = MPI_Wtime ();
  const PetscScalar  *xa;
  PetscScalar        *ya;

  if (c->mode == LGH_CHOL_A_IS_Z
      && (op == LGH_CHOL_OP_APPLYZ || op == LGH_CHOL_OP_APPLYZT)) {
    PetscCall (MatMult (p->Z, x, y));
  }
  else {
    PetscCall (VecGetArrayRead (x, &xa));
    PetscCallMPI (MPI_Allgatherv (xa, (int) c->nloc, MPIU_SCALAR, c->xg,
                                  c->rcounts, c->rdispls, MPIU_SCALAR,
                                  c->comm));
    PetscCall (VecRestoreArrayRead (x, &xa));
    PetscCall (lgh_chol_full_op (c, op, c->xg, c->t1, 1));
    PetscCall (VecGetArrayWrite (y, &ya));
    PetscCall (PetscArraycpy (ya, c->xg + c->rstart, (size_t) c->nloc));
    PetscCall (VecRestoreArrayWrite (y, &ya));
  }
  c->st.t_single += MPI_Wtime () - t0;
  c->st.n_single++;
  return PETSC_SUCCESS;
}

static void
lgh_chol_cb_applyZ (Vec x, Vec y, void *vctx)
{
  lgh_prior_t        *p = (lgh_prior_t *) vctx;
  PetscErrorCode      ierr = lgh_chol_vec_op (p, LGH_CHOL_OP_APPLYZ, x, y);
  CHKERRABORT (p->chol->comm, ierr);
}

static void
lgh_chol_cb_applyZt (Vec x, Vec y, void *vctx)
{
  lgh_prior_t        *p = (lgh_prior_t *) vctx;
  PetscErrorCode      ierr = lgh_chol_vec_op (p, LGH_CHOL_OP_APPLYZT, x, y);
  CHKERRABORT (p->chol->comm, ierr);
}

static void
lgh_chol_cb_solveZ (Vec x, Vec y, void *vctx)
{
  lgh_prior_t        *p = (lgh_prior_t *) vctx;
  PetscErrorCode      ierr = lgh_chol_vec_op (p, LGH_CHOL_OP_SOLVEZ, x, y);
  CHKERRABORT (p->chol->comm, ierr);
}

static void
lgh_chol_cb_solveZt (Vec x, Vec y, void *vctx)
{
  lgh_prior_t        *p = (lgh_prior_t *) vctx;
  PetscErrorCode      ierr = lgh_chol_vec_op (p, LGH_CHOL_OP_SOLVEZT, x, y);
  CHKERRABORT (p->chol->comm, ierr);
}

/* ---- blocked path: whole columns per rank ----------------------------- */

static PetscErrorCode
lgh_chol_grow (double **buf, size_t *len, size_t need)
{
  if (*len >= need) return PETSC_SUCCESS;
  PetscCall (PetscFree (*buf));
  PetscCall (PetscMalloc1 (PetscMax (need, (size_t) 1), buf));
  *len = need;
  return PETSC_SUCCESS;
}

/* Y = op(X) on row-distributed MATDENSE blocks (nloc x ncols each).
 * Columns [cstart[r], cstart[r+1]) go to rank r (balanced contiguous
 * split; with ncols < size some ranks get none).                       */
static PetscErrorCode
lgh_chol_block_op (lgh_prior_t *p, lgh_chol_op_t op, Mat X, Mat Y)
{
  struct lgh_chol_ctx *c = p->chol;
  const double        t0 = MPI_Wtime ();
  double              tc = 0., tc0;
  PetscInt            ncols, nrow_chk, xlda, ylda, mycols, nloc = c->nloc;
  const PetscScalar  *xa;
  PetscScalar        *ya;
  PetscMPIInt         r, size = c->size, rank = c->rank;
  long long           tot;

  PetscCall (MatGetSize (X, &nrow_chk, &ncols));
  PetscCheck (nrow_chk == c->n, c->comm, PETSC_ERR_ARG_SIZ,
              "lgh_prior_cholmod: block has %" PetscInt_FMT " rows, the "
              "factor %" PetscInt_FMT, nrow_chk, c->n);
  for (r = 0; r <= size; r++)
    c->cstart[r] = (int) (((long long) ncols * r) / size);
  mycols = c->cstart[rank + 1] - c->cstart[rank];

  /* counts: to rank r, my local rows of r's columns; from rank q, q's
   * local rows of my columns                                           */
  tot = 0;
  for (r = 0; r < size; r++) {
    const long long     cnt = (long long) nloc * (c->cstart[r + 1] - c->cstart[r]);
    c->scnt[r] = (int) cnt;
    c->sdsp[r] = (int) tot;
    tot += cnt;
  }
  PetscCheck (tot <= INT_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP,
              "lgh_prior_cholmod: block too large for MPI int counts");
  tot = 0;
  for (r = 0; r < size; r++) {
    const long long     cnt = (long long) c->rcounts[r] * mycols;
    c->rcnt[r] = (int) cnt;
    c->rdsp[r] = (int) tot;
    tot += cnt;
  }
  PetscCheck (tot <= INT_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP,
              "lgh_prior_cholmod: block too large for MPI int counts");
  PetscCall (lgh_chol_grow (&c->bsend, &c->bsend_len,
                            (size_t) nloc * (size_t) ncols));
  PetscCall (lgh_chol_grow (&c->brecv, &c->brecv_len,
                            (size_t) c->n * (size_t) mycols));
  if (c->bfull_len < (size_t) c->n * (size_t) mycols) {
    PetscCall (PetscFree (c->bfull));
    PetscCall (PetscFree (c->bfull2));
    c->bfull_len = PetscMax ((size_t) c->n * (size_t) mycols, (size_t) 1);
    PetscCall (PetscMalloc1 (c->bfull_len, &c->bfull));
    PetscCall (PetscMalloc1 (c->bfull_len, &c->bfull2));
  }

  /* pack: local columns in global column order (already grouped by dest) */
  PetscCall (MatDenseGetLDA (X, &xlda));
  PetscCall (MatDenseGetArrayRead (X, &xa));
  for (PetscInt j = 0; j < ncols; j++)
    PetscCall (PetscArraycpy (c->bsend + (size_t) j * nloc,
                              xa + (size_t) j * xlda, (size_t) nloc));
  PetscCall (MatDenseRestoreArrayRead (X, &xa));
  tc0 = MPI_Wtime ();
  PetscCallMPI (MPI_Alltoallv (c->bsend, c->scnt, c->sdsp, MPIU_SCALAR,
                               c->brecv, c->rcnt, c->rdsp, MPIU_SCALAR,
                               c->comm));
  tc += MPI_Wtime () - tc0;
  /* unpack into whole columns: from q, [col][q's rows]                   */
  for (r = 0; r < size; r++)
    for (PetscInt j = 0; j < mycols; j++)
      PetscCall (PetscArraycpy (c->bfull + (size_t) j * c->n + c->rdispls[r],
                                c->brecv + c->rdsp[r]
                                  + (size_t) j * c->rcounts[r],
                                (size_t) c->rcounts[r]));

  if (mycols > 0)
    PetscCall (lgh_chol_full_op (c, op, c->bfull, c->bfull2, mycols));

  /* back: the same exchange reversed                                    */
  for (r = 0; r < size; r++)
    for (PetscInt j = 0; j < mycols; j++)
      PetscCall (PetscArraycpy (c->brecv + c->rdsp[r]
                                  + (size_t) j * c->rcounts[r],
                                c->bfull + (size_t) j * c->n + c->rdispls[r],
                                (size_t) c->rcounts[r]));
  tc0 = MPI_Wtime ();
  PetscCallMPI (MPI_Alltoallv (c->brecv, c->rcnt, c->rdsp, MPIU_SCALAR,
                               c->bsend, c->scnt, c->sdsp, MPIU_SCALAR,
                               c->comm));
  tc += MPI_Wtime () - tc0;
  PetscCall (MatDenseGetLDA (Y, &ylda));
  PetscCall (MatDenseGetArrayWrite (Y, &ya));
  for (PetscInt j = 0; j < ncols; j++)
    PetscCall (PetscArraycpy (ya + (size_t) j * ylda,
                              c->bsend + (size_t) j * nloc, (size_t) nloc));
  PetscCall (MatDenseRestoreArrayWrite (Y, &ya));

  c->st.t_blocked += MPI_Wtime () - t0;
  c->st.t_blocked_comm += tc;
  c->st.n_blocked++;
  c->st.n_blocked_cols += (long) ncols;
  return PETSC_SUCCESS;
}

static void
lgh_chol_cb_solveZ_blocked (Mat X, Mat Y, void *vctx)
{
  lgh_prior_t        *p = (lgh_prior_t *) vctx;
  PetscErrorCode      ierr = lgh_chol_block_op (p, LGH_CHOL_OP_SOLVEZ, X, Y);
  CHKERRABORT (p->chol->comm, ierr);
}

static void
lgh_chol_cb_solveZt_blocked (Mat X, Mat Y, void *vctx)
{
  lgh_prior_t        *p = (lgh_prior_t *) vctx;
  PetscErrorCode      ierr = lgh_chol_block_op (p, LGH_CHOL_OP_SOLVEZT, X, Y);
  CHKERRABORT (p->chol->comm, ierr);
}

/* ---- setup ------------------------------------------------------------ */

lgh_prior_cholmod_opts_t
lgh_prior_cholmod_opts_default (void)
{
  lgh_prior_cholmod_opts_t o;
  o.verbose = 1;
  return o;
}

/* Two deterministic probes: |y'Ax - x'Ay| relative to |x||Ay| + |y||Ax|.
 * CHOLMOD reads one triangle only, so a non-symmetric A would be silently
 * symmetrized; refuse it instead.                                       */
static PetscErrorCode
lgh_chol_check_symmetric (Mat A, double *asym)
{
  Vec                 x, y, Ax, Ay;
  PetscInt            lo, hi;
  PetscScalar        *a, *b, yAx, xAy;
  PetscReal           nx, ny, nAx, nAy;

  PetscCall (MatCreateVecs (A, &x, &Ax));
  PetscCall (VecDuplicate (x, &y));
  PetscCall (VecDuplicate (x, &Ay));
  PetscCall (VecGetOwnershipRange (x, &lo, &hi));
  PetscCall (VecGetArray (x, &a));
  PetscCall (VecGetArray (y, &b));
  for (PetscInt i = lo; i < hi; i++) {
    a[i - lo] = lgh_randn_at (0xC401E5UL, i, 0);
    b[i - lo] = lgh_randn_at (0xC401E5UL, i, 1);
  }
  PetscCall (VecRestoreArray (x, &a));
  PetscCall (VecRestoreArray (y, &b));
  PetscCall (MatMult (A, x, Ax));
  PetscCall (MatMult (A, y, Ay));
  PetscCall (VecDot (Ax, y, &yAx));
  PetscCall (VecDot (Ay, x, &xAy));
  PetscCall (VecNorm (x, NORM_2, &nx));
  PetscCall (VecNorm (y, NORM_2, &ny));
  PetscCall (VecNorm (Ax, NORM_2, &nAx));
  PetscCall (VecNorm (Ay, NORM_2, &nAy));
  *asym = PetscAbsScalar (yAx - xAy) / (double) (nx * nAy + ny * nAx);
  PetscCall (VecDestroy (&x));
  PetscCall (VecDestroy (&y));
  PetscCall (VecDestroy (&Ax));
  PetscCall (VecDestroy (&Ay));
  return PETSC_SUCCESS;
}

/* Gather the upper triangle of every row of A (entries j >= i of row i)
 * onto every rank as the lower-stored CSC matrix CHOLMOD factors: by
 * symmetry, row i's entries j >= i are column i's entries below the
 * diagonal.  Ownership ranges are contiguous and rank-ordered, so the
 * gathered arrays are in global order.                                  */
static PetscErrorCode
lgh_chol_gather (struct lgh_chol_ctx *c, Mat A, cholmod_sparse **Aout)
{
  PetscInt            i, ncols;
  const PetscInt     *cols;
  const PetscScalar  *vals;
  int64_t            *lcnt, *lidx, lnnz = 0, pos, *Ap, *Ai;
  double             *lval, *Ax;
  int                *ncnt, *ndsp, myn;
  long long           tot;
  cholmod_sparse     *As;
  PetscMPIInt         r;

  /* pass 1: counts */
  PetscCall (PetscMalloc1 (PetscMax (c->nloc, 1), &lcnt));
  for (i = 0; i < c->nloc; i++) {
    const PetscInt      gi = c->rstart + i;
    int64_t             k = 0;
    PetscCall (MatGetRow (A, gi, &ncols, &cols, NULL));
    for (PetscInt j = 0; j < ncols; j++) if (cols[j] >= gi) k++;
    PetscCall (MatRestoreRow (A, gi, &ncols, &cols, NULL));
    lcnt[i] = k;
    lnnz += k;
  }
  /* pass 2: indices (sorted within each row) and values */
  PetscCall (PetscMalloc2 (PetscMax (lnnz, 1), &lidx, PetscMax (lnnz, 1), &lval));
  pos = 0;
  for (i = 0; i < c->nloc; i++) {
    const PetscInt      gi = c->rstart + i;
    const int64_t       p0 = pos;
    PetscCall (MatGetRow (A, gi, &ncols, &cols, &vals));
    for (PetscInt j = 0; j < ncols; j++)
      if (cols[j] >= gi) {
        lidx[pos] = (int64_t) cols[j];
        lval[pos] = (double) vals[j];
        pos++;
      }
    PetscCall (MatRestoreRow (A, gi, &ncols, &cols, &vals));
    for (int64_t a = p0 + 1; a < pos; a++) {     /* insertion sort; AIJ rows
                                                    arrive sorted already */
      const int64_t       ki = lidx[a];
      const double        kv = lval[a];
      int64_t             b = a - 1;
      while (b >= p0 && lidx[b] > ki) {
        lidx[b + 1] = lidx[b]; lval[b + 1] = lval[b]; b--;
      }
      lidx[b + 1] = ki; lval[b + 1] = kv;
    }
  }

  PetscCall (PetscMalloc2 (c->size, &ncnt, c->size, &ndsp));
  PetscCheck (lnnz <= INT_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP,
              "lgh_prior_cholmod: local triangle too large for MPI int counts");
  myn = (int) lnnz;
  PetscCallMPI (MPI_Allgather (&myn, 1, MPI_INT, ncnt, 1, MPI_INT, c->comm));
  tot = 0;
  for (r = 0; r < c->size; r++) {
    ndsp[r] = (int) tot;
    tot += ncnt[r];
    PetscCheck (tot <= INT_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP,
                "lgh_prior_cholmod: A has too many entries for MPI int counts");
  }
  As = cholmod_l_allocate_sparse ((size_t) c->n, (size_t) c->n, (size_t) tot,
                                  1, 1, -1, CHOLMOD_REAL + CHOLMOD_DOUBLE,
                                  &c->cc);
  PetscCheck (As != NULL, PETSC_COMM_SELF, PETSC_ERR_MEM,
              "lgh_prior_cholmod: cannot allocate the gathered A (%lld entries)",
              tot);
  Ap = (int64_t *) As->p;
  Ai = (int64_t *) As->i;
  Ax = (double *) As->x;
  /* column counts land in Ap[1..n], then prefix-sum                     */
  PetscCallMPI (MPI_Allgatherv (lcnt, (int) c->nloc, MPI_INT64_T, Ap + 1,
                                c->rcounts, c->rdispls, MPI_INT64_T, c->comm));
  Ap[0] = 0;
  for (PetscInt k = 0; k < c->n; k++) Ap[k + 1] += Ap[k];
  PetscCheck (Ap[c->n] == (int64_t) tot, PETSC_COMM_SELF, PETSC_ERR_PLIB,
              "lgh_prior_cholmod: gathered counts disagree");
  PetscCallMPI (MPI_Allgatherv (lidx, myn, MPI_INT64_T, Ai, ncnt, ndsp,
                                MPI_INT64_T, c->comm));
  PetscCallMPI (MPI_Allgatherv (lval, myn, MPI_DOUBLE, Ax, ncnt, ndsp,
                                MPI_DOUBLE, c->comm));
  PetscCall (PetscFree2 (ncnt, ndsp));
  PetscCall (PetscFree2 (lidx, lval));
  PetscCall (PetscFree (lcnt));
  c->st.nnz_A = (double) tot;
  *Aout = As;
  return PETSC_SUCCESS;
}

int
lgh_prior_create_cholmod (Mat A, Vec mass_lumps, int mode,
                          const lgh_prior_cholmod_opts_t *opts,
                          lgh_prior_t **prior)
{
  lgh_prior_cholmod_opts_t o = (opts != NULL) ? *opts
                                              : lgh_prior_cholmod_opts_default ();
  struct lgh_chol_ctx *c;
  lgh_prior_t        *p;
  cholmod_sparse     *As = NULL;
  PetscInt            M, N, mlo, mhi, vlo, vhi, nloc;
  PetscMPIInt         r;
  double              t0, asym;
  lgh_chol_threads_t  th;
  int                 ok;

  PetscCheck (mode == LGH_CHOL_A_IS_R || mode == LGH_CHOL_A_IS_Z,
              PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
              "lgh_prior_create_cholmod: mode must be LGH_CHOL_A_IS_R or "
              "LGH_CHOL_A_IS_Z (got %d)", mode);
  PetscCall (MatGetSize (A, &M, &N));
  PetscCheck (M == N, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
              "lgh_prior_create_cholmod: A is %" PetscInt_FMT " x %"
              PetscInt_FMT ", not square", M, N);
  PetscCall (MatGetOwnershipRange (A, &mlo, &mhi));
  PetscCall (VecGetOwnershipRange (mass_lumps, &vlo, &vhi));
  PetscCheck (mlo == vlo && mhi == vhi, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
              "lgh_prior_create_cholmod: A's rows [%" PetscInt_FMT ", %"
              PetscInt_FMT ") and mass_lumps' [%" PetscInt_FMT ", %"
              PetscInt_FMT ") differ", mlo, mhi, vlo, vhi);

  PetscCall (PetscNew (&c));
  PetscCall (PetscObjectGetComm ((PetscObject) A, &c->comm));
  PetscCallMPI (MPI_Comm_size (c->comm, &c->size));
  PetscCallMPI (MPI_Comm_rank (c->comm, &c->rank));
  c->mode = mode;
  c->n = N;
  c->rstart = mlo;
  c->nloc = nloc = mhi - mlo;
  c->st.mode = mode;
  c->st.n = (long) N;
  PetscCall (PetscMalloc2 (c->size, &c->rcounts, c->size, &c->rdispls));
  {
    int                 myn = (int) nloc;
    PetscCallMPI (MPI_Allgather (&myn, 1, MPI_INT, c->rcounts, 1, MPI_INT,
                                 c->comm));
    c->rdispls[0] = 0;
    for (r = 1; r < c->size; r++)
      c->rdispls[r] = c->rdispls[r - 1] + c->rcounts[r - 1];
  }
  PetscCall (PetscMalloc4 (c->size, &c->scnt, c->size, &c->sdsp,
                           c->size, &c->rcnt, c->size, &c->rdsp));
  PetscCall (PetscMalloc1 (c->size + 1, &c->cstart));

  PetscCall (lgh_chol_check_symmetric (A, &asym));
  PetscCheck (asym <= 1e-10, c->comm, PETSC_ERR_ARG_WRONG,
              "lgh_prior_create_cholmod: A is not symmetric (probe asymmetry "
              "%.3e)", asym);

  /* CHOLMOD: AMD only (no METIS: deterministic, and the same everywhere),
   * supernodal LL^T, one OpenMP thread.                                  */
  ok = cholmod_l_start (&c->cc);
  PetscCheck (ok, PETSC_COMM_SELF, PETSC_ERR_LIB, "cholmod_l_start failed");
  c->cc_started = 1;
  c->cc.print = 0;
  c->cc.nmethods = 1;
  c->cc.method[0].ordering = CHOLMOD_AMD;
  c->cc.postorder = 1;
  c->cc.supernodal = CHOLMOD_SUPERNODAL;
  c->cc.final_asis = 1;
  c->cc.final_super = 1;
  c->cc.final_ll = 1;
  c->cc.nthreads_max = 1;

  t0 = MPI_Wtime ();
  PetscCall (lgh_chol_gather (c, A, &As));
  c->st.t_gather = MPI_Wtime () - t0;

  th = lgh_chol_threads_enter ();
  t0 = MPI_Wtime ();
  c->L = cholmod_l_analyze (As, &c->cc);
  c->st.t_analyse = MPI_Wtime () - t0;
  if (c->L != NULL && c->cc.status == CHOLMOD_OK) {
    t0 = MPI_Wtime ();
    ok = cholmod_l_factorize (As, c->L, &c->cc);
    c->st.t_factor = MPI_Wtime () - t0;
  }
  else ok = 0;
  lgh_chol_threads_leave (th);
  (void) cholmod_l_free_sparse (&As, &c->cc);   /* the gathered A, now */
  PetscCheck (c->L != NULL, PETSC_COMM_SELF, PETSC_ERR_LIB,
              "lgh_prior_create_cholmod: cholmod_l_analyze failed (status %d)",
              c->cc.status);
  PetscCheck (ok && c->cc.status == CHOLMOD_OK
              && c->L->minor == (size_t) c->n, PETSC_COMM_SELF, PETSC_ERR_LIB,
              "lgh_prior_create_cholmod: factorization failed (status %d, "
              "minor %ld of %" PetscInt_FMT "): A not positive definite?",
              c->cc.status, (long) c->L->minor, c->n);
  /* supernodal => LL^T; a simplicial LDL^T factor would make every
   * half-solve below silently omit D */
  PetscCheck (c->L->is_ll && c->L->is_super && c->L->itype == CHOLMOD_LONG
              && c->L->xtype == CHOLMOD_REAL && c->L->dtype == CHOLMOD_DOUBLE,
              PETSC_COMM_SELF, PETSC_ERR_PLIB,
              "lgh_prior_create_cholmod: expected a supernodal real double "
              "LL^T factor (is_ll %d, is_super %d)", c->L->is_ll,
              c->L->is_super);
  c->perm = (const int64_t *) c->L->Perm;
  c->st.nnz_L = c->cc.lnz;
  c->st.factor_bytes = 8.0 * ((double) c->L->xsize + (double) c->L->ssize
                              + 3.0 * ((double) c->L->nsuper + 1.0)
                              + 2.0 * (double) c->n);
  c->st.peak_bytes = (double) c->cc.memory_usage;

  PetscCall (PetscMalloc2 (PetscMax (c->n, 1), &c->xg, PetscMax (c->n, 1),
                           &c->t1));

  if (o.verbose) {
    double              loc[4] = { c->st.t_gather, c->st.t_analyse,
                                   c->st.t_factor, c->st.peak_bytes };
    double              mx[4];

    PetscCallMPI (MPI_Allreduce (loc, mx, 4, MPI_DOUBLE, MPI_MAX, c->comm));
    PetscCall (PetscPrintf (c->comm,
      "[LGH-PRIOR-CHOL] mode %s  n %" PetscInt_FMT "  nnz(tril A) %.0f  "
      "nnz(L) %.0f (%.1f per row)  supernodes %ld  ordering AMD  LL^T %s\n",
      mode == LGH_CHOL_A_IS_R ? "A_IS_R (Z = P^T L P)" : "A_IS_Z (Z = A)",
      c->n, c->st.nnz_A, c->st.nnz_L,
      c->n > 0 ? c->st.nnz_L / (double) c->n : 0., (long) c->L->nsuper,
      c->L->is_ll ? "yes" : "NO"));
    PetscCall (PetscPrintf (c->comm,
      "[LGH-PRIOR-CHOL] factor %.1f MB per rank (x %d ranks)  CHOLMOD peak "
      "%.1f MB (incl. the gathered A)  gather %.3f s  analyse %.3f s  "
      "factor %.3f s  (max over ranks)  symmetry probe %.1e\n",
      c->st.factor_bytes / 1e6, (int) c->size, mx[3] / 1e6, mx[0], mx[1],
      mx[2], asym));
  }

  /* the prior */
  PetscCall (lgh_prior_init_common (mass_lumps, &p));
  p->chol = c;
  if (mode == LGH_CHOL_A_IS_Z) {
    p->Z = A;                        /* applies are MatMult with A        */
    PetscCall (PetscObjectReference ((PetscObject) A));
    p->cb.applyZ = lgh_chol_cb_applyZ;
    p->cb.solveZ = lgh_chol_cb_solveZ;
    p->cb.solveZ_blocked = lgh_chol_cb_solveZ_blocked;
    /* symmetric: the t-variants fall back */
  }
  else {
    p->cb.applyZ = lgh_chol_cb_applyZ;
    p->cb.applyZt = lgh_chol_cb_applyZt;
    p->cb.solveZ = lgh_chol_cb_solveZ;
    p->cb.solveZt = lgh_chol_cb_solveZt;
    p->cb.solveZ_blocked = lgh_chol_cb_solveZ_blocked;
    p->cb.solveZt_blocked = lgh_chol_cb_solveZt_blocked;
  }
  p->cb.ctx = p;
  *prior = p;
  return PETSC_SUCCESS;
}

int
lgh_prior_cholmod_get_stats (const lgh_prior_t *prior,
                             lgh_prior_cholmod_stats_t *stats)
{
  if (prior == NULL || prior->chol == NULL || stats == NULL) return 1;
  *stats = prior->chol->st;
  return 0;
}

static void
lgh_chol_prior_teardown (lgh_prior_t *p)
{
  struct lgh_chol_ctx *c = p->chol;

  if (c == NULL) return;
  if (c->L != NULL) (void) cholmod_l_free_factor (&c->L, &c->cc);
  if (c->cc_started) (void) cholmod_l_finish (&c->cc);
  (void) PetscFree2 (c->xg, c->t1);
  (void) PetscFree (c->ework);
  (void) PetscFree (c->bsend);
  (void) PetscFree (c->brecv);
  (void) PetscFree (c->bfull);
  (void) PetscFree (c->bfull2);
  (void) PetscFree2 (c->rcounts, c->rdispls);
  (void) PetscFree4 (c->scnt, c->sdsp, c->rcnt, c->rdsp);
  (void) PetscFree (c->cstart);
  (void) PetscFree (c);
  p->chol = NULL;
}

#endif /* LGPSF_HESSIAN_CHOLMOD_IMPL_H */

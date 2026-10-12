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
#include <stdio.h>
#include <stdlib.h>
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
  /* the node-shared factor (A4, 2026-10-12; shared_copies > 0): one copy
   * per copycomm, its leader (copy_rank 0) factors; x and s live in `win`,
   * the small arrays in `small`; L then points at the alias `Lalias`      */
  int                 shared_copies, copies_per_node, socket_colour;
  MPI_Comm            copycomm;
  int                 copy_rank, copy_size;
  MPI_Win             win;
  cholmod_factor     *Lalias;
  int64_t            *small;
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

/* Y = op(X) on row-distributed MATDENSE blocks (nloc x ncols each), ONE
 * exchange: columns [cstart[r], cstart[r+1]) go to rank r (balanced
 * contiguous split; with ncols < size some ranks get none).  Its buffers
 * are sized to the block it is given; lgh_chol_block_op below feeds it
 * column tiles so that they stay bounded.                              */
static PetscErrorCode
lgh_chol_block_op_one (lgh_prior_t *p, lgh_chol_op_t op, Mat X, Mat Y)
{
  struct lgh_chol_ctx *c = p->chol;
  const double        t0 = MPI_Wtime ();
  double              tc = 0., tc0, tl = 0., tl0;
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
  if (lgh_gt_registered) PetscCall (PetscLogEventBegin (lgh_gt_ev[LGH_GT_ZS_EXCH], 0, 0, 0, 0));
  tc0 = MPI_Wtime ();
  PetscCallMPI (MPI_Alltoallv (c->bsend, c->scnt, c->sdsp, MPIU_SCALAR,
                               c->brecv, c->rcnt, c->rdsp, MPIU_SCALAR,
                               c->comm));
  tc += MPI_Wtime () - tc0;
  if (lgh_gt_registered) PetscCall (PetscLogEventEnd (lgh_gt_ev[LGH_GT_ZS_EXCH], 0, 0, 0, 0));
  /* unpack into whole columns: from q, [col][q's rows]                   */
  for (r = 0; r < size; r++)
    for (PetscInt j = 0; j < mycols; j++)
      PetscCall (PetscArraycpy (c->bfull + (size_t) j * c->n + c->rdispls[r],
                                c->brecv + c->rdsp[r]
                                  + (size_t) j * c->rcounts[r],
                                (size_t) c->rcounts[r]));

  if (lgh_gt_registered) PetscCall (PetscLogEventBegin (lgh_gt_ev[LGH_GT_ZS_LOCAL], 0, 0, 0, 0));
  tl0 = MPI_Wtime ();
  if (mycols > 0)
    PetscCall (lgh_chol_full_op (c, op, c->bfull, c->bfull2, mycols));
  tl = MPI_Wtime () - tl0;
  if (lgh_gt_registered) PetscCall (PetscLogEventEnd (lgh_gt_ev[LGH_GT_ZS_LOCAL], 0, 0, 0, 0));

  /* back: the same exchange reversed                                    */
  for (r = 0; r < size; r++)
    for (PetscInt j = 0; j < mycols; j++)
      PetscCall (PetscArraycpy (c->brecv + c->rdsp[r]
                                  + (size_t) j * c->rcounts[r],
                                c->bfull + (size_t) j * c->n + c->rdispls[r],
                                (size_t) c->rcounts[r]));
  if (lgh_gt_registered) PetscCall (PetscLogEventBegin (lgh_gt_ev[LGH_GT_ZS_EXCH], 0, 0, 0, 0));
  tc0 = MPI_Wtime ();
  PetscCallMPI (MPI_Alltoallv (c->brecv, c->rcnt, c->rdsp, MPIU_SCALAR,
                               c->bsend, c->scnt, c->sdsp, MPIU_SCALAR,
                               c->comm));
  tc += MPI_Wtime () - tc0;
  if (lgh_gt_registered) PetscCall (PetscLogEventEnd (lgh_gt_ev[LGH_GT_ZS_EXCH], 0, 0, 0, 0));
  PetscCall (MatDenseGetLDA (Y, &ylda));
  PetscCall (MatDenseGetArrayWrite (Y, &ya));
  for (PetscInt j = 0; j < ncols; j++)
    PetscCall (PetscArraycpy (ya + (size_t) j * ylda,
                              c->bsend + (size_t) j * nloc, (size_t) nloc));
  PetscCall (MatDenseRestoreArrayWrite (Y, &ya));

  {
    const double        tt = MPI_Wtime () - t0;
    c->st.t_blocked += tt;
    c->st.t_blocked_comm += tc;
    c->st.t_blocked_local += tl;
    c->st.t_blocked_copy += tt - tc - tl;   /* pack/unpack + bookkeeping */
    c->st.b_blocked_comm += 2. * sizeof (PetscScalar) * (double) nloc
                            * (double) ncols;   /* both ways, incl. self  */
    c->st.n_blocked_exch += 2;
  }
  return PETSC_SUCCESS;
}

/* Y = op(X) in column TILES of at most `tile` columns, one exchange each.
 * Without tiling every rank held four buffers sized to the whole block
 * (nloc x ncols to send, three of n x ncols/size for the whole columns):
 * 1.38 GB per rank at ncols = 20200 on 192 ranks, an out-of-memory in the
 * GLR build (2026-09-28).  Tiles bound them by nloc x tile + 3 n x
 * tile/size.  Every column gets the same solve; only the number of columns
 * CHOLMOD handles together changes, so results agree to rounding (the test
 * checks blocked == column-by-column to 1e-12, also with forced tiles).  tile = max(LGH_CHOL_BLOCK_TILE_MIN, 8 x size)
 * (at least 8 columns per rank per tile); the environment variable
 * LGH_CHOL_BLOCK_TILE overrides it (tests force small tiles with it).   */
#ifndef LGH_CHOL_BLOCK_TILE_MIN
#define LGH_CHOL_BLOCK_TILE_MIN 2048
#endif
static PetscErrorCode
lgh_chol_block_op (lgh_prior_t *p, lgh_chol_op_t op, Mat X, Mat Y)
{
  struct lgh_chol_ctx *c = p->chol;
  PetscInt            ncols, tile, c0;
  const char         *env = getenv ("LGH_CHOL_BLOCK_TILE");

  PetscCall (MatGetSize (X, NULL, &ncols));
  tile = PetscMax ((PetscInt) LGH_CHOL_BLOCK_TILE_MIN, (PetscInt) 8 * (PetscInt) c->size);
  if (env != NULL && atoi (env) > 0) tile = (PetscInt) atoi (env);
  if (ncols <= tile) PetscCall (lgh_chol_block_op_one (p, op, X, Y));
  else {
    for (c0 = 0; c0 < ncols; c0 += tile) {
      const PetscInt      c1 = PetscMin (ncols, c0 + tile);
      Mat                 Xs, Ys;
      PetscCall (MatDenseGetSubMatrix (X, PETSC_DECIDE, PETSC_DECIDE, c0, c1, &Xs));
      PetscCall (MatDenseGetSubMatrix (Y, PETSC_DECIDE, PETSC_DECIDE, c0, c1, &Ys));
      PetscCall (lgh_chol_block_op_one (p, op, Xs, Ys));
      PetscCall (MatDenseRestoreSubMatrix (Y, &Ys));
      PetscCall (MatDenseRestoreSubMatrix (X, &Xs));
    }
  }
  c->st.n_blocked++;                       /* per caller's call, not per tile */
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

/* ---- the node-shared factor (GLR-MEMORY-REDESIGN A4, 2026-10-12) ------
 * With opts->shared_copies > 0 the ranks of one node (1) or one socket (2)
 * share ONE factor.  The copy's leader factors; the two large arrays of the
 * supernodal factor, x (xsize doubles) and s (ssize int64), are copied once
 * into an MPI shared-memory window; the small arrays (super, pi, px: nsuper
 * + 1 each; Perm: n) are broadcast and kept per rank; every rank then reads
 * the factor through a cholmod_factor struct of its own that aliases the
 * window.  CHOLMOD's supernodal solves and lgh_chol_lmul only READ the
 * factor (SuiteSparse 7.9: t_cholmod_super_solve_worker reads n, nsuper,
 * super, pi, px, s, x, maxesize, dtype; the wrapper xtype, is_super, is_ll),
 * so every solve stays bitwise the same.  The alias is calloc'ed and freed
 * by hand, never through cholmod_l_free_factor.  Saves (ranks per copy - 1)
 * x (x + s bytes) per copy -- 27 GB per node at albmap_40 with 48 ranks --
 * and CHOLMOD's factorization workspace on every rank but the leaders. */

/* the socket this process runs on: /proc/self/stat field 39 (the cpu) and
 * that cpu's topology/physical_package_id; -1 when either is unreadable  */
static int
lgh_chol_cpu_package (void)
{
  FILE               *f = fopen ("/proc/self/stat", "r");
  char                buf[1024];
  int                 cpu = -1, pkg = -1;

  if (f == NULL) return -1;
  if (fgets (buf, sizeof (buf), f) != NULL) {
    const char         *q = strrchr (buf, ')');   /* the comm field may hold spaces */
    int                 field = 2;
    if (q != NULL) {
      q++;
      while (*q != '\0' && field < 39) { if (*q == ' ') field++; q++; }
      if (field == 39) cpu = atoi (q);
    }
  }
  fclose (f);
  if (cpu < 0) return -1;
  snprintf (buf, sizeof (buf), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
  f = fopen (buf, "r");
  if (f == NULL) return -1;
  if (fscanf (f, "%d", &pkg) != 1) pkg = -1;
  fclose (f);
  return pkg;
}

/* the communicator of this rank's copy; copies_per_node for the record.
 * LGH_CHOL_FORCE_SLICES=k (tests) slices every node into k rank ranges
 * instead of reading the sockets.                                        */
static PetscErrorCode
lgh_chol_copycomm (struct lgh_chol_ctx *c)
{
  MPI_Comm            nodecomm;
  int                 nrank, nsize, colour = 0, cmin, cmax;
  const char         *env = getenv ("LGH_CHOL_FORCE_SLICES");
  const int           slices = (env != NULL && atoi (env) > 0) ? atoi (env) : 0;

  PetscCallMPI (MPI_Comm_split_type (c->comm, MPI_COMM_TYPE_SHARED, c->rank,
                                     MPI_INFO_NULL, &nodecomm));
  PetscCallMPI (MPI_Comm_rank (nodecomm, &nrank));
  PetscCallMPI (MPI_Comm_size (nodecomm, &nsize));
  c->socket_colour = 0;
  if (slices > 0) {
    colour = (int) (((long) nrank * slices) / nsize);
  }
  else if (c->shared_copies >= 2) {
    int                 pkg = lgh_chol_cpu_package (), pmin;
    PetscCallMPI (MPI_Allreduce (&pkg, &pmin, 1, MPI_INT, MPI_MIN, nodecomm));
    if (pmin >= 0) { colour = pkg; c->socket_colour = 1; }
    else colour = (int) (((long) nrank * c->shared_copies) / nsize);   /* fallback: rank ranges */
  }
  PetscCallMPI (MPI_Allreduce (&colour, &cmin, 1, MPI_INT, MPI_MIN, nodecomm));
  PetscCallMPI (MPI_Allreduce (&colour, &cmax, 1, MPI_INT, MPI_MAX, nodecomm));
  c->copies_per_node = cmax - cmin + 1;
  if (c->copies_per_node == 1) {
    c->copycomm = nodecomm;
  }
  else {
    PetscCallMPI (MPI_Comm_split (nodecomm, colour, c->rank, &c->copycomm));
    PetscCallMPI (MPI_Comm_free (&nodecomm));
  }
  PetscCallMPI (MPI_Comm_rank (c->copycomm, &c->copy_rank));
  PetscCallMPI (MPI_Comm_size (c->copycomm, &c->copy_size));
  return PETSC_SUCCESS;
}

/* after the leader factored (c->L on the leader, NULL elsewhere; `ok` the
 * leader's factorize return): validate collectively, publish x and s in
 * the window, broadcast the small arrays and the struct, build the alias */
static PetscErrorCode
lgh_chol_share_factor (struct lgh_chol_ctx *c, int ok)
{
  cholmod_factor      hdr;
  int                 flags[2] = { 0, 0 };     /* [0] valid, [1] cc.status */
  double              st3[2] = { 0., 0. };     /* lnz, peak                */
  size_t              xbytes, sbytes, off_s, nsup1, nsmall;
  MPI_Aint            total, qsz;
  int                 disp;
  char               *base = NULL;
  int64_t            *small;

  memset (&hdr, 0, sizeof (hdr));
  if (c->copy_rank == 0) {
    flags[1] = (int) c->cc.status;
    flags[0] = (ok && c->L != NULL && c->cc.status == CHOLMOD_OK
                && c->L->minor == (size_t) c->n && c->L->is_ll && c->L->is_super
                && c->L->itype == CHOLMOD_LONG && c->L->xtype == CHOLMOD_REAL
                && c->L->dtype == CHOLMOD_DOUBLE
                && c->L->p == NULL && c->L->i == NULL && c->L->z == NULL) ? 1 : 0;
    if (c->L != NULL) hdr = *c->L;
    st3[0] = c->cc.lnz;
    st3[1] = (double) c->cc.memory_usage;
  }
  PetscCallMPI (MPI_Bcast (flags, 2, MPI_INT, 0, c->copycomm));
  PetscCallMPI (MPI_Bcast (&hdr, (int) sizeof (hdr), MPI_BYTE, 0, c->copycomm));
  PetscCallMPI (MPI_Bcast (st3, 2, MPI_DOUBLE, 0, c->copycomm));
  PetscCheck (flags[0], c->copycomm, PETSC_ERR_LIB,
              "lgh_prior_create_cholmod: the copy leader's factorization failed or is not a "
              "supernodal real double LL^T (CHOLMOD status %d, minor %ld of %" PetscInt_FMT
              "): A not positive definite?", flags[1], (long) hdr.minor, c->n);
  c->st.nnz_L = st3[0];
  c->st.peak_bytes = (c->copy_rank == 0) ? st3[1] : 0.;

  /* the window: x, then s at a 64-byte boundary */
  xbytes = hdr.xsize * sizeof (double);
  sbytes = hdr.ssize * sizeof (int64_t);
  off_s = (xbytes + 63) & ~((size_t) 63);
  total = (MPI_Aint) (off_s + sbytes);
  PetscCallMPI (MPI_Win_allocate_shared (c->copy_rank == 0 ? total : 0, 1, MPI_INFO_NULL,
                                         c->copycomm, &base, &c->win));
  PetscCallMPI (MPI_Win_shared_query (c->win, 0, &qsz, &disp, &base));
  PetscCheck ((size_t) qsz >= off_s + sbytes, c->copycomm, PETSC_ERR_LIB,
              "lgh_prior_create_cholmod: shared window smaller than the factor");
  PetscCallMPI (MPI_Win_lock_all (MPI_MODE_NOCHECK, c->win));
  if (c->copy_rank == 0) {
    memcpy (base, c->L->x, xbytes);
    memcpy (base + off_s, c->L->s, sbytes);
  }
  PetscCallMPI (MPI_Win_sync (c->win));
  PetscCallMPI (MPI_Barrier (c->copycomm));
  PetscCallMPI (MPI_Win_sync (c->win));

  /* the small arrays, one per rank: super | pi | px | Perm */
  nsup1 = hdr.nsuper + 1;
  nsmall = 3 * nsup1 + (size_t) c->n;
  PetscCheck (nsmall <= (size_t) INT_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP,
              "lgh_prior_create_cholmod: factor index arrays too large for one broadcast");
  PetscCall (PetscMalloc1 (nsmall, &small));
  if (c->copy_rank == 0) {
    memcpy (small, c->L->super, nsup1 * sizeof (int64_t));
    memcpy (small + nsup1, c->L->pi, nsup1 * sizeof (int64_t));
    memcpy (small + 2 * nsup1, c->L->px, nsup1 * sizeof (int64_t));
    memcpy (small + 3 * nsup1, c->L->Perm, (size_t) c->n * sizeof (int64_t));
  }
  PetscCallMPI (MPI_Bcast (small, (int) nsmall, MPI_INT64_T, 0, c->copycomm));

  /* the alias: the leader's struct with the pointers re-aimed */
  c->Lalias = (cholmod_factor *) calloc (1, sizeof (cholmod_factor));
  PetscCheck (c->Lalias != NULL, PETSC_COMM_SELF, PETSC_ERR_MEM, "calloc failed");
  *c->Lalias = hdr;
  c->Lalias->x = base;
  c->Lalias->s = base + off_s;
  c->Lalias->super = small;
  c->Lalias->pi = small + nsup1;
  c->Lalias->px = small + 2 * nsup1;
  c->Lalias->Perm = small + 3 * nsup1;
  c->Lalias->ColCount = NULL;   /* read by no solve path */
  c->Lalias->IPerm = NULL;
  c->Lalias->p = NULL; c->Lalias->i = NULL; c->Lalias->z = NULL;
  c->Lalias->nz = NULL; c->Lalias->next = NULL; c->Lalias->prev = NULL;
  c->small = small;
  if (c->copy_rank == 0) (void) cholmod_l_free_factor (&c->L, &c->cc);   /* CHOLMOD's own copy */
  c->L = c->Lalias;
  c->perm = (const int64_t *) c->Lalias->Perm;
  c->st.factor_bytes = (double) (xbytes + sbytes) + 8.0 * (double) nsmall;
  c->st.private_bytes = 8.0 * (double) nsmall;
  return PETSC_SUCCESS;
}

lgh_prior_cholmod_opts_t
lgh_prior_cholmod_opts_default (void)
{
  lgh_prior_cholmod_opts_t o;
  o.verbose = 1;
  o.shared_copies = 2;        /* one factor per socket (Nick, 2026-10-12); 0 = the replicated factor */
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
  PetscCheck (o.shared_copies >= 0 && o.shared_copies <= 2, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
              "lgh_prior_create_cholmod: shared_copies must be 0, 1 or 2 (got %d)", o.shared_copies);
  c->shared_copies = o.shared_copies;
  c->copycomm = MPI_COMM_NULL; c->win = MPI_WIN_NULL; c->Lalias = NULL; c->small = NULL;
  c->copy_rank = 0; c->copy_size = 1; c->copies_per_node = 1; c->socket_colour = 0;
  c->st.shared_copies = o.shared_copies;
  if (o.shared_copies > 0) PetscCall (lgh_chol_copycomm (c));
  c->st.copies_per_node = (o.shared_copies > 0) ? c->copies_per_node : 1;
  c->st.copy_ranks = (o.shared_copies > 0) ? c->copy_size : 1;
  c->st.private_bytes = 0.;

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

  if (c->shared_copies == 0 || c->copy_rank == 0) {    /* every rank, or the copy's leader */
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
  }
  else { c->L = NULL; ok = 1; }
  (void) cholmod_l_free_sparse (&As, &c->cc);   /* the gathered A, now */
  if (c->shared_copies > 0) {
    PetscCall (lgh_chol_share_factor (c, ok));    /* validates collectively; sets L, perm, stats */
  }
  else {
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
  }

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
    if (c->shared_copies == 0) {
      PetscCall (PetscPrintf (c->comm,
        "[LGH-PRIOR-CHOL] factor %.1f MB per rank (x %d ranks)  CHOLMOD peak "
        "%.1f MB (incl. the gathered A)  gather %.3f s  analyse %.3f s  "
        "factor %.3f s  (max over ranks)  symmetry probe %.1e\n",
        c->st.factor_bytes / 1e6, (int) c->size, mx[3] / 1e6, mx[0], mx[1],
        mx[2], asym));
    }
    else {
      int                 loc2[2] = { c->copies_per_node, c->socket_colour }, mx2[2];
      PetscCallMPI (MPI_Allreduce (loc2, mx2, 2, MPI_INT, MPI_MAX, c->comm));
      PetscCall (PetscPrintf (c->comm,
        "[LGH-PRIOR-CHOL] factor %.1f MB x %d copies per node (shared over %d ranks each, %s; "
        "%.1f MB per rank besides)  CHOLMOD peak %.1f MB on the leaders (incl. the gathered A)  "
        "gather %.3f s  analyse %.3f s  factor %.3f s  (max over ranks)  symmetry probe %.1e\n",
        c->st.factor_bytes / 1e6, mx2[0], (int) c->copy_size,
        mx2[1] ? "sockets from the cpu topology"
               : (getenv ("LGH_CHOL_FORCE_SLICES") ? "rank slices (forced)" : "rank slices"),
        c->st.private_bytes / 1e6, mx[3] / 1e6, mx[0], mx[1], mx[2], asym));
    }
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
  if (c->Lalias != NULL) {             /* the shared factor: by hand, never cholmod_l_free_factor */
    free (c->Lalias);
    c->Lalias = NULL;
    c->L = NULL;
    (void) PetscFree (c->small);
    if (c->win != MPI_WIN_NULL) {
      (void) MPI_Win_unlock_all (c->win);
      (void) MPI_Win_free (&c->win);
    }
  }
  if (c->L != NULL) (void) cholmod_l_free_factor (&c->L, &c->cc);
  if (c->copycomm != MPI_COMM_NULL) (void) MPI_Comm_free (&c->copycomm);
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

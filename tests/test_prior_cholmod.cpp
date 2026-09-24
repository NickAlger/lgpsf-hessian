/* test_prior_cholmod.cpp — gates for lgh_prior_create_cholmod (the CHOLMOD
 * prior), runnable at any communicator size (registered at n = 1, 2, 4).
 *
 * A = 2D 5-point Laplacian + a varying positive diagonal shift on a
 * GRID_M x GRID_M grid (SPD; its AMD ordering is not an involution, so every
 * test below is sensitive to the direction of the permutation).  Checks,
 * on two row layouts (PETSc's default, and an uneven one where rank 1 owns
 * no rows):
 *
 *   A_IS_R (Z = P^T L P, M = I/gamma):
 *     - the factor is supernodal LL^T;
 *     - the permutation convention is pinned against CHOLMOD itself
 *       (our P x == cholmod_solve(CHOLMOD_P)) and P A P^T = L L^T holds in
 *       our convention (L (L^T (P x)) == P (A x));
 *     - R_lib = Z M^{-1} Z^T == gamma A on random vectors (rel 1e-12);
 *     - Z^{-1} Z = I, Z Z^{-1} = I, Z^{-T} Z^T = I, Z^{-T} Z^{-1} A = I;
 *     - blocked solves (Z^{-1} and Z^{-T}) == column-by-column, for block
 *       widths 1, 2, 3, 5, 9 (widths below the rank count leave ranks idle).
 *   A_IS_Z (Z = A):
 *     - solves == the Mat path's solves (tight KSP), single and blocked;
 *     - R_lib identical to the Mat path's; Z^{-1} Z = I; blocked ==
 *       column-by-column.
 *   The Z/Z^T mix-up guard (both GLR backends): R = A M_L^{-1} A.  Build 1:
 *     symmetric Z = A, M = M_L (A_IS_Z).  Build 2: the NON-symmetric
 *     Cholesky factor of R itself (A_IS_R on R/gamma with M = I/gamma).
 *     The two whitenings differ by an orthogonal factor (S2 = S1 Q), so
 *     every invariant must agree between the builds and with dense
 *     oracles: the spectrum (generalized eigenvalues of (B, R)),
 *     lgh_glr_solve vs a dense (B + c R)^{-1}, lgh_glr_apply, the four
 *     lgh_glr_factor actions through their invariants (G G^T = H,
 *     G^{-T} G^{-1} = H^{-1}, |G^T v|^2 = v'Hv, |G^{-1} v|^2 = v'H^{-1}v,
 *     G^{-1} G = I, G^T G^{-T} = I; the raw outputs legitimately differ by
 *     Q), and lgh_glr_correct_probes (report, corrected spectrum, corrected
 *     apply/solve vs dense (H_d + c R)^{+-1}, factor invariants again).
 */

#include <lgpsf_hessian/lgpsf_hessian.h>
#include <lgpsf_hessian/impl.hpp>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define GRID_M   14
#define N_GLOBAL (GRID_M * GRID_M)
#define RANK_B   8
#define RANK_C   3

static int          n_fail = 0;
static const double GAMMA = 2.5;

static void
check (int ok, const char *what, double val)
{
  if (!ok) {
    n_fail++;
    PetscPrintf (PETSC_COMM_WORLD, "FAIL: %s (%.3e)\n", what, val);
  }
}

/* deterministic, partition-independent problem data ------------------- */

static double
a_entry (int i, int j)
{
  const int           ix = i % GRID_M, iy = i / GRID_M;
  const int           jx = j % GRID_M, jy = j / GRID_M;

  if (i == j) return 4.3 + 0.05 * (i % 5);
  if ((iy == jy && abs (ix - jx) == 1) || (ix == jx && abs (iy - jy) == 1))
    return -1.0;
  return 0.0;
}

static double
m_entry (int i) { return 0.5 + 0.1 * (i % 7); }

static double
factor_entry (int k, int i)
{
  return sin (0.1 * (i + 1) * (k + 3)) + 0.2 * cos (0.37 * i * k + k);
}

static double
b_entry (int i, int j)
{
  double              s = 0.;
  for (int k = 0; k < RANK_B; k++)
    s += 2.0 / (k + 1.0) * factor_entry (k, i) * factor_entry (k, j);
  return s;
}

static double
perr_entry (int k, int i)
{
  return cos (0.21 * (i + 2) * (k + 1)) + 0.3 * sin (0.4 * i - k);
}

static double
hd_entry (int i, int j)           /* B + a small signed rank-3 error */
{
  double              s = b_entry (i, j);
  for (int k = 0; k < RANK_C; k++)
    s += 1e-2 / (k + 1.0) * ((k % 2) ? -1.0 : 1.0) * perr_entry (k, i)
      * perr_entry (k, j);
  return s;
}

/* layouts: 0 = PETSC_DECIDE; 1 = uneven, rank 1 owns nothing (n > 1)   */
static PetscInt
local_rows (int layout)
{
  PetscMPIInt         size, rank;
  PetscInt            nonempty, base;

  MPI_Comm_size (PETSC_COMM_WORLD, &size);
  MPI_Comm_rank (PETSC_COMM_WORLD, &rank);
  if (layout == 0 || size == 1) {
    PetscInt            n = PETSC_DECIDE, Ng = N_GLOBAL;
    PetscSplitOwnership (PETSC_COMM_WORLD, &n, &Ng);
    return n;
  }
  if (rank == 1) return 0;
  nonempty = size - 1;
  base = N_GLOBAL / nonempty;
  /* the last rank takes the remainder */
  return (rank == size - 1) ? N_GLOBAL - base * (nonempty - 1) : base;
}

static PetscErrorCode
build_A (PetscInt mloc, Mat *A_out)
{
  Mat                 A;
  PetscInt            rs, re;

  PetscCall (MatCreateAIJ (PETSC_COMM_WORLD, mloc, mloc, N_GLOBAL, N_GLOBAL,
                           5, NULL, 4, NULL, &A));
  PetscCall (MatGetOwnershipRange (A, &rs, &re));
  for (PetscInt i = rs; i < re; i++)
    for (PetscInt j = PetscMax (0, i - GRID_M);
         j <= PetscMin (N_GLOBAL - 1, i + GRID_M); j++) {
      const double        a = a_entry ((int) i, (int) j);
      if (a != 0.0) PetscCall (MatSetValue (A, i, j, a, INSERT_VALUES));
    }
  PetscCall (MatAssemblyBegin (A, MAT_FINAL_ASSEMBLY));
  PetscCall (MatAssemblyEnd (A, MAT_FINAL_ASSEMBLY));
  *A_out = A;
  return PETSC_SUCCESS;
}

static PetscErrorCode
fill_randn (Vec v, unsigned long seed, int col)
{
  PetscInt            rs, re;
  PetscScalar        *a;

  PetscCall (VecGetOwnershipRange (v, &rs, &re));
  PetscCall (VecGetArray (v, &a));
  for (PetscInt i = rs; i < re; i++) a[i - rs] = lgh_randn_at (seed, i, col);
  PetscCall (VecRestoreArray (v, &a));
  return PETSC_SUCCESS;
}

static PetscErrorCode
fill_fn (Vec v, double (*f) (int))
{
  PetscInt            rs, re;
  PetscScalar        *a;

  PetscCall (VecGetOwnershipRange (v, &rs, &re));
  PetscCall (VecGetArray (v, &a));
  for (PetscInt i = rs; i < re; i++) a[i - rs] = f ((int) i);
  PetscCall (VecRestoreArray (v, &a));
  return PETSC_SUCCESS;
}

static PetscErrorCode
rel_diff (Vec a, Vec b, double *out)
{
  Vec                 d;
  PetscReal           nd, nb;

  PetscCall (VecDuplicate (a, &d));
  PetscCall (VecCopy (a, d));
  PetscCall (VecAXPY (d, -1.0, b));
  PetscCall (VecNorm (d, NORM_2, &nd));
  PetscCall (VecNorm (b, NORM_2, &nb));
  PetscCall (VecDestroy (&d));
  *out = (double) (nd / nb);
  return PETSC_SUCCESS;
}

/* full copy of a distributed Vec on every rank */
static PetscErrorCode
gather_full (Vec v, double *full)
{
  Vec                 seq;
  VecScatter          sc;
  const PetscScalar  *a;

  PetscCall (VecScatterCreateToAll (v, &sc, &seq));
  PetscCall (VecScatterBegin (sc, v, seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall (VecScatterEnd (sc, v, seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall (VecGetArrayRead (seq, &a));
  for (int i = 0; i < N_GLOBAL; i++) full[i] = a[i];
  PetscCall (VecRestoreArrayRead (seq, &a));
  PetscCall (VecScatterDestroy (&sc));
  PetscCall (VecDestroy (&seq));
  return PETSC_SUCCESS;
}

static PetscErrorCode
set_from_full (Vec v, const double *full)
{
  PetscInt            rs, re;
  PetscScalar        *a;

  PetscCall (VecGetOwnershipRange (v, &rs, &re));
  PetscCall (VecGetArray (v, &a));
  for (PetscInt i = rs; i < re; i++) a[i - rs] = full[i];
  PetscCall (VecRestoreArray (v, &a));
  return PETSC_SUCCESS;
}

static double
full_rel (const double *a, const double *b, int n)
{
  double              nd = 0., nb = 0.;
  for (int i = 0; i < n; i++) {
    nd += (a[i] - b[i]) * (a[i] - b[i]);
    nb += b[i] * b[i];
  }
  return sqrt (nd / nb);
}

/* ---- dense oracle helpers (column-major N x N, redundant per rank) --- */

/* H <- chol(H) lower in place; returns info */
static int
dense_chol (double *H)
{
  int                 n = N_GLOBAL, info = 0;
  LGH_LAPACK_DPOTRF ("L", &n, H, &n, &info);
  return info;
}

/* x <- H^{-1} x with C = chol(H) lower */
static void
dense_chol_solve (const double *C, double *x)
{
  int                 n = N_GLOBAL, one = 1;
  double              alpha = 1.0;
  LGH_BLAS_DTRSM ("L", "L", "N", "N", &n, &one, &alpha, C, &n, x, &n);
  LGH_BLAS_DTRSM ("L", "L", "T", "N", &n, &one, &alpha, C, &n, x, &n);
}

static void
dense_matvec (const double *H, const double *x, double *y)
{
  for (int i = 0; i < N_GLOBAL; i++) {
    double              s = 0.;
    for (int j = 0; j < N_GLOBAL; j++) s += H[i + j * N_GLOBAL] * x[j];
    y[i] = s;
  }
}

/* R = A M^{-1} A, dense */
static void
dense_R (double *R)
{
  for (int j = 0; j < N_GLOBAL; j++)
    for (int i = 0; i < N_GLOBAL; i++) {
      double              s = 0.;
      for (int k = PetscMax (0, i - GRID_M);
           k <= PetscMin (N_GLOBAL - 1, i + GRID_M); k++)
        s += a_entry (i, k) * a_entry (k, j) / m_entry (k);
      R[i + j * N_GLOBAL] = s;
    }
}

/* signed generalized eigenvalues of (K, R), descending by value */
static int
dense_gen_eigs (double (*kfn) (int, int), const double *R, double *w)
{
  static double       C[N_GLOBAL * N_GLOBAL], F[N_GLOBAL * N_GLOBAL];
  int                 n = N_GLOBAL, info = 0, lwork = -1, liwork = -1;
  int                 iwkopt = 0;
  double              wkopt, alpha = 1.0;

  for (int i = 0; i < n * n; i++) C[i] = R[i];
  if (dense_chol (C) != 0) return 1;
  for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++) F[i + j * n] = kfn (i, j);
  /* F = C^{-1} K C^{-T} */
  LGH_BLAS_DTRSM ("L", "L", "N", "N", &n, &n, &alpha, C, &n, F, &n);
  LGH_BLAS_DTRSM ("R", "L", "T", "N", &n, &n, &alpha, C, &n, F, &n);
  for (int j = 0; j < n; j++)            /* symmetrize round-off */
    for (int i = j + 1; i < n; i++) {
      double              s = 0.5 * (F[i + j * n] + F[j + i * n]);
      F[i + j * n] = F[j + i * n] = s;
    }
  LGH_LAPACK_DSYEVD ("N", "U", &n, F, &n, w, &wkopt, &lwork, &iwkopt,
                     &liwork, &info);
  lwork = (int) wkopt; liwork = iwkopt;
  {
    double             *work = (double *) malloc (sizeof (double) * lwork);
    int                *iwork = (int *) malloc (sizeof (int) * liwork);
    LGH_LAPACK_DSYEVD ("N", "U", &n, F, &n, w, work, &lwork, iwork,
                       &liwork, &info);
    free (work); free (iwork);
  }
  /* dsyevd: ascending -> descending */
  for (int i = 0; i < n / 2; i++) {
    double              t = w[i]; w[i] = w[n - 1 - i]; w[n - 1 - i] = t;
  }
  return info;
}

static void
sort_desc (double *a, int n)
{
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
      if (a[j] > a[i]) { double t = a[i]; a[i] = a[j]; a[j] = t; }
}

/* ---- block helpers ----------------------------------------------------- */

static PetscErrorCode
make_block (Vec like, PetscInt ncols, unsigned long seed, Mat *X)
{
  PetscInt            nloc, rs, re, lda;
  PetscScalar        *xa;

  PetscCall (VecGetLocalSize (like, &nloc));
  PetscCall (VecGetOwnershipRange (like, &rs, &re));
  PetscCall (MatCreateDense (PETSC_COMM_WORLD, nloc, PETSC_DECIDE, N_GLOBAL,
                             ncols, NULL, X));
  PetscCall (MatDenseGetLDA (*X, &lda));
  PetscCall (MatDenseGetArrayWrite (*X, &xa));
  for (PetscInt j = 0; j < ncols; j++)
    for (PetscInt i = rs; i < re; i++)
      xa[(i - rs) + j * lda] = lgh_randn_at (seed, i, (int) j)
        + (j == 1 ? 3.0 : 0.0);           /* one column with a mean */
  PetscCall (MatDenseRestoreArrayWrite (*X, &xa));
  PetscCall (MatAssemblyBegin (*X, MAT_FINAL_ASSEMBLY));
  PetscCall (MatAssemblyEnd (*X, MAT_FINAL_ASSEMBLY));
  return PETSC_SUCCESS;
}

/* worst column-relative difference between the blocked solve of p and the
 * Vec-wise solve of q applied column by column                          */
static PetscErrorCode
blocked_vs_columns (lgh_prior_t *p, lgh_prior_t *q,
                    lgh_prior_block_which_t which, Vec like, PetscInt ncols,
                    double *worst)
{
  Mat                 X, Y;
  Vec                 ref;

  PetscCall (make_block (like, ncols, 0xB10CUL + (unsigned long) ncols, &X));
  PetscCall (MatDuplicate (X, MAT_DO_NOT_COPY_VALUES, &Y));
  PetscCall (VecDuplicate (like, &ref));
  PetscCall (lgh_prior_solve_block (p, which, X, Y));
  *worst = 0.;
  for (PetscInt j = 0; j < ncols; j++) {
    Vec                 xj, yj;
    double              rd;
    PetscCall (MatDenseGetColumnVecRead (X, j, &xj));
    if (which == LGH_PRIOR_SOLVEZ) PetscCall (lgh_prior_solveZ_vec (q, xj, ref));
    else PetscCall (lgh_prior_solveZt_vec (q, xj, ref));
    PetscCall (MatDenseRestoreColumnVecRead (X, j, &xj));
    PetscCall (MatDenseGetColumnVecRead (Y, j, &yj));
    PetscCall (rel_diff (yj, ref, &rd));
    PetscCall (MatDenseRestoreColumnVecRead (Y, j, &yj));
    *worst = fmax (*worst, rd);
  }
  PetscCall (VecDestroy (&ref));
  PetscCall (MatDestroy (&X));
  PetscCall (MatDestroy (&Y));
  return PETSC_SUCCESS;
}

/* ======================================================================= */
/* part 1: A_IS_R and A_IS_Z at the prior level                             */

static PetscErrorCode
run_prior_level (int layout)
{
  Mat                 A;
  Vec                 minv_gamma, mass, v, w, x, y, Av;
  lgh_prior_t        *pr, *pz, *pm;
  lgh_prior_cholmod_opts_t co = lgh_prior_cholmod_opts_default ();
  lgh_prior_mat_opts_t mo = lgh_prior_mat_opts_default ();
  char                what[160];
  double              rd;
  const int           widths[5] = { 1, 2, 3, 5, 9 };

  PetscCall (PetscPrintf (PETSC_COMM_WORLD, "-- layout %s\n",
                          layout == 0 ? "default" : "uneven (rank 1 empty)"));
  PetscCall (build_A (local_rows (layout), &A));
  PetscCall (MatCreateVecs (A, &mass, NULL));
  PetscCall (VecDuplicate (mass, &minv_gamma));
  PetscCall (VecDuplicate (mass, &v));
  PetscCall (VecDuplicate (mass, &w));
  PetscCall (VecDuplicate (mass, &x));
  PetscCall (VecDuplicate (mass, &y));
  PetscCall (VecDuplicate (mass, &Av));
  PetscCall (fill_fn (mass, m_entry));
  PetscCall (VecSet (minv_gamma, 1.0 / GAMMA));

  co.verbose = (layout == 0);
  PetscCall ((PetscErrorCode) lgh_prior_create_cholmod (A, minv_gamma,
                                                        LGH_CHOL_A_IS_R, &co,
                                                        &pr));
  /* ---- the factor */
  check (pr->chol != NULL && pr->chol->L->is_ll && pr->chol->L->is_super,
         "A_IS_R: factor is supernodal LL^T", 0.);

  /* ---- permutation pinned against CHOLMOD, and P A P^T = L L^T */
  {
    struct lgh_chol_ctx *c = pr->chol;
    static double       xf[N_GLOBAL], px[N_GLOBAL], t[N_GLOBAL],
                        s[N_GLOBAL], af[N_GLOBAL], paf[N_GLOBAL];
    int                 nmoved = 0, ninvol = 0;
    cholmod_dense      *Xd, *Yd;

    for (int k = 0; k < N_GLOBAL; k++) {
      if (c->perm[k] != k) nmoved++;
      if (c->perm[c->perm[k]] != k) ninvol++;
    }
    check (nmoved > 0 && ninvol > 0,
           "AMD permutation is neither identity nor an involution",
           (double) ninvol);
    for (int k = 0; k < N_GLOBAL; k++) xf[k] = lgh_randn_at (77UL, k, 0);
    lgh_chol_perm_fwd (c, xf, px, 1);
    Xd = cholmod_l_allocate_dense (N_GLOBAL, 1, N_GLOBAL,
                                   CHOLMOD_REAL + CHOLMOD_DOUBLE, &c->cc);
    for (int k = 0; k < N_GLOBAL; k++) ((double *) Xd->x)[k] = xf[k];
    Yd = cholmod_l_solve (CHOLMOD_P, c->L, Xd, &c->cc);
    rd = full_rel (px, (double *) Yd->x, N_GLOBAL);
    check (rd == 0.0, "our P x == cholmod_solve(CHOLMOD_P)", rd);
    cholmod_l_free_dense (&Yd, &c->cc);
    /* and P^T: CHOLMOD_Pt of P x gives x back through our bwd */
    lgh_chol_perm_bwd (c, px, t, 1);
    rd = full_rel (t, xf, N_GLOBAL);
    check (rd == 0.0, "our P^T P x == x", rd);
    for (int k = 0; k < N_GLOBAL; k++) ((double *) Xd->x)[k] = px[k];
    Yd = cholmod_l_solve (CHOLMOD_Pt, c->L, Xd, &c->cc);
    rd = full_rel (t, (double *) Yd->x, N_GLOBAL);
    check (rd == 0.0, "our P^T y == cholmod_solve(CHOLMOD_Pt)", rd);
    cholmod_l_free_dense (&Yd, &c->cc);
    cholmod_l_free_dense (&Xd, &c->cc);
    /* L (L^T (P x)) == P (A x) */
    PetscCall (set_from_full (v, xf));
    PetscCall (MatMult (A, v, Av));
    PetscCall (gather_full (Av, af));
    lgh_chol_perm_fwd (c, af, paf, 1);
    lgh_chol_lmul (c, 1, px, t, 1);
    lgh_chol_lmul (c, 0, t, s, 1);
    rd = full_rel (s, paf, N_GLOBAL);
    check (rd < 1e-13, "L L^T (P x) == P A x (convention of L->Perm)", rd);
  }

  /* ---- R_lib = Z M^{-1} Z^T == gamma A */
  for (int t = 0; t < 3; t++) {
    PetscCall (fill_randn (v, 0xA11CEUL, t));
    PetscCall ((PetscErrorCode) lgh_prior_apply (pr, v, y));
    PetscCall (MatMult (A, v, Av));
    PetscCall (VecScale (Av, GAMMA));
    PetscCall (rel_diff (y, Av, &rd));
    snprintf (what, sizeof (what), "A_IS_R: R_lib v == gamma A v (probe %d)", t);
    check (rd < 1e-12, what, rd);
  }

  /* ---- inverse pairs */
  PetscCall (fill_randn (v, 0x1D1D1UL, 0));
  PetscCall (lgh_prior_applyZ_vec (pr, v, w));
  PetscCall (lgh_prior_solveZ_vec (pr, w, x));
  PetscCall (rel_diff (x, v, &rd));
  check (rd < 1e-12, "A_IS_R: Z^{-1} Z v == v", rd);
  PetscCall (lgh_prior_solveZ_vec (pr, v, w));
  PetscCall (lgh_prior_applyZ_vec (pr, w, x));
  PetscCall (rel_diff (x, v, &rd));
  check (rd < 1e-12, "A_IS_R: Z Z^{-1} v == v", rd);
  PetscCall (lgh_prior_applyZt_vec (pr, v, w));
  PetscCall (lgh_prior_solveZt_vec (pr, w, x));
  PetscCall (rel_diff (x, v, &rd));
  check (rd < 1e-12, "A_IS_R: Z^{-T} Z^T v == v", rd);
  PetscCall (MatMult (A, v, Av));
  PetscCall (lgh_prior_solveZ_vec (pr, Av, w));
  PetscCall (lgh_prior_solveZt_vec (pr, w, x));
  PetscCall (rel_diff (x, v, &rd));
  check (rd < 1e-12, "A_IS_R: Z^{-T} Z^{-1} A v == v", rd);
  /* Z != Z^T: the two applies must differ (non-symmetric factor) */
  PetscCall (lgh_prior_applyZ_vec (pr, v, w));
  PetscCall (lgh_prior_applyZt_vec (pr, v, x));
  PetscCall (rel_diff (x, w, &rd));
  check (rd > 1e-3, "A_IS_R: Z v and Z^T v differ", rd);

  /* ---- blocked == column-by-column */
  for (int k = 0; k < 5; k++) {
    PetscCall (blocked_vs_columns (pr, pr, LGH_PRIOR_SOLVEZ, v, widths[k], &rd));
    snprintf (what, sizeof (what), "A_IS_R: blocked Z^{-1} == columns (width %d)", widths[k]);
    check (rd < 1e-12, what, rd);
    PetscCall (blocked_vs_columns (pr, pr, LGH_PRIOR_SOLVEZT, v, widths[k], &rd));
    snprintf (what, sizeof (what), "A_IS_R: blocked Z^{-T} == columns (width %d)", widths[k]);
    check (rd < 1e-12, what, rd);
  }

  /* ---- stats accessor */
  {
    lgh_prior_cholmod_stats_t st;
    int                 e = lgh_prior_cholmod_get_stats (pr, &st);
    check (e == 0 && st.n == N_GLOBAL && st.n_single > 0 && st.n_blocked == 10
           && st.n_blocked_cols == 2 * (1 + 2 + 3 + 5 + 9) && st.nnz_L > 0
           && st.mode == LGH_CHOL_A_IS_R,
           "A_IS_R: stats accessor", (double) st.n_blocked);
  }

  /* ---- A_IS_Z vs the Mat path */
  PetscCall ((PetscErrorCode) lgh_prior_create_cholmod (A, mass,
                                                        LGH_CHOL_A_IS_Z, &co,
                                                        &pz));
  mo.ksp_rtol = 1e-14;
  PetscCall ((PetscErrorCode) lgh_prior_create_mat (A, mass, &mo, &pm));
  {
    lgh_prior_cholmod_stats_t st;
    check (lgh_prior_cholmod_get_stats (pm, &st) != 0,
           "stats accessor refuses a non-CHOLMOD prior", 0.);
  }
  for (int t = 0; t < 2; t++) {
    PetscCall (fill_randn (v, 0x2222UL, t));
    if (t == 1) PetscCall (VecShift (v, 5.0));
    PetscCall (lgh_prior_solveZ_vec (pz, v, x));
    PetscCall (lgh_prior_solveZ_vec (pm, v, y));
    PetscCall (rel_diff (x, y, &rd));
    snprintf (what, sizeof (what), "A_IS_Z: solve == Mat-path solve (probe %d)", t);
    check (rd < 1e-11, what, rd);
    PetscCall (lgh_prior_solveZt_vec (pz, v, x));
    PetscCall (rel_diff (x, y, &rd));
    snprintf (what, sizeof (what), "A_IS_Z: solveZt == Mat-path solve (probe %d)", t);
    check (rd < 1e-11, what, rd);
    PetscCall ((PetscErrorCode) lgh_prior_apply (pz, v, x));
    PetscCall ((PetscErrorCode) lgh_prior_apply (pm, v, y));
    PetscCall (rel_diff (x, y, &rd));
    snprintf (what, sizeof (what), "A_IS_Z: R_lib == Mat-path R_lib (probe %d)", t);
    check (rd < 1e-15, what, rd);
  }
  PetscCall (fill_randn (v, 0x3333UL, 0));
  PetscCall (lgh_prior_applyZ_vec (pz, v, w));
  PetscCall (lgh_prior_solveZ_vec (pz, w, x));
  PetscCall (rel_diff (x, v, &rd));
  check (rd < 1e-12, "A_IS_Z: Z^{-1} Z v == v", rd);
  for (int k = 0; k < 5; k++) {
    PetscCall (blocked_vs_columns (pz, pz, LGH_PRIOR_SOLVEZ, v, widths[k], &rd));
    snprintf (what, sizeof (what), "A_IS_Z: blocked == columns (width %d)", widths[k]);
    check (rd < 1e-12, what, rd);
    PetscCall (blocked_vs_columns (pz, pz, LGH_PRIOR_SOLVEZT, v, widths[k], &rd));
    snprintf (what, sizeof (what), "A_IS_Z: blocked Zt == columns (width %d)", widths[k]);
    check (rd < 1e-12, what, rd);
    PetscCall (blocked_vs_columns (pz, pm, LGH_PRIOR_SOLVEZ, v, widths[k], &rd));
    snprintf (what, sizeof (what), "A_IS_Z: blocked == Mat-path columns (width %d)", widths[k]);
    check (rd < 1e-11, what, rd);
  }

  lgh_prior_destroy (pr);
  lgh_prior_destroy (pz);
  lgh_prior_destroy (pm);
  PetscCall (VecDestroy (&minv_gamma));
  PetscCall (VecDestroy (&mass));
  PetscCall (VecDestroy (&v));
  PetscCall (VecDestroy (&w));
  PetscCall (VecDestroy (&x));
  PetscCall (VecDestroy (&y));
  PetscCall (VecDestroy (&Av));
  PetscCall (MatDestroy (&A));
  return PETSC_SUCCESS;
}

/* ======================================================================= */
/* part 2: the Z/Z^T mix-up guard through the GLR engines                   */

typedef struct
{
  int                 nloc, rstart;
  int                *counts, *displs;
  double             *xglob;
}
hd_ctx_t;

static void
hd_apply (const double *in_local, double *out_local, void *ctx)
{
  hd_ctx_t           *c = (hd_ctx_t *) ctx;

  MPI_Allgatherv ((void *) in_local, c->nloc, MPI_DOUBLE, c->xglob,
                  c->counts, c->displs, MPI_DOUBLE, PETSC_COMM_WORLD);
  for (int i = 0; i < c->nloc; i++) {
    double              s = 0.;
    for (int j = 0; j < N_GLOBAL; j++)
      s += hd_entry (c->rstart + i, j) * c->xglob[j];
    out_local[i] = s;
  }
}

/* H = K + c R dense; C = chol(H) */
static int
dense_H_chol (double (*kfn) (int, int), const double *R, double c,
              double *H, double *C)
{
  for (int j = 0; j < N_GLOBAL; j++)
    for (int i = 0; i < N_GLOBAL; i++)
      H[i + j * N_GLOBAL] = kfn (i, j) + c * R[i + j * N_GLOBAL];
  for (int i = 0; i < N_GLOBAL * N_GLOBAL; i++) C[i] = H[i];
  return dense_chol (C);
}

/* every downstream invariant of one build vs the dense H = K + c R.
 * out[0..3] collect scalars to compare across builds.                   */
static PetscErrorCode
glr_invariants (lgh_glr_t *g, double c, const double *H, const double *C,
                Vec v, const char *tag, double *scal)
{
  static double       vf[N_GLOBAL], ref[N_GLOBAL], got[N_GLOBAL];
  Vec                 x, y, z;
  double              rd, vHv, vHiv;
  PetscReal           nrm;
  char                what[160];

  PetscCall (VecDuplicate (v, &x));
  PetscCall (VecDuplicate (v, &y));
  PetscCall (VecDuplicate (v, &z));
  PetscCall (gather_full (v, vf));

  /* solve vs dense H^{-1} */
  PetscCall ((PetscErrorCode) lgh_glr_solve (g, c, v, x));
  PetscCall (gather_full (x, got));
  for (int i = 0; i < N_GLOBAL; i++) ref[i] = vf[i];
  dense_chol_solve (C, ref);
  rd = full_rel (got, ref, N_GLOBAL);
  snprintf (what, sizeof (what), "%s: glr_solve == dense (B + cR)^{-1}", tag);
  check (rd < 1e-9, what, rd);
  vHiv = 0.;
  for (int i = 0; i < N_GLOBAL; i++) vHiv += vf[i] * ref[i];

  /* apply vs dense H */
  PetscCall ((PetscErrorCode) lgh_glr_apply (g, c, v, x));
  PetscCall (gather_full (x, got));
  dense_matvec (H, vf, ref);
  rd = full_rel (got, ref, N_GLOBAL);
  snprintf (what, sizeof (what), "%s: glr_apply == dense (B + cR)", tag);
  check (rd < 1e-9, what, rd);
  vHv = 0.;
  for (int i = 0; i < N_GLOBAL; i++) vHv += vf[i] * ref[i];

  /* G (G^T v) == H v */
  PetscCall ((PetscErrorCode) lgh_glr_factor (g, c, LGH_FACTOR_GT, v, x));
  PetscCall (VecNorm (x, NORM_2, &nrm));
  scal[0] = (double) (nrm * nrm);
  snprintf (what, sizeof (what), "%s: |GT v|^2 == v'Hv", tag);
  check (fabs (scal[0] - vHv) < 1e-9 * fabs (vHv), what, scal[0] - vHv);
  PetscCall ((PetscErrorCode) lgh_glr_factor (g, c, LGH_FACTOR_G, x, y));
  PetscCall (gather_full (y, got));
  rd = full_rel (got, ref, N_GLOBAL);
  snprintf (what, sizeof (what), "%s: G(GT v) == dense H v", tag);
  check (rd < 1e-9, what, rd);
  /* G^T (G^{-T} v) == v */
  PetscCall ((PetscErrorCode) lgh_glr_factor (g, c, LGH_FACTOR_GINVT, v, x));
  PetscCall ((PetscErrorCode) lgh_glr_factor (g, c, LGH_FACTOR_GT, x, y));
  PetscCall (rel_diff (y, v, &rd));
  snprintf (what, sizeof (what), "%s: GT(GINVT v) == v", tag);
  check (rd < 1e-9, what, rd);

  /* G^{-T} (G^{-1} v) == H^{-1} v, |G^{-1} v|^2 == v'H^{-1}v */
  PetscCall ((PetscErrorCode) lgh_glr_factor (g, c, LGH_FACTOR_GINV, v, x));
  PetscCall (VecNorm (x, NORM_2, &nrm));
  scal[1] = (double) (nrm * nrm);
  snprintf (what, sizeof (what), "%s: |GINV v|^2 == v'H^{-1}v", tag);
  check (fabs (scal[1] - vHiv) < 1e-9 * fabs (vHiv), what, scal[1] - vHiv);
  PetscCall ((PetscErrorCode) lgh_glr_factor (g, c, LGH_FACTOR_GINVT, x, y));
  PetscCall (gather_full (y, got));
  for (int i = 0; i < N_GLOBAL; i++) ref[i] = vf[i];
  dense_chol_solve (C, ref);
  rd = full_rel (got, ref, N_GLOBAL);
  snprintf (what, sizeof (what), "%s: GINVT(GINV v) == dense H^{-1} v", tag);
  check (rd < 1e-9, what, rd);
  /* G^{-1} (G v) == v */
  PetscCall ((PetscErrorCode) lgh_glr_factor (g, c, LGH_FACTOR_G, v, x));
  PetscCall ((PetscErrorCode) lgh_glr_factor (g, c, LGH_FACTOR_GINV, x, y));
  PetscCall (rel_diff (y, v, &rd));
  snprintf (what, sizeof (what), "%s: GINV(G v) == v", tag);
  check (rd < 1e-9, what, rd);

  PetscCall (VecDestroy (&x));
  PetscCall (VecDestroy (&y));
  PetscCall (VecDestroy (&z));
  return PETSC_SUCCESS;
}

static PetscErrorCode
run_mixup_guard (lgh_glr_backend_t backend)
{
  const double        c = 0.7;
  const int           kprobe = 30, n_qc = 6;
  Mat                 A, Ms, R, B;
  Vec                 mass, minv, minv_gamma, v, x1, x2;
  lgh_prior_t        *pZ, *pR;
  lgh_prior_cholmod_opts_t co = lgh_prior_cholmod_opts_default ();
  lgh_glr_t          *g[2];
  lgh_glr_opts_t      go = lgh_glr_opts_default ();
  lgh_glr_report_t    rep[2];
  static double       Rd[N_GLOBAL * N_GLOBAL], H[N_GLOBAL * N_GLOBAL],
                      C[N_GLOBAL * N_GLOBAL], w[N_GLOBAL];
  double              scal[2][2], rd;
  PetscInt            rs, re, nloc;
  const char         *bname = backend == LGH_GLR_REPLICATED ? "replicated"
                                                            : "scalapack";
  char                tag[2][64], what[160];
  hd_ctx_t            hctx;
  double             *V, *HV;
  lgh_glr_correct_opts_t cop = lgh_glr_correct_opts_default ();
  lgh_glr_correct_report_t crep[2];

  PetscCall (PetscPrintf (PETSC_COMM_WORLD, "-- mix-up guard, %s backend\n",
                          bname));
  PetscCall (build_A (PETSC_DECIDE, &A));
  PetscCall (MatCreateVecs (A, &mass, NULL));
  PetscCall (VecDuplicate (mass, &minv));
  PetscCall (VecDuplicate (mass, &minv_gamma));
  PetscCall (VecDuplicate (mass, &v));
  PetscCall (VecDuplicate (mass, &x1));
  PetscCall (VecDuplicate (mass, &x2));
  PetscCall (fill_fn (mass, m_entry));
  PetscCall (VecCopy (mass, minv));
  PetscCall (VecReciprocal (minv));
  PetscCall (VecSet (minv_gamma, 1.0 / GAMMA));
  PetscCall (VecGetOwnershipRange (mass, &rs, &re));
  nloc = re - rs;

  /* R/gamma = A M^{-1} A / gamma, assembled */
  PetscCall (MatDuplicate (A, MAT_COPY_VALUES, &Ms));
  PetscCall (MatDiagonalScale (Ms, minv, NULL));
  PetscCall (MatMatMult (A, Ms, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &R));
  PetscCall (MatScale (R, 1.0 / GAMMA));

  co.verbose = 0;
  PetscCall ((PetscErrorCode) lgh_prior_create_cholmod (A, mass,
                                                        LGH_CHOL_A_IS_Z, &co,
                                                        &pZ));
  PetscCall ((PetscErrorCode) lgh_prior_create_cholmod (R, minv_gamma,
                                                        LGH_CHOL_A_IS_R, &co,
                                                        &pR));
  /* same R_lib */
  PetscCall (fill_randn (v, 0x5EEDUL, 0));
  PetscCall ((PetscErrorCode) lgh_prior_apply (pZ, v, x1));
  PetscCall ((PetscErrorCode) lgh_prior_apply (pR, v, x2));
  PetscCall (rel_diff (x2, x1, &rd));
  check (rd < 1e-12, "guard: both builds' R_lib agree", rd);

  /* B (exact rank RANK_B), dense */
  PetscCall (MatCreateDense (PETSC_COMM_WORLD, nloc, nloc, N_GLOBAL, N_GLOBAL,
                             NULL, &B));
  {
    PetscScalar        *ba;
    PetscInt            lda;
    PetscCall (MatDenseGetLDA (B, &lda));
    PetscCall (MatDenseGetArrayWrite (B, &ba));
    for (PetscInt j = 0; j < N_GLOBAL; j++)
      for (PetscInt i = rs; i < re; i++)
        ba[(i - rs) + j * lda] = b_entry ((int) i, (int) j);
    PetscCall (MatDenseRestoreArrayWrite (B, &ba));
  }
  PetscCall (MatAssemblyBegin (B, MAT_FINAL_ASSEMBLY));
  PetscCall (MatAssemblyEnd (B, MAT_FINAL_ASSEMBLY));

  go.ell = 30;
  go.q_power = 1;
  go.trunc_rel = 1e-10;
  go.backend = backend;
  go.nb = 8;
  go.panel = 16;
  PetscCall ((PetscErrorCode) lgh_glr_compute (B, pZ, &go, &g[0], &rep[0]));
  PetscCall ((PetscErrorCode) lgh_glr_compute (B, pR, &go, &g[1], &rep[1]));
  snprintf (tag[0], sizeof (tag[0]), "guard/%s/Z=A", bname);
  snprintf (tag[1], sizeof (tag[1]), "guard/%s/Z=chol(R)", bname);

  /* spectrum: both builds vs the dense generalized eigenvalues of (B, R) */
  dense_R (Rd);
  check (dense_gen_eigs (b_entry, Rd, w) == 0, "guard: dense oracle eigs", 0.);
  {
    int                 nk[2];
    const double       *lam[2];
    double              worst = 0., worst_x = 0.;
    for (int b = 0; b < 2; b++)
      PetscCall ((PetscErrorCode) lgh_glr_eigs (g[b], &nk[b], &lam[b]));
    check (nk[0] == RANK_B && nk[1] == RANK_B, "guard: kept == rank(B)",
           (double) nk[1]);
    for (int i = 0; i < RANK_B && i < nk[0] && i < nk[1]; i++) {
      worst = fmax (worst, fmax (fabs (lam[0][i] - w[i]),
                                 fabs (lam[1][i] - w[i])) / w[0]);
      worst_x = fmax (worst_x, fabs (lam[0][i] - lam[1][i]) / w[0]);
    }
    check (worst < 1e-9, "guard: spectra == dense oracle", worst);
    check (worst_x < 1e-9, "guard: spectra agree between builds", worst_x);
  }

  /* downstream, before correction: H = B + c R */
  check (dense_H_chol (b_entry, Rd, c, H, C) == 0, "guard: dense H SPD", 0.);
  PetscCall (fill_randn (v, 0xF00DUL, 0));
  for (int b = 0; b < 2; b++)
    PetscCall (glr_invariants (g[b], c, H, C, v, tag[b], scal[b]));
  check (fabs (scal[0][0] - scal[1][0]) < 1e-9 * scal[0][0],
         "guard: |GT v|^2 agrees between builds", scal[0][0] - scal[1][0]);
  check (fabs (scal[0][1] - scal[1][1]) < 1e-9 * scal[0][1],
         "guard: |GINV v|^2 agrees between builds", scal[0][1] - scal[1][1]);
  PetscCall ((PetscErrorCode) lgh_glr_solve (g[0], c, v, x1));
  PetscCall ((PetscErrorCode) lgh_glr_solve (g[1], c, v, x2));
  PetscCall (rel_diff (x2, x1, &rd));
  check (rd < 1e-10, "guard: glr_solve agrees between builds", rd);

  /* correction from probes against H_d = B + DH */
  {
    PetscMPIInt         P;
    PetscCallMPI (MPI_Comm_size (PETSC_COMM_WORLD, &P));
    hctx.nloc = (int) nloc;
    hctx.rstart = (int) rs;
    hctx.counts = (int *) malloc (sizeof (int) * P);
    hctx.displs = (int *) malloc (sizeof (int) * P);
    hctx.xglob = (double *) malloc (sizeof (double) * N_GLOBAL);
    PetscCallMPI (MPI_Allgather (&hctx.nloc, 1, MPI_INT, hctx.counts, 1,
                                 MPI_INT, PETSC_COMM_WORLD));
    hctx.displs[0] = 0;
    for (int r = 1; r < P; r++)
      hctx.displs[r] = hctx.displs[r - 1] + hctx.counts[r - 1];
  }
  V = (double *) malloc (sizeof (double) * (size_t) nloc * kprobe + 8);
  HV = (double *) malloc (sizeof (double) * (size_t) nloc * kprobe + 8);
  for (int j = 0; j < kprobe; j++)
    for (PetscInt i = 0; i < nloc; i++) {
      double              s = 0.;
      V[(size_t) j * nloc + i] = lgh_randn_at (0x9B0BEUL, rs + i, j);
      for (int l = 0; l < N_GLOBAL; l++)
        s += hd_entry ((int) (rs + i), l) * lgh_randn_at (0x9B0BEUL, l, j);
      HV[(size_t) j * nloc + i] = s;
    }
  cop.c0 = 1.0;
  cop.applies = 8;
  cop.n_qc = n_qc;
  for (int b = 0; b < 2; b++)
    PetscCall ((PetscErrorCode) lgh_glr_correct_probes (g[b], kprobe, V, HV,
                                                        hd_apply, &hctx, &cop,
                                                        &crep[b]));
  check (crep[0].kept == crep[1].kept && crep[0].basis == crep[1].basis
         && crep[0].hd_applies == crep[1].hd_applies
         && crep[0].clamped == crep[1].clamped,
         "guard: correction report counts agree", (double) crep[1].kept);
  check (crep[0].basis == RANK_C, "guard: correction basis == rank(DH)",
         (double) crep[0].basis);
  check (fabs (crep[0].d_min - crep[1].d_min) < 1e-8
         && fabs (crep[0].d_max - crep[1].d_max) < 1e-8,
         "guard: correction eigenvalues agree",
         fabs (crep[0].d_min - crep[1].d_min));
  check (fabs (crep[0].floor - crep[1].floor) < 1e-8,
         "guard: floor agrees", crep[0].floor - crep[1].floor);
  check (fabs (crep[0].qc_before - crep[1].qc_before)
           < 1e-8 * crep[0].qc_before
         && crep[0].qc_after < 1e-7 && crep[1].qc_after < 1e-7,
         "guard: qc_before agrees, qc_after ~ 0 in both",
         crep[0].qc_before - crep[1].qc_before);
  check (crep[0].qc_before > 1e-4, "guard: the planted error is visible",
         crep[0].qc_before);
  {
    static double       wd[N_GLOBAL];
    int                 nk[2];
    const double       *lam[2];
    double              l0[N_GLOBAL], l1[N_GLOBAL], worst = 0., worst_x = 0.;

    check (dense_gen_eigs (hd_entry, Rd, wd) == 0, "guard: dense H_d eigs", 0.);
    for (int b = 0; b < 2; b++)
      PetscCall ((PetscErrorCode) lgh_glr_eigs (g[b], &nk[b], &lam[b]));
    check (nk[0] == nk[1] && nk[0] == RANK_B + RANK_C,
           "guard: corrected kept agrees (= rank B + rank DH)", (double) nk[1]);
    for (int i = 0; i < nk[0]; i++) l0[i] = lam[0][i];
    for (int i = 0; i < nk[1]; i++) l1[i] = lam[1][i];
    sort_desc (l0, nk[0]);
    sort_desc (l1, nk[1]);
    /* the oracle's nonzero eigenvalues: the nk largest by |.|, by value */
    {
      double              top[N_GLOBAL];
      int                 lo = 0, hi = N_GLOBAL - 1, m = 0;
      while (m < nk[0] && lo <= hi) {    /* w is value-descending */
        if (fabs (wd[lo]) >= fabs (wd[hi])) top[m++] = wd[lo++];
        else top[m++] = wd[hi--];
      }
      sort_desc (top, m);
      for (int i = 0; i < nk[0] && i < nk[1]; i++) {
        worst = fmax (worst, fmax (fabs (l0[i] - top[i]),
                                   fabs (l1[i] - top[i])) / fabs (wd[0]));
        worst_x = fmax (worst_x, fabs (l0[i] - l1[i]) / fabs (wd[0]));
      }
    }
    check (worst < 1e-7, "guard: corrected spectra == dense oracle of H_d",
           worst);
    check (worst_x < 1e-8, "guard: corrected spectra agree between builds",
           worst_x);
  }
  /* downstream after correction: H = H_d + c R */
  check (dense_H_chol (hd_entry, Rd, c, H, C) == 0, "guard: dense H_d SPD", 0.);
  PetscCall (fill_randn (v, 0xBEEFUL, 0));
  for (int b = 0; b < 2; b++) {
    snprintf (what, sizeof (what), "%s/corrected", tag[b]);
    PetscCall (glr_invariants (g[b], c, H, C, v, what, scal[b]));
  }
  check (fabs (scal[0][0] - scal[1][0]) < 1e-9 * scal[0][0],
         "guard: corrected |GT v|^2 agrees", scal[0][0] - scal[1][0]);
  check (fabs (scal[0][1] - scal[1][1]) < 1e-9 * scal[0][1],
         "guard: corrected |GINV v|^2 agrees", scal[0][1] - scal[1][1]);
  PetscCall ((PetscErrorCode) lgh_glr_apply (g[0], c, v, x1));
  PetscCall ((PetscErrorCode) lgh_glr_apply (g[1], c, v, x2));
  PetscCall (rel_diff (x2, x1, &rd));
  check (rd < 1e-10, "guard: corrected glr_apply agrees between builds", rd);
  PetscCall ((PetscErrorCode) lgh_glr_solve (g[0], c, v, x1));
  PetscCall ((PetscErrorCode) lgh_glr_solve (g[1], c, v, x2));
  PetscCall (rel_diff (x2, x1, &rd));
  check (rd < 1e-10, "guard: corrected glr_solve agrees between builds", rd);

  free (V);
  free (HV);
  free (hctx.counts);
  free (hctx.displs);
  free (hctx.xglob);
  lgh_glr_destroy (g[0]);
  lgh_glr_destroy (g[1]);
  lgh_prior_destroy (pZ);
  lgh_prior_destroy (pR);
  PetscCall (MatDestroy (&B));
  PetscCall (MatDestroy (&R));
  PetscCall (MatDestroy (&Ms));
  PetscCall (MatDestroy (&A));
  PetscCall (VecDestroy (&mass));
  PetscCall (VecDestroy (&minv));
  PetscCall (VecDestroy (&minv_gamma));
  PetscCall (VecDestroy (&v));
  PetscCall (VecDestroy (&x1));
  PetscCall (VecDestroy (&x2));
  return PETSC_SUCCESS;
}

/* A non-symmetric A must be refused. */
static PetscErrorCode
run_refuse_nonsymmetric (void)
{
  Mat                 A;
  Vec                 mass;
  lgh_prior_t        *p = NULL;
  PetscInt            rs, re;
  int                 ierr;

  PetscCall (build_A (PETSC_DECIDE, &A));
  PetscCall (MatGetOwnershipRange (A, &rs, &re));
  if (rs <= 5 && 5 < re)
    PetscCall (MatSetValue (A, 5, 6, -0.5, INSERT_VALUES));
  PetscCall (MatAssemblyBegin (A, MAT_FINAL_ASSEMBLY));
  PetscCall (MatAssemblyEnd (A, MAT_FINAL_ASSEMBLY));
  PetscCall (MatCreateVecs (A, &mass, NULL));
  PetscCall (VecSet (mass, 1.0));
  PetscCall (PetscPushErrorHandler (PetscReturnErrorHandler, NULL));
  ierr = lgh_prior_create_cholmod (A, mass, LGH_CHOL_A_IS_R, NULL, &p);
  PetscCall (PetscPopErrorHandler ());
  check (ierr != 0, "non-symmetric A is refused", (double) ierr);
  PetscCall (VecDestroy (&mass));
  PetscCall (MatDestroy (&A));
  return PETSC_SUCCESS;
}

int
main (int argc, char **argv)
{
  PetscCall (PetscInitialize (&argc, &argv, NULL, NULL));
  PetscCall (run_prior_level (0));
  PetscCall (run_prior_level (1));
  PetscCall (run_mixup_guard (LGH_GLR_REPLICATED));
#ifdef LGH_WITH_SCALAPACK
  PetscCall (run_mixup_guard (LGH_GLR_SCALAPACK));
#endif
  PetscCall (run_refuse_nonsymmetric ());
  if (n_fail == 0)
    PetscCall (PetscPrintf (PETSC_COMM_WORLD, "PASS test_prior_cholmod\n"));
  PetscCall (PetscFinalize ());
  return n_fail == 0 ? 0 : 1;
}

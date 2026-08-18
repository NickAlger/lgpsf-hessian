/* correct_impl.h — stage 2.5: the deflation correction (lgh_glr_correct).
 *
 * Math.  The GLR object holds U diag(lam) U^T ~= F_B (the prior-whitened
 * sparse fit).  The TRUE whitened misfit operator is
 *     F_true = M^{1/2} Z^{-1} H_d Z^{-T} M^{1/2},
 * and the whitened error operator at the reference shift c0 is
 *     E_w = S^- (F_true + c0 I) S^- - I,       S^± = (U lam U^T + c0 I)^{±1/2},
 * which equals S^- (F_true - U lam U^T) S^- — symmetric, and c-free up to the
 * S^- metric (the regularization cancels in H - Htilde).  Its dominant
 * eigenpairs (V, D) give the deflation correction
 *     F_corr = U lam U^T + Vhat D Vhat^T,      Vhat = S^+ V,
 * which is grafted into a fresh (U', lam') by orthogonalization against U
 * plus a small dense eigensolve — after which every downstream operation
 * serves the corrected operator at every shift c > floor = max(0, -min lam').
 *
 * Two constructions (provenance: the slice-40 deflation study — directions
 * are nearly free, VALUES are the binding constraint):
 *   - value pass (lgh_glr_correct_probes): the error basis comes for FREE
 *     from the fit stage's probe pairs (Omega, H_d Omega) — E_w applied to
 *     the transformed probes is pure recombination of stored data; the
 *     apply budget is spent on exact Rayleigh values on the top basis
 *     directions, with one power step past the basis when budget remains;
 *   - fresh randomized (lgh_glr_correct): hashed-Gaussian sketch of E_w,
 *     ~2x the applies for the same rank.
 *
 * Eigenvalue signs are kept (no FLIP): d < 0 is legitimate information —
 * the fit came out too big in that direction.  Exact generalized error
 * eigenvalues satisfy 1 + d > 0 automatically; the clamp at -1 + clamp_eps
 * guards only against estimation noise in the Rayleigh values.
 *
 * Everything is block-based (MATDENSE sweeps): per-column Z solves always
 * take the expensive KSP path, and the blocked prior tiers only fire on
 * Mat blocks.
 */

#ifndef LGPSF_HESSIAN_CORRECT_IMPL_H
#define LGPSF_HESSIAN_CORRECT_IMPL_H

#ifndef LGPSF_HESSIAN_GLR_COMMON_IMPL_H
#error "include impl/glr_common_impl.h first (impl.hpp does this)"
#endif

lgh_glr_correct_opts_t
lgh_glr_correct_opts_default (void)
{
  lgh_glr_correct_opts_t o;

  o.c0 = 1.0;
  o.rank = -1;
  o.applies = 30;
  o.n_qc = 0;
  o.q_power = 0;
  o.clamp_eps = 0.05;
  o.drop_tol = 1e-12;
  o.seed = 424242UL;
  o.verbose = 0;
  return o;
}

/* replicated symmetric eigensolve: A (n x n, column-major) in, eigenvectors
 * in A and eigenvalues ASCENDING in ev out.                               */
static PetscErrorCode
lgh_correct_dsyevd (int n, double *A, double *ev)
{
  int                 lwork = -1, liwork = -1, iwkopt = 0, info = 0;
  double              wkopt;
  double             *work;
  int                *iwork;

  LGH_LAPACK_DSYEVD ("V", "U", &n, A, &n, ev, &wkopt, &lwork, &iwkopt,
                     &liwork, &info);
  lwork = (int) wkopt; liwork = iwkopt;
  PetscCall (PetscMalloc1 (lwork, &work));
  PetscCall (PetscMalloc1 (liwork, &iwork));
  LGH_LAPACK_DSYEVD ("V", "U", &n, A, &n, ev, work, &lwork, iwork, &liwork,
                     &info);
  PetscCall (PetscFree (work));
  PetscCall (PetscFree (iwork));
  PetscCheck (info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB,
              "lgh_glr_correct: dsyevd failed");
  return PETSC_SUCCESS;
}

/* G[p x q] = X^T Y, replicated (local GEMM + one allreduce).             */
static PetscErrorCode
lgh_correct_xtY (Mat X, Mat Y, double *G)
{
  const PetscScalar  *xa, *ya;
  PetscInt            nloc, ldx, ldy, p, q, i;
  double             *Gl;
  MPI_Comm            comm;

  PetscCall (MatGetLocalSize (X, &nloc, NULL));
  PetscCall (MatGetSize (X, NULL, &p));
  PetscCall (MatGetSize (Y, NULL, &q));
  PetscCall (PetscMalloc1 ((size_t) PetscMax (p * q, 1), &Gl));
  PetscCall (MatDenseGetLDA (X, &ldx));
  PetscCall (MatDenseGetLDA (Y, &ldy));
  PetscCall (MatDenseGetArrayRead (X, &xa));
  PetscCall (MatDenseGetArrayRead (Y, &ya));
  if (nloc > 0) {
    int                 bm = (int) nloc, bp = (int) p, bq = (int) q;
    int                 bldx = (int) ldx, bldy = (int) ldy;
    double              one = 1.0, zero = 0.0;
    LGH_BLAS_DGEMM ("T", "N", &bp, &bq, &bm, &one, xa, &bldx, ya, &bldy,
                    &zero, Gl, &bp);
  }
  else { for (i = 0; i < p * q; i++) Gl[i] = 0.; }
  PetscCall (MatDenseRestoreArrayRead (Y, &ya));
  PetscCall (MatDenseRestoreArrayRead (X, &xa));
  PetscCall (PetscObjectGetComm ((PetscObject) X, &comm));
  PetscCallMPI (MPI_Allreduce (Gl, G, (int) (p * q), MPI_DOUBLE, MPI_SUM,
                               comm));
  PetscCall (PetscFree (Gl));
  return PETSC_SUCCESS;
}

/* C = alpha * A * B + beta * C, purely local: A is N x p distributed
 * MATDENSE, B a replicated column-major block with leading dimension ldb
 * (>= p; lets callers pass row sub-blocks of a taller matrix), C N x q.  */
static PetscErrorCode
lgh_correct_gemm_local (double alpha, Mat A, const double *B, int p, int ldb,
                        int q, double beta, Mat C)
{
  const PetscScalar  *aa;
  PetscScalar        *ca;
  PetscInt            nloc, lda, ldc;

  PetscCall (MatGetLocalSize (A, &nloc, NULL));
  PetscCall (MatDenseGetLDA (A, &lda));
  PetscCall (MatDenseGetLDA (C, &ldc));
  PetscCall (MatDenseGetArrayRead (A, &aa));
  PetscCall (MatDenseGetArray (C, &ca));
  if (nloc > 0) {
    int                 bm = (int) nloc, bp = p, bq = q, bldb = ldb;
    int                 blda = (int) lda, bldc = (int) ldc;
    LGH_BLAS_DGEMM ("N", "N", &bm, &bq, &bp, &alpha, aa, &blda, B, &bldb,
                    &beta, ca, &bldc);
  }
  PetscCall (MatDenseRestoreArray (C, &ca));
  PetscCall (MatDenseRestoreArrayRead (A, &aa));
  return PETSC_SUCCESS;
}

/* new Mat = the leading m columns of A (copy).                            */
static PetscErrorCode
lgh_correct_head_cols (Mat A, int m, Mat *out)
{
  const PetscScalar  *aa;
  PetscScalar        *oa;
  PetscInt            nloc, Nglob, lda, ldo, i, j;
  MPI_Comm            comm;

  PetscCall (PetscObjectGetComm ((PetscObject) A, &comm));
  PetscCall (MatGetLocalSize (A, &nloc, NULL));
  PetscCall (MatGetSize (A, &Nglob, NULL));
  PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob,
                             PetscMax (m, 1), NULL, out));
  PetscCall (MatDenseGetLDA (A, &lda));
  PetscCall (MatDenseGetLDA (*out, &ldo));
  PetscCall (MatDenseGetArrayRead (A, &aa));
  PetscCall (MatDenseGetArrayWrite (*out, &oa));
  for (j = 0; j < m; j++)
    for (i = 0; i < nloc; i++)
      oa[i + (size_t) j * ldo] = aa[i + (size_t) j * lda];
  PetscCall (MatDenseRestoreArrayWrite (*out, &oa));
  PetscCall (MatDenseRestoreArrayRead (A, &aa));
  return PETSC_SUCCESS;
}

/* new Mat = [A | B] (copy).                                               */
static PetscErrorCode
lgh_correct_concat (Mat A, Mat B, Mat *out)
{
  const PetscScalar  *aa, *ba;
  PetscScalar        *oa;
  PetscInt            nloc, Nglob, ma, mb, lda, ldb, ldo, i, j;
  MPI_Comm            comm;

  PetscCall (PetscObjectGetComm ((PetscObject) A, &comm));
  PetscCall (MatGetLocalSize (A, &nloc, NULL));
  PetscCall (MatGetSize (A, &Nglob, &ma));
  PetscCall (MatGetSize (B, NULL, &mb));
  PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, ma + mb, NULL,
                             out));
  PetscCall (MatDenseGetLDA (A, &lda));
  PetscCall (MatDenseGetLDA (B, &ldb));
  PetscCall (MatDenseGetLDA (*out, &ldo));
  PetscCall (MatDenseGetArrayRead (A, &aa));
  PetscCall (MatDenseGetArrayRead (B, &ba));
  PetscCall (MatDenseGetArrayWrite (*out, &oa));
  for (j = 0; j < ma; j++)
    for (i = 0; i < nloc; i++)
      oa[i + (size_t) j * ldo] = aa[i + (size_t) j * lda];
  for (j = 0; j < mb; j++)
    for (i = 0; i < nloc; i++)
      oa[i + (size_t) (ma + j) * ldo] = ba[i + (size_t) j * ldb];
  PetscCall (MatDenseRestoreArrayWrite (*out, &oa));
  PetscCall (MatDenseRestoreArrayRead (B, &ba));
  PetscCall (MatDenseRestoreArrayRead (A, &aa));
  return PETSC_SUCCESS;
}

/* Y = F_true X = Msqrt . Z^{-1} . H_d . Z^{-T} . Msqrt . X around the user
 * misfit callback; W is scratch of the same shape.  Adds ncols to
 * *napplies.  (The mirror of lgh_glr_apply_F_block with MatMult(B)
 * replaced by the true Hessian.)                                          */
static PetscErrorCode
lgh_correct_Ftrue_block (lgh_glr_t *g, lgh_hessian_fn hess, void *ctx,
                         Mat X, Mat Y, Mat W, int *napplies)
{
  PetscInt            j, ncols;

  PetscCall (MatCopy (X, W, SAME_NONZERO_PATTERN));
  PetscCall (MatDiagonalScale (W, g->prior->msqrt, NULL));
  PetscCall (lgh_prior_solve_block (g->prior, LGH_PRIOR_SOLVEZT, W, Y));
  PetscCall (MatGetSize (X, NULL, &ncols));
  for (j = 0; j < ncols; j++) {
    Vec                 yj, wj;
    const PetscScalar  *ya;
    PetscScalar        *wa;

    PetscCall (MatDenseGetColumnVecRead (Y, j, &yj));
    PetscCall (MatDenseGetColumnVecWrite (W, j, &wj));
    PetscCall (VecGetArrayRead (yj, &ya));
    PetscCall (VecGetArrayWrite (wj, &wa));
    hess ((const double *) ya, (double *) wa, ctx);
    PetscCall (VecRestoreArrayWrite (wj, &wa));
    PetscCall (VecRestoreArrayRead (yj, &ya));
    PetscCall (MatDenseRestoreColumnVecWrite (W, j, &wj));
    PetscCall (MatDenseRestoreColumnVecRead (Y, j, &yj));
  }
  *napplies += (int) ncols;
  PetscCall (lgh_prior_solve_block (g->prior, LGH_PRIOR_SOLVEZ, W, Y));
  PetscCall (MatDiagonalScale (Y, g->prior->msqrt, NULL));
  return PETSC_SUCCESS;
}

/* E = E_w T = S^-((F_true + c0 I)(S^- T)) - T;  S1, S2 scratch (same
 * shape).  Costs ncols true applies.                                      */
static PetscErrorCode
lgh_correct_Ew_block (lgh_glr_t *g, double c0, lgh_hessian_fn hess,
                      void *ctx, Mat T, Mat E, Mat S1, Mat S2, int *napplies)
{
  PetscCall (lgh_glr_pow_block (g, c0, -0.5, T, S1));
  PetscCall (lgh_correct_Ftrue_block (g, hess, ctx, S1, S2, E, napplies));
  PetscCall (MatAXPY (S2, c0, S1, SAME_NONZERO_PATTERN));
  PetscCall (lgh_glr_pow_block (g, c0, -0.5, S2, E));
  PetscCall (MatAXPY (E, -1.0, T, SAME_NONZERO_PATTERN));
  return PETSC_SUCCESS;
}

/* Energy-ordered orthonormal basis of the columns of IMG: eigendecompose
 * the Gram matrix, drop directions with energy <= drop_tol * the scale
 * (max Gram eigenvalue, floored by scale_floor for near-zero inputs), and
 * return Q with columns in DESCENDING energy order.                       */
static PetscErrorCode
lgh_correct_energy_basis (Mat IMG, double drop_tol, double scale_floor,
                          Mat *Qout, int *dim)
{
  PetscInt            nloc, Nglob, k;
  double             *G, *ev, *C;
  double              scale;
  int                 nkeep, j, i;
  MPI_Comm            comm;

  PetscCall (PetscObjectGetComm ((PetscObject) IMG, &comm));
  PetscCall (MatGetLocalSize (IMG, &nloc, NULL));
  PetscCall (MatGetSize (IMG, &Nglob, &k));
  PetscCall (PetscMalloc2 ((size_t) k * k, &G, k, &ev));
  PetscCall (lgh_correct_xtY (IMG, IMG, G));
  PetscCall (lgh_correct_dsyevd ((int) k, G, ev));
  scale = PetscMax (ev[k - 1], scale_floor);
  nkeep = 0;
  for (j = (int) k - 1; j >= 0; j--)
    if (ev[j] > drop_tol * scale && ev[j] > 0.) nkeep++;
    else break;
  PetscCall (PetscMalloc1 ((size_t) k * PetscMax (nkeep, 1), &C));
  for (j = 0; j < nkeep; j++) {
    const int           idx = (int) k - 1 - j;    /* descending energy */
    const double        s = 1.0 / sqrt (ev[idx]);
    for (i = 0; i < (int) k; i++)
      C[i + (size_t) j * k] = G[i + (size_t) idx * k] * s;
  }
  PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob,
                             PetscMax (nkeep, 1), NULL, Qout));
  PetscCall (lgh_correct_gemm_local (1.0, IMG, C, (int) k, (int) k,
                                     PetscMax (nkeep, 1), 0.0, *Qout));
  PetscCall (PetscFree (C));
  PetscCall (PetscFree2 (G, ev));
  *dim = nkeep;
  return PETSC_SUCCESS;
}

/* Whitened energy ratio of the CURRENT low-rank model against held-out
 * responses: sqrt( |U lam U^T Xw - Yw|_F^2 / |Yw|_F^2 ).                  */
static PetscErrorCode
lgh_correct_qc (lgh_glr_t *g, Mat Xw, Mat Yw, double *qc)
{
  Mat                 R;
  PetscInt            nloc, Nglob, m, i;
  double             *fv, num, den;

  PetscCall (MatGetLocalSize (Xw, &nloc, NULL));
  PetscCall (MatGetSize (Xw, &Nglob, &m));
  PetscCall (MatCreateDense (g->comm, nloc, PETSC_DECIDE, Nglob, m, NULL,
                             &R));
  PetscCall (PetscMalloc1 (PetscMax (g->kept, 1), &fv));
  for (i = 0; i < g->kept; i++) fv[i] = g->lam[i];
  PetscCall (lgh_glr_filter_block (g, fv, 0.0, Xw, R));
  PetscCall (PetscFree (fv));
  PetscCall (MatAXPY (R, -1.0, Yw, SAME_NONZERO_PATTERN));
  PetscCall (MatNorm (R, NORM_FROBENIUS, &num));
  PetscCall (MatNorm (Yw, NORM_FROBENIUS, &den));
  PetscCall (MatDestroy (&R));
  *qc = (den > 0.) ? num / den : 0.;
  return PETSC_SUCCESS;
}

/* Fold Vhat diag(d) Vhat^T into (U, lam): orthogonalize Vhat against U,
 * eigendecompose the combined low-rank coefficient matrix, keep by |lam'|
 * with the object's truncation rules, SIGNED (no treatment).  Leaves the
 * object with an explicit U regardless of backend.                        */
static PetscErrorCode
lgh_glr_graft (lgh_glr_t *g, Mat Vhat, const double *d, int r,
               const lgh_glr_correct_opts_t *opts,
               lgh_glr_correct_report_t *rep)
{
  const PetscInt      nloc = g->nloc, Nglob = g->Nglob;
  PetscInt            kept0;
  Mat                 Vperp = NULL, Qv = NULL, Unew = NULL;
  double             *Cuv = NULL, *Cqv = NULL, *tmp = NULL, *T2 = NULL;
  double             *ev2 = NULL;
  int                 r2 = 0, n2, keptn, i, j, jc;
  double              scale_floor;

#ifdef LGH_WITH_SCALAPACK
  if (g->U == NULL) {   /* ScaLAPACK representation: make U explicit, drop
                         * the sketch state (extension is over anyway).    */
    Mat                 Ue;
    PetscCall (lgh_glrd_materialize_U (g, &Ue));
    lgh_glrd_destroy_state (g);
    g->U = Ue;
  }
#endif
  kept0 = g->kept;

  /* Vperp = (I - U U^T) Vhat, then an energy-orthonormal basis of it.
   * scale_floor: mean column energy of Vhat itself, so that a correction
   * lying inside span(U) yields r2 = 0 instead of noise columns.          */
  {
    double             *Gv;
    PetscCall (PetscMalloc1 ((size_t) r * r, &Gv));
    PetscCall (lgh_correct_xtY (Vhat, Vhat, Gv));
    scale_floor = 0.;
    for (i = 0; i < r; i++) scale_floor += Gv[i + (size_t) i * r];
    scale_floor /= PetscMax (r, 1);
    PetscCall (PetscFree (Gv));
  }
  PetscCall (lgh_correct_head_cols (Vhat, r, &Vperp));
  if (kept0 > 0) {
    double             *negC;
    PetscCall (PetscMalloc1 ((size_t) kept0 * r, &Cuv));
    PetscCall (PetscMalloc1 ((size_t) kept0 * r, &negC));
    PetscCall (lgh_glr_UTmult_block (g, Vhat, Cuv));
    for (i = 0; i < (int) kept0 * r; i++) negC[i] = -Cuv[i];
    PetscCall (lgh_glr_Umult_block (g, negC, Vperp, PETSC_TRUE));
    PetscCall (PetscFree (negC));
  }
  PetscCall (lgh_correct_energy_basis (Vperp, opts->drop_tol, scale_floor,
                                       &Qv, &r2));
  PetscCall (MatDestroy (&Vperp));
  if (r2 > 0) {
    PetscCall (PetscMalloc1 ((size_t) r2 * r, &Cqv));
    PetscCall (lgh_correct_xtY (Qv, Vhat, Cqv));
  }

  /* combined coefficient matrix on the orthonormal basis [U | Qv]:
   * T2 = [[diag(lam) + Cuv D Cuv^T,  Cuv D Cqv^T],
   *       [Cqv D Cuv^T,              Cqv D Cqv^T]]                        */
  n2 = (int) kept0 + r2;
  PetscCall (PetscCalloc1 ((size_t) PetscMax (n2 * n2, 1), &T2));
  PetscCall (PetscMalloc1 (PetscMax (n2, 1), &ev2));
  for (i = 0; i < (int) kept0; i++) T2[i + (size_t) i * n2] = g->lam[i];
  {
    /* tmp = [Cuv; Cqv] * diag(d)  (n2 x r), then T2 += tmp * [Cuv; Cqv]^T */
    double             *CB;
    PetscCall (PetscMalloc2 ((size_t) PetscMax (n2 * r, 1), &tmp,
                             (size_t) PetscMax (n2 * r, 1), &CB));
    for (jc = 0; jc < r; jc++) {
      for (i = 0; i < (int) kept0; i++)
        CB[i + (size_t) jc * n2] = Cuv[i + (size_t) jc * kept0];
      for (i = 0; i < r2; i++)
        CB[kept0 + i + (size_t) jc * n2] = Cqv[i + (size_t) jc * r2];
    }
    for (jc = 0; jc < r; jc++)
      for (i = 0; i < n2; i++)
        tmp[i + (size_t) jc * n2] = CB[i + (size_t) jc * n2] * d[jc];
    if (n2 > 0 && r > 0) {
      int                 bn = n2, br = r;
      double              one = 1.0;
      LGH_BLAS_DGEMM ("N", "T", &bn, &bn, &br, &one, tmp, &bn, CB, &bn,
                      &one, T2, &bn);
    }
    PetscCall (PetscFree2 (tmp, CB));
    tmp = NULL;
  }
  for (j = 0; j < n2; j++)          /* symmetrize against roundoff */
    for (i = 0; i < j; i++) {
      const double        avg = 0.5 * (T2[i + (size_t) j * n2]
                                       + T2[j + (size_t) i * n2]);
      T2[i + (size_t) j * n2] = avg;
      T2[j + (size_t) i * n2] = avg;
    }
  if (n2 > 0) PetscCall (lgh_correct_dsyevd (n2, T2, ev2));

  /* order |lam'| descending, truncate with the object's rules, keep SIGNED */
  {
    int                *order;
    double              absmax, cut, lmin;
    double             *lam_new;

    PetscCall (PetscMalloc1 (PetscMax (n2, 1), &order));
    for (i = 0; i < n2; i++) order[i] = i;
    for (i = 0; i < n2; i++)
      for (j = i + 1; j < n2; j++)
        if (fabs (ev2[order[j]]) > fabs (ev2[order[i]])) {
          const int           t = order[i]; order[i] = order[j]; order[j] = t;
        }
    absmax = (n2 > 0) ? fabs (ev2[order[0]]) : 0.;
    cut = 0.;
    if (g->opts.trunc_abs > 0.) cut = PetscMax (cut, g->opts.trunc_abs);
    if (g->opts.trunc_rel > 0.)
      cut = PetscMax (cut, g->opts.trunc_rel * absmax);
    keptn = 0;
    while (keptn < n2 && fabs (ev2[order[keptn]]) > cut) keptn++;

    PetscCall (PetscMalloc1 (PetscMax (keptn, 1), &lam_new));
    lmin = 0.;
    for (i = 0; i < keptn; i++) {
      lam_new[i] = ev2[order[i]];
      lmin = PetscMin (lmin, lam_new[i]);
    }

    /* U' = [U | Qv] * W_selected: two local GEMMs into one slab */
    PetscCall (MatCreateDense (g->comm, nloc, PETSC_DECIDE, Nglob,
                               PetscMax (keptn, 1), NULL, &Unew));
    {
      double             *Wsel;
      PetscCall (PetscMalloc1 ((size_t) PetscMax (n2 * keptn, 1), &Wsel));
      for (j = 0; j < keptn; j++)
        for (i = 0; i < n2; i++)
          Wsel[i + (size_t) j * n2] = T2[i + (size_t) order[j] * n2];
      if (kept0 > 0)
        PetscCall (lgh_correct_gemm_local (1.0, g->U, Wsel, (int) kept0,
                                           n2, PetscMax (keptn, 1), 0.0,
                                           Unew));
      else
        PetscCall (MatZeroEntries (Unew));
      if (r2 > 0)
        PetscCall (lgh_correct_gemm_local (1.0, Qv, Wsel + kept0, r2, n2,
                                           PetscMax (keptn, 1),
                                           kept0 > 0 ? 1.0 : 0.0, Unew));
      PetscCall (PetscFree (Wsel));
    }

    PetscCall (MatDestroy (&g->U));
    PetscCall (PetscFree (g->lam));
    g->U = Unew;
    PetscCall (PetscMalloc1 (PetscMax (keptn, 1), &g->lam));
    for (i = 0; i < keptn; i++) g->lam[i] = lam_new[i];
    g->kept = keptn;
    g->corrected = 1;
    g->floor = PetscMax (0., -lmin);
    rep->floor = g->floor;
    PetscCall (PetscFree (lam_new));
    PetscCall (PetscFree (order));
  }

  PetscCall (MatDestroy (&Qv));
  PetscCall (PetscFree (Cuv));
  PetscCall (PetscFree (Cqv));
  PetscCall (PetscFree (T2));
  PetscCall (PetscFree (ev2));
  return PETSC_SUCCESS;
}

/* Rayleigh-price the orthonormal directions QB with their images EB = E_w QB,
 * clamp, keep by |d|, build Vhat = S^+ (QB Vs), and graft.                */
static PetscErrorCode
lgh_correct_price_and_graft (lgh_glr_t *g,
                             const lgh_glr_correct_opts_t *opts,
                             lgh_glr_correct_report_t *rep, Mat QB, Mat EB)
{
  PetscInt            r, i, j, nloc, Nglob;
  double             *S, *ev, *dsel = NULL, *Csel = NULL;
  int                *order;
  int                 rkeep;
  Mat                 Vw = NULL, Vhat = NULL;
  double              t0;

  PetscCall (MatGetLocalSize (QB, &nloc, NULL));
  PetscCall (MatGetSize (QB, &Nglob, &r));
  t0 = MPI_Wtime ();
  PetscCall (PetscMalloc2 ((size_t) r * r, &S, r, &ev));
  PetscCall (lgh_correct_xtY (QB, EB, S));
  for (j = 0; j < r; j++)
    for (i = 0; i < j; i++) {
      const double        avg = 0.5 * (S[i + (size_t) j * r]
                                       + S[j + (size_t) i * r]);
      S[i + (size_t) j * r] = avg;
      S[j + (size_t) i * r] = avg;
    }
  PetscCall (lgh_correct_dsyevd ((int) r, S, ev));

  PetscCall (PetscMalloc1 (r, &order));
  for (i = 0; i < r; i++) order[i] = (int) i;
  for (i = 0; i < r; i++)
    for (j = i + 1; j < r; j++)
      if (fabs (ev[order[j]]) > fabs (ev[order[i]])) {
        const int           t = order[i]; order[i] = order[j]; order[j] = t;
      }
  rkeep = (opts->rank < 0) ? (int) r : PetscMin ((int) r, opts->rank);

  PetscCall (PetscMalloc2 (PetscMax (rkeep, 1), &dsel,
                           (size_t) r * PetscMax (rkeep, 1), &Csel));
  rep->clamped = 0;
  rep->d_min = 0.; rep->d_max = 0.;
  for (j = 0; j < rkeep; j++) {
    double              dj = ev[order[j]];
    if (dj < -1. + opts->clamp_eps) {
      dj = -1. + opts->clamp_eps;
      rep->clamped++;
    }
    dsel[j] = dj;
    rep->d_min = (j == 0) ? dj : PetscMin (rep->d_min, dj);
    rep->d_max = (j == 0) ? dj : PetscMax (rep->d_max, dj);
    for (i = 0; i < r; i++)
      Csel[i + (size_t) j * r] = S[i + (size_t) order[j] * r];
  }
  PetscCall (PetscFree (order));
  PetscCall (PetscFree2 (S, ev));

  PetscCall (MatCreateDense (g->comm, nloc, PETSC_DECIDE, Nglob,
                             PetscMax (rkeep, 1), NULL, &Vw));
  PetscCall (lgh_correct_gemm_local (1.0, QB, Csel, (int) r, (int) r,
                                     PetscMax (rkeep, 1), 0.0, Vw));
  PetscCall (MatCreateDense (g->comm, nloc, PETSC_DECIDE, Nglob,
                             PetscMax (rkeep, 1), NULL, &Vhat));
  PetscCall (lgh_glr_pow_block (g, opts->c0, +0.5, Vw, Vhat));
  PetscCall (MatDestroy (&Vw));

  PetscCall (lgh_glr_graft (g, Vhat, dsel, rkeep, opts, rep));
  rep->kept = rkeep;
  rep->t_dense += MPI_Wtime () - t0;

  PetscCall (MatDestroy (&Vhat));
  PetscCall (PetscFree2 (dsel, Csel));
  return PETSC_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* entry points                                                        */

int
lgh_glr_correct_probes (lgh_glr_t *glr, int k, const double *V,
                        const double *HV, lgh_hessian_fn hessian_apply,
                        void *ctx, const lgh_glr_correct_opts_t *opts_in,
                        lgh_glr_correct_report_t *report)
{
  lgh_glr_correct_opts_t opts = (opts_in != NULL)
      ? *opts_in : lgh_glr_correct_opts_default ();
  lgh_glr_correct_report_t rep;
  const PetscInt      nloc = glr->nloc, Nglob = glr->Nglob;
  const int           kf = k - opts.n_qc;
  Mat                 OM = NULL, YM = NULL, X = NULL, Yw = NULL;
  Mat                 TMP = NULL, Rw = NULL, IMG = NULL;
  Mat                 QB0 = NULL, Q1 = NULL, E1 = NULL, S1 = NULL, S2 = NULL;
  Mat                 QB = NULL, EB = NULL;
  Mat                 Xq = NULL, Yqw = NULL;
  int                 bdim = 0, m1, napplies = 0;
  PetscInt            j;
  double              t0;

  PetscCall (PetscMemzero (&rep, sizeof (rep)));
  rep.qc_before = -1.; rep.qc_after = -1.;
  PetscCheck (!glr->corrected, glr->comm, PETSC_ERR_ORDER,
              "lgh_glr_correct_probes: the object is already corrected");
  PetscCheck (kf > 0, glr->comm, PETSC_ERR_ARG_OUTOFRANGE,
              "lgh_glr_correct_probes: need k - n_qc > 0 probes (k=%d, "
              "n_qc=%d)", k, opts.n_qc);
  PetscCheck (opts.applies > 0, glr->comm, PETSC_ERR_ARG_OUTOFRANGE,
              "lgh_glr_correct_probes: applies budget must be positive");
  PetscCheck (opts.c0 > 0., glr->comm, PETSC_ERR_ARG_OUTOFRANGE,
              "lgh_glr_correct_probes: c0 must be positive");

  /* wrap the caller's arrays (read-only; nloc x k column-major)           */
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, kf,
                             (PetscScalar *) V, &OM));
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, kf,
                             (PetscScalar *) HV, &YM));

  /* transformed probes X = M^{-1/2} Z^T Omega (cheap applies) and true
   * whitened images Yw = M^{1/2} Z^{-1} (H_d Omega) (block Z solves) —
   * F_true X = Yw with ZERO new true applies.                             */
  t0 = MPI_Wtime ();
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, kf, NULL,
                             &X));
  for (j = 0; j < kf; j++) {
    Vec                 vj, xj;
    PetscCall (MatDenseGetColumnVecRead (OM, j, &vj));
    PetscCall (MatDenseGetColumnVecWrite (X, j, &xj));
    PetscCall (lgh_prior_applyZt_vec (glr->prior, vj, xj));
    PetscCall (MatDenseRestoreColumnVecWrite (X, j, &xj));
    PetscCall (MatDenseRestoreColumnVecRead (OM, j, &vj));
  }
  PetscCall (MatDiagonalScale (X, glr->prior->minvsqrt, NULL));
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, kf, NULL,
                             &Yw));
  PetscCall (lgh_prior_solve_block (glr->prior, LGH_PRIOR_SOLVEZ, YM, Yw));
  PetscCall (MatDiagonalScale (Yw, glr->prior->msqrt, NULL));
  rep.t_operator += MPI_Wtime () - t0;

  /* held-out QC block, and the before-correction QC                       */
  if (opts.n_qc > 0) {
    Mat                 OMq = NULL, YMq = NULL;
    PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob,
                               opts.n_qc,
                               (PetscScalar *) (V + (size_t) kf * nloc),
                               &OMq));
    PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob,
                               opts.n_qc,
                               (PetscScalar *) (HV + (size_t) kf * nloc),
                               &YMq));
    PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob,
                               opts.n_qc, NULL, &Xq));
    for (j = 0; j < opts.n_qc; j++) {
      Vec                 vj, xj;
      PetscCall (MatDenseGetColumnVecRead (OMq, j, &vj));
      PetscCall (MatDenseGetColumnVecWrite (Xq, j, &xj));
      PetscCall (lgh_prior_applyZt_vec (glr->prior, vj, xj));
      PetscCall (MatDenseRestoreColumnVecWrite (Xq, j, &xj));
      PetscCall (MatDenseRestoreColumnVecRead (OMq, j, &vj));
    }
    PetscCall (MatDiagonalScale (Xq, glr->prior->minvsqrt, NULL));
    PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob,
                               opts.n_qc, NULL, &Yqw));
    PetscCall (lgh_prior_solve_block (glr->prior, LGH_PRIOR_SOLVEZ, YMq,
                                      Yqw));
    PetscCall (MatDiagonalScale (Yqw, glr->prior->msqrt, NULL));
    PetscCall (MatDestroy (&OMq));
    PetscCall (MatDestroy (&YMq));
    PetscCall (lgh_correct_qc (glr, Xq, Yqw, &rep.qc_before));
  }

  /* residual Rw = Yw - U lam U^T X, whitened images IMG = S^- Rw — the
   * error basis, FREE of true applies.                                    */
  t0 = MPI_Wtime ();
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, kf, NULL,
                             &TMP));
  {
    double             *fv;
    PetscCall (PetscMalloc1 (PetscMax (glr->kept, 1), &fv));
    for (j = 0; j < glr->kept; j++) fv[j] = glr->lam[j];
    PetscCall (lgh_glr_filter_block (glr, fv, 0.0, X, TMP));
    PetscCall (PetscFree (fv));
  }
  PetscCall (lgh_correct_head_cols (Yw, kf, &Rw));
  PetscCall (MatAXPY (Rw, -1.0, TMP, SAME_NONZERO_PATTERN));
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, kf, NULL,
                             &IMG));
  PetscCall (lgh_glr_pow_block (glr, opts.c0, -0.5, Rw, IMG));

  PetscCall (lgh_correct_energy_basis (IMG, opts.drop_tol, 0., &QB0, &bdim));
  rep.basis = bdim;
  rep.t_dense += MPI_Wtime () - t0;
  PetscCheck (bdim > 0, glr->comm, PETSC_ERR_CONV_FAILED,
              "lgh_glr_correct_probes: the probe residuals span nothing — "
              "either the approximation is already exact on the probes or "
              "the probe data is inconsistent");

  /* price the top m1 basis directions exactly (m1 true applies)           */
  m1 = PetscMin (opts.applies, bdim);
  PetscCall (lgh_correct_head_cols (QB0, m1, &Q1));
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, m1, NULL,
                             &E1));
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, m1, NULL,
                             &S1));
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, m1, NULL,
                             &S2));
  t0 = MPI_Wtime ();
  PetscCall (lgh_correct_Ew_block (glr, opts.c0, hessian_apply, ctx, Q1, E1,
                                   S1, S2, &napplies));
  rep.t_operator += MPI_Wtime () - t0;

  if (opts.applies > m1 && m1 > 0) {
    /* one power step past the basis: fresh directions from E_w's images   */
    Mat                 P = NULL, Q2b = NULL, Q2 = NULL, E2 = NULL;
    Mat                 T1 = NULL, T2m = NULL;
    double             *C;
    int                 b2 = 0, m2;

    PetscCall (lgh_correct_head_cols (E1, m1, &P));
    PetscCall (PetscMalloc1 ((size_t) m1 * m1, &C));
    PetscCall (lgh_correct_xtY (Q1, E1, C));
    PetscCall (lgh_correct_gemm_local (-1.0, Q1, C, m1, m1, m1, 1.0, P));
    PetscCall (PetscFree (C));
    {
      double              sf;
      double             *Ge;
      PetscCall (PetscMalloc1 ((size_t) m1 * m1, &Ge));
      PetscCall (lgh_correct_xtY (E1, E1, Ge));
      sf = 0.;
      for (j = 0; j < m1; j++) sf += Ge[j + (size_t) j * m1];
      sf /= PetscMax (m1, 1);
      PetscCall (PetscFree (Ge));
      PetscCall (lgh_correct_energy_basis (P, opts.drop_tol, sf, &Q2b,
                                           &b2));
    }
    PetscCall (MatDestroy (&P));
    m2 = PetscMin (opts.applies - m1, b2);
    if (m2 > 0) {
      PetscCall (lgh_correct_head_cols (Q2b, m2, &Q2));
      PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, m2,
                                 NULL, &E2));
      PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, m2,
                                 NULL, &T1));
      PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, m2,
                                 NULL, &T2m));
      t0 = MPI_Wtime ();
      PetscCall (lgh_correct_Ew_block (glr, opts.c0, hessian_apply, ctx, Q2,
                                       E2, T1, T2m, &napplies));
      rep.t_operator += MPI_Wtime () - t0;
      PetscCall (lgh_correct_concat (Q1, Q2, &QB));
      PetscCall (lgh_correct_concat (E1, E2, &EB));
      PetscCall (MatDestroy (&Q2));
      PetscCall (MatDestroy (&E2));
      PetscCall (MatDestroy (&T1));
      PetscCall (MatDestroy (&T2m));
    }
    PetscCall (MatDestroy (&Q2b));
  }
  if (QB == NULL) {
    PetscCall (lgh_correct_head_cols (Q1, m1, &QB));
    PetscCall (lgh_correct_head_cols (E1, m1, &EB));
  }

  PetscCall (lgh_correct_price_and_graft (glr, &opts, &rep, QB, EB));
  PetscCall (lgh_glr_setup_scratch (glr));
  rep.hd_applies = napplies;

  if (opts.n_qc > 0)
    PetscCall (lgh_correct_qc (glr, Xq, Yqw, &rep.qc_after));

  if (opts.verbose) {
    PetscCall (PetscPrintf (glr->comm,
                            "[lgh correct] basis %d, priced %d applies, "
                            "kept %d (clamped %d), d in [%.3g, %.3g], "
                            "floor %.3g, qc %.3g -> %.3g\n", rep.basis,
                            rep.hd_applies, rep.kept, rep.clamped, rep.d_min,
                            rep.d_max, rep.floor, rep.qc_before,
                            rep.qc_after));
  }

  PetscCall (MatDestroy (&QB));
  PetscCall (MatDestroy (&EB));
  PetscCall (MatDestroy (&Q1));
  PetscCall (MatDestroy (&E1));
  PetscCall (MatDestroy (&S1));
  PetscCall (MatDestroy (&S2));
  PetscCall (MatDestroy (&QB0));
  PetscCall (MatDestroy (&IMG));
  PetscCall (MatDestroy (&Rw));
  PetscCall (MatDestroy (&TMP));
  PetscCall (MatDestroy (&Yw));
  PetscCall (MatDestroy (&X));
  PetscCall (MatDestroy (&YM));
  PetscCall (MatDestroy (&OM));
  PetscCall (MatDestroy (&Xq));
  PetscCall (MatDestroy (&Yqw));
  if (report != NULL) *report = rep;
  return PETSC_SUCCESS;
}

int
lgh_glr_correct (lgh_glr_t *glr, lgh_hessian_fn hessian_apply, void *ctx,
                 const lgh_glr_correct_opts_t *opts_in,
                 lgh_glr_correct_report_t *report)
{
  lgh_glr_correct_opts_t opts = (opts_in != NULL)
      ? *opts_in : lgh_glr_correct_opts_default ();
  lgh_glr_correct_report_t rep;
  const PetscInt      nloc = glr->nloc, Nglob = glr->Nglob;
  const int           passes = 2 + PetscMax (opts.q_power, 0);
  const int           mr = PetscMax (opts.applies / passes, 1);
  Mat                 T = NULL, IMG = NULL, S1 = NULL, S2 = NULL;
  Mat                 QB0 = NULL, QB = NULL, EB = NULL;
  int                 bdim = 0, m1, napplies = 0, p;
  PetscInt            i, j;
  double              t0;

  PetscCall (PetscMemzero (&rep, sizeof (rep)));
  rep.qc_before = -1.; rep.qc_after = -1.;
  PetscCheck (!glr->corrected, glr->comm, PETSC_ERR_ORDER,
              "lgh_glr_correct: the object is already corrected");
  PetscCheck (opts.applies >= 2, glr->comm, PETSC_ERR_ARG_OUTOFRANGE,
              "lgh_glr_correct: the fresh path needs applies >= 2 "
              "(range pass + Rayleigh pass)");
  PetscCheck (opts.c0 > 0., glr->comm, PETSC_ERR_ARG_OUTOFRANGE,
              "lgh_glr_correct: c0 must be positive");

  /* hashed Gaussian test block in the whitened frame                      */
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, mr, NULL,
                             &T));
  {
    PetscScalar        *ta;
    PetscInt            rlo, lda;
    PetscCall (MatGetOwnershipRange (T, &rlo, NULL));
    PetscCall (MatDenseGetLDA (T, &lda));
    PetscCall (MatDenseGetArrayWrite (T, &ta));
    for (j = 0; j < mr; j++)
      for (i = 0; i < nloc; i++)
        ta[i + (size_t) j * lda] = lgh_randn_at (opts.seed, rlo + i, j);
    PetscCall (MatDenseRestoreArrayWrite (T, &ta));
  }
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, mr, NULL,
                             &IMG));
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, mr, NULL,
                             &S1));
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, mr, NULL,
                             &S2));

  t0 = MPI_Wtime ();
  PetscCall (lgh_correct_Ew_block (glr, opts.c0, hessian_apply, ctx, T, IMG,
                                   S1, S2, &napplies));
  rep.t_operator += MPI_Wtime () - t0;
  for (p = 0; p < opts.q_power; p++) {
    Mat                 Tn = NULL;
    int                 bd = 0;
    PetscCall (lgh_correct_energy_basis (IMG, opts.drop_tol, 0., &Tn, &bd));
    PetscCall (MatDestroy (&T));
    T = Tn;
    if (bd == 0) break;
    t0 = MPI_Wtime ();
    PetscCall (lgh_correct_Ew_block (glr, opts.c0, hessian_apply, ctx, T,
                                     IMG, S1, S2, &napplies));
    rep.t_operator += MPI_Wtime () - t0;
  }

  PetscCall (lgh_correct_energy_basis (IMG, opts.drop_tol, 0., &QB0, &bdim));
  rep.basis = bdim;
  PetscCheck (bdim > 0, glr->comm, PETSC_ERR_CONV_FAILED,
              "lgh_glr_correct: the error sketch spans nothing — the "
              "approximation may already be exact");

  m1 = PetscMin (PetscMax (opts.applies - napplies, 1), bdim);
  PetscCall (lgh_correct_head_cols (QB0, m1, &QB));
  PetscCall (MatDestroy (&S1));
  PetscCall (MatDestroy (&S2));
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, m1, NULL,
                             &EB));
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, m1, NULL,
                             &S1));
  PetscCall (MatCreateDense (glr->comm, nloc, PETSC_DECIDE, Nglob, m1, NULL,
                             &S2));
  t0 = MPI_Wtime ();
  PetscCall (lgh_correct_Ew_block (glr, opts.c0, hessian_apply, ctx, QB, EB,
                                   S1, S2, &napplies));
  rep.t_operator += MPI_Wtime () - t0;

  PetscCall (lgh_correct_price_and_graft (glr, &opts, &rep, QB, EB));
  PetscCall (lgh_glr_setup_scratch (glr));
  rep.hd_applies = napplies;

  PetscCall (MatDestroy (&QB));
  PetscCall (MatDestroy (&EB));
  PetscCall (MatDestroy (&QB0));
  PetscCall (MatDestroy (&S1));
  PetscCall (MatDestroy (&S2));
  PetscCall (MatDestroy (&IMG));
  PetscCall (MatDestroy (&T));
  if (report != NULL) *report = rep;
  return PETSC_SUCCESS;
}

#endif /* LGPSF_HESSIAN_CORRECT_IMPL_H */

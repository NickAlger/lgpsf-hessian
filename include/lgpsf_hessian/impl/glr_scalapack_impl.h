/* glr_scalapack_impl.h — the distributed-feature-space (ScaLAPACK) backend
 * of lgh_glr_t.  Compiled inside the single consumer TU, after
 * glr_common_impl.h, when LGH_WITH_SCALAPACK is defined.
 *
 * Provenance: the production distributed GLR engine of a large-scale
 * ice-sheet inversion, per its design record (docs/ once migrated):
 *   - orthogonalization by panel-blocked BCGS2: no ell x ell Gram matrix is
 *     ever materialized;
 *   - T = sym(Q^T F Q) is formed in column panels and reduced directly into
 *     a 2D block-cyclic layout on a BLACS process grid (feature space owns
 *     its own partition — by feature index, never by mesh geometry);
 *   - eigendecomposition by ScaLAPACK pdsyevd; the eigenvector matrix V
 *     STAYS IN PLACE (block-cyclic); eigenvalues are replicated (small);
 *   - U = Q V_kept is IMPLICIT: U w = Q(V w), U^T x = V^T(Q^T x), with
 *     O(ell) coefficient traffic per apply;
 *   - incremental extension: fresh hashed columns continue the sketch, are
 *     BCGS2-orthogonalized against the existing Q, and border the stored
 *     raw A = Q^T F Q (1D cyclic column store); re-eigensolve, re-select.
 *
 * Process grid: DEFAULT 1 x P (always safe); opts.grid2d = 1 requests the
 * near-square grid.  ROOT CAUSE of historical 2D-grid crashes: on 2D grids
 * pdlarfb's row-scope BLACS broadcast of the trapezoidal T factor can
 * produce an EMPTY piece on some rank; BLACS then sends a zero-byte derived
 * datatype, and MPICH (ch3, observed on 4.2.1) hits an unguarded division
 * 0/0 (SIGFPE) in MPIR_Datatype_get_density.  BLACS has carried a
 * workaround since the MPICH-1 era — compile ScaLAPACK with
 * -DZeroByteTypeBug — but neither the reference CMake build nor PETSc's
 * --download-scalapack enables it.  Use 1 x P unless your ScaLAPACK is
 * known healthy (see docs/platform-notes.md).
 *
 * Eigenvector signs / near-degenerate rotations depend on the process
 * grid: cross-P and cross-backend comparisons must use invariants only
 * (Lambda, filtered applies, logdet).
 */

#ifndef LGPSF_HESSIAN_GLR_SCALAPACK_IMPL_H
#define LGPSF_HESSIAN_GLR_SCALAPACK_IMPL_H

#ifndef LGPSF_HESSIAN_GLR_COMMON_IMPL_H
#error "include impl/glr_common_impl.h first (impl.hpp does this)"
#endif

#ifdef __cplusplus
extern "C"
{
#endif
/* BLACS C interface (reference ScaLAPACK) */
extern int          Csys2blacs_handle (MPI_Comm comm);
extern void         Cblacs_gridinit (int *ictxt, const char *layout,
                                     int nprow, int npcol);
extern void         Cblacs_gridinfo (int ictxt, int *nprow, int *npcol,
                                     int *myrow, int *mycol);
extern void         Cblacs_gridexit (int ictxt);

#define LGH_SCALAPACK_PDSYEVD LGH_F77_FUNC (pdsyevd, PDSYEVD)
#define LGH_SCALAPACK_DESCINIT LGH_F77_FUNC (descinit, DESCINIT)
extern void LGH_SCALAPACK_PDSYEVD (const char *jobz, const char *uplo,
                                   int *n, double *a, int *ia, int *ja,
                                   int *desca, double *w, double *z, int *iz,
                                   int *jz, int *descz, double *work,
                                   int *lwork, int *iwork, int *liwork,
                                   int *info);
extern void LGH_SCALAPACK_DESCINIT (int *desc, int *m, int *n, int *mb,
                                    int *nb, int *irsrc, int *icsrc,
                                    int *ictxt, int *lld, int *info);
#ifdef __cplusplus
}
#endif

/* ------------------------------------------------------------------ */

struct lgh_glrd_state
{
  int                 P, rank;
  int                 ictxt, prow, pcol, myrow, mycol;  /* BLACS grid    */
  int                 nb;          /* block-cyclic block size            */
  int                 ell;         /* sketch size (mirrors g->opts.ell)  */
  int                 desc[9];     /* descriptor for T / V               */
  int                 tl_rows, tl_cols;  /* local block sizes            */
  double             *Vloc;        /* eigenvectors, block-cyclic, in place */
  double             *lam_raw;     /* ell, pdsyevd-ascending (replicated) */
  int                *keep;        /* kept -> raw index                  */
  int                *kpos;        /* raw index -> kept position or -1   */
  Mat                 Q;           /* N x ell MATDENSE row slab (owned)  */
  double             *cwork;       /* scratch: 3*ell                     */
  /* 1D cyclic-by-column store of the raw (unsymmetrized) A = Q^T F Q,
   * ell rows x acn local columns (global j = lc*P + rank): the bordered-T
   * source for incremental extension.  New rows of old columns are filled
   * by symmetry of F.                                                   */
  double             *Acols;
  int                 acn;
};

/* block-cyclic ownership helpers (row-src / col-src fixed at process 0) */
static int
lgh_glrd_owner (int gidx, int nb, int np)
{
  return (gidx / nb) % np;
}

static int
lgh_glrd_lidx (int gidx, int nb, int np)
{
  return (gidx / (nb * np)) * nb + gidx % nb;
}

/* ------------------------------------------------------------------ */
/* panel-blocked BCGS2 on the local slab of an N x ell MATDENSE.        */

static PetscErrorCode
lgh_glrd_panel_cholqr (double *pa, PetscInt nloc, PetscInt lda, int bw,
                       MPI_Comm comm)
{
  double             *G, *Gl;
  int                 pass, i, j, info;
  int                 bm = (int) nloc, bn = bw, blda = (int) lda;
  double              one = 1.0, zero = 0.0;

  PetscCall (PetscMalloc2 ((size_t) bw * bw, &G, (size_t) bw * bw, &Gl));
  for (pass = 0; pass < 2; pass++) {   /* CholQR2 */
    if (nloc > 0) {
      LGH_BLAS_DGEMM ("T", "N", &bn, &bn, &bm, &one, pa, &blda, pa, &blda,
                      &zero, Gl, &bn);
    }
    else { for (i = 0; i < bw * bw; i++) Gl[i] = 0.; }
    PetscCallMPI (MPI_Allreduce (Gl, G, bw * bw, MPI_DOUBLE, MPI_SUM, comm));
    info = 0;
    LGH_LAPACK_DPOTRF ("U", &bn, G, &bn, &info);
    if (info != 0) {                 /* jitter once */
      double              dmax = 0.;
      PetscCallMPI (MPI_Allreduce (Gl, G, bw * bw, MPI_DOUBLE, MPI_SUM,
                                   comm));
      for (i = 0; i < bw; i++) dmax = PetscMax (dmax, G[i + i * bw]);
      for (i = 0; i < bw; i++) G[i + i * bw] += 1.e-12 * PetscMax (dmax, 1.);
      info = 0;
      LGH_LAPACK_DPOTRF ("U", &bn, G, &bn, &info);
      PetscCheck (info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB,
                  "lgh_glrd_panel_cholqr: dpotrf failed twice");
    }
    for (j = 0; j < bw; j++)           /* zero strict lower of R */
      for (i = j + 1; i < bw; i++) G[i + j * bw] = 0.;
    if (nloc > 0) {
      LGH_BLAS_DTRSM ("R", "U", "N", "N", &bm, &bn, &one, G, &bn, pa, &blda);
    }
  }
  PetscCall (PetscFree2 (G, Gl));
  return PETSC_SUCCESS;
}

/* Orthonormalize columns [p_start, ell) of Y against columns [0, p_start)
 * (assumed already orthonormal) and each other; p_start = 0 is the full
 * factorization.  Used with p_start = ell1 for incremental extension.   */
static PetscErrorCode
lgh_glrd_bcgs2 (Mat Y, int ell, int b, int p_start, MPI_Comm comm)
{
  PetscScalar        *ya;
  PetscInt            nloc, lda;
  double             *C, *Cl;
  int                 p0;

  PetscCall (MatGetLocalSize (Y, &nloc, NULL));
  PetscCall (MatDenseGetLDA (Y, &lda));
  PetscCall (MatDenseGetArray (Y, &ya));
  PetscCall (PetscMalloc2 ((size_t) ell * b, &C, (size_t) ell * b, &Cl));
  for (p0 = p_start; p0 < ell; p0 += b) {
    const int           bw = PetscMin (b, ell - p0);
    double             *pa = ya + (size_t) p0 * lda;
    int                 sweep;

    for (sweep = 0; sweep < 2; sweep++) {   /* BCGS2: two full sweeps */
      if (p0 > 0) {
        int                 bm = (int) nloc, bj = p0, bb = bw;
        int                 blda = (int) lda;
        double              one = 1.0, zero = 0.0, mone = -1.0;
        int                 i;

        if (nloc > 0) {
          LGH_BLAS_DGEMM ("T", "N", &bj, &bb, &bm, &one, ya, &blda, pa,
                          &blda, &zero, Cl, &bj);
        }
        else { for (i = 0; i < p0 * bw; i++) Cl[i] = 0.; }
        PetscCallMPI (MPI_Allreduce (Cl, C, p0 * bw, MPI_DOUBLE, MPI_SUM,
                                     comm));
        if (nloc > 0) {
          LGH_BLAS_DGEMM ("N", "N", &bm, &bb, &bj, &mone, ya, &blda, C,
                          &bj, &one, pa, &blda);
        }
      }
      PetscCall (lgh_glrd_panel_cholqr (pa, nloc, lda, bw, comm));
    }
  }
  PetscCall (PetscFree2 (C, Cl));
  PetscCall (MatDenseRestoreArray (Y, &ya));
  return PETSC_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* T = sym(Q^T Y), streamed in column panels of width b, reduced with an
 * allreduce per panel and scattered into the 2D block-cyclic Tloc with
 * symmetrization applied during the scatter.  Optionally accumulates a
 * rank-0 replicated copy of A (the pdsyevd-vs-dsyevd cross-check).      */
static PetscErrorCode
lgh_glrd_form_T (lgh_glr_t *g, Mat Q, Mat Y, int b, double *Tloc,
                 double *Arep)
{
  struct lgh_glrd_state *s = g->dist;
  const int           ell = s->ell, nb = s->nb;
  const PetscScalar  *qa, *ya;
  PetscInt            nloc, ldq, ldy;
  double             *C, *Cl;
  int                 p0, i, j, jj;

  PetscCall (MatGetLocalSize (Q, &nloc, NULL));
  PetscCall (MatDenseGetLDA (Q, &ldq));
  PetscCall (MatDenseGetLDA (Y, &ldy));
  PetscCall (MatDenseGetArrayRead (Q, &qa));
  PetscCall (MatDenseGetArrayRead (Y, &ya));
  PetscCall (PetscMalloc2 ((size_t) ell * b, &C, (size_t) ell * b, &Cl));
  for (i = 0; i < s->tl_rows * s->tl_cols; i++) Tloc[i] = 0.;

  for (p0 = 0; p0 < ell; p0 += b) {
    const int           bw = PetscMin (b, ell - p0);
    int                 bm = (int) nloc, bl = ell, bb = bw;
    int                 bldq = (int) ldq, bldy = (int) ldy;
    double              one = 1.0, zero = 0.0;

    if (nloc > 0) {
      LGH_BLAS_DGEMM ("T", "N", &bl, &bb, &bm, &one, qa, &bldq,
                      ya + (size_t) p0 * ldy, &bldy, &zero, Cl, &bl);
    }
    else { for (i = 0; i < ell * bw; i++) Cl[i] = 0.; }
    PetscCallMPI (MPI_Allreduce (Cl, C, ell * bw, MPI_DOUBLE, MPI_SUM,
                                 g->comm));

    if (Arep != NULL && s->rank == 0) {     /* cross-check replicated A */
      for (jj = 0; jj < bw; jj++)
        for (i = 0; i < ell; i++)
          Arep[i + (size_t) (p0 + jj) * ell] = C[i + (size_t) jj * ell];
    }

    /* 1D cyclic column store of raw A (extension source): j = lc*P + rank */
    for (jj = 0; jj < bw; jj++) {
      j = p0 + jj;
      if (j % s->P == s->rank)
        for (i = 0; i < ell; i++)
          s->Acols[i + (size_t) (j / s->P) * ell] = C[i + (size_t) jj * ell];
    }

    /* scatter with symmetrization: owned (i,j), j in panel -> 0.5*A(i,j) */
    for (jj = 0; jj < bw; jj++) {
      j = p0 + jj;
      if (lgh_glrd_owner (j, nb, s->pcol) != s->mycol) continue;
      {
        const int           lj = lgh_glrd_lidx (j, nb, s->pcol);
        for (i = 0; i < ell; i++) {
          if (lgh_glrd_owner (i, nb, s->prow) != s->myrow) continue;
          Tloc[lgh_glrd_lidx (i, nb, s->prow) + (size_t) lj * s->tl_rows] +=
              0.5 * C[i + (size_t) jj * ell];
        }
      }
    }
    /* owned (i,j), i in panel -> 0.5*A(j,i) = 0.5*C[j, i-p0] */
    for (jj = 0; jj < bw; jj++) {
      i = p0 + jj;
      if (lgh_glrd_owner (i, nb, s->prow) != s->myrow) continue;
      {
        const int           li = lgh_glrd_lidx (i, nb, s->prow);
        for (j = 0; j < ell; j++) {
          if (lgh_glrd_owner (j, nb, s->pcol) != s->mycol) continue;
          Tloc[li + (size_t) lgh_glrd_lidx (j, nb, s->pcol) * s->tl_rows] +=
              0.5 * C[j + (size_t) jj * ell];
        }
      }
    }
  }
  PetscCall (PetscFree2 (C, Cl));
  PetscCall (MatDenseRestoreArrayRead (Y, &ya));
  PetscCall (MatDenseRestoreArrayRead (Q, &qa));
  return PETSC_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* implicit-U primitives (coefficients replicated; O(ell) traffic)      */

/* w[kept] = U^T x = V_kept^T (Q^T x) */
static PetscErrorCode
lgh_glrd_UTmult (lgh_glr_t *g, Vec x, double *w)
{
  struct lgh_glrd_state *s = g->dist;
  const PetscScalar  *qa, *xa;
  PetscInt            nloc, ldq;
  double             *cl = s->cwork, *cf = s->cwork + s->ell;
  double             *wl = s->cwork + 2 * s->ell;
  int                 i, lj, k;

  PetscCall (MatGetLocalSize (s->Q, &nloc, NULL));
  PetscCall (MatDenseGetLDA (s->Q, &ldq));
  PetscCall (MatDenseGetArrayRead (s->Q, &qa));
  PetscCall (VecGetArrayRead (x, &xa));
  if (nloc > 0) {
    int                 bm = (int) nloc, bl = s->ell;
    int                 bldq = (int) ldq, ione = 1;
    double              one = 1.0, zero = 0.0;
    LGH_BLAS_DGEMV ("T", &bm, &bl, &one, qa, &bldq, (const double *) xa,
                    &ione, &zero, cl, &ione);
  }
  else { for (i = 0; i < s->ell; i++) cl[i] = 0.; }
  PetscCall (VecRestoreArrayRead (x, &xa));
  PetscCall (MatDenseRestoreArrayRead (s->Q, &qa));
  PetscCallMPI (MPI_Allreduce (cl, cf, s->ell, MPI_DOUBLE, MPI_SUM,
                               g->comm));

  for (k = 0; k < g->kept; k++) wl[k] = 0.;
  for (lj = 0; lj < s->tl_cols; lj++) {
    const int           gj = (lj / s->nb) * s->nb * s->pcol
                              + s->mycol * s->nb + lj % s->nb;
    if (gj >= s->ell || (k = s->kpos[gj]) < 0) continue;
    for (i = 0; i < s->tl_rows; i++) {
      const int           gi = (i / s->nb) * s->nb * s->prow
                                + s->myrow * s->nb + i % s->nb;
      if (gi >= s->ell) continue;
      wl[k] += s->Vloc[i + (size_t) lj * s->tl_rows] * cf[gi];
    }
  }
  PetscCallMPI (MPI_Allreduce (wl, w, (int) g->kept, MPI_DOUBLE, MPI_SUM,
                               g->comm));
  return PETSC_SUCCESS;
}

/* y (+)= U w = Q (V_kept w); if add == PETSC_FALSE, y is overwritten */
static PetscErrorCode
lgh_glrd_Umult (lgh_glr_t *g, const double *w, Vec y, PetscBool add)
{
  struct lgh_glrd_state *s = g->dist;
  const PetscScalar  *qa;
  PetscScalar        *yal;
  PetscInt            nloc, ldq;
  double             *cl = s->cwork, *cf = s->cwork + s->ell;
  int                 i, lj, k;

  for (i = 0; i < s->ell; i++) cl[i] = 0.;
  for (lj = 0; lj < s->tl_cols; lj++) {
    const int           gj = (lj / s->nb) * s->nb * s->pcol
                              + s->mycol * s->nb + lj % s->nb;
    if (gj >= s->ell || (k = s->kpos[gj]) < 0) continue;
    for (i = 0; i < s->tl_rows; i++) {
      const int           gi = (i / s->nb) * s->nb * s->prow
                                + s->myrow * s->nb + i % s->nb;
      if (gi >= s->ell) continue;
      cl[gi] += s->Vloc[i + (size_t) lj * s->tl_rows] * w[k];
    }
  }
  PetscCallMPI (MPI_Allreduce (cl, cf, s->ell, MPI_DOUBLE, MPI_SUM,
                               g->comm));

  PetscCall (MatGetLocalSize (s->Q, &nloc, NULL));
  PetscCall (MatDenseGetLDA (s->Q, &ldq));
  PetscCall (MatDenseGetArrayRead (s->Q, &qa));
  PetscCall (VecGetArray (y, &yal));
  if (nloc > 0) {
    int                 bm = (int) nloc, bl = s->ell;
    int                 bldq = (int) ldq, ione = 1;
    double              one = 1.0, beta = add ? 1.0 : 0.0;
    LGH_BLAS_DGEMV ("N", &bm, &bl, &one, qa, &bldq, cf, &ione, &beta, yal,
                    &ione);
  }
  PetscCall (VecRestoreArray (y, &yal));
  PetscCall (MatDenseRestoreArrayRead (s->Q, &qa));
  return PETSC_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* selection: rank by |lam_raw| descending, truncate, treat.  Fills
 * g->{lam, kept} and s->{keep, kpos}; reports.  Shared by build/extend. */
static PetscErrorCode
lgh_glrd_select (lgh_glr_t *g, lgh_glr_report_t *rep)
{
  struct lgh_glrd_state *s = g->dist;
  const int           ell = s->ell;
  int                *order;
  double              absmax, cut;
  int                 kept = 0, i, j;

  PetscCall (PetscMalloc1 (ell, &order));
  for (i = 0; i < ell; i++) order[i] = i;
  for (i = 0; i < ell; i++)
    for (j = i + 1; j < ell; j++)
      if (fabs (s->lam_raw[order[j]]) > fabs (s->lam_raw[order[i]])) {
        int                 t = order[i]; order[i] = order[j]; order[j] = t;
      }
  absmax = fabs (s->lam_raw[order[0]]);
  cut = 0.;
  if (g->opts.trunc_abs > 0.) cut = PetscMax (cut, g->opts.trunc_abs);
  if (g->opts.trunc_rel > 0.) cut = PetscMax (cut, g->opts.trunc_rel * absmax);
  while (kept < ell && fabs (s->lam_raw[order[kept]]) > cut) kept++;
  g->kept = kept;
  rep->kept = kept;
  rep->lam_abs_max = absmax;
  rep->next_abs = (kept < ell) ? fabs (s->lam_raw[order[kept]]) : 0.;
  rep->lam_raw_max = s->lam_raw[ell - 1];   /* pdsyevd ascending */
  rep->lam_raw_min = s->lam_raw[0];
  rep->n_negative_raw = 0;
  for (i = 0; i < ell; i++)
    if (s->lam_raw[i] < 0.) rep->n_negative_raw++;

  PetscCall (PetscFree (g->lam));
  PetscCall (PetscFree (s->keep));
  PetscCall (PetscFree (s->kpos));
  PetscCall (PetscMalloc1 (PetscMax (kept, 1), &g->lam));
  PetscCall (PetscMalloc1 (PetscMax (kept, 1), &s->keep));
  PetscCall (PetscMalloc1 (ell, &s->kpos));
  for (i = 0; i < ell; i++) s->kpos[i] = -1;
  for (i = 0; i < kept; i++) {
    double              l = s->lam_raw[order[i]];
    s->keep[i] = order[i];
    s->kpos[order[i]] = i;
    switch (g->opts.treat) {
    case LGH_TREAT_FLIP: g->lam[i] = fabs (l); break;
    case LGH_TREAT_RELU: g->lam[i] = PetscMax (l, 0.); break;
    default:             g->lam[i] = l; break;
    }
  }
  PetscCall (PetscFree (order));
  return PETSC_SUCCESS;
}

/* block-cyclic local extents + descriptor at the current s->ell */
static PetscErrorCode
lgh_glrd_set_desc (struct lgh_glrd_state *s)
{
  int                 izero = 0, info = 0, lld;
  int                 nell = s->ell, nnb = s->nb;

  s->tl_rows = (s->ell / (s->nb * s->prow)) * s->nb;
  s->tl_cols = (s->ell / (s->nb * s->pcol)) * s->nb;
  /* numroc by hand: remaining block */
  {
    int                 extra = s->ell - s->tl_rows * s->prow;
    int                 myoff = s->myrow * s->nb;
    s->tl_rows += PetscMin (PetscMax (extra - myoff, 0), s->nb);
  }
  {
    int                 extra = s->ell - s->tl_cols * s->pcol;
    int                 myoff = s->mycol * s->nb;
    s->tl_cols += PetscMin (PetscMax (extra - myoff, 0), s->nb);
  }
  lld = PetscMax (s->tl_rows, 1);
  LGH_SCALAPACK_DESCINIT (s->desc, &nell, &nell, &nnb, &nnb, &izero, &izero,
                          &s->ictxt, &lld, &info);
  PetscCheck (info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB,
              "lgh_glrd: descinit failed");
  return PETSC_SUCCESS;
}

static PetscErrorCode
lgh_glrd_pdsyevd (struct lgh_glrd_state *s, double *Tloc)
{
  int                 nell = s->ell, ione = 1, info = 0;
  int                 lwork = -1, liwork = -1, iwkopt = 0;
  double              wkopt;
  double             *work;
  int                *iwork;

  LGH_SCALAPACK_PDSYEVD ("V", "L", &nell, Tloc, &ione, &ione, s->desc,
                         s->lam_raw, s->Vloc, &ione, &ione, s->desc, &wkopt,
                         &lwork, &iwkopt, &liwork, &info);
  lwork = (int) wkopt; liwork = iwkopt;
  PetscCall (PetscMalloc1 (lwork, &work));
  PetscCall (PetscMalloc1 (liwork, &iwork));
  LGH_SCALAPACK_PDSYEVD ("V", "L", &nell, Tloc, &ione, &ione, s->desc,
                         s->lam_raw, s->Vloc, &ione, &ione, s->desc, work,
                         &lwork, iwork, &liwork, &info);
  PetscCall (PetscFree (work));
  PetscCall (PetscFree (iwork));
  PetscCheck (info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB,
              "lgh_glrd: pdsyevd failed");
  return PETSC_SUCCESS;
}

/* ------------------------------------------------------------------ */

static void
lgh_glrd_destroy_state (lgh_glr_t *g)
{
  struct lgh_glrd_state *s = g->dist;

  if (s == NULL) return;
  if (s->Q != NULL) (void) MatDestroy (&s->Q);
  (void) PetscFree (s->Vloc);
  (void) PetscFree (s->lam_raw);
  (void) PetscFree (s->keep);
  (void) PetscFree (s->kpos);
  (void) PetscFree (s->cwork);
  (void) PetscFree (s->Acols);
  Cblacs_gridexit (s->ictxt);
  (void) PetscFree (s);
  g->dist = NULL;
}

/* Build: hashed sketch -> BCGS2 -> block-cyclic T -> pdsyevd -> select.
 * opts.check additionally runs the streamed BCGS2 orthonormality check
 * and the rank-0 replicated-dsyevd cross-check; the cross-check's max
 * |dlam| (relative) is surfaced in rep->resid_max for this backend.     */
static PetscErrorCode
lgh_glrd_build (lgh_glr_t *g, lgh_glr_report_t *rep)
{
  struct lgh_glrd_state *s;
  Mat                 Q, Y, W;
  double             *Tloc = NULL, *Arep = NULL;
  const int           ell = (int) g->opts.ell;
  const int           panel_b = g->opts.panel;
  const PetscInt      nloc = g->nloc, Nglob = g->Nglob;
  double              t0;
  int                 i, j, p;

  PetscCall (PetscMemzero (rep, sizeof (*rep)));
  PetscCall (PetscNew (&s));
  g->dist = s;
  PetscCallMPI (MPI_Comm_size (g->comm, &s->P));
  PetscCallMPI (MPI_Comm_rank (g->comm, &s->rank));
  s->nb = g->opts.nb;
  s->ell = ell;

  /* BLACS grid: 1 x P default; near-square when opts.grid2d (see header) */
  {
    int                 pr = 1;
    if (g->opts.grid2d) {
      for (i = 1; (double) i * i <= (double) s->P; i++)
        if (s->P % i == 0) pr = i;
    }
    s->prow = pr;
    s->pcol = s->P / pr;
    s->ictxt = Csys2blacs_handle (g->comm);
    Cblacs_gridinit (&s->ictxt, "R", s->prow, s->pcol);
    Cblacs_gridinfo (s->ictxt, &s->prow, &s->pcol, &s->myrow, &s->mycol);
  }
  PetscCall (lgh_glrd_set_desc (s));
  PetscCall (PetscCalloc1 ((size_t) PetscMax (s->tl_rows * s->tl_cols, 1),
                           &Tloc));
  PetscCall (PetscCalloc1 ((size_t) PetscMax (s->tl_rows * s->tl_cols, 1),
                           &s->Vloc));
  PetscCall (PetscMalloc1 (ell, &s->lam_raw));
  PetscCall (PetscMalloc1 (3 * ell, &s->cwork));
  s->acn = (ell + s->P - 1 - s->rank) / s->P;
  PetscCall (PetscMalloc1 ((size_t) PetscMax (s->acn, 1) * ell, &s->Acols));
  if (g->opts.check) {
    PetscCall (PetscCalloc1 (s->rank == 0 ? (size_t) ell * ell : 1, &Arep));
  }

  /* tall blocks + hashed Omega */
  PetscCall (MatCreateDense (g->comm, nloc, PETSC_DECIDE, Nglob, ell, NULL,
                             &Q));
  PetscCall (MatCreateDense (g->comm, nloc, PETSC_DECIDE, Nglob, ell, NULL,
                             &Y));
  PetscCall (MatCreateDense (g->comm, nloc, PETSC_DECIDE, Nglob, ell, NULL,
                             &W));
  {
    PetscScalar        *qa;
    PetscInt            rlo, rhi, lda;

    PetscCall (MatGetOwnershipRange (Q, &rlo, &rhi));
    PetscCall (MatDenseGetLDA (Q, &lda));
    PetscCall (MatDenseGetArrayWrite (Q, &qa));
    for (j = 0; j < ell; j++)
      for (i = 0; i < (int) (rhi - rlo); i++)
        qa[i + (size_t) j * lda] = lgh_randn_at (g->opts.seed, rlo + i, j);
    PetscCall (MatDenseRestoreArrayWrite (Q, &qa));
  }

  /* Y = F Omega; q power passes with BCGS2 re-orthonormalization */
  t0 = MPI_Wtime ();
  PetscCall (lgh_glr_apply_F_block (g, Q, Y, W));
  rep->t_operator += MPI_Wtime () - t0;
  for (p = 0; p < g->opts.q_power; p++) {
    t0 = MPI_Wtime ();
    PetscCall (lgh_glrd_bcgs2 (Y, ell, panel_b, 0, g->comm));
    rep->t_dense += MPI_Wtime () - t0;
    PetscCall (MatCopy (Y, Q, SAME_NONZERO_PATTERN));
    t0 = MPI_Wtime ();
    PetscCall (lgh_glr_apply_F_block (g, Q, Y, W));
    rep->t_operator += MPI_Wtime () - t0;
  }
  t0 = MPI_Wtime ();
  PetscCall (lgh_glrd_bcgs2 (Y, ell, panel_b, 0, g->comm));
  rep->t_dense += MPI_Wtime () - t0;
  PetscCall (MatCopy (Y, Q, SAME_NONZERO_PATTERN));
  t0 = MPI_Wtime ();
  PetscCall (lgh_glr_apply_F_block (g, Q, Y, W));
  rep->t_operator += MPI_Wtime () - t0;

  if (g->opts.check) {   /* streamed orthonormality check, no ell^2 storage */
    const PetscScalar  *qa;
    PetscInt            ldq;
    double             *C, *Cl, dmax = 0., gmax;
    int                 p0, jj;

    PetscCall (MatDenseGetLDA (Q, &ldq));
    PetscCall (MatDenseGetArrayRead (Q, &qa));
    PetscCall (PetscMalloc2 ((size_t) ell * panel_b, &C,
                             (size_t) ell * panel_b, &Cl));
    for (p0 = 0; p0 < ell; p0 += panel_b) {
      const int           bw = PetscMin (panel_b, ell - p0);
      int                 bm = (int) nloc, bl = ell, bb = bw;
      int                 bldq = (int) ldq;
      double              one = 1.0, zero = 0.0;

      if (nloc > 0) {
        LGH_BLAS_DGEMM ("T", "N", &bl, &bb, &bm, &one, qa, &bldq,
                        qa + (size_t) p0 * ldq, &bldq, &zero, Cl, &bl);
      }
      else { for (i = 0; i < ell * bw; i++) Cl[i] = 0.; }
      PetscCallMPI (MPI_Allreduce (Cl, C, ell * bw, MPI_DOUBLE, MPI_SUM,
                                   g->comm));
      for (jj = 0; jj < bw; jj++)
        for (i = 0; i < ell; i++)
          dmax = PetscMax (dmax, fabs (C[i + (size_t) jj * ell]
                                       - (i == p0 + jj ? 1. : 0.)));
    }
    PetscCall (PetscFree2 (C, Cl));
    PetscCall (MatDenseRestoreArrayRead (Q, &qa));
    PetscCallMPI (MPI_Allreduce (&dmax, &gmax, 1, MPI_DOUBLE, MPI_MAX,
                                 g->comm));
    PetscCall (PetscInfo ((PetscObject) Q,
                          "lgh_glrd: BCGS2 orthonormality |Q^T Q - I|_max "
                          "= %.3e\n", gmax));
  }

  /* T = sym(Q^T Y), streamed into block-cyclic; then pdsyevd (V in place) */
  t0 = MPI_Wtime ();
  PetscCall (lgh_glrd_form_T (g, Q, Y, panel_b, Tloc, Arep));
  PetscCall (lgh_glrd_pdsyevd (s, Tloc));
  rep->t_dense += MPI_Wtime () - t0;

  if (g->opts.check) {          /* replicated dsyevd cross-check, rank 0 */
    double              dmax = 0., amax = 0., rel = 0.;

    if (s->rank == 0) {
      double             *evr, *work, wkopt;
      int                *iwork, iwkopt = 0;
      int                 nell = ell, lwork = -1, liwork = -1, info = 0;

      for (j = 0; j < ell; j++)           /* symmetrize the replicated A */
        for (i = 0; i < j; i++) {
          double              avg = 0.5 * (Arep[i + (size_t) j * ell]
                                           + Arep[j + (size_t) i * ell]);
          Arep[i + (size_t) j * ell] = avg;
          Arep[j + (size_t) i * ell] = avg;
        }
      PetscCall (PetscMalloc1 (ell, &evr));
      LGH_LAPACK_DSYEVD ("N", "L", &nell, Arep, &nell, evr, &wkopt, &lwork,
                         &iwkopt, &liwork, &info);
      lwork = (int) wkopt; liwork = iwkopt;
      PetscCall (PetscMalloc1 (lwork, &work));
      PetscCall (PetscMalloc1 (liwork, &iwork));
      LGH_LAPACK_DSYEVD ("N", "L", &nell, Arep, &nell, evr, work, &lwork,
                         iwork, &liwork, &info);
      if (info == 0) {
        for (i = 0; i < ell; i++) {
          dmax = PetscMax (dmax, fabs (evr[i] - s->lam_raw[i]));
          amax = PetscMax (amax, fabs (evr[i]));
        }
        rel = dmax / PetscMax (amax, 1e-300);
      }
      PetscCall (PetscFree (evr));
      PetscCall (PetscFree (work));
      PetscCall (PetscFree (iwork));
    }
    PetscCallMPI (MPI_Bcast (&rel, 1, MPI_DOUBLE, 0, g->comm));
    rep->resid_max = rel;   /* this backend: pdsyevd-vs-dsyevd rel |dlam| */
    PetscCall (PetscFree (Arep));
  }

  PetscCall (lgh_glrd_select (g, rep));

  PetscCall (PetscFree (Tloc));
  PetscCall (MatDestroy (&Y));
  PetscCall (MatDestroy (&W));
  s->Q = Q;
  return PETSC_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Incremental rank growth: fresh hashed columns continue the sketch
 * (deterministic, partition-independent), q power passes on the new block
 * (internal orth only — span-equivalent to one-shot), BCGS2 against the
 * existing Q, border the stored A with C = Q_all^T (F Q_new) (the
 * (new,old) block comes from (old,new)^T by symmetry of F), rebuild the
 * block-cyclic T from the column store, re-eigensolve, re-select.       */
static PetscErrorCode
lgh_glrd_extend_incr (lgh_glr_t *g, int k_new, lgh_glr_report_t *rep)
{
  struct lgh_glrd_state *s = g->dist;
  const int           ell1 = s->ell, ell2 = s->ell + k_new;
  const int           panel_b = g->opts.panel;
  Mat                 Qb, Qn, Yn, Wn;
  double             *Tloc, *Cnew, *Cl;
  double              t0;
  int                 i, j, jj, p, p0, lc;

  PetscCall (PetscMemzero (rep, sizeof (*rep)));

  /* Q grows: new N x ell2 slab, old columns copied, new ones hashed */
  PetscCall (MatCreateDense (g->comm, g->nloc, PETSC_DECIDE, g->Nglob, ell2,
                             NULL, &Qb));
  PetscCall (MatCreateDense (g->comm, g->nloc, PETSC_DECIDE, g->Nglob,
                             k_new, NULL, &Qn));
  PetscCall (MatCreateDense (g->comm, g->nloc, PETSC_DECIDE, g->Nglob,
                             k_new, NULL, &Yn));
  PetscCall (MatCreateDense (g->comm, g->nloc, PETSC_DECIDE, g->Nglob,
                             k_new, NULL, &Wn));
  {
    const PetscScalar  *qa;
    PetscScalar        *qb, *qn;
    PetscInt            rlo, rhi, ldq, ldb, ldn;

    PetscCall (MatGetOwnershipRange (Qb, &rlo, &rhi));
    PetscCall (MatDenseGetLDA (s->Q, &ldq));
    PetscCall (MatDenseGetLDA (Qb, &ldb));
    PetscCall (MatDenseGetLDA (Qn, &ldn));
    PetscCall (MatDenseGetArrayRead (s->Q, &qa));
    PetscCall (MatDenseGetArrayWrite (Qb, &qb));
    PetscCall (MatDenseGetArrayWrite (Qn, &qn));
    for (j = 0; j < ell1; j++)
      for (i = 0; i < (int) (rhi - rlo); i++)
        qb[i + (size_t) j * ldb] = qa[i + (size_t) j * ldq];
    for (j = 0; j < k_new; j++)
      for (i = 0; i < (int) (rhi - rlo); i++)
        qn[i + (size_t) j * ldn] =
            lgh_randn_at (g->opts.seed, rlo + i, ell1 + j);
    PetscCall (MatDenseRestoreArrayWrite (Qn, &qn));
    PetscCall (MatDenseRestoreArrayWrite (Qb, &qb));
    PetscCall (MatDenseRestoreArrayRead (s->Q, &qa));
  }

  /* power passes on the new block (internal orth only) */
  t0 = MPI_Wtime ();
  PetscCall (lgh_glr_apply_F_block (g, Qn, Yn, Wn));
  rep->t_operator += MPI_Wtime () - t0;
  for (p = 0; p < g->opts.q_power; p++) {
    t0 = MPI_Wtime ();
    PetscCall (lgh_glrd_bcgs2 (Yn, k_new, panel_b, 0, g->comm));
    rep->t_dense += MPI_Wtime () - t0;
    PetscCall (MatCopy (Yn, Qn, SAME_NONZERO_PATTERN));
    t0 = MPI_Wtime ();
    PetscCall (lgh_glr_apply_F_block (g, Qn, Yn, Wn));
    rep->t_operator += MPI_Wtime () - t0;
  }

  /* place F^{q+1} Omega_new into Qb's new columns; BCGS2 vs old + internal */
  {
    const PetscScalar  *yn;
    PetscScalar        *qb;
    PetscInt            ldn, ldb, rlo, rhi;

    PetscCall (MatGetOwnershipRange (Qb, &rlo, &rhi));
    PetscCall (MatDenseGetLDA (Yn, &ldn));
    PetscCall (MatDenseGetLDA (Qb, &ldb));
    PetscCall (MatDenseGetArrayRead (Yn, &yn));
    PetscCall (MatDenseGetArray (Qb, &qb));
    for (j = 0; j < k_new; j++)
      for (i = 0; i < (int) (rhi - rlo); i++)
        qb[i + (size_t) (ell1 + j) * ldb] = yn[i + (size_t) j * ldn];
    PetscCall (MatDenseRestoreArray (Qb, &qb));
    PetscCall (MatDenseRestoreArrayRead (Yn, &yn));
  }
  t0 = MPI_Wtime ();
  PetscCall (lgh_glrd_bcgs2 (Qb, ell2, panel_b, ell1, g->comm));
  rep->t_dense += MPI_Wtime () - t0;

  /* one more F apply on the orthonormalized new block; border columns
   * Cnew = Q_all^T (F Q_new), replicated panel-wise                     */
  {
    PetscScalar        *qn;
    const PetscScalar  *qb;
    PetscInt            ldn, ldb, rlo, rhi;

    PetscCall (MatGetOwnershipRange (Qb, &rlo, &rhi));
    PetscCall (MatDenseGetLDA (Qn, &ldn));
    PetscCall (MatDenseGetLDA (Qb, &ldb));
    PetscCall (MatDenseGetArrayRead (Qb, &qb));
    PetscCall (MatDenseGetArrayWrite (Qn, &qn));
    for (j = 0; j < k_new; j++)
      for (i = 0; i < (int) (rhi - rlo); i++)
        qn[i + (size_t) j * ldn] = qb[i + (size_t) (ell1 + j) * ldb];
    PetscCall (MatDenseRestoreArrayWrite (Qn, &qn));
    PetscCall (MatDenseRestoreArrayRead (Qb, &qb));
  }
  t0 = MPI_Wtime ();
  PetscCall (lgh_glr_apply_F_block (g, Qn, Yn, Wn));
  rep->t_operator += MPI_Wtime () - t0;

  PetscCall (PetscMalloc2 ((size_t) ell2 * PetscMax (k_new, panel_b), &Cnew,
                           (size_t) ell2 * PetscMax (k_new, panel_b), &Cl));
  {
    const PetscScalar  *qb, *yn;
    PetscInt            ldb, ldn, nloc = g->nloc;
    int                 bm = (int) nloc, bl2 = ell2, bk = k_new;
    double              one = 1.0, zero = 0.0;

    PetscCall (MatDenseGetLDA (Qb, &ldb));
    PetscCall (MatDenseGetLDA (Yn, &ldn));
    PetscCall (MatDenseGetArrayRead (Qb, &qb));
    PetscCall (MatDenseGetArrayRead (Yn, &yn));
    if (nloc > 0) {
      int                 bldb = (int) ldb, bldn = (int) ldn;
      LGH_BLAS_DGEMM ("T", "N", &bl2, &bk, &bm, &one, qb, &bldb, yn, &bldn,
                      &zero, Cl, &bl2);
    }
    else { for (i = 0; i < ell2 * k_new; i++) Cl[i] = 0.; }
    PetscCallMPI (MPI_Allreduce (Cl, Cnew, ell2 * k_new, MPI_DOUBLE,
                                 MPI_SUM, g->comm));
    PetscCall (MatDenseRestoreArrayRead (Yn, &yn));
    PetscCall (MatDenseRestoreArrayRead (Qb, &qb));
  }

  /* grow the 1D column store to ell2 rows; fill old columns' new rows by
   * symmetry (A(i_new, j_old) := A(j_old, i_new) = Cnew[j_old, i_new-ell1]);
   * append owned new columns                                            */
  {
    double             *Anew;
    const int           acn2 = (ell2 + s->P - 1 - s->rank) / s->P;

    PetscCall (PetscMalloc1 ((size_t) PetscMax (acn2, 1) * ell2, &Anew));
    for (lc = 0; lc < s->acn; lc++) {
      for (i = 0; i < ell1; i++)
        Anew[i + (size_t) lc * ell2] = s->Acols[i + (size_t) lc * ell1];
      for (i = ell1; i < ell2; i++)
        Anew[i + (size_t) lc * ell2] =
            Cnew[(lc * s->P + s->rank) + (size_t) (i - ell1) * ell2];
    }
    for (lc = s->acn; lc < acn2; lc++) {
      const int           gj = lc * s->P + s->rank;   /* a new column */
      for (i = 0; i < ell2; i++)
        Anew[i + (size_t) lc * ell2] =
            Cnew[i + (size_t) (gj - ell1) * ell2];
    }
    PetscCall (PetscFree (s->Acols));
    s->Acols = Anew;
    s->acn = acn2;
  }
  PetscCall (MatDestroy (&Qn));
  PetscCall (MatDestroy (&Yn));
  PetscCall (MatDestroy (&Wn));
  PetscCall (MatDestroy (&s->Q));
  s->Q = Qb;
  s->ell = ell2;
  g->opts.ell = ell2;

  /* new block-cyclic extents + descriptor at ell2 */
  PetscCall (lgh_glrd_set_desc (s));
  PetscCall (PetscFree (s->Vloc));
  PetscCall (PetscCalloc1 ((size_t) PetscMax (s->tl_rows * s->tl_cols, 1),
                           &s->Vloc));
  PetscCall (PetscCalloc1 ((size_t) PetscMax (s->tl_rows * s->tl_cols, 1),
                           &Tloc));

  /* rebuild sym(T) block-cyclic from the column store: per panel, owners
   * write their columns into a zeroed replicated buffer, allreduce(SUM)
   * replicates it, then the same two-sided 0.5 scatter as form_T        */
  t0 = MPI_Wtime ();
  for (p0 = 0; p0 < ell2; p0 += panel_b) {
    const int           bw = PetscMin (panel_b, ell2 - p0);
    double             *Cp = Cnew;   /* reuse as replicated panel buffer */

    for (i = 0; i < ell2 * bw; i++) Cl[i] = 0.;
    for (jj = 0; jj < bw; jj++) {
      j = p0 + jj;
      if (j % s->P == s->rank)
        for (i = 0; i < ell2; i++)
          Cl[i + (size_t) jj * ell2] =
              s->Acols[i + (size_t) (j / s->P) * ell2];
    }
    PetscCallMPI (MPI_Allreduce (Cl, Cp, ell2 * bw, MPI_DOUBLE, MPI_SUM,
                                 g->comm));
    for (jj = 0; jj < bw; jj++) {
      j = p0 + jj;
      if (lgh_glrd_owner (j, s->nb, s->pcol) != s->mycol) continue;
      {
        const int           lj = lgh_glrd_lidx (j, s->nb, s->pcol);
        for (i = 0; i < ell2; i++) {
          if (lgh_glrd_owner (i, s->nb, s->prow) != s->myrow) continue;
          Tloc[lgh_glrd_lidx (i, s->nb, s->prow)
               + (size_t) lj * s->tl_rows] +=
              0.5 * Cp[i + (size_t) jj * ell2];
        }
      }
    }
    for (jj = 0; jj < bw; jj++) {
      i = p0 + jj;
      if (lgh_glrd_owner (i, s->nb, s->prow) != s->myrow) continue;
      {
        const int           li = lgh_glrd_lidx (i, s->nb, s->prow);
        for (j = 0; j < ell2; j++) {
          if (lgh_glrd_owner (j, s->nb, s->pcol) != s->mycol) continue;
          Tloc[li + (size_t) lgh_glrd_lidx (j, s->nb, s->pcol)
               * s->tl_rows] += 0.5 * Cp[j + (size_t) jj * ell2];
        }
      }
    }
  }
  PetscCall (PetscFree2 (Cnew, Cl));

  /* eig at ell2 + re-selection */
  PetscCall (PetscFree (s->lam_raw));
  PetscCall (PetscMalloc1 (ell2, &s->lam_raw));
  PetscCall (lgh_glrd_pdsyevd (s, Tloc));
  rep->t_dense += MPI_Wtime () - t0;
  PetscCall (PetscFree (Tloc));

  PetscCall (lgh_glrd_select (g, rep));

  PetscCall (PetscFree (s->cwork));
  PetscCall (PetscMalloc1 (3 * ell2, &s->cwork));
  return PETSC_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* block (MATDENSE) forms of the implicit-U primitives: one GEMM + two
 * allreduces per call regardless of the column count.                   */

/* W[kept x m] = U^T X = V_kept^T (Q^T X); W replicated, column-major.   */
static PetscErrorCode
lgh_glrd_UTmult_block (lgh_glr_t *g, Mat X, double *W)
{
  struct lgh_glrd_state *s = g->dist;
  const PetscScalar  *qa, *xa;
  PetscInt            nloc, ldq, ldx, m;
  double             *cl, *cf, *wl;
  int                 i, lj, k, jc;

  PetscCall (MatGetSize (X, NULL, &m));
  PetscCall (PetscMalloc3 ((size_t) s->ell * m, &cl,
                           (size_t) s->ell * m, &cf,
                           (size_t) PetscMax (g->kept * m, 1), &wl));
  PetscCall (MatGetLocalSize (s->Q, &nloc, NULL));
  PetscCall (MatDenseGetLDA (s->Q, &ldq));
  PetscCall (MatDenseGetLDA (X, &ldx));
  PetscCall (MatDenseGetArrayRead (s->Q, &qa));
  PetscCall (MatDenseGetArrayRead (X, &xa));
  if (nloc > 0) {
    int                 bm = (int) nloc, bl = s->ell, bmm = (int) m;
    int                 bldq = (int) ldq, bldx = (int) ldx;
    double              one = 1.0, zero = 0.0;
    LGH_BLAS_DGEMM ("T", "N", &bl, &bmm, &bm, &one, qa, &bldq, xa, &bldx,
                    &zero, cl, &bl);
  }
  else { for (i = 0; i < s->ell * (int) m; i++) cl[i] = 0.; }
  PetscCall (MatDenseRestoreArrayRead (X, &xa));
  PetscCall (MatDenseRestoreArrayRead (s->Q, &qa));
  PetscCallMPI (MPI_Allreduce (cl, cf, (int) (s->ell * m), MPI_DOUBLE,
                               MPI_SUM, g->comm));

  for (i = 0; i < (int) (g->kept * m); i++) wl[i] = 0.;
  for (lj = 0; lj < s->tl_cols; lj++) {
    const int           gj = (lj / s->nb) * s->nb * s->pcol
                              + s->mycol * s->nb + lj % s->nb;
    if (gj >= s->ell || (k = s->kpos[gj]) < 0) continue;
    for (i = 0; i < s->tl_rows; i++) {
      const int           gi = (i / s->nb) * s->nb * s->prow
                                + s->myrow * s->nb + i % s->nb;
      double              vij;
      if (gi >= s->ell) continue;
      vij = s->Vloc[i + (size_t) lj * s->tl_rows];
      for (jc = 0; jc < (int) m; jc++)
        wl[k + (size_t) jc * g->kept] += vij * cf[gi + (size_t) jc * s->ell];
    }
  }
  PetscCallMPI (MPI_Allreduce (wl, W, (int) (g->kept * m), MPI_DOUBLE,
                               MPI_SUM, g->comm));
  PetscCall (PetscFree3 (cl, cf, wl));
  return PETSC_SUCCESS;
}

/* Y (+)= U W = Q (V_kept W); W replicated kept x m, column-major.       */
static PetscErrorCode
lgh_glrd_Umult_block (lgh_glr_t *g, const double *W, Mat Y, PetscBool add)
{
  struct lgh_glrd_state *s = g->dist;
  const PetscScalar  *qa;
  PetscScalar        *ya;
  PetscInt            nloc, ldq, ldy, m;
  double             *cl, *cf;
  int                 i, lj, k, jc;

  PetscCall (MatGetSize (Y, NULL, &m));
  PetscCall (PetscMalloc2 ((size_t) s->ell * m, &cl,
                           (size_t) s->ell * m, &cf));
  for (i = 0; i < (int) (s->ell * m); i++) cl[i] = 0.;
  for (lj = 0; lj < s->tl_cols; lj++) {
    const int           gj = (lj / s->nb) * s->nb * s->pcol
                              + s->mycol * s->nb + lj % s->nb;
    if (gj >= s->ell || (k = s->kpos[gj]) < 0) continue;
    for (i = 0; i < s->tl_rows; i++) {
      const int           gi = (i / s->nb) * s->nb * s->prow
                                + s->myrow * s->nb + i % s->nb;
      double              vij;
      if (gi >= s->ell) continue;
      vij = s->Vloc[i + (size_t) lj * s->tl_rows];
      for (jc = 0; jc < (int) m; jc++)
        cl[gi + (size_t) jc * s->ell] += vij * W[k + (size_t) jc * g->kept];
    }
  }
  PetscCallMPI (MPI_Allreduce (cl, cf, (int) (s->ell * m), MPI_DOUBLE,
                               MPI_SUM, g->comm));

  PetscCall (MatGetLocalSize (s->Q, &nloc, NULL));
  PetscCall (MatDenseGetLDA (s->Q, &ldq));
  PetscCall (MatDenseGetLDA (Y, &ldy));
  PetscCall (MatDenseGetArrayRead (s->Q, &qa));
  PetscCall (MatDenseGetArray (Y, &ya));
  if (nloc > 0) {
    int                 bm = (int) nloc, bl = s->ell, bmm = (int) m;
    int                 bldq = (int) ldq, bldy = (int) ldy;
    double              one = 1.0, beta = add ? 1.0 : 0.0;
    LGH_BLAS_DGEMM ("N", "N", &bm, &bmm, &bl, &one, qa, &bldq, cf, &bl,
                    &beta, ya, &bldy);
  }
  PetscCall (MatDenseRestoreArray (Y, &ya));
  PetscCall (MatDenseRestoreArrayRead (s->Q, &qa));
  PetscCall (PetscFree2 (cl, cf));
  return PETSC_SUCCESS;
}

/* Materialize the implicit U = Q V_kept as an explicit N x kept row slab
 * (panel-wise: replicate an ell x bw coefficient panel from Vloc's owners,
 * one allreduce + one GEMM per panel).  Used by the correction graft,
 * which afterwards abandons the sketch representation entirely.          */
static PetscErrorCode
lgh_glrd_materialize_U (lgh_glr_t *g, Mat *Uout)
{
  struct lgh_glrd_state *s = g->dist;
  const int           pb = PetscMax (g->opts.panel, 1);
  Mat                 U;
  const PetscScalar  *qa;
  PetscScalar        *ua;
  PetscInt            nloc, ldq, ldu;
  double             *cl, *cf;
  int                 p0, i, lj, k;

  PetscCall (MatCreateDense (g->comm, g->nloc, PETSC_DECIDE, g->Nglob,
                             PetscMax (g->kept, 1), NULL, &U));
  PetscCall (PetscMalloc2 ((size_t) s->ell * pb, &cl,
                           (size_t) s->ell * pb, &cf));
  PetscCall (MatGetLocalSize (s->Q, &nloc, NULL));
  PetscCall (MatDenseGetLDA (s->Q, &ldq));
  PetscCall (MatDenseGetLDA (U, &ldu));
  PetscCall (MatDenseGetArrayRead (s->Q, &qa));
  PetscCall (MatDenseGetArray (U, &ua));
  for (p0 = 0; p0 < g->kept; p0 += pb) {
    const int           bw = PetscMin (pb, (int) g->kept - p0);

    for (i = 0; i < s->ell * bw; i++) cl[i] = 0.;
    for (lj = 0; lj < s->tl_cols; lj++) {
      const int           gj = (lj / s->nb) * s->nb * s->pcol
                                + s->mycol * s->nb + lj % s->nb;
      if (gj >= s->ell || (k = s->kpos[gj]) < 0) continue;
      if (k < p0 || k >= p0 + bw) continue;
      for (i = 0; i < s->tl_rows; i++) {
        const int           gi = (i / s->nb) * s->nb * s->prow
                                  + s->myrow * s->nb + i % s->nb;
        if (gi >= s->ell) continue;
        cl[gi + (size_t) (k - p0) * s->ell] =
            s->Vloc[i + (size_t) lj * s->tl_rows];
      }
    }
    PetscCallMPI (MPI_Allreduce (cl, cf, s->ell * bw, MPI_DOUBLE, MPI_SUM,
                                 g->comm));
    if (nloc > 0) {
      int                 bm = (int) nloc, bl = s->ell, bb = bw;
      int                 bldq = (int) ldq, bldu = (int) ldu;
      double              one = 1.0, zero = 0.0;
      LGH_BLAS_DGEMM ("N", "N", &bm, &bb, &bl, &one, qa, &bldq, cf, &bl,
                      &zero, ua + (size_t) p0 * ldu, &bldu);
    }
  }
  PetscCall (MatDenseRestoreArray (U, &ua));
  PetscCall (MatDenseRestoreArrayRead (s->Q, &qa));
  PetscCall (PetscFree2 (cl, cf));
  *Uout = U;
  return PETSC_SUCCESS;
}

#endif /* LGPSF_HESSIAN_GLR_SCALAPACK_IMPL_H */

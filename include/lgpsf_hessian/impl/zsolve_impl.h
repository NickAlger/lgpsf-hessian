/* zsolve_impl.h — the Z-solve machinery behind lgh_prior_create_mat.
 * Compiled inside the single consumer TU, after glr_common_impl.h.
 *
 * Provenance: the production Z-solve provider of a large-scale ice-sheet
 * inversion (pure PETSc), ported with one deliberate change: the original
 * folded D = scale * M^{1/2} into its solve callbacks because the old
 * engine contract required mass-folded solves; in this library the mass
 * scalings are composed by the GLR layer, so everything here is a RAW
 * A^{-1} = Z^{-1} action.
 *
 * Tier ladder (same contract, internal upgrades):
 *   mode 0/1: per-column KSPSolve on dense-column Vecs;
 *   mode 2:   blocked fixed-count Chebyshev with the AMG preconditioner
 *             applied through PCMatApply (BLAS-3 across the block);
 *   mode 3:   blocked Chebyshev around a block V-cycle on the READ-ONLY
 *             harvested AMG hierarchy — all smoothing is CORRECTION FORM
 *             (X += S(B - A X)), equivalent to the affine smoother and
 *             guess-free.  Tiny coarse operators are dense-LU factored
 *             (block MatMatSolve); at communicator size > 1 the coarse
 *             solve is REDUNDANT: every rank factors a full sequential
 *             copy of A_0 and the coarse RHS block is allgathered.
 * Chebyshev iteration counts derive from the once-estimated preconditioned
 * spectral interval (Lanczos estimates are a LOWER bound on kappa, so the
 * interval is widened by a margin before use).
 */

#ifndef LGPSF_HESSIAN_ZSOLVE_IMPL_H
#define LGPSF_HESSIAN_ZSOLVE_IMPL_H

#include <stdlib.h>
#include <stdio.h>
#if defined(PETSC_HAVE_HYPRE)
#include <petscmathypre.h>
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <_hypre_parcsr_ls.h>
#endif

#ifndef LGPSF_HESSIAN_GLR_COMMON_IMPL_H
#error "include impl/glr_common_impl.h first (impl.hpp does this)"
#endif

/* ------------------------------------------------------------------ */

typedef PetscErrorCode (*lgh_zs_pcapply_fn) (Mat R, Mat Z, void *ctx);

typedef struct lgh_zs_mglevel
{
  Mat                 A, P;      /* borrowed; P maps l-1 -> l, NULL at 0 */
  Vec                 dinv;
  PetscReal           smax;      /* lambda_max(dinv A) estimate          */
  PetscReal           slo, shi;  /* smoothing interval                   */
  PetscInt            nu;        /* smoother degree                      */
  Mat                 B, X;      /* restricted problem (levels < finest) */
  Mat                 R, Z, D;   /* workspace, n_l x tile                */
  Mat                 AD;        /* product cache A*D                    */
  Mat                 PZw;       /* product cache P*X_coarse             */
  int                 Bt_flag, PZ_flag;
}
lgh_zs_mglevel_t;

typedef struct lgh_zs_mgh
{
  PetscInt            nl, tile;
  lgh_zs_mglevel_t   *lev;
  KSP                 coarse;      /* borrowed level-0 KSP               */
  Mat                 coarse_fact; /* dense LU of A_0                    */
  PetscMPIInt         csize;
  PetscInt            n0, c_rstart, c_nloc;
  int                *c_counts, *c_displs;
  PetscScalar        *cB, *cX;
  Mat                 cBmat, cXmat;
  int                 owns_ops;    /* 1: A, P are ours (hypre-built)     */
  int                 source;      /* LGH_HIERARCHY_PCMG | _HYPRE        */
}
lgh_zs_mgh_t;

struct lgh_zs_ctx
{
  KSP                 ksp;    /* borrowed (the prior owns it)           */
  int                 mode;   /* 1 = per-column KSP; 2/3 = blocked      */
  Mat                 Aface;  /* borrowed operator of ksp               */
  PC                  amg;    /* borrowed PC (mode 2)                   */
  lgh_zs_mgh_t       *mgh;    /* harvested hierarchy (mode 3)           */
  int                 hierarchy;  /* resolved LGH_HIERARCHY_* (mode 3)  */
  PetscInt            tile, nit;
  PetscReal           lmin, lmax;      /* margined outer interval       */
  Mat                 tB, tX, tR, tZ, tD, tAD;   /* n x tile workspace  */
};

/* ------------------------------------------------------------------ */

static PetscErrorCode
lgh_zs_create (KSP ksp, struct lgh_zs_ctx **out)
{
  struct lgh_zs_ctx  *zs;

  PetscCall (PetscNew (&zs));
  zs->mode = 1;
  zs->ksp = ksp;
  *out = zs;
  return PETSC_SUCCESS;
}

/* iteration count for accuracy tol on interval [lmin, lmax]            */
static PetscInt
lgh_zs_cheb_count (PetscReal lmin, PetscReal lmax, PetscReal tol)
{
  const PetscReal     kap = lmax / lmin;
  const PetscReal     rho = (PetscSqrtReal (kap) - 1.) /
                            (PetscSqrtReal (kap) + 1.);
  return (PetscInt) PetscCeilReal (PetscLogReal (2. / tol) /
                                   PetscLogReal (1. / rho));
}

/* Chebyshev core on blocks.  CONTRACT: on entry R holds the residual
 * B - A*X of the accumulator X (fresh solve: X = 0, R = copy of B); runs
 * nit polynomial terms, updating X and keeping R consistent.  AD is a
 * product cache for A x D (created on first use, REUSED).               */
static PetscErrorCode
lgh_zs_cheb_core (Mat A, lgh_zs_pcapply_fn pcap, void *pcctx,
                  PetscReal lmin, PetscReal lmax, PetscInt nit,
                  Mat X, Mat R, Mat Z, Mat D, Mat *ADp)
{
  const PetscReal     theta = 0.5 * (lmax + lmin);
  const PetscReal     delta = 0.5 * (lmax - lmin);
  const PetscReal     sigma1 = theta / delta;
  PetscReal           rho_old = 1.0 / sigma1, rho;
  PetscInt            k;

  PetscCall (pcap (R, Z, pcctx));
  PetscCall (MatCopy (Z, D, SAME_NONZERO_PATTERN));
  PetscCall (MatScale (D, 1.0 / theta));
  PetscCall (MatAXPY (X, 1.0, D, SAME_NONZERO_PATTERN));
  if (*ADp == NULL) {
    PetscCall (MatMatMult (A, D, MAT_INITIAL_MATRIX, PETSC_DEFAULT, ADp));
  }
  else {
    PetscCall (MatMatMult (A, D, MAT_REUSE_MATRIX, PETSC_DEFAULT, ADp));
  }
  PetscCall (MatAXPY (R, -1.0, *ADp, SAME_NONZERO_PATTERN));
  for (k = 2; k <= nit; k++) {
    PetscCall (pcap (R, Z, pcctx));
    rho = 1.0 / (2.0 * sigma1 - rho_old);
    PetscCall (MatScale (D, rho * rho_old));
    PetscCall (MatAXPY (D, 2.0 * rho / delta, Z, SAME_NONZERO_PATTERN));
    PetscCall (MatAXPY (X, 1.0, D, SAME_NONZERO_PATTERN));
    PetscCall (MatMatMult (A, D, MAT_REUSE_MATRIX, PETSC_DEFAULT, ADp));
    PetscCall (MatAXPY (R, -1.0, *ADp, SAME_NONZERO_PATTERN));
    rho_old = rho;
  }
  return PETSC_SUCCESS;
}

/* row-scaling PC apply: Z = diag(dinv) R                                */
static PetscErrorCode
lgh_zs_rowscale_pc (Mat R, Mat Z, void *ctx)
{
  Vec                 dinv = (Vec) ctx;

  PetscCall (MatCopy (R, Z, SAME_NONZERO_PATTERN));
  PetscCall (MatDiagonalScale (Z, dinv, NULL));
  return PETSC_SUCCESS;
}

/* Worker KSP of the given type sharing the solver's operators AND PC
 * object (refcounted; the solver KSP is untouched).  Caller destroys.   */
static PetscErrorCode
lgh_zs_make_worker_ksp (struct lgh_zs_ctx *zs, KSPType type, KSP *out)
{
  Mat                 A, P;
  PC                  pc;
  MPI_Comm            comm;
  KSP                 k2;

  PetscCall (PetscObjectGetComm ((PetscObject) zs->ksp, &comm));
  PetscCall (KSPGetOperators (zs->ksp, &A, &P));
  PetscCall (KSPGetPC (zs->ksp, &pc));
  PetscCall (KSPCreate (comm, &k2));
  PetscCall (KSPSetType (k2, type));
  PetscCall (KSPSetPC (k2, pc));
  PetscCall (KSPSetOperators (k2, A, P));
  *out = k2;
  return PETSC_SUCCESS;
}

/* Spectral bounds of the PRECONDITIONED operator: tight CG solve on a
 * given RHS with singular-value computation.  Lanczos estimates from the
 * explored subspace — a LOWER bound on kappa; consumers add margins.    */
static PetscErrorCode
lgh_zs_spectral_bounds (struct lgh_zs_ctx *zs, Vec rhs, PetscInt maxit,
                        PetscReal *emin, PetscReal *emax, PetscInt *its)
{
  KSP                 k2;
  Vec                 y;

  PetscCall (lgh_zs_make_worker_ksp (zs, KSPCG, &k2));
  PetscCall (KSPSetComputeSingularValues (k2, PETSC_TRUE));
  PetscCall (KSPSetTolerances (k2, 1e-12, 0., PETSC_DEFAULT, maxit));
  PetscCall (VecDuplicate (rhs, &y));
  PetscCall (VecZeroEntries (y));
  PetscCall (KSPSolve (k2, rhs, y));
  PetscCall (KSPComputeExtremeSingularValues (k2, emax, emin));
  PetscCall (KSPGetIterationNumber (k2, its));
  PetscCall (VecDestroy (&y));
  PetscCall (KSPDestroy (&k2));
  return PETSC_SUCCESS;
}

/* -------- harvested hierarchy (mode 3) -------- */

/* Every rank gets a sequential dense copy of the small distributed A0:
 * local rows via MatGetRow, one MPI_Allgatherv.  Replaces
 * MatCreateRedundantMatrix (2026-09-05): that path SEGV'd inside PETSc's
 * MatCreateSubMatrices at 192 ranks on a hypre coarse grid whose 158 rows
 * were spread over all ranks (most owning 0-1 rows); this is layout-agnostic.
 * Ownership ranges are contiguous and rank-ordered, so the gathered
 * row-major buffer is already in global row order.                        */
static PetscErrorCode
lgh_zs_gather_dense (Mat A, Mat *Aseq)
{
  MPI_Comm            comm;
  PetscMPIInt         size, r;
  PetscInt            n0, nloc, rs, re, i, j, ncols;
  const PetscInt     *cols;
  const PetscScalar  *vals;
  PetscScalar        *loc, *all, *a;
  int                *counts, *displs, myn;

  PetscCall (PetscObjectGetComm ((PetscObject) A, &comm));
  PetscCallMPI (MPI_Comm_size (comm, &size));
  PetscCall (MatGetSize (A, &n0, NULL));
  PetscCall (MatGetOwnershipRange (A, &rs, &re));
  nloc = re - rs;
  PetscCall (PetscCalloc1 ((size_t) PetscMax (nloc, 1) * (size_t) n0, &loc));
  for (i = rs; i < re; i++) {
    PetscCall (MatGetRow (A, i, &ncols, &cols, &vals));
    for (j = 0; j < ncols; j++) loc[(size_t) (i - rs) * n0 + cols[j]] = vals[j];
    PetscCall (MatRestoreRow (A, i, &ncols, &cols, &vals));
  }
  PetscCall (PetscMalloc2 (size, &counts, size, &displs));
  myn = (int) (nloc * n0);
  PetscCallMPI (MPI_Allgather (&myn, 1, MPI_INT, counts, 1, MPI_INT, comm));
  displs[0] = 0;
  for (r = 1; r < size; r++) displs[r] = displs[r - 1] + counts[r - 1];
  {
    long                tot = (long) displs[size - 1] + counts[size - 1];
    PetscInt            N, M;

    PetscCall (MatGetSize (A, &M, &N));
    PetscCheck (tot == (long) n0 * n0, comm, PETSC_ERR_PLIB,
                "gather_dense: row counts sum to %ld entries but the matrix is "
                "%" PetscInt_FMT " x %" PetscInt_FMT " (this rank: rows [%"
                PetscInt_FMT ", %" PetscInt_FMT "))", tot, M, N, rs, re);
  }
  PetscCall (PetscMalloc1 ((size_t) n0 * (size_t) n0, &all));
  PetscCallMPI (MPI_Allgatherv (loc, myn, MPIU_SCALAR, all, counts, displs,
                                MPIU_SCALAR, comm));
  PetscCall (MatCreateSeqDense (PETSC_COMM_SELF, n0, n0, NULL, Aseq));
  PetscCall (MatDenseGetArrayWrite (*Aseq, &a));
  for (i = 0; i < n0; i++)
    for (j = 0; j < n0; j++) a[i + (size_t) j * n0] = all[(size_t) i * n0 + j];
  PetscCall (MatDenseRestoreArrayWrite (*Aseq, &a));
  PetscCall (PetscFree (all));
  PetscCall (PetscFree (loc));
  PetscCall (PetscFree2 (counts, displs));
  return PETSC_SUCCESS;
}

/* Build the blocked V-cycle data from level operators A[l] and
 * prolongators P[l] (l = 0 coarsest .. nl-1 finest; P[l] maps l-1 -> l,
 * P[0] unused), plus per-level smoother bounds (deterministic seed; run
 * constants).  Work blocks at tile width.  coarse_ksp (may be NULL) is the
 * fallback coarse solver when A_0 is too large for the dense LU; nu_hint[l]
 * (may be NULL) is the level's suggested smoother degree.  With owned = 1
 * the hierarchy takes ownership of the A/P handles.                      */
static PetscErrorCode
lgh_zs_mg_build (PetscInt nl, Mat *Aops, Mat *Pops, KSP coarse_ksp,
                 const PetscInt *nu_hint, int owned, int source,
                 PetscInt tile, int smax_krylov, PetscReal smoother_top,
                 PetscInt nu_force, lgh_zs_mgh_t **h_out)
{
  lgh_zs_mgh_t       *h;
  PetscInt            l;

  PetscCall (PetscNew (&h));
  h->nl = nl; h->tile = tile; h->owns_ops = owned; h->source = source;
  PetscCall (PetscCalloc1 (nl, &h->lev));
  for (l = 0; l < nl; l++) {
    lgh_zs_mglevel_t   *lv = &h->lev[l];
    PetscInt            nloc, Nglob;
    MPI_Comm            comm;

    lv->A = Aops[l];
    if (l == 0) {
      PetscInt            n0;
      MPI_Comm            comm0;
      PetscMPIInt         csize;

      h->coarse = coarse_ksp;
      PetscCall (MatGetSize (lv->A, &n0, NULL));
      PetscCall (PetscObjectGetComm ((PetscObject) lv->A, &comm0));
      PetscCallMPI (MPI_Comm_size (comm0, &csize));
      h->csize = csize; h->n0 = n0;
      PetscCheck (n0 <= 2000 || coarse_ksp != NULL, comm0, PETSC_ERR_SUP,
                  "coarsest level has %" PetscInt_FMT " rows (> 2000) and "
                  "there is no coarse KSP to fall back on; lower "
                  "hypre_max_coarse", n0);
      if (n0 <= 2000 && csize == 1) {
        /* tiny coarse operator: dense LU once, block MatMatSolve */
        PetscCall (MatConvert (lv->A, MATDENSE, MAT_INITIAL_MATRIX,
                               &h->coarse_fact));
        PetscCall (MatLUFactor (h->coarse_fact, NULL, NULL, NULL));
      }
      else if (n0 <= 2000) {
        /* size > 1: redundant coarse — full seq copy on every rank */
        Mat                 Adense;
        PetscMPIInt         r;
        PetscInt            rs, re;

        PetscCall (lgh_zs_gather_dense (lv->A, &Adense));
        PetscCall (MatLUFactor (Adense, NULL, NULL, NULL));
        h->coarse_fact = Adense;
        PetscCall (MatGetOwnershipRange (lv->A, &rs, &re));
        h->c_rstart = rs; h->c_nloc = re - rs;
        PetscCall (PetscMalloc2 (csize, &h->c_counts, csize, &h->c_displs));
        {
          int                 myn = (int) h->c_nloc;
          PetscCallMPI (MPI_Allgather (&myn, 1, MPI_INT, h->c_counts, 1,
                                       MPI_INT, comm0));
        }
        h->c_displs[0] = 0;
        for (r = 1; r < csize; r++)
          h->c_displs[r] = h->c_displs[r - 1] + h->c_counts[r - 1];
        PetscCall (PetscMalloc2 ((size_t) n0 * tile, &h->cB,
                                 (size_t) n0 * tile, &h->cX));
        PetscCall (MatCreateSeqDense (PETSC_COMM_SELF, n0, tile, h->cB,
                                      &h->cBmat));
        PetscCall (MatCreateSeqDense (PETSC_COMM_SELF, n0, tile, h->cX,
                                      &h->cXmat));
      }
    }
    if (l > 0) {
      lv->P = Pops[l];
    }
    PetscCall (PetscObjectGetComm ((PetscObject) lv->A, &comm));
    PetscCall (MatGetLocalSize (lv->A, &nloc, NULL));
    PetscCall (MatGetSize (lv->A, &Nglob, NULL));
    PetscCall (MatCreateVecs (lv->A, NULL, &lv->dinv));
    PetscCall (MatGetDiagonal (lv->A, lv->dinv));
    PetscCall (VecReciprocal (lv->dinv));
    if (l > 0) {
      const PetscInt      maxits = (nu_hint != NULL) ? nu_hint[l] : 0;
      lv->nu = (nu_force > 0) ? nu_force
                              : ((maxits > 0 && maxits < 20) ? maxits : 2);
      if (smax_krylov) {
        /* lambda_max(dinv A) via CG-Lanczos singular values */
        KSP                 ke;
        PC                  pe;
        Vec                 rb, ye;
        PetscRandom         rnd;
        PetscReal           emine, emaxe;

        PetscCall (KSPCreate (comm, &ke));
        PetscCall (KSPSetType (ke, KSPCG));
        PetscCall (KSPGetPC (ke, &pe));
        PetscCall (PCSetType (pe, PCJACOBI));
        PetscCall (KSPSetOperators (ke, lv->A, lv->A));
        PetscCall (KSPSetComputeSingularValues (ke, PETSC_TRUE));
        PetscCall (KSPSetTolerances (ke, 1e-12, 0., PETSC_DEFAULT, 40));
        PetscCall (MatCreateVecs (lv->A, NULL, &rb));
        PetscCall (VecDuplicate (rb, &ye));
        PetscCall (PetscRandomCreate (comm, &rnd));
        PetscCall (PetscRandomSetSeed (rnd, 20260813 + (unsigned long) l));
        PetscCall (PetscRandomSeed (rnd));
        PetscCall (VecSetRandom (rb, rnd));
        PetscCall (PetscRandomDestroy (&rnd));
        PetscCall (VecZeroEntries (ye));
        PetscCall (KSPSolve (ke, rb, ye));
        PetscCall (KSPComputeExtremeSingularValues (ke, &emaxe, &emine));
        lv->smax = emaxe;
        PetscCall (VecDestroy (&rb));
        PetscCall (VecDestroy (&ye));
        PetscCall (KSPDestroy (&ke));
      }
      else {
        /* lambda_max(dinv A) by power iteration (20 its, fixed seed) */
        Vec                 v, w;
        PetscRandom         rnd;
        PetscReal           nrm = 1.;
        PetscInt            it;

        PetscCall (MatCreateVecs (lv->A, NULL, &v));
        PetscCall (VecDuplicate (v, &w));
        PetscCall (PetscRandomCreate (comm, &rnd));
        PetscCall (PetscRandomSetSeed (rnd, 20260813 + (unsigned long) l));
        PetscCall (PetscRandomSeed (rnd));
        PetscCall (VecSetRandom (v, rnd));
        PetscCall (PetscRandomDestroy (&rnd));
        PetscCall (VecNormalize (v, NULL));
        for (it = 0; it < 20; it++) {
          PetscCall (MatMult (lv->A, v, w));
          PetscCall (VecPointwiseMult (w, w, lv->dinv));
          PetscCall (VecNorm (w, NORM_2, &nrm));
          PetscCall (VecCopy (w, v));
          PetscCall (VecScale (v, 1.0 / nrm));
        }
        lv->smax = nrm;
        PetscCall (VecDestroy (&v));
        PetscCall (VecDestroy (&w));
      }
      lv->slo = 0.1 * lv->smax;
      lv->shi = smoother_top * lv->smax;
    }
    PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, tile, NULL,
                               &lv->R));
    PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, tile, NULL,
                               &lv->Z));
    PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, tile, NULL,
                               &lv->D));
    if (l < nl - 1) {
      PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, tile,
                                 NULL, &lv->X));
    }
  }
  *h_out = h;
  return PETSC_SUCCESS;
}

/* READ-ONLY harvest of a PCMG/GAMG hierarchy (the original source).      */
static PetscErrorCode
lgh_zs_mg_harvest (PC pcmg, PetscInt tile, int smax_krylov,
                   PetscReal smoother_top, PetscInt nu_force,
                   lgh_zs_mgh_t **h_out)
{
  PetscInt            l, nl;
  Mat                *Aops, *Pops;
  PetscInt           *nuh;
  KSP                 kcoarse = NULL;
  PetscBool           ismg;
  const char         *tname;

  PetscCall (PetscObjectTypeCompareAny ((PetscObject) pcmg, &ismg, PCMG,
                                        PCGAMG, PCML, ""));
  PetscCall (PetscObjectGetType ((PetscObject) pcmg, &tname));
  PetscCheck (ismg, PetscObjectComm ((PetscObject) pcmg), PETSC_ERR_SUP,
              "hierarchy = pcmg needs a PCMG-type preconditioner on the "
              "Z solver (got %s); use hierarchy = hypre or configure GAMG",
              tname);
  PetscCall (PCMGGetLevels (pcmg, &nl));
  PetscCall (PetscMalloc3 (nl, &Aops, nl, &Pops, nl, &nuh));
  for (l = 0; l < nl; l++) {
    KSP                 kl;

    PetscCall (PCMGGetSmoother (pcmg, l, &kl));
    PetscCall (KSPGetOperators (kl, &Aops[l], NULL));
    Pops[l] = NULL;
    if (l > 0) PetscCall (PCMGGetInterpolation (pcmg, l, &Pops[l]));
    if (l == 0) kcoarse = kl;
    PetscCall (KSPGetTolerances (kl, NULL, NULL, NULL, &nuh[l]));
  }
  PetscCall (lgh_zs_mg_build (nl, Aops, Pops, kcoarse, nuh, 0,
                              LGH_HIERARCHY_PCMG, tile, smax_krylov,
                              smoother_top, nu_force, h_out));
  PetscCall (PetscFree3 (Aops, Pops, nuh));
  return PETSC_SUCCESS;
}

#if defined(PETSC_HAVE_HYPRE)
/* hypre ParCSR -> PETSc MPIAIJ (a copy), built from hypre's diag/offd CSR
 * blocks and its row/column partitions.  NOT PETSc's MatCreateFromParCSR:
 * in PETSc 3.21 that routine turns a rank with ZERO local rows into a
 * one-row rank (it clamps the empty range, then still adds 1 for the PETSc
 * convention), shifting every later rank's range and leaving the layout
 * inconsistent with the global size -- the coarse levels of a hypre
 * hierarchy have many empty ranks, so this crashed at 192 ranks
 * (2026-09-05).  Host memory only (CPU hypre).                           */
static PetscErrorCode
lgh_parcsr_to_aij (hypre_ParCSRMatrix *par, MPI_Comm comm, Mat *out)
{
  hypre_CSRMatrix    *diag = hypre_ParCSRMatrixDiag (par);
  hypre_CSRMatrix    *offd = hypre_ParCSRMatrixOffd (par);
  HYPRE_BigInt       *rows = hypre_ParCSRMatrixRowStarts (par);
  HYPRE_BigInt       *cols = hypre_ParCSRMatrixColStarts (par);
  HYPRE_BigInt       *cmap = hypre_ParCSRMatrixColMapOffd (par);
  const PetscInt      M = (PetscInt) hypre_ParCSRMatrixGlobalNumRows (par);
  const PetscInt      N = (PetscInt) hypre_ParCSRMatrixGlobalNumCols (par);
  const PetscInt      m = (PetscInt) (rows[1] - rows[0]);
  const PetscInt      n = (PetscInt) (cols[1] - cols[0]);
  HYPRE_Int          *di = hypre_CSRMatrixI (diag), *dj = hypre_CSRMatrixJ (diag);
  HYPRE_Int          *oi = hypre_CSRMatrixI (offd), *oj = hypre_CSRMatrixJ (offd);
  HYPRE_Complex      *dv = hypre_CSRMatrixData (diag), *ov = hypre_CSRMatrixData (offd);
  PetscInt           *dnnz, *onnz, i, k, maxnz = 0;
  PetscInt           *cbuf;
  PetscScalar        *vbuf;
  Mat                 B;

  PetscCall (MatCreate (comm, &B));
  PetscCall (MatSetSizes (B, m, n, M, N));
  PetscCall (MatSetType (B, MATAIJ));
  PetscCall (PetscMalloc2 (PetscMax (m, 1), &dnnz, PetscMax (m, 1), &onnz));
  for (i = 0; i < m; i++) {
    dnnz[i] = (PetscInt) (di[i + 1] - di[i]);
    onnz[i] = (oi != NULL) ? (PetscInt) (oi[i + 1] - oi[i]) : 0;
    if (dnnz[i] + onnz[i] > maxnz) maxnz = dnnz[i] + onnz[i];
  }
  PetscCall (MatSeqAIJSetPreallocation (B, 0, dnnz));
  PetscCall (MatMPIAIJSetPreallocation (B, 0, dnnz, 0, onnz));
  PetscCall (MatSetOption (B, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall (PetscMalloc2 (PetscMax (maxnz, 1), &cbuf, PetscMax (maxnz, 1), &vbuf));
  for (i = 0; i < m; i++) {
    const PetscInt      r = (PetscInt) rows[0] + i;
    PetscInt            nz = 0;

    for (k = di[i]; k < di[i + 1]; k++) {
      cbuf[nz] = (PetscInt) cols[0] + (PetscInt) dj[k];
      vbuf[nz++] = (PetscScalar) dv[k];
    }
    if (oi != NULL) {
      for (k = oi[i]; k < oi[i + 1]; k++) {
        cbuf[nz] = (PetscInt) cmap[oj[k]];
        vbuf[nz++] = (PetscScalar) ov[k];
      }
    }
    PetscCall (MatSetValues (B, 1, &r, nz, cbuf, vbuf, INSERT_VALUES));
  }
  PetscCall (MatAssemblyBegin (B, MAT_FINAL_ASSEMBLY));
  PetscCall (MatAssemblyEnd (B, MAT_FINAL_ASSEMBLY));
  PetscCall (PetscFree2 (cbuf, vbuf));
  PetscCall (PetscFree2 (dnnz, onnz));
  *out = B;
  return PETSC_SUCCESS;
}

/* Build a BoomerAMG hierarchy for A with hypre, copy its level operators
 * and prolongators into PETSc matrices, discard the hypre solver.  hypre's
 * coarse operators are Galerkin (R = P^T, restriction type 0), which is
 * what the blocked cycle assumes.  hypre level 0 is the finest; ours is
 * the coarsest, hence the index flip.                                    */
static PetscErrorCode
lgh_zs_mg_harvest_hypre (Mat A, int coarsen, int interp, PetscReal strong,
                         int agg_nl, int max_coarse, PetscInt tile,
                         int smax_krylov, PetscReal smoother_top,
                         PetscInt nu_force, lgh_zs_mgh_t **h_out)
{
  Mat                 Ah;
  hypre_ParCSRMatrix *par;
  HYPRE_Solver        solver;
  hypre_ParAMGData   *amg;
  hypre_ParVector    *fv, *xv;
  hypre_ParCSRMatrix **A_array, **P_array;
  PetscInt            nl, k, l;
  Mat                *Aops, *Pops;
  MPI_Comm            comm;

  PetscCall (PetscObjectGetComm ((PetscObject) A, &comm));
  PetscCall (MatConvert (A, MATHYPRE, MAT_INITIAL_MATRIX, &Ah));
  PetscCall (MatHYPREGetParCSR (Ah, &par));
  PetscCallExternal (HYPRE_BoomerAMGCreate, &solver);
  PetscCallExternal (HYPRE_BoomerAMGSetCoarsenType, solver, coarsen);
  PetscCallExternal (HYPRE_BoomerAMGSetInterpType, solver, interp);
  PetscCallExternal (HYPRE_BoomerAMGSetStrongThreshold, solver, strong);
  PetscCallExternal (HYPRE_BoomerAMGSetAggNumLevels, solver, agg_nl);
  PetscCallExternal (HYPRE_BoomerAMGSetMaxCoarseSize, solver, max_coarse);
  PetscCallExternal (HYPRE_BoomerAMGSetMaxLevels, solver, 25);
  PetscCallExternal (HYPRE_BoomerAMGSetRestriction, solver, 0);
  PetscCallExternal (HYPRE_BoomerAMGSetRelaxType, solver, 0);  /* setup only */
  PetscCallExternal (HYPRE_BoomerAMGSetPrintLevel, solver, 0);
  PetscCallExternal (HYPRE_BoomerAMGSetMaxIter, solver, 1);
  PetscCallExternal (HYPRE_BoomerAMGSetTol, solver, 0.);
  fv = hypre_ParVectorCreate (hypre_ParCSRMatrixComm (par),
                              hypre_ParCSRMatrixGlobalNumRows (par),
                              hypre_ParCSRMatrixRowStarts (par));
  xv = hypre_ParVectorCreate (hypre_ParCSRMatrixComm (par),
                              hypre_ParCSRMatrixGlobalNumRows (par),
                              hypre_ParCSRMatrixRowStarts (par));
  hypre_ParVectorInitialize (fv);
  hypre_ParVectorInitialize (xv);
  PetscCallExternal (HYPRE_BoomerAMGSetup, solver, (HYPRE_ParCSRMatrix) par,
                     (HYPRE_ParVector) fv, (HYPRE_ParVector) xv);
  amg = (hypre_ParAMGData *) solver;
  nl = (PetscInt) hypre_ParAMGDataNumLevels (amg);
  A_array = hypre_ParAMGDataAArray (amg);
  P_array = hypre_ParAMGDataPArray (amg);
  PetscCall (PetscMalloc2 (nl, &Aops, nl, &Pops));
  for (k = 0; k < nl; k++) {
    l = nl - 1 - k;
    if (k == 0) {
      Aops[l] = A;
      PetscCall (PetscObjectReference ((PetscObject) A));
    }
    else {
      PetscCall (lgh_parcsr_to_aij (A_array[k], comm, &Aops[l]));
    }
    Pops[l] = NULL;
    if (k < nl - 1) {   /* P_array[k]: hypre level k+1 -> k, i.e. ours l-1 -> l */
      PetscCall (lgh_parcsr_to_aij (P_array[k], comm, &Pops[l]));
    }
  }
  hypre_ParVectorDestroy (fv);
  hypre_ParVectorDestroy (xv);
  PetscCallExternal (HYPRE_BoomerAMGDestroy, solver);
  PetscCall (MatDestroy (&Ah));
  PetscCall (lgh_zs_mg_build (nl, Aops, Pops, NULL, NULL, 1,
                              LGH_HIERARCHY_HYPRE, tile, smax_krylov,
                              smoother_top, nu_force, h_out));
  PetscCall (PetscFree2 (Aops, Pops));
  return PETSC_SUCCESS;
}
#endif

/* one line: source, levels, sizes, operator complexity                   */
static PetscErrorCode
lgh_zs_mgh_report (const lgh_zs_mgh_t *h, MPI_Comm comm)
{
  PetscInt            l;
  double              nnz_fine = 0., nnz_all = 0.;
  char                sizes[512] = "";
  size_t              used = 0;

  for (l = h->nl - 1; l >= 0; l--) {
    MatInfo             info;
    PetscInt            N;

    PetscCall (MatGetInfo (h->lev[l].A, MAT_GLOBAL_SUM, &info));
    PetscCall (MatGetSize (h->lev[l].A, &N, NULL));
    nnz_all += info.nz_used;
    if (l == h->nl - 1) nnz_fine = info.nz_used;
    if (used < sizeof (sizes) - 32)
      used += (size_t) snprintf (sizes + used, sizeof (sizes) - used,
                                 "%s%" PetscInt_FMT, (used ? "-" : ""), N);
  }
  PetscCall (PetscPrintf (comm, "lgh_prior: hierarchy %s: %" PetscInt_FMT
                          " levels %s, operator complexity %.2f\n",
                          h->source == LGH_HIERARCHY_HYPRE ? "hypre" : "pcmg",
                          h->nl, sizes,
                          nnz_fine > 0. ? nnz_all / nnz_fine : 0.));
  return PETSC_SUCCESS;
}

static void
lgh_zs_mgh_destroy (lgh_zs_mgh_t *h)
{
  PetscInt            l;

  if (h == NULL) return;
  for (l = 0; l < h->nl; l++) {
    lgh_zs_mglevel_t   *lv = &h->lev[l];
    /* A, P, coarse KSP are borrowed unless the hierarchy built them */
    if (h->owns_ops) {
      (void) MatDestroy (&lv->A);
      (void) MatDestroy (&lv->P);
    }
    (void) VecDestroy (&lv->dinv);
    (void) MatDestroy (&lv->R);
    (void) MatDestroy (&lv->Z);
    (void) MatDestroy (&lv->D);
    (void) MatDestroy (&lv->X);
    (void) MatDestroy (&lv->AD);
    (void) MatDestroy (&lv->PZw);
    (void) MatDestroy (&lv->B);
  }
  (void) MatDestroy (&h->coarse_fact);
  (void) MatDestroy (&h->cBmat);
  (void) MatDestroy (&h->cXmat);
  if (h->cB != NULL) (void) PetscFree2 (h->cB, h->cX);
  if (h->c_counts != NULL) (void) PetscFree2 (h->c_counts, h->c_displs);
  (void) PetscFree (h->lev);
  (void) PetscFree (h);
}

/* correction-form smoothing: X += p_nu(dinv A)(applied to B - A X)      */
static PetscErrorCode
lgh_zs_smooth_block (lgh_zs_mglevel_t *lv, Mat B, Mat X)
{
  PetscCall (MatCopy (X, lv->D, SAME_NONZERO_PATTERN));
  if (lv->AD == NULL) {
    PetscCall (MatMatMult (lv->A, lv->D, MAT_INITIAL_MATRIX, PETSC_DEFAULT,
                           &lv->AD));
  }
  else {
    PetscCall (MatMatMult (lv->A, lv->D, MAT_REUSE_MATRIX, PETSC_DEFAULT,
                           &lv->AD));
  }
  PetscCall (MatCopy (B, lv->R, SAME_NONZERO_PATTERN));
  PetscCall (MatAXPY (lv->R, -1.0, lv->AD, SAME_NONZERO_PATTERN));
  PetscCall (lgh_zs_cheb_core (lv->A, lgh_zs_rowscale_pc, (void *) lv->dinv,
                               lv->slo, lv->shi, lv->nu,
                               X, lv->R, lv->Z, lv->D, &lv->AD));
  return PETSC_SUCCESS;
}

/* the block V-cycle (recursive, correction form).  X_l zeroed on entry. */
static PetscErrorCode
lgh_zs_bvcycle_run (lgh_zs_mgh_t *h, PetscInt l, Mat B, Mat X)
{
  lgh_zs_mglevel_t   *lv = &h->lev[l];

  if (l == 0) {
    PetscInt            j, ncols;

    if (h->coarse_fact != NULL && h->csize == 1) {
      PetscCall (MatMatSolve (h->coarse_fact, B, X));
      return PETSC_SUCCESS;
    }
    if (h->coarse_fact != NULL) {
      /* size > 1 redundant coarse: allgather RHS, solve locally, copy
       * owned rows back                                                 */
      const PetscScalar  *ba;
      PetscScalar        *xa;
      PetscInt            ldb, ldx;
      MPI_Comm            comm0;

      PetscCall (MatGetSize (B, NULL, &ncols));
      PetscCheck (ncols == h->tile, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
                  "coarse block width != tile");
      PetscCall (PetscObjectGetComm ((PetscObject) lv->A, &comm0));
      PetscCall (MatDenseGetLDA (B, &ldb));
      PetscCall (MatDenseGetArrayRead (B, &ba));
      for (j = 0; j < ncols; j++) {
        PetscCallMPI (MPI_Allgatherv (ba + j * ldb, (int) h->c_nloc,
                                      MPIU_SCALAR, h->cB + j * h->n0,
                                      h->c_counts, h->c_displs, MPIU_SCALAR,
                                      comm0));
      }
      PetscCall (MatDenseRestoreArrayRead (B, &ba));
      PetscCall (MatMatSolve (h->coarse_fact, h->cBmat, h->cXmat));
      PetscCall (MatDenseGetLDA (X, &ldx));
      PetscCall (MatDenseGetArray (X, &xa));
      for (j = 0; j < ncols; j++)
        PetscCall (PetscMemcpy (xa + j * ldx,
                                h->cX + j * h->n0 + h->c_rstart,
                                h->c_nloc * sizeof (PetscScalar)));
      PetscCall (MatDenseRestoreArray (X, &xa));
      return PETSC_SUCCESS;
    }
    PetscCall (MatGetSize (B, NULL, &ncols));
    for (j = 0; j < ncols; j++) {
      Vec                 bj, xj;

      PetscCall (MatDenseGetColumnVecRead (B, j, &bj));
      PetscCall (MatDenseGetColumnVecWrite (X, j, &xj));
      PetscCall (VecZeroEntries (xj));
      PetscCall (KSPSolve (h->coarse, bj, xj));
      PetscCall (MatDenseRestoreColumnVecWrite (X, j, &xj));
      PetscCall (MatDenseRestoreColumnVecRead (B, j, &bj));
    }
    return PETSC_SUCCESS;
  }
  PetscCall (lgh_zs_smooth_block (lv, B, X));                   /* pre  */
  PetscCall (MatCopy (X, lv->D, SAME_NONZERO_PATTERN));
  PetscCall (MatMatMult (lv->A, lv->D, MAT_REUSE_MATRIX, PETSC_DEFAULT,
                         &lv->AD));
  PetscCall (MatCopy (B, lv->R, SAME_NONZERO_PATTERN));
  PetscCall (MatAXPY (lv->R, -1.0, lv->AD, SAME_NONZERO_PATTERN));
  PetscCall (MatTransposeMatMult (lv->P, lv->R,
                                  h->lev[l - 1].Bt_flag ? MAT_REUSE_MATRIX
                                                        : MAT_INITIAL_MATRIX,
                                  PETSC_DEFAULT, &h->lev[l - 1].B));
  h->lev[l - 1].Bt_flag = 1;
  PetscCall (MatZeroEntries (h->lev[l - 1].X));
  PetscCall (lgh_zs_bvcycle_run (h, l - 1, h->lev[l - 1].B,
                                 h->lev[l - 1].X));
  PetscCall (MatMatMult (lv->P, h->lev[l - 1].X,
                         lv->PZ_flag ? MAT_REUSE_MATRIX
                                     : MAT_INITIAL_MATRIX,
                         PETSC_DEFAULT, &lv->PZw));
  lv->PZ_flag = 1;
  PetscCall (MatAXPY (X, 1.0, lv->PZw, SAME_NONZERO_PATTERN));
  PetscCall (lgh_zs_smooth_block (lv, B, X));                   /* post */
  return PETSC_SUCCESS;
}

static PetscErrorCode
lgh_zs_bvcycle_pcapply (Mat R, Mat Z, void *ctx)
{
  lgh_zs_mgh_t       *h = (lgh_zs_mgh_t *) ctx;

  PetscCall (MatZeroEntries (Z));
  PetscCall (lgh_zs_bvcycle_run (h, h->nl - 1, R, Z));
  return PETSC_SUCCESS;
}

static PetscErrorCode
lgh_zs_pcmatapply_pcapply (Mat R, Mat Z, void *ctx)
{
  PC                  pc = (PC) ctx;

  PetscCall (PCMatApply (pc, R, Z));
  return PETSC_SUCCESS;
}

/* -------- blocked-mode setup and drivers -------- */

static PetscErrorCode
lgh_zs_blocked_setup (struct lgh_zs_ctx *zs, const lgh_prior_mat_opts_t *o)
{
  PetscInt            nloc, Nglob;
  MPI_Comm            comm;
  const int           mode = o->blocked_mode;
  const PetscInt      tile = o->tile;

  zs->mode = mode; zs->tile = tile;
  PetscCall (KSPGetOperators (zs->ksp, &zs->Aface, NULL));
  PetscCall (KSPGetPC (zs->ksp, &zs->amg));
  PetscCall (PetscObjectGetComm ((PetscObject) zs->Aface, &comm));
  if (mode == 3) {
    int                 want = o->hierarchy;

#if defined(PETSC_HAVE_HYPRE)
    if (want == LGH_HIERARCHY_AUTO) want = LGH_HIERARCHY_HYPRE;
#else
    PetscCheck (want != LGH_HIERARCHY_HYPRE, comm, PETSC_ERR_SUP,
                "hierarchy = hypre requested, but this PETSc was built "
                "without hypre (PETSC_HAVE_HYPRE undefined)");
    want = LGH_HIERARCHY_PCMG;
#endif
    zs->hierarchy = want;
#if defined(PETSC_HAVE_HYPRE)
    if (want == LGH_HIERARCHY_HYPRE) {
      PetscCall (lgh_zs_mg_harvest_hypre (zs->Aface, o->hypre_coarsen,
                                          o->hypre_interp, o->hypre_strong,
                                          o->hypre_agg_nl, o->hypre_max_coarse,
                                          tile, o->smax_krylov,
                                          o->smoother_top, o->nu_force,
                                          &zs->mgh));
    }
    else
#endif
    {
      PetscCall (lgh_zs_mg_harvest (zs->amg, tile, o->smax_krylov,
                                    o->smoother_top, o->nu_force, &zs->mgh));
    }
    if (o->verbose) PetscCall (lgh_zs_mgh_report (zs->mgh, comm));
  }
  PetscCall (MatGetLocalSize (zs->Aface, &nloc, NULL));
  PetscCall (MatGetSize (zs->Aface, &Nglob, NULL));
  PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, tile, NULL,
                             &zs->tB));
  PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, tile, NULL,
                             &zs->tX));
  PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, tile, NULL,
                             &zs->tR));
  PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, tile, NULL,
                             &zs->tZ));
  PetscCall (MatCreateDense (comm, nloc, PETSC_DECIDE, Nglob, tile, NULL,
                             &zs->tD));
  return PETSC_SUCCESS;
}

/* single-vector V-cycle adapter (bounds estimation of OUR mode-3 PC)    */
static PetscErrorCode
lgh_zs_bvcycle_pcapply_vec (PC pc, Vec r, Vec z)
{
  struct lgh_zs_ctx  *zs;
  PetscInt            nloc, j;
  PetscInt            ldb;
  const PetscScalar  *ra, *xa;
  PetscScalar        *ba, *za;

  PetscCall (PCShellGetContext (pc, &zs));
  PetscCall (VecGetLocalSize (r, &nloc));
  PetscCall (MatDenseGetLDA (zs->tB, &ldb));
  PetscCall (VecGetArrayRead (r, &ra));
  PetscCall (MatDenseGetArrayWrite (zs->tB, &ba));
  PetscCall (PetscMemcpy (ba, ra, nloc * sizeof (PetscScalar)));
  for (j = 1; j < zs->tile; j++)
    PetscCall (PetscMemzero (ba + j * ldb, nloc * sizeof (PetscScalar)));
  PetscCall (MatDenseRestoreArrayWrite (zs->tB, &ba));
  PetscCall (VecRestoreArrayRead (r, &ra));
  PetscCall (MatZeroEntries (zs->tX));
  PetscCall (lgh_zs_bvcycle_run (zs->mgh, zs->mgh->nl - 1, zs->tB, zs->tX));
  PetscCall (MatDenseGetArrayRead (zs->tX, &xa));
  PetscCall (VecGetArrayWrite (z, &za));
  PetscCall (PetscMemcpy (za, xa, nloc * sizeof (PetscScalar)));
  PetscCall (VecRestoreArrayWrite (z, &za));
  PetscCall (MatDenseRestoreArrayRead (zs->tX, &xa));
  return PETSC_SUCCESS;
}

/* spectral bounds of the mode-3 preconditioned operator                 */
static PetscErrorCode
lgh_zs_bounds_mode3 (struct lgh_zs_ctx *zs, Vec rhs, PetscInt maxit,
                     PetscReal *emin, PetscReal *emax, PetscInt *its,
                     int verbose)
{
  MPI_Comm            comm;
  KSP                 k2;
  PC                  psh;
  Vec                 y;

  PetscCall (PetscObjectGetComm ((PetscObject) zs->Aface, &comm));
  PetscCall (KSPCreate (comm, &k2));
  PetscCall (KSPSetType (k2, KSPCG));
  PetscCall (KSPGetPC (k2, &psh));
  PetscCall (PCSetType (psh, PCSHELL));
  PetscCall (PCShellSetContext (psh, zs));
  PetscCall (PCShellSetApply (psh, lgh_zs_bvcycle_pcapply_vec));
  PetscCall (KSPSetOperators (k2, zs->Aface, zs->Aface));
  PetscCall (KSPSetComputeSingularValues (k2, PETSC_TRUE));
  PetscCall (KSPSetTolerances (k2, 1e-12, 0., PETSC_DEFAULT, maxit));
  PetscCall (VecDuplicate (rhs, &y));
  PetscCall (VecZeroEntries (y));
  PetscCall (KSPSolve (k2, rhs, y));
  PetscCall (KSPComputeExtremeSingularValues (k2, emax, emin));
  PetscCall (KSPGetIterationNumber (k2, its));
  if (verbose && *its > 0) {
    /* the Lanczos Ritz values of the preconditioned operator: is the low
     * end an isolated outlier (deflatable) or a continuum?               */
    PetscInt            n = *its, neig, i;
    PetscReal          *re, *im;

    PetscCall (PetscMalloc2 (n, &re, n, &im));
    PetscCall (KSPComputeEigenvalues (k2, n, re, im, &neig));
    PetscCall (PetscSortReal (neig, re));
    PetscCall (PetscPrintf (comm, "lgh_prior: mode 3 ritz (%d): low",
                            (int) neig));
    for (i = 0; i < PetscMin (12, neig); i++)
      PetscCall (PetscPrintf (comm, " %.3e", (double) re[i]));
    PetscCall (PetscPrintf (comm, " | high"));
    for (i = PetscMax (0, neig - 3); i < neig; i++)
      PetscCall (PetscPrintf (comm, " %.3e", (double) re[i]));
    PetscCall (PetscPrintf (comm, "\n"));
    PetscCall (PetscFree2 (re, im));
  }
  PetscCall (VecDestroy (&y));
  PetscCall (KSPDestroy (&k2));
  return PETSC_SUCCESS;
}

/* Y = A^{-1} X on blocks, tiled; src == dst allowed (tile buffering).   */
static PetscErrorCode
lgh_zs_blocked_asolve (struct lgh_zs_ctx *zs, Mat Xsrc, Mat Ydst)
{
  PetscInt            nloc, ncols, c0, w_, j;
  PetscInt            ldas, ldad, ldat;
  lgh_zs_pcapply_fn   pcap;
  void               *pcctx;

  if (zs->mode == 3) { pcap = lgh_zs_bvcycle_pcapply; pcctx = zs->mgh; }
  else               { pcap = lgh_zs_pcmatapply_pcapply; pcctx = zs->amg; }
  PetscCall (MatGetLocalSize (Xsrc, &nloc, NULL));
  PetscCall (MatGetSize (Xsrc, NULL, &ncols));
  for (c0 = 0; c0 < ncols; c0 += zs->tile) {
    const PetscScalar  *xs;
    PetscScalar        *tb, *yd;
    const PetscScalar  *tx;

    w_ = PetscMin (zs->tile, ncols - c0);
    PetscCall (MatDenseGetLDA (Xsrc, &ldas));
    PetscCall (MatDenseGetLDA (zs->tB, &ldat));
    PetscCall (MatDenseGetArrayRead (Xsrc, &xs));
    PetscCall (MatDenseGetArrayWrite (zs->tB, &tb));
    for (j = 0; j < w_; j++)
      PetscCall (PetscMemcpy (tb + j * ldat, xs + (c0 + j) * ldas,
                              nloc * sizeof (PetscScalar)));
    for (j = w_; j < zs->tile; j++)
      PetscCall (PetscMemzero (tb + j * ldat, nloc * sizeof (PetscScalar)));
    PetscCall (MatDenseRestoreArrayWrite (zs->tB, &tb));
    PetscCall (MatDenseRestoreArrayRead (Xsrc, &xs));
    PetscCall (MatZeroEntries (zs->tX));
    PetscCall (MatCopy (zs->tB, zs->tR, SAME_NONZERO_PATTERN));
    PetscCall (lgh_zs_cheb_core (zs->Aface, pcap, pcctx, zs->lmin, zs->lmax,
                                 zs->nit, zs->tX, zs->tR, zs->tZ, zs->tD,
                                 &zs->tAD));
    PetscCall (MatDenseGetLDA (Ydst, &ldad));
    PetscCall (MatDenseGetArrayRead (zs->tX, &tx));
    PetscCall (MatDenseGetArray (Ydst, &yd));
    for (j = 0; j < w_; j++)
      PetscCall (PetscMemcpy (yd + (c0 + j) * ldad, tx + j * ldat,
                              nloc * sizeof (PetscScalar)));
    PetscCall (MatDenseRestoreArray (Ydst, &yd));
    PetscCall (MatDenseRestoreArrayRead (zs->tX, &tx));
  }
  return PETSC_SUCCESS;
}

/* Diagnostic (env LGH_ZS_DUMP=<prefix>; serial, modes 2/3, N <= 12000 only):
 * write the face operator A and the block V-cycle applied to the identity,
 * both as raw column-major float64 N x N files, for an offline exact
 * eigen-analysis of the V-cycle-preconditioned operator (deflation
 * studies).  Never runs unless the environment variable is set.          */
static PetscErrorCode
lgh_zs_dump_dense (struct lgh_zs_ctx *zs, Mat A, const char *prefix)
{
  MPI_Comm            comm;
  PetscMPIInt         size;
  PetscInt            N, c0, w, j, lda, ldaz;
  Mat                 Adense;
  const PetscScalar  *a, *tz;
  PetscScalar        *tb;
  FILE               *fp;
  char                name[PETSC_MAX_PATH_LEN];
  lgh_zs_pcapply_fn   pcap;
  void               *pcctx;

  PetscCall (PetscObjectGetComm ((PetscObject) A, &comm));
  PetscCallMPI (MPI_Comm_size (comm, &size));
  PetscCall (MatGetSize (A, &N, NULL));
  if (size != 1 || N > 12000 || zs->mode < 2) {
    PetscCall (PetscPrintf (comm, "lgh_zs_dump_dense: skipped (size %d, N %d, "
                            "mode %d)\n", (int) size, (int) N, zs->mode));
    return PETSC_SUCCESS;
  }
  if (zs->mode == 3) { pcap = lgh_zs_bvcycle_pcapply;    pcctx = zs->mgh; }
  else               { pcap = lgh_zs_pcmatapply_pcapply; pcctx = zs->amg; }
  PetscCall (MatConvert (A, MATDENSE, MAT_INITIAL_MATRIX, &Adense));
  PetscCall (MatDenseGetLDA (Adense, &lda));
  PetscCall (MatDenseGetArrayRead (Adense, &a));
  snprintf (name, sizeof name, "%s_A.f64", prefix);
  fp = fopen (name, "wb");
  for (j = 0; j < N; j++)
    fwrite (a + (size_t) j * lda, sizeof (PetscScalar), (size_t) N, fp);
  fclose (fp);
  PetscCall (MatDenseRestoreArrayRead (Adense, &a));
  PetscCall (MatDestroy (&Adense));

  snprintf (name, sizeof name, "%s_Minv.f64", prefix);
  fp = fopen (name, "wb");
  PetscCall (MatDenseGetLDA (zs->tB, &lda));
  PetscCall (MatDenseGetLDA (zs->tZ, &ldaz));
  for (c0 = 0; c0 < N; c0 += zs->tile) {
    w = PetscMin (zs->tile, N - c0);
    PetscCall (MatZeroEntries (zs->tB));
    PetscCall (MatDenseGetArrayWrite (zs->tB, &tb));
    for (j = 0; j < w; j++) tb[(c0 + j) + (size_t) j * lda] = 1.0;
    PetscCall (MatDenseRestoreArrayWrite (zs->tB, &tb));
    if (zs->mode == 3) {
      PetscCall (pcap (zs->tB, zs->tZ, pcctx));
    }
    else {   /* column-wise PCApply: PCMatApply is not available for every PC */
      for (j = 0; j < w; j++) {
        Vec                 bj, zj;

        PetscCall (MatDenseGetColumnVecRead (zs->tB, j, &bj));
        PetscCall (MatDenseGetColumnVecWrite (zs->tZ, j, &zj));
        PetscCall (PCApply (zs->amg, bj, zj));
        PetscCall (MatDenseRestoreColumnVecWrite (zs->tZ, j, &zj));
        PetscCall (MatDenseRestoreColumnVecRead (zs->tB, j, &bj));
      }
    }
    PetscCall (MatDenseGetArrayRead (zs->tZ, &tz));
    for (j = 0; j < w; j++)
      fwrite (tz + (size_t) j * ldaz, sizeof (PetscScalar), (size_t) N, fp);
    PetscCall (MatDenseRestoreArrayRead (zs->tZ, &tz));
  }
  fclose (fp);
  PetscCall (PetscPrintf (comm, "lgh_zs_dump_dense: wrote %s_A.f64 and "
                          "%s_Minv.f64 (N=%d, column-major float64)\n",
                          prefix, prefix, (int) N));
  return PETSC_SUCCESS;
}

static void
lgh_zs_destroy (struct lgh_zs_ctx *zs)
{
  if (zs == NULL) return;
  (void) MatDestroy (&zs->tB);
  (void) MatDestroy (&zs->tX);
  (void) MatDestroy (&zs->tR);
  (void) MatDestroy (&zs->tZ);
  (void) MatDestroy (&zs->tD);
  (void) MatDestroy (&zs->tAD);
  lgh_zs_mgh_destroy (zs->mgh);
  /* ksp is borrowed (the prior owns it) */
  (void) PetscFree (zs);
}

/* ================================================================== */
/* the prior's Mat path: adapters + create                              */

static void
lgh_prior_mat_applyZ (Vec x, Vec y, void *vctx)
{
  lgh_prior_t        *p = (lgh_prior_t *) vctx;
  PetscErrorCode      ierr = MatMult (p->Z, x, y);

  CHKERRABORT (PetscObjectComm ((PetscObject) p->Z), ierr);
}

static void
lgh_prior_mat_solveZ (Vec x, Vec y, void *vctx)
{
  lgh_prior_t        *p = (lgh_prior_t *) vctx;
  PetscErrorCode      ierr;

  ierr = VecZeroEntries (y);
  CHKERRABORT (PetscObjectComm ((PetscObject) p->Z), ierr);
  ierr = KSPSolve (p->zksp, x, y);
  CHKERRABORT (PetscObjectComm ((PetscObject) p->Z), ierr);
}

static void
lgh_prior_mat_solveZ_blocked (Mat X, Mat Y, void *vctx)
{
  lgh_prior_t        *p = (lgh_prior_t *) vctx;
  PetscErrorCode      ierr;

  if (p->zs->mode >= 2) {
    ierr = lgh_zs_blocked_asolve (p->zs, X, Y);
    CHKERRABORT (PetscObjectComm ((PetscObject) p->Z), ierr);
    return;
  }
  {
    PetscInt            j, ncols;

    ierr = MatGetSize (X, NULL, &ncols);
    CHKERRABORT (PetscObjectComm ((PetscObject) p->Z), ierr);
    for (j = 0; j < ncols; j++) {
      Vec                 xj, yj;

      ierr = MatDenseGetColumnVecRead (X, j, &xj);
      CHKERRABORT (PetscObjectComm ((PetscObject) p->Z), ierr);
      ierr = MatDenseGetColumnVecWrite (Y, j, &yj);
      CHKERRABORT (PetscObjectComm ((PetscObject) p->Z), ierr);
      lgh_prior_mat_solveZ (xj, yj, vctx);
      ierr = MatDenseRestoreColumnVecWrite (Y, j, &yj);
      CHKERRABORT (PetscObjectComm ((PetscObject) p->Z), ierr);
      ierr = MatDenseRestoreColumnVecRead (X, j, &xj);
      CHKERRABORT (PetscObjectComm ((PetscObject) p->Z), ierr);
    }
  }
}

/* Shared tail of the Mat and KSP paths: reference the solver and its
 * operator, stand up the Z-solve machinery + requested blocked tier, wire
 * the prior's callbacks.  ksp is referenced (never mutated).            */
static PetscErrorCode
lgh_prior_setup_from_ksp (KSP ksp, Vec mass_lumps,
                          const lgh_prior_mat_opts_t *o, lgh_prior_t **prior)
{
  lgh_prior_t        *p;
  MPI_Comm            comm;
  Mat                 Zop;

  PetscCall (lgh_prior_init_common (mass_lumps, &p));
  p->zksp = ksp;
  PetscCall (PetscObjectReference ((PetscObject) ksp));
  PetscCall (KSPGetOperators (ksp, &Zop, NULL));
  p->Z = Zop;
  PetscCall (PetscObjectReference ((PetscObject) Zop));
  PetscCall (PetscObjectGetComm ((PetscObject) Zop, &comm));

  PetscCall (lgh_zs_create (p->zksp, &p->zs));
  if (o->blocked_mode >= 2) {
    Vec                 rhs;
    PetscReal           emin, emax;
    PetscInt            its;
    PetscInt            rlo, rhi, i;
    PetscScalar        *ra;

    PetscCall (lgh_zs_blocked_setup (p->zs, o));
    PetscCall (MatCreateVecs (Zop, &rhs, NULL));
    PetscCall (VecGetOwnershipRange (rhs, &rlo, &rhi));
    PetscCall (VecGetArray (rhs, &ra));
    for (i = rlo; i < rhi; i++)
      ra[i - rlo] = lgh_randn_at (0xB0BD5UL, i, 0);
    PetscCall (VecRestoreArray (rhs, &ra));
    if (o->blocked_mode == 3) {
      PetscCall (lgh_zs_bounds_mode3 (p->zs, rhs, 200, &emin, &emax, &its,
                                      o->verbose));
    }
    else {
      PetscCall (lgh_zs_spectral_bounds (p->zs, rhs, 200, &emin, &emax,
                                         &its));
    }
    PetscCall (VecDestroy (&rhs));
    /* Lanczos bounds underestimate kappa: widen before trusting them */
    p->zs->lmin = 0.9 * emin;
    p->zs->lmax = 1.1 * emax;
    p->zs->nit = lgh_zs_cheb_count (p->zs->lmin, p->zs->lmax, o->cheb_rtol);
    if (o->verbose) {
      PetscCall (PetscPrintf (comm,
                              "lgh_prior: mode %d bounds "
                              "[%.3e, %.3e] (%d its) -> chebyshev nit %d\n",
                              o->blocked_mode, (double) emin, (double) emax,
                              (int) its, (int) p->zs->nit));
    }
    {
      const char         *dump = getenv ("LGH_ZS_DUMP");

      if (dump != NULL)
        PetscCall (lgh_zs_dump_dense (p->zs, Zop, dump));
    }
  }

  /* wire the callbacks: Z symmetric, so the t-variants fall back */
  p->cb.applyZ = lgh_prior_mat_applyZ;
  p->cb.solveZ = lgh_prior_mat_solveZ;
  p->cb.solveZ_blocked = lgh_prior_mat_solveZ_blocked;
  p->cb.ctx = p;

  *prior = p;
  return PETSC_SUCCESS;
}

int
lgh_prior_create_ksp (KSP ksp, Vec mass_lumps,
                      const lgh_prior_mat_opts_t *opts, lgh_prior_t **prior)
{
  lgh_prior_mat_opts_t o = (opts != NULL) ? *opts
                                          : lgh_prior_mat_opts_default ();
  Mat                 Zop = NULL;

  PetscCall (KSPGetOperators (ksp, &Zop, NULL));
  PetscCheck (Zop != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE,
              "lgh_prior_create_ksp: the KSP has no operator set");
  return lgh_prior_setup_from_ksp (ksp, mass_lumps, &o, prior);
}

int
lgh_prior_create_mat (Mat Z, Vec mass_lumps,
                      const lgh_prior_mat_opts_t *opts, lgh_prior_t **prior)
{
  lgh_prior_mat_opts_t o = (opts != NULL) ? *opts
                                          : lgh_prior_mat_opts_default ();
  MPI_Comm            comm;
  KSP                 ksp;
  PC                  pc;

  /* the solver for Z: CG + AMG, overridable via -lgh_prior_* options */
  PetscCall (PetscObjectGetComm ((PetscObject) Z, &comm));
  PetscCall (KSPCreate (comm, &ksp));
  PetscCall (KSPSetType (ksp, KSPCG));
  PetscCall (KSPGetPC (ksp, &pc));
  PetscCall (PCSetType (pc, PCGAMG));
  PetscCall (KSPSetOperators (ksp, Z, Z));
  PetscCall (KSPSetTolerances (ksp, o.ksp_rtol, 0., PETSC_DEFAULT, 10000));
  PetscCall (KSPSetOptionsPrefix (ksp, "lgh_prior_"));
  PetscCall (KSPSetFromOptions (ksp));
  PetscCall (KSPSetUp (ksp));

  PetscCall (lgh_prior_setup_from_ksp (ksp, mass_lumps, &o, prior));
  PetscCall (KSPDestroy (&ksp));   /* the prior holds its own reference */
  return PETSC_SUCCESS;
}

static void
lgh_zs_prior_teardown (lgh_prior_t *p)
{
  if (p->zs != NULL) { lgh_zs_destroy (p->zs); p->zs = NULL; }
  if (p->zksp != NULL) (void) KSPDestroy (&p->zksp);
}

int
lgh_have_hypre (void)
{
#if defined(PETSC_HAVE_HYPRE)
  return 1;
#else
  return 0;
#endif
}

#endif /* LGPSF_HESSIAN_ZSOLVE_IMPL_H */

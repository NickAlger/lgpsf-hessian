/* test_prior_mat.cpp — gates for lgh_prior_create_mat (the Z-solve
 * machinery), runnable at any communicator size (registered at n=1,2,4).
 *
 * Z = 2D 5-point Laplacian + 0.5 I on an m x m grid (SPD, AMG-friendly).
 * Reference: a tight CG solve on the same operator.  Checks:
 *   - per-column (mode 0), blocked-Chebyshev + PCMatApply (mode 2), and
 *     blocked-Chebyshev + harvested block V-cycle (mode 3) solves all
 *     match the reference;
 *   - blocked == per-column on a multi-column block;
 *   - solveZ(applyZ(v)) == v;
 *   - end-to-end: a replicated GLR build over a low-rank B with the
 *     mode-2 prior satisfies solve(apply(v)) == v.
 */

#include <lgpsf_hessian/lgpsf_hessian.h>
#include <lgpsf_hessian/impl.hpp>

#include <math.h>

#define GRID_M   24
#define N_GLOBAL (GRID_M * GRID_M)
#define RANK_B   6

static int          n_fail = 0;

static void
check (int ok, const char *what, double val)
{
  if (!ok) {
    n_fail++;
    PetscPrintf (PETSC_COMM_WORLD, "FAIL: %s (%.3e)\n", what, val);
  }
}

static double
m_entry (int i) { return 0.5 + 0.1 * (i % 7); }

static double
probe_entry (int i) { return cos (0.05 * i + 1.0); }

static double
factor_entry (int k, int i)
{
  return sin (0.1 * (i + 1) * (k + 3)) + 0.2 * cos (0.37 * i * k + k);
}

static PetscErrorCode
build_Z (Mat *Z_out)
{
  Mat                 Z;
  PetscInt            rstart, rend, i;

  PetscCall (MatCreateAIJ (PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
                           N_GLOBAL, N_GLOBAL, 5, NULL, 3, NULL, &Z));
  PetscCall (MatGetOwnershipRange (Z, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    const PetscInt      ix = i % GRID_M, iy = i / GRID_M;
    const PetscScalar   diag = 4.5, off = -1.0;   /* Laplacian + 0.5 I */

    PetscCall (MatSetValue (Z, i, i, diag, INSERT_VALUES));
    if (ix > 0) PetscCall (MatSetValue (Z, i, i - 1, off, INSERT_VALUES));
    if (ix < GRID_M - 1)
      PetscCall (MatSetValue (Z, i, i + 1, off, INSERT_VALUES));
    if (iy > 0)
      PetscCall (MatSetValue (Z, i, i - GRID_M, off, INSERT_VALUES));
    if (iy < GRID_M - 1)
      PetscCall (MatSetValue (Z, i, i + GRID_M, off, INSERT_VALUES));
  }
  PetscCall (MatAssemblyBegin (Z, MAT_FINAL_ASSEMBLY));
  PetscCall (MatAssemblyEnd (Z, MAT_FINAL_ASSEMBLY));
  *Z_out = Z;
  return PETSC_SUCCESS;
}

static PetscErrorCode
fill_vec (Vec v, double (*f) (int))
{
  PetscInt            rstart, rend;
  PetscScalar        *a;

  PetscCall (VecGetOwnershipRange (v, &rstart, &rend));
  PetscCall (VecGetArray (v, &a));
  for (PetscInt i = rstart; i < rend; i++) a[i - rstart] = f ((int) i);
  PetscCall (VecRestoreArray (v, &a));
  return PETSC_SUCCESS;
}

static PetscErrorCode
rel_diff (Vec a, Vec b, Vec scratch, double *out)
{
  PetscReal           nd, nb;

  PetscCall (VecCopy (a, scratch));
  PetscCall (VecAXPY (scratch, -1.0, b));
  PetscCall (VecNorm (scratch, NORM_2, &nd));
  PetscCall (VecNorm (b, NORM_2, &nb));
  *out = (double) (nd / nb);
  return PETSC_SUCCESS;
}

int
main (int argc, char **argv)
{
  Mat                 Z;
  Vec                 mass, b, xref, x, s1, b2, xref2;
  KSP                 kref;
  PC                  pcref;

  PetscCall (PetscInitialize (&argc, &argv, NULL, NULL));
  PetscCall (build_Z (&Z));
  PetscCall (MatCreateVecs (Z, &mass, NULL));
  PetscCall (VecDuplicate (mass, &b));
  PetscCall (VecDuplicate (mass, &xref));
  PetscCall (VecDuplicate (mass, &x));
  PetscCall (VecDuplicate (mass, &s1));
  PetscCall (fill_vec (mass, m_entry));
  PetscCall (fill_vec (b, probe_entry));

  /* reference solve */
  PetscCall (KSPCreate (PETSC_COMM_WORLD, &kref));
  PetscCall (KSPSetType (kref, KSPCG));
  PetscCall (KSPGetPC (kref, &pcref));
  PetscCall (PCSetType (pcref, PCGAMG));
  PetscCall (KSPSetOperators (kref, Z, Z));
  PetscCall (KSPSetTolerances (kref, 1e-13, 0., PETSC_DEFAULT, 10000));
  PetscCall (VecZeroEntries (xref));
  PetscCall (KSPSolve (kref, b, xref));
  /* a right-hand side with a large mean: exercises the constant deflation
   * (exact for any nonsingular Z; Z here is not shifted-singular, so this
   * checks the bookkeeping, not the benefit) */
  PetscCall (VecDuplicate (b, &b2));
  PetscCall (VecDuplicate (b, &xref2));
  PetscCall (VecCopy (b, b2));
  PetscCall (VecShift (b2, 7.0));
  PetscCall (VecZeroEntries (xref2));
  PetscCall (KSPSolve (kref, b2, xref2));

  /* passes: per-column KSP (0), blocked PCMatApply (2), blocked V-cycle on
   * the PCMG hierarchy (3), and -- when PETSc has hypre -- the blocked
   * V-cycle on a hypre-built hierarchy (3h).  Same gates for all.        */
  const int           n_pass = lgh_have_hypre () ? 4 : 3;
  for (int pass = 0; pass < n_pass; pass++) {
    lgh_prior_mat_opts_t po = lgh_prior_mat_opts_default ();
    lgh_prior_t        *prior;
    char                what[128];
    double              rd;
    const int           mode = (pass == 0) ? 0 : (pass == 1) ? 2 : 3;

    po.blocked_mode = mode;
    po.hierarchy = (pass == 3) ? LGH_HIERARCHY_HYPRE : LGH_HIERARCHY_PCMG;
    po.tile = 8;
    po.verbose = (mode == 3);
    if (mode == 3)
      PetscCall (PetscPrintf (PETSC_COMM_WORLD, "-- mode 3, hierarchy %s\n",
                              pass == 3 ? "hypre" : "pcmg"));
    PetscCall ((PetscErrorCode) lgh_prior_create_mat (Z, mass, &po, &prior));

    /* single-vector solve vs reference */
    lgh_prior_mat_solveZ (b, x, prior);
    PetscCall (rel_diff (x, xref, s1, &rd));
    snprintf (what, sizeof (what), "pass %d (mode %d) vec solve vs reference", pass, mode);
    check (rd < 1e-8, what, rd);
    lgh_prior_mat_solveZ (b2, x, prior);
    PetscCall (rel_diff (x, xref2, s1, &rd));
    snprintf (what, sizeof (what), "pass %d (mode %d) vec solve, mean-7 rhs, vs reference", pass, mode);
    check (rd < 1e-8, what, rd);
    {
      Mat                 X2, Y2;
      PetscInt            nloc;
      Vec                 cj;

      PetscCall (VecGetLocalSize (mass, &nloc));
      PetscCall (MatCreateDense (PETSC_COMM_WORLD, nloc, PETSC_DECIDE, N_GLOBAL, 3, NULL, &X2));
      PetscCall (MatCreateDense (PETSC_COMM_WORLD, nloc, PETSC_DECIDE, N_GLOBAL, 3, NULL, &Y2));
      for (int j = 0; j < 3; j++) {
        PetscCall (MatDenseGetColumnVecWrite (X2, j, &cj));
        PetscCall (VecCopy (j == 1 ? b2 : b, cj));
        if (j == 2) PetscCall (VecScale (cj, -3.0));
        PetscCall (MatDenseRestoreColumnVecWrite (X2, j, &cj));
      }
      PetscCall (lgh_prior_solve_block (prior, LGH_PRIOR_SOLVEZ, X2, Y2));
      PetscCall (MatDenseGetColumnVecRead (Y2, 1, &cj));
      PetscCall (VecCopy (cj, x));
      PetscCall (MatDenseRestoreColumnVecRead (Y2, 1, &cj));
      PetscCall (rel_diff (x, xref2, s1, &rd));
      snprintf (what, sizeof (what), "pass %d (mode %d) blocked solve, mean-7 column, vs reference", pass, mode);
      check (rd < 1e-7, what, rd);
      PetscCall (MatDenseGetColumnVecRead (Y2, 2, &cj));
      PetscCall (VecCopy (cj, x));
      PetscCall (MatDenseRestoreColumnVecRead (Y2, 2, &cj));
      PetscCall (VecScale (x, -1.0 / 3.0));
      PetscCall (rel_diff (x, xref, s1, &rd));
      snprintf (what, sizeof (what), "pass %d (mode %d) blocked solve, scaled column, vs reference", pass, mode);
      check (rd < 1e-7, what, rd);
      PetscCall (MatDestroy (&X2));
      PetscCall (MatDestroy (&Y2));
    }

    /* blocked solve vs reference (multi-column, width > tile) */
    {
      Mat                 X, Y;
      PetscInt            nloc;
      const int           ncols = 11;

      PetscCall (VecGetLocalSize (mass, &nloc));
      PetscCall (MatCreateDense (PETSC_COMM_WORLD, nloc, PETSC_DECIDE,
                                 N_GLOBAL, ncols, NULL, &X));
      PetscCall (MatCreateDense (PETSC_COMM_WORLD, nloc, PETSC_DECIDE,
                                 N_GLOBAL, ncols, NULL, &Y));
      for (int j = 0; j < ncols; j++) {
        Vec                 xj;
        PetscCall (MatDenseGetColumnVecWrite (X, j, &xj));
        PetscCall (VecCopy (b, xj));
        PetscCall (VecScale (xj, 1.0 + 0.1 * j));
        PetscCall (MatDenseRestoreColumnVecWrite (X, j, &xj));
      }
      PetscCall (lgh_prior_solve_block (prior, LGH_PRIOR_SOLVEZ, X, Y));
      {
        Vec                 yj;
        double              worst = 0.;
        for (int j = 0; j < ncols; j++) {
          PetscCall (MatDenseGetColumnVecRead (Y, j, &yj));
          PetscCall (VecCopy (yj, x));
          PetscCall (MatDenseRestoreColumnVecRead (Y, j, &yj));
          PetscCall (VecScale (x, 1.0 / (1.0 + 0.1 * j)));
          PetscCall (rel_diff (x, xref, s1, &rd));
          if (rd > worst) worst = rd;
        }
        snprintf (what, sizeof (what),
                  "pass %d (mode %d) blocked solve vs reference", pass, mode);
        check (worst < 1e-7, what, worst);
      }
      PetscCall (MatDestroy (&X));
      PetscCall (MatDestroy (&Y));
    }

    /* solveZ(applyZ(v)) == v */
    PetscCall ((PetscErrorCode) lgh_prior_apply (prior, b, x)); /* touches applyZ path */
    lgh_prior_mat_applyZ (b, s1, prior);
    lgh_prior_mat_solveZ (s1, x, prior);
    PetscCall (rel_diff (x, b, s1, &rd));
    snprintf (what, sizeof (what), "pass %d (mode %d) solveZ(applyZ(v)) == v", pass, mode);
    check (rd < 1e-8, what, rd);

    lgh_prior_destroy (prior);
  }

  /* bring-your-own-KSP path: wrap an already-configured solver (mode 2) */
  {
    KSP                 kown;
    PC                  pcown;
    lgh_prior_mat_opts_t po = lgh_prior_mat_opts_default ();
    lgh_prior_t        *prior;
    double              rd;

    PetscCall (KSPCreate (PETSC_COMM_WORLD, &kown));
    PetscCall (KSPSetType (kown, KSPCG));
    PetscCall (KSPGetPC (kown, &pcown));
    PetscCall (PCSetType (pcown, PCGAMG));
    PetscCall (KSPSetOperators (kown, Z, Z));
    PetscCall (KSPSetTolerances (kown, 1e-12, 0., PETSC_DEFAULT, 10000));
    PetscCall (KSPSetUp (kown));
    po.blocked_mode = 2;
    po.tile = 8;
    PetscCall ((PetscErrorCode) lgh_prior_create_ksp (kown, mass, &po,
                                                      &prior));
    lgh_prior_mat_solveZ (b, x, prior);
    PetscCall (rel_diff (x, xref, s1, &rd));
    check (rd < 1e-8, "create_ksp vec solve vs reference", rd);
    lgh_prior_destroy (prior);
    PetscCall (KSPDestroy (&kown));   /* borrowed by the prior, refcounted */
  }

  /* end-to-end: replicated GLR over a low-rank B with a mode-2 prior */
  {
    Mat                 B;
    lgh_prior_mat_opts_t po = lgh_prior_mat_opts_default ();
    lgh_prior_t        *prior;
    lgh_glr_t          *glr;
    lgh_glr_opts_t      go = lgh_glr_opts_default ();
    lgh_glr_report_t    rep;
    PetscInt            rstart, rend;
    const double        c = 0.7;
    double              rd;
    Vec                 v, y;

    PetscCall (MatCreateDense (PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
                               N_GLOBAL, N_GLOBAL, NULL, &B));
    PetscCall (MatGetOwnershipRange (B, &rstart, &rend));
    {
      PetscScalar        *ba;
      PetscInt            lda;
      PetscCall (MatDenseGetLDA (B, &lda));
      PetscCall (MatDenseGetArrayWrite (B, &ba));
      for (PetscInt j = 0; j < N_GLOBAL; j++)
        for (PetscInt i = rstart; i < rend; i++) {
          double              s = 0.;
          for (int k = 0; k < RANK_B; k++)
            s += 2.0 / (k + 1.0) * factor_entry (k, (int) i)
              * factor_entry (k, (int) j);
          ba[(i - rstart) + j * lda] = s;
        }
      PetscCall (MatDenseRestoreArrayWrite (B, &ba));
    }
    PetscCall (MatAssemblyBegin (B, MAT_FINAL_ASSEMBLY));
    PetscCall (MatAssemblyEnd (B, MAT_FINAL_ASSEMBLY));

    po.blocked_mode = 2;
    po.tile = 8;
    PetscCall ((PetscErrorCode) lgh_prior_create_mat (Z, mass, &po, &prior));
    go.ell = 16;
    go.trunc_rel = 1e-6;
    go.backend = LGH_GLR_REPLICATED;
    PetscCall ((PetscErrorCode) lgh_glr_compute (B, prior, &go, &glr, &rep));
    check (rep.kept == RANK_B, "e2e kept == rank(B)", (double) rep.kept);

    PetscCall (VecDuplicate (mass, &v));
    PetscCall (VecDuplicate (mass, &y));
    PetscCall (fill_vec (v, probe_entry));
    PetscCall ((PetscErrorCode) lgh_glr_apply (glr, c, v, y));
    PetscCall ((PetscErrorCode) lgh_glr_solve (glr, c, y, x));
    PetscCall (rel_diff (x, v, s1, &rd));
    check (rd < 1e-6, "e2e solve(apply(v)) == v (mode-2 prior)", rd);

    lgh_glr_destroy (glr);
    lgh_prior_destroy (prior);
    PetscCall (VecDestroy (&v));
    PetscCall (VecDestroy (&y));
    PetscCall (MatDestroy (&B));
  }

  if (n_fail == 0)
    PetscCall (PetscPrintf (PETSC_COMM_WORLD, "PASS test_prior_mat\n"));
  PetscCall (KSPDestroy (&kref));
  PetscCall (VecDestroy (&b2));
  PetscCall (VecDestroy (&xref2));
  PetscCall (VecDestroy (&mass));
  PetscCall (VecDestroy (&b));
  PetscCall (VecDestroy (&xref));
  PetscCall (VecDestroy (&x));
  PetscCall (VecDestroy (&s1));
  PetscCall (MatDestroy (&Z));
  PetscCall (PetscFinalize ());
  return n_fail == 0 ? 0 : 1;
}

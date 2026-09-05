/* prior_zsolve_bench.cpp — how good is the blocked prior solve at scale?
 *
 * Builds Z = -Delta_h + mass * M on an m x m Neumann grid of side L (the P1
 * stiffness matrix on the uniform right-triangle mesh, which is the
 * 4-neighbour graph Laplacian, plus the lumped mass h^2 w_x w_y), wraps it
 * with lgh_prior_create_mat in the blocked tier (mode 3 by default), and
 * reports what the setup found: hierarchy (source, levels, sizes, operator
 * complexity), the preconditioned spectral interval and the Chebyshev
 * count derived from it, the setup time, and the time and residual of a
 * blocked solve on `-tile` random columns.  Every hierarchy knob of
 * lgh_prior_mat_opts_t is an option, so the effect of level count, domain
 * size, near-null shift, rank count and hypre's coarsening parameters on
 * the Chebyshev interval can be mapped without an application.
 *
 * The ice-sheet prior it imitates: -Delta + 1e-5 M on a km-unit basal mesh
 * of ~5000 km extent (409k nodes at the continent, h ~ 1-50 km); at
 * m=640, L=5000 the shift sits ~25 Laplacian modes above the bottom, as
 * there.  Structured, uniform and strip-partitioned, so it tests
 * hypotheses, not the exact numbers of an unstructured graded mesh.
 *
 * Options (PETSc style; the internal Z solver takes the prefix
 * -lgh_prior_, e.g. -lgh_prior_pc_gamg_threshold 0.01 for the PCMG path):
 *   -m 160  -L 5000  -mass 1e-5
 *   -hierarchy auto|pcmg|hypre   -coarsen 10 -interp 6 -strong 0.25
 *   -agg_nl 0 -max_coarse 200    -nu 0 (force smoother degree)
 *   -mode 3 -tile 64 -cheb_rtol 1e-3 -smoother_top 1.1 -nsolve 1
 *   -load <file>  a production operator instead of the grid: PETSc binary
 *                 holding the Mat then the mass-lump Vec, as written by the
 *                 library's LGH_ZS_DUMP_SPARSE=<file> hook at prior setup
 *
 * Examples:  mpiexec -n 4 ./prior_zsolve_bench -m 640 -L 5000 -mass 1e-5
 *            mpiexec -n 4 ./prior_zsolve_bench -load prior_continent.petsc -strong 0.5
 */

#include <lgpsf_hessian/lgpsf_hessian.h>
#include <lgpsf_hessian/impl.hpp>

#include <math.h>
#include <string.h>

static PetscErrorCode
build_Z (PetscInt m, PetscReal L, PetscReal mass, Mat *Z_out, Vec *lump_out)
{
  const PetscInt      N = m * m;
  const PetscReal     h = L / (PetscReal) (m - 1), h2 = h * h;
  Mat                 Z;
  Vec                 lump;
  PetscInt            rstart, rend, i;

  PetscCall (MatCreateAIJ (PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
                           N, N, 5, NULL, 3, NULL, &Z));
  PetscCall (MatCreateVecs (Z, &lump, NULL));
  PetscCall (MatGetOwnershipRange (Z, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    const PetscInt      ix = i % m, iy = i / m;
    const PetscReal     wx = (ix == 0 || ix == m - 1) ? 0.5 : 1.0;
    const PetscReal     wy = (iy == 0 || iy == m - 1) ? 0.5 : 1.0;
    const PetscReal     ml = h2 * wx * wy;
    PetscInt            deg = 0;

    if (ix > 0)     { deg++; PetscCall (MatSetValue (Z, i, i - 1, -1.0, INSERT_VALUES)); }
    if (ix < m - 1) { deg++; PetscCall (MatSetValue (Z, i, i + 1, -1.0, INSERT_VALUES)); }
    if (iy > 0)     { deg++; PetscCall (MatSetValue (Z, i, i - m, -1.0, INSERT_VALUES)); }
    if (iy < m - 1) { deg++; PetscCall (MatSetValue (Z, i, i + m, -1.0, INSERT_VALUES)); }
    PetscCall (MatSetValue (Z, i, i, (PetscReal) deg + mass * ml, INSERT_VALUES));
    PetscCall (VecSetValue (lump, i, ml, INSERT_VALUES));
  }
  PetscCall (MatAssemblyBegin (Z, MAT_FINAL_ASSEMBLY));
  PetscCall (MatAssemblyEnd (Z, MAT_FINAL_ASSEMBLY));
  PetscCall (VecAssemblyBegin (lump));
  PetscCall (VecAssemblyEnd (lump));
  *Z_out = Z;
  *lump_out = lump;
  return PETSC_SUCCESS;
}

int
main (int argc, char **argv)
{
  PetscInt            m = 160, nsolve = 1;
  PetscReal           L = 5000., mass = 1e-5;
  char                hier[32] = "auto", load[PETSC_MAX_PATH_LEN] = "";
  PetscBool           have_load = PETSC_FALSE;
  lgh_prior_mat_opts_t po = lgh_prior_mat_opts_default ();
  Mat                 Z;
  Vec                 lump;
  lgh_prior_t        *prior = NULL;
  PetscMPIInt         size;
  double              t0, t_setup, t_solve;

  PetscCall (PetscInitialize (&argc, &argv, NULL,
                              "blocked prior Z-solve benchmark on -Delta_h + mass M\n"));
  PetscCallMPI (MPI_Comm_size (PETSC_COMM_WORLD, &size));
  po.blocked_mode = 3;
  po.cheb_rtol = 1e-3;
  po.verbose = 1;
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-m", &m, NULL));
  PetscCall (PetscOptionsGetReal (NULL, NULL, "-L", &L, NULL));
  PetscCall (PetscOptionsGetReal (NULL, NULL, "-mass", &mass, NULL));
  PetscCall (PetscOptionsGetString (NULL, NULL, "-hierarchy", hier, sizeof (hier), NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-coarsen", &po.hypre_coarsen, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-interp", &po.hypre_interp, NULL));
  PetscCall (PetscOptionsGetReal (NULL, NULL, "-strong", &po.hypre_strong, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-agg_nl", &po.hypre_agg_nl, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-max_coarse", &po.hypre_max_coarse, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-nu", &po.nu_force, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-mode", &po.blocked_mode, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-tile", &po.tile, NULL));
  PetscCall (PetscOptionsGetReal (NULL, NULL, "-cheb_rtol", &po.cheb_rtol, NULL));
  PetscCall (PetscOptionsGetReal (NULL, NULL, "-smoother_top", &po.smoother_top, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-nsolve", &nsolve, NULL));
  PetscCall (PetscOptionsGetString (NULL, NULL, "-load", load, sizeof (load), &have_load));
  if (!strcmp (hier, "pcmg") || !strcmp (hier, "gamg")) po.hierarchy = LGH_HIERARCHY_PCMG;
  else if (!strcmp (hier, "hypre")) po.hierarchy = LGH_HIERARCHY_HYPRE;
  else po.hierarchy = LGH_HIERARCHY_AUTO;

  if (have_load) {
    PetscViewer         v;
    PetscInt            N;

    PetscCall (PetscViewerBinaryOpen (PETSC_COMM_WORLD, load, FILE_MODE_READ, &v));
    PetscCall (MatCreate (PETSC_COMM_WORLD, &Z));
    PetscCall (MatSetType (Z, MATAIJ));
    PetscCall (MatLoad (Z, v));
    PetscCall (MatCreateVecs (Z, &lump, NULL));
    PetscCall (VecLoad (lump, v));
    PetscCall (PetscViewerDestroy (&v));
    PetscCall (MatGetSize (Z, &N, NULL));
    m = (PetscInt) PetscSqrtReal ((PetscReal) N);   /* only for the column count below */
    PetscCall (PetscPrintf (PETSC_COMM_WORLD,
      "bench: loaded %s: N = %d, ranks %d, hierarchy %s (hypre compiled in: %s)\n",
      load, (int) N, (int) size, hier, lgh_have_hypre () ? "yes" : "no"));
  } else PetscCall (build_Z (m, L, mass, &Z, &lump));
  if (!have_load) {
    const PetscReal     h = L / (PetscReal) (m - 1);
    /* Laplacian modes below the shift: (k pi / L)^2 < mass  =>  k < L sqrt(mass) / pi
     * per direction (continuum estimate, Neumann) */
    const PetscReal     kmax = L * PetscSqrtReal (mass) / PETSC_PI;
    PetscCall (PetscPrintf (PETSC_COMM_WORLD,
      "bench: N = %d (m = %d), L = %g, h = %g, mass = %g, ranks %d, hierarchy %s (hypre compiled in: %s)\n"
      "bench: shift / lowest nonzero Laplacian eigenvalue ~ %.1f  (~%.0f modes under the shift)\n",
      (int) (m * m), (int) m, (double) L, (double) h, (double) mass, (int) size, hier,
      lgh_have_hypre () ? "yes" : "no",
      (double) (kmax * kmax), (double) (0.25 * PETSC_PI * kmax * kmax)));
  }

  PetscCallMPI (MPI_Barrier (PETSC_COMM_WORLD));
  t0 = MPI_Wtime ();
  PetscCall ((PetscErrorCode) lgh_prior_create_mat (Z, lump, &po, &prior));
  PetscCallMPI (MPI_Barrier (PETSC_COMM_WORLD));
  t_setup = MPI_Wtime () - t0;

  /* one blocked solve on random columns; residual of column 0 against Z */
  {
    Mat                 X, Y;
    Vec                 x0, y0, r;
    PetscInt            nloc;
    PetscReal           nr, nx;
    PetscInt            N;
    PetscRandom         rnd;

    PetscCall (VecGetSize (lump, &N));
    PetscCall (VecGetLocalSize (lump, &nloc));
    PetscCall (MatCreateDense (PETSC_COMM_WORLD, nloc, PETSC_DECIDE, N, po.tile, NULL, &X));
    PetscCall (MatCreateDense (PETSC_COMM_WORLD, nloc, PETSC_DECIDE, N, po.tile, NULL, &Y));
    PetscCall (PetscRandomCreate (PETSC_COMM_WORLD, &rnd));
    PetscCall (PetscRandomSetSeed (rnd, 7));
    PetscCall (PetscRandomSeed (rnd));
    PetscCall (MatSetRandom (X, rnd));
    PetscCall (PetscRandomDestroy (&rnd));
    PetscCallMPI (MPI_Barrier (PETSC_COMM_WORLD));
    t0 = MPI_Wtime ();
    for (PetscInt s = 0; s < nsolve; s++)
      PetscCall (lgh_prior_solve_block (prior, LGH_PRIOR_SOLVEZ, X, Y));
    PetscCallMPI (MPI_Barrier (PETSC_COMM_WORLD));
    t_solve = (MPI_Wtime () - t0) / (double) nsolve;

    PetscCall (VecDuplicate (lump, &r));
    PetscCall (MatDenseGetColumnVecRead (X, 0, &x0));
    PetscCall (MatDenseGetColumnVecRead (Y, 0, &y0));
    PetscCall (MatMult (Z, y0, r));
    PetscCall (VecAXPY (r, -1.0, x0));
    PetscCall (VecNorm (r, NORM_2, &nr));
    PetscCall (VecNorm (x0, NORM_2, &nx));
    PetscCall (MatDenseRestoreColumnVecRead (X, 0, &x0));
    PetscCall (MatDenseRestoreColumnVecRead (Y, 0, &y0));
    PetscCall (PetscPrintf (PETSC_COMM_WORLD,
      "bench: setup %.2f s | blocked solve of %d columns %.3f s (%.2f ms per column) | "
      "column-0 residual |Z y - x| / |x| = %.2e (cheb_rtol %g)\n",
      t_setup, (int) po.tile, t_solve, 1e3 * t_solve / (double) po.tile,
      (double) (nr / nx), (double) po.cheb_rtol));
    PetscCall (VecDestroy (&r));
    PetscCall (MatDestroy (&X));
    PetscCall (MatDestroy (&Y));
  }

  lgh_prior_destroy (prior);
  PetscCall (VecDestroy (&lump));
  PetscCall (MatDestroy (&Z));
  PetscCall (PetscFinalize ());
  return 0;
}

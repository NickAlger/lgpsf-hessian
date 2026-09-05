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
 *   -zero_mean    block entries in [-1, 1) instead of [0, 1)
 *   -energy_norm  also print the residual in the Z^-1 norm (the one the
 *                 Chebyshev bound controls; the 2-norm can exceed it by up
 *                 to sqrt(theta_max / theta_min) on a mean-heavy block)
 *   -nit_list 3,4,5,6,8   then force these counts on the same setup and
 *                 print time and both residuals per count: the
 *                 time-to-accuracy curve, which is what compares two
 *                 hierarchies fairly (their nominal counts differ in slack)
 *   -load <file>  a production operator instead of the grid: PETSc binary
 *                 holding the Mat then the mass-lump Vec, as written by the
 *                 library's LGH_ZS_DUMP_SPARSE=<file> hook at prior setup
 *   -layout <file>  replay the recorded row distribution (<dump>.layout,
 *                 same rank count as the recording run) instead of PETSc's
 *                 contiguous default; -repartition parmetis then rebalances
 *                 the loaded operator with a graph partition of its own
 *                 adjacency (what a basal-mesh repartition in the
 *                 application would do) before the setup.
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
  char                layout[PETSC_MAX_PATH_LEN] = "", repart[32] = "none";
  PetscBool           have_load = PETSC_FALSE, have_layout = PETSC_FALSE;
  PetscBool           zero_mean = PETSC_FALSE;   /* -zero_mean: block entries in [-1, 1) instead of [0, 1) */
  PetscBool           energy_norm = PETSC_FALSE; /* -energy_norm: also |r|_{Z^-1} / |x|_{Z^-1} (the norm Chebyshev controls) */
  PetscInt            nit_list[32], n_nit = 32;  /* -nit_list 3,4,5,...: after the nominal solve, force these Chebyshev
                                                    counts on the same setup and report time + residuals per count: the
                                                    time-to-accuracy curve, which compares configurations fairly */
  PetscMPIInt         rank;
  lgh_prior_mat_opts_t po = lgh_prior_mat_opts_default ();
  Mat                 Z;
  Vec                 lump;
  lgh_prior_t        *prior = NULL;
  PetscMPIInt         size;
  double              t0, t_setup, t_solve;

  PetscCall (PetscInitialize (&argc, &argv, NULL,
                              "blocked prior Z-solve benchmark on -Delta_h + mass M\n"));
  PetscCallMPI (MPI_Comm_size (PETSC_COMM_WORLD, &size));
  PetscCallMPI (MPI_Comm_rank (PETSC_COMM_WORLD, &rank));
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
  PetscCall (PetscOptionsGetString (NULL, NULL, "-layout", layout, sizeof (layout), &have_layout));
  PetscCall (PetscOptionsGetString (NULL, NULL, "-repartition", repart, sizeof (repart), NULL));
  PetscCall (PetscOptionsGetBool (NULL, NULL, "-zero_mean", &zero_mean, NULL));
  PetscCall (PetscOptionsGetBool (NULL, NULL, "-energy_norm", &energy_norm, NULL));
  PetscCall (PetscOptionsGetIntArray (NULL, NULL, "-nit_list", nit_list, &n_nit, NULL));
  if (!strcmp (hier, "pcmg") || !strcmp (hier, "gamg")) po.hierarchy = LGH_HIERARCHY_PCMG;
  else if (!strcmp (hier, "hypre")) po.hierarchy = LGH_HIERARCHY_HYPRE;
  else po.hierarchy = LGH_HIERARCHY_AUTO;

  if (have_load) {
    PetscViewer         v;
    PetscInt            N;

    PetscCall (PetscViewerBinaryOpen (PETSC_COMM_WORLD, load, FILE_MODE_READ, &v));
    PetscCall (MatCreate (PETSC_COMM_WORLD, &Z));
    PetscCall (MatSetType (Z, MATAIJ));
    if (have_layout) {
      /* recorded local row counts: rank 0 reads, everyone gets its own */
      PetscInt           *counts = NULL, nl, Nl;
      int                 nr = 0;

      if (rank == 0) {
        FILE               *f = fopen (layout, "r");

        PetscCheck (f != NULL, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "cannot read %s", layout);
        PetscCheck (fscanf (f, "%d %" PetscInt_FMT, &nr, &Nl) == 2, PETSC_COMM_SELF,
                    PETSC_ERR_FILE_READ, "%s: bad header", layout);
        PetscCheck (nr == (int) size, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
                    "%s was recorded on %d ranks, this run has %d", layout, nr, (int) size);
        PetscCall (PetscMalloc1 (size, &counts));
        for (int r = 0; r < nr; r++)
          PetscCheck (fscanf (f, "%" PetscInt_FMT, &counts[r]) == 1, PETSC_COMM_SELF,
                      PETSC_ERR_FILE_READ, "%s: short", layout);
        fclose (f);
      }
      PetscCallMPI (MPI_Scatter (counts, 1, MPIU_INT, &nl, 1, MPIU_INT, 0, PETSC_COMM_WORLD));
      if (rank == 0) PetscCall (PetscFree (counts));
      PetscCall (MatSetSizes (Z, nl, nl, PETSC_DETERMINE, PETSC_DETERMINE));
    }
    PetscCall (MatLoad (Z, v));
    PetscCall (MatCreateVecs (Z, &lump, NULL));
    PetscCall (VecLoad (lump, v));
    PetscCall (PetscViewerDestroy (&v));
    PetscCall (MatGetSize (Z, &N, NULL));
    m = (PetscInt) PetscSqrtReal ((PetscReal) N);   /* only for the column count below */
    PetscCall (PetscPrintf (PETSC_COMM_WORLD,
      "bench: loaded %s: N = %d, ranks %d, hierarchy %s (hypre compiled in: %s)%s\n",
      load, (int) N, (int) size, hier, lgh_have_hypre () ? "yes" : "no",
      have_layout ? ", recorded row layout" : ""));
  } else PetscCall (build_Z (m, L, mass, &Z, &lump));

  if (strcmp (repart, "none") && size > 1) {
    /* rebalance: graph partition of Z's own adjacency (ParMETIS through
     * PETSc's MatPartitioning), then the rows move to their new owners and
     * are renumbered contiguously per rank -- the operator the blocked
     * cycle would see after a basal-mesh repartition in the application */
    Mat                 adj, Z2;
    Vec                 lump2;
    MatPartitioning     part;
    IS                  is_part, is_num, is_rows;
    PetscInt           *counts;
    VecScatter          sc;

    PetscCheck (!strcmp (repart, "parmetis"), PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE,
                "-repartition: %s (parmetis | none)", repart);
    PetscCall (MatConvert (Z, MATMPIADJ, MAT_INITIAL_MATRIX, &adj));
    PetscCall (MatPartitioningCreate (PETSC_COMM_WORLD, &part));
    PetscCall (MatPartitioningSetAdjacency (part, adj));
    PetscCall (MatPartitioningSetType (part, MATPARTITIONINGPARMETIS));
    PetscCall (MatPartitioningSetFromOptions (part));
    PetscCall (MatPartitioningApply (part, &is_part));
    PetscCall (ISPartitioningToNumbering (is_part, &is_num));
    PetscCall (PetscMalloc1 (size, &counts));
    PetscCall (ISPartitioningCount (is_part, size, counts));
    PetscCall (ISInvertPermutation (is_num, counts[rank], &is_rows));
    PetscCall (MatCreateSubMatrix (Z, is_rows, is_rows, MAT_INITIAL_MATRIX, &Z2));
    PetscCall (VecCreateMPI (PETSC_COMM_WORLD, counts[rank], PETSC_DETERMINE, &lump2));
    PetscCall (VecScatterCreate (lump, is_rows, lump2, NULL, &sc));
    PetscCall (VecScatterBegin (sc, lump, lump2, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall (VecScatterEnd (sc, lump, lump2, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall (VecScatterDestroy (&sc));
    PetscCall (ISDestroy (&is_rows));
    PetscCall (ISDestroy (&is_num));
    PetscCall (ISDestroy (&is_part));
    PetscCall (PetscFree (counts));
    PetscCall (MatPartitioningDestroy (&part));
    PetscCall (MatDestroy (&adj));
    PetscCall (MatDestroy (&Z));
    PetscCall (VecDestroy (&lump));
    Z = Z2;
    lump = lump2;
    PetscCall (PetscPrintf (PETSC_COMM_WORLD, "bench: rows repartitioned with %s\n", repart));
  }
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
    if (zero_mean) PetscCall (PetscRandomSetInterval (rnd, -1.0, 1.0));
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
    if (energy_norm) {
      /* the Chebyshev bound is on the error's energy norm, i.e. the residual's
       * Z^{-1} norm; on a right-hand side rich in the strongly amplified low
       * modes the 2-norm residual can exceed it by up to sqrt(theta_max /
       * theta_min).  Two tight CG solves (Z^{-1} r and Z^{-1} x) measure it. */
      KSP                 kt;
      PC                  pct;
      Vec                 t, x0c;
      PetscReal           rr, xx;

      PetscCall (KSPCreate (PETSC_COMM_WORLD, &kt));
      PetscCall (KSPSetType (kt, KSPCG));
      PetscCall (KSPGetPC (kt, &pct));
      PetscCall (PCSetType (pct, PCGAMG));
      PetscCall (KSPSetOperators (kt, Z, Z));
      PetscCall (KSPSetTolerances (kt, 1e-12, 0., PETSC_DEFAULT, 500));
      PetscCall (KSPSetOptionsPrefix (kt, "energy_"));
      PetscCall (KSPSetFromOptions (kt));
      PetscCall (VecDuplicate (r, &t));
      PetscCall (VecDuplicate (r, &x0c));
      PetscCall (MatDenseGetColumnVecRead (X, 0, &x0));
      PetscCall (VecCopy (x0, x0c));
      PetscCall (MatDenseRestoreColumnVecRead (X, 0, &x0));
      PetscCall (VecZeroEntries (t));
      PetscCall (KSPSolve (kt, r, t));
      PetscCall (VecDot (r, t, &rr));
      PetscCall (VecZeroEntries (t));
      PetscCall (KSPSolve (kt, x0c, t));
      PetscCall (VecDot (x0c, t, &xx));
      PetscCall (PetscPrintf (PETSC_COMM_WORLD,
        "bench: column-0 residual in the Z^-1 norm: |r|_{Z^-1} / |x|_{Z^-1} = %.2e\n",
        (double) PetscSqrtReal (rr / xx)));
      PetscCall (VecDestroy (&t));
      PetscCall (VecDestroy (&x0c));
      PetscCall (KSPDestroy (&kt));
    }
    if (n_nit > 0 && n_nit < 32) {
      /* time-to-accuracy curve on the same setup: force the count, re-solve */
      KSP                 kt;
      PC                  pct;
      Vec                 t, x0c;
      PetscReal           xx = 1., rr;
      const PetscInt      nit_nominal = prior->zs->nit;

      PetscCall (KSPCreate (PETSC_COMM_WORLD, &kt));
      PetscCall (KSPSetType (kt, KSPCG));
      PetscCall (KSPGetPC (kt, &pct));
      PetscCall (PCSetType (pct, PCGAMG));
      PetscCall (KSPSetOperators (kt, Z, Z));
      PetscCall (KSPSetTolerances (kt, 1e-12, 0., PETSC_DEFAULT, 500));
      PetscCall (KSPSetOptionsPrefix (kt, "energy_"));
      PetscCall (KSPSetFromOptions (kt));
      PetscCall (VecDuplicate (r, &t));
      PetscCall (VecDuplicate (r, &x0c));
      PetscCall (MatDenseGetColumnVecRead (X, 0, &x0));
      PetscCall (VecCopy (x0, x0c));
      PetscCall (MatDenseRestoreColumnVecRead (X, 0, &x0));
      PetscCall (VecZeroEntries (t));
      PetscCall (KSPSolve (kt, x0c, t));
      PetscCall (VecDot (x0c, t, &xx));
      for (PetscInt q = 0; q < n_nit; q++) {
        prior->zs->nit = nit_list[q];
        PetscCallMPI (MPI_Barrier (PETSC_COMM_WORLD));
        t0 = MPI_Wtime ();
        PetscCall (lgh_prior_solve_block (prior, LGH_PRIOR_SOLVEZ, X, Y));
        PetscCallMPI (MPI_Barrier (PETSC_COMM_WORLD));
        t_solve = MPI_Wtime () - t0;
        PetscCall (MatDenseGetColumnVecRead (Y, 0, &y0));
        PetscCall (MatMult (Z, y0, r));
        PetscCall (MatDenseRestoreColumnVecRead (Y, 0, &y0));
        PetscCall (VecAXPY (r, -1.0, x0c));
        PetscCall (VecNorm (r, NORM_2, &nr));
        PetscCall (VecZeroEntries (t));
        PetscCall (KSPSolve (kt, r, t));
        PetscCall (VecDot (r, t, &rr));
        PetscCall (PetscPrintf (PETSC_COMM_WORLD,
          "bench: nit %2d | solve of %d columns %.3f s | residual 2-norm %.2e | Z^-1-norm %.2e\n",
          (int) nit_list[q], (int) po.tile, t_solve, (double) (nr / nx),
          (double) PetscSqrtReal (rr / xx)));
      }
      prior->zs->nit = nit_nominal;
      PetscCall (VecDestroy (&t));
      PetscCall (VecDestroy (&x0c));
      PetscCall (KSPDestroy (&kt));
    }
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

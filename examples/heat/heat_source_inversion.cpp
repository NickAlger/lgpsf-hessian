/* heat_source_inversion.cpp — the complete lgpsf-hessian pipeline on a
 * heat-equation source-inversion problem with scattered observations.
 *
 * Problem.  A source s on a rectangular domain (regular grid, 5-point
 * finite differences, lumped mass M = h^2 I) diffuses to time T:
 * u(T) = L s with L = ((M + dt kappa K)^{-1} M)^{nt} (implicit Euler; K =
 * the 5-point stiffness).  Noisy point observations y = B L s + noise,
 * B = evaluation at n_obs scattered locations drawn from a Gaussian
 * centered in the right half of the domain (dense there, sparse on the
 * left).  With prior precision R = Z M^{-1} Z, Z = a K + b M (a shifted
 * Laplacian; correlation length ~ sqrt(a/b)), the deterministic problem is
 *
 *     min_s 0.5 |y - B L s|^2 / sigma^2 + 0.5 s^T R s ,
 *
 * whose Hessian is H = Hd + R with Hd = L^T B^T B L / sigma^2 — an
 * operator with local point-spread structure (each row is a blurred
 * splat of the observation pattern near that node).  The Laplace
 * posterior is N(s*, H^{-1}).
 *
 * Pipeline (all three lgpsf-hessian stages):
 *   1. lgh_fit_probes: fit the sparse Hd_lg ~= Hd from hashed probe pairs,
 *      recording the energy-QC as the probe count grows (the QC-vs-k
 *      curve), with a-priori isotropic ellipsoids from the physics
 *      (ell = 2 sqrt(2 kappa T), the 2-sigma width of L delta; the x5
 *      support margin is lgpsf's window).
 *   2. lgh_prior_create_mat(Z): the prior wrapped with the blocked
 *      Chebyshev tier; lgh_glr_compute: the GLR eigendecomposition of the
 *      prior-preconditioned Hd (spectrum written out for plotting).
 *   3. s* = lgh_glr_solve(rhs) (the approximate-Hessian solve);
 *      posterior draws s* + lgh_glr_sample(xi); an empirical pointwise-
 *      sigma map from many draws; lgh_glr_logdet for flavor.
 *
 * Outputs (out/): truth.pgm, obs_points.txt, data_misfit.txt, recon.pgm,
 * sample_0..3.pgm, std.pgm, eigs.txt, qc_vs_k.txt.  Render with plot.py.
 *
 * Run:  ./heat_source_inversion            (or mpiexec -n 4 ...)
 * Key options: -height 128|256, -nobs 2000, -noise_rel 0.05,
 * -prior_len 0.25, -prior_b <strength>, -kmax 60, -ell 2200, -threads 4.
 */

#include <lgpsf_hessian/lgpsf_hessian.h>
#include <lgpsf_hessian/impl.hpp>

#include <petscdmda.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <vector>

/* ---------------------------------------------------------------- */

typedef struct
{
  DM                  da;
  Mat                 Aheat;    /* M + dt kappa K                     */
  KSP                 kheat;
  Vec                 utmp, rtmp;
  double              hmass;    /* lumped mass h^2                    */
  int                 nt;       /* implicit Euler steps               */
  int                 n_obs;
  const double       *obs_x, *obs_y;      /* observation locations    */
  const int          *obs_i, *obs_j;      /* nearest grid node        */
  double              inv_sigma2;         /* 1/sigma_noise^2          */
  int                 hd_applies;
}
heat_ctx_t;

/* u <- L u : nt implicit Euler steps (rhs = M u_n) */
static PetscErrorCode
heat_L (heat_ctx_t *hc, Vec u)
{
  for (int t = 0; t < hc->nt; t++) {
    PetscCall (VecCopy (u, hc->utmp));
    PetscCall (VecScale (hc->utmp, hc->hmass));
    PetscCall (KSPSolve (hc->kheat, hc->utmp, u));
  }
  return PETSC_SUCCESS;
}

/* r[n_obs] = B u  (nearest-node samples; allreduced so every rank has r) */
static PetscErrorCode
heat_B (heat_ctx_t *hc, Vec u, double *r)
{
  const PetscScalar **ua;
  PetscInt            xs, ys, xm, ym;
  std::vector<double> rl ((size_t) hc->n_obs, 0.0);

  PetscCall (DMDAGetCorners (hc->da, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall (DMDAVecGetArrayRead (hc->da, u, &ua));
  for (int k = 0; k < hc->n_obs; k++) {
    const int           i = hc->obs_i[k], j = hc->obs_j[k];
    if (i >= xs && i < xs + xm && j >= ys && j < ys + ym)
      rl[(size_t) k] = ua[j][i];
  }
  PetscCall (DMDAVecRestoreArrayRead (hc->da, u, &ua));
  PetscCallMPI (MPI_Allreduce (rl.data (), r, hc->n_obs, MPI_DOUBLE,
                               MPI_SUM, PETSC_COMM_WORLD));
  return PETSC_SUCCESS;
}

/* u <- B^T r  (scatter-add at owned observation nodes) */
static PetscErrorCode
heat_Bt (heat_ctx_t *hc, const double *r, Vec u)
{
  PetscScalar       **ua;
  PetscInt            xs, ys, xm, ym;

  PetscCall (VecZeroEntries (u));
  PetscCall (DMDAGetCorners (hc->da, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall (DMDAVecGetArray (hc->da, u, &ua));
  for (int k = 0; k < hc->n_obs; k++) {
    const int           i = hc->obs_i[k], j = hc->obs_j[k];
    if (i >= xs && i < xs + xm && j >= ys && j < ys + ym)
      ua[j][i] += r[k];
  }
  PetscCall (DMDAVecRestoreArray (hc->da, u, &ua));
  return PETSC_SUCCESS;
}

/* rhs <- L^T B^T (r / sigma^2): the adjoint chain.  With scalar lumped
 * mass the Euclidean transpose of L equals L, so the same solve chain
 * serves forward and adjoint. */
static PetscErrorCode
heat_LtBt (heat_ctx_t *hc, const double *r, Vec out)
{
  std::vector<double> rs ((size_t) hc->n_obs);

  for (int k = 0; k < hc->n_obs; k++) rs[(size_t) k] = r[k] * hc->inv_sigma2;
  PetscCall (heat_Bt (hc, rs.data (), out));
  PetscCall (heat_L (hc, out));
  return PETSC_SUCCESS;
}

/* the misfit-Hessian matvec for lgh_fit: out = L^T B^T B L in / sigma^2 */
static void
hd_apply (const double *in_local, double *out_local, void *vctx)
{
  heat_ctx_t         *hc = (heat_ctx_t *) vctx;
  PetscScalar        *a;
  PetscInt            nl;
  Vec                 u = hc->rtmp;
  std::vector<double> r ((size_t) hc->n_obs);
  PetscErrorCode      ierr;

  ierr = VecGetLocalSize (u, &nl);
  CHKERRABORT (PETSC_COMM_WORLD, ierr);
  ierr = VecGetArray (u, &a);
  CHKERRABORT (PETSC_COMM_WORLD, ierr);
  memcpy (a, in_local, (size_t) nl * sizeof (double));
  ierr = VecRestoreArray (u, &a);
  CHKERRABORT (PETSC_COMM_WORLD, ierr);

  ierr = heat_L (hc, u);              CHKERRABORT (PETSC_COMM_WORLD, ierr);
  ierr = heat_B (hc, u, r.data ());   CHKERRABORT (PETSC_COMM_WORLD, ierr);
  ierr = heat_LtBt (hc, r.data (), u); CHKERRABORT (PETSC_COMM_WORLD, ierr);

  ierr = VecGetArray (u, &a);
  CHKERRABORT (PETSC_COMM_WORLD, ierr);
  memcpy (out_local, a, (size_t) nl * sizeof (double));
  ierr = VecRestoreArray (u, &a);
  CHKERRABORT (PETSC_COMM_WORLD, ierr);
  hc->hd_applies++;
}

/* ---------------------------------------------------------------- */

/* read a binary P5 PGM (every rank; local filesystem) */
static int
read_pgm (const char *fn, std::vector<unsigned char> &px, int *w, int *h)
{
  FILE               *f = fopen (fn, "rb");
  int                 maxv;

  if (f == NULL) return 1;
  if (fscanf (f, "P5 %d %d %d", w, h, &maxv) != 3 || maxv != 255) {
    fclose (f);
    return 2;
  }
  fgetc (f);   /* single whitespace after header */
  px.resize ((size_t) (*w) * (size_t) (*h));
  if (fread (px.data (), 1, px.size (), f) != px.size ()) {
    fclose (f);
    return 3;
  }
  fclose (f);
  return 0;
}

/* gather a DMDA field to rank 0 and write it as an 8-bit PGM, mapping
 * [lo, hi] -> [0, 255] (fixed range => images are comparable) */
static PetscErrorCode
write_pgm (DM da, Vec v, const char *fn, double lo, double hi)
{
  DM                  da0;
  Vec                 nat, v0;
  VecScatter          sc;
  PetscInt            nx, ny;
  PetscMPIInt         rank;

  PetscCall (DMDAGetInfo (da, NULL, &nx, &ny, NULL, NULL, NULL, NULL, NULL,
                          NULL, NULL, NULL, NULL, NULL));
  MPI_Comm_rank (PETSC_COMM_WORLD, &rank);
  /* PETSc order -> natural order -> rank 0 */
  PetscCall (DMDACreateNaturalVector (da, &nat));
  PetscCall (DMDAGlobalToNaturalBegin (da, v, INSERT_VALUES, nat));
  PetscCall (DMDAGlobalToNaturalEnd (da, v, INSERT_VALUES, nat));
  PetscCall (VecScatterCreateToZero (nat, &sc, &v0));
  PetscCall (VecScatterBegin (sc, nat, v0, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall (VecScatterEnd (sc, nat, v0, INSERT_VALUES, SCATTER_FORWARD));
  if (rank == 0) {
    const PetscScalar  *a;
    FILE               *f = fopen (fn, "wb");

    PetscCheck (f != NULL, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
                "cannot open %s", fn);
    fprintf (f, "P5\n%d %d\n255\n", (int) nx, (int) ny);
    PetscCall (VecGetArrayRead (v0, &a));
    /* natural order is row-major bottom-up; images are top-down */
    for (PetscInt j = ny - 1; j >= 0; j--)
      for (PetscInt i = 0; i < nx; i++) {
        double              t = (a[j * nx + i] - lo) / (hi - lo);
        int                 g = (int) lround (255.0 * (t < 0 ? 0 : (t > 1 ? 1 : t)));
        fputc (g, f);
      }
    PetscCall (VecRestoreArrayRead (v0, &a));
    fclose (f);
    (void) da0;
  }
  PetscCall (VecScatterDestroy (&sc));
  PetscCall (VecDestroy (&v0));
  PetscCall (VecDestroy (&nat));
  return PETSC_SUCCESS;
}

/* ---------------------------------------------------------------- */

int
main (int argc, char **argv)
{
  PetscCall (PetscInitialize (&argc, &argv, NULL, NULL));
  {
  PetscMPIInt         rank;
  MPI_Comm_rank (PETSC_COMM_WORLD, &rank);

  /* ---- options -------------------------------------------------- */
  PetscInt            height = 128, nobs = 2000, kmax = 60, nqc = 5;
  PetscInt            nt = 10, nsamples = 4, nstd = 100, ell_glr = 2200;
  PetscInt            threads = 4;
  PetscReal           noise_rel = 0.05, prior_len = 0.25, prior_b = -1.0;
  PetscReal           trunc_abs = 1e-2, kappa = 1.0, obs_cx = -1., obs_sd = 0.18;
  PetscInt            seed = 20260817;
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-height", &height, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-nobs", &nobs, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-kmax", &kmax, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-nqc", &nqc, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-nt", &nt, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-nsamples", &nsamples, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-nstd", &nstd, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-ell", &ell_glr, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-threads", &threads, NULL));
  PetscCall (PetscOptionsGetInt (NULL, NULL, "-seed", &seed, NULL));
  PetscCall (PetscOptionsGetReal (NULL, NULL, "-noise_rel", &noise_rel, NULL));
  PetscCall (PetscOptionsGetReal (NULL, NULL, "-prior_len", &prior_len, NULL));
  PetscCall (PetscOptionsGetReal (NULL, NULL, "-prior_b", &prior_b, NULL));
  PetscCall (PetscOptionsGetReal (NULL, NULL, "-trunc_abs", &trunc_abs, NULL));
  PetscCall (PetscOptionsGetReal (NULL, NULL, "-obs_sd", &obs_sd, NULL));

  /* ---- the source image ----------------------------------------- */
  char                srcfn[256];
  std::vector<unsigned char> px;
  int                 iw = 0, ih = 0;
  snprintf (srcfn, sizeof srcfn, "source_%d.pgm", (int) height);
  PetscCheck (read_pgm (srcfn, px, &iw, &ih) == 0, PETSC_COMM_WORLD,
              PETSC_ERR_FILE_OPEN,
              "cannot read %s (run prepare_image.py --height %d)", srcfn,
              (int) height);
  const PetscInt      nx = iw, ny = ih;
  const double        h = 1.0 / (double) (ny - 1);
  const double        Lx = (double) (nx - 1) * h;
  if (obs_cx < 0.) obs_cx = 0.72 * Lx;

  /* T from the target PSF scale.  sigma_heat = sqrt(2 kappa T) is the
   * impulse-response width; 1.5 grid cells keeps the fitted operator's
   * windows at a tractable ~900 columns per row (the window footprint in
   * PIXELS is resolution-independent, so this choice fixes the per-row
   * fit cost at every -height) while the PSF still spans several
   * observation spacings on the dense side.  The a-priori ellipsoid is
   * the 2-sigma width of L delta: ell = 2 sqrt(2 kappa T); the support
   * margin is lgpsf's window (tau_window). */
  const double        sigma_heat = 1.5 * h;
  const double        ell_lg = 2.0 * sqrt (2.0) * sigma_heat;
  const double        T = sigma_heat * sigma_heat / (2.0 * kappa);
  const double        dt = T / (double) nt;

  /* ---- grid, mass, stiffness, heat operator ---------------------- */
  DM                  da;
  PetscCall (DMDACreate2d (PETSC_COMM_WORLD, DM_BOUNDARY_NONE,
                           DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, nx, ny,
                           PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL,
                           &da));
  PetscCall (DMSetUp (da));
  const double        hmass = h * h;

  Mat                 K;
  PetscCall (DMCreateMatrix (da, &K));
  {
    PetscInt            xs, ys, xm, ym;
    PetscCall (DMDAGetCorners (da, &xs, &ys, NULL, &xm, &ym, NULL));
    for (PetscInt j = ys; j < ys + ym; j++)
      for (PetscInt i = xs; i < xs + xm; i++) {
        MatStencil          row = {0, j, i, 0}, col[5];
        PetscScalar         v[5];
        PetscInt            nc = 0;
        PetscScalar         diag = 0.;

        if (i > 0)      { col[nc].k=0; col[nc].j=j;   col[nc].i=i-1; col[nc].c=0; v[nc++] = -1.0; diag += 1.0; }
        if (i < nx - 1) { col[nc].k=0; col[nc].j=j;   col[nc].i=i+1; col[nc].c=0; v[nc++] = -1.0; diag += 1.0; }
        if (j > 0)      { col[nc].k=0; col[nc].j=j-1; col[nc].i=i;   col[nc].c=0; v[nc++] = -1.0; diag += 1.0; }
        if (j < ny - 1) { col[nc].k=0; col[nc].j=j+1; col[nc].i=i;   col[nc].c=0; v[nc++] = -1.0; diag += 1.0; }
        col[nc].k=0; col[nc].j=j; col[nc].i=i; col[nc].c=0; v[nc++] = diag;
        PetscCall (MatSetValuesStencil (K, 1, &row, nc, col, v,
                                        INSERT_VALUES));
      }
    PetscCall (MatAssemblyBegin (K, MAT_FINAL_ASSEMBLY));
    PetscCall (MatAssemblyEnd (K, MAT_FINAL_ASSEMBLY));
  }

  heat_ctx_t          hc;
  memset (&hc, 0, sizeof hc);
  hc.da = da;
  hc.hmass = hmass;
  hc.nt = (int) nt;
  PetscCall (MatDuplicate (K, MAT_COPY_VALUES, &hc.Aheat));
  PetscCall (MatScale (hc.Aheat, dt * kappa));
  PetscCall (MatShift (hc.Aheat, hmass));
  PetscCall (KSPCreate (PETSC_COMM_WORLD, &hc.kheat));
  PetscCall (KSPSetType (hc.kheat, KSPCG));
  {
    PC                  pc;
    PetscCall (KSPGetPC (hc.kheat, &pc));
    PetscCall (PCSetType (pc, PCGAMG));
  }
  PetscCall (KSPSetOperators (hc.kheat, hc.Aheat, hc.Aheat));
  PetscCall (KSPSetTolerances (hc.kheat, 1e-12, 0., PETSC_DEFAULT, 1000));
  PetscCall (KSPSetOptionsPrefix (hc.kheat, "heat_"));
  PetscCall (KSPSetFromOptions (hc.kheat));
  PetscCall (DMCreateGlobalVector (da, &hc.utmp));
  PetscCall (DMCreateGlobalVector (da, &hc.rtmp));

  /* ---- observation locations (deterministic, replicated draw) ---- */
  std::vector<double> ox ((size_t) nobs), oy ((size_t) nobs);
  std::vector<int>    oi ((size_t) nobs), oj ((size_t) nobs);
  {
    long                d = 0;
    for (int k = 0; k < nobs; k++) {
      double              x, y;
      do {
        x = obs_cx + obs_sd * lgh::hashed_normal ((unsigned long) seed,
                                                  9000000L + d, 0);
        y = 0.5 + obs_sd * lgh::hashed_normal ((unsigned long) seed,
                                               9000000L + d, 1);
        d++;
      } while (x < 0. || x > Lx || y < 0. || y > 1.);
      ox[(size_t) k] = x;
      oy[(size_t) k] = y;
      oi[(size_t) k] = (int) lround (x / h);
      oj[(size_t) k] = (int) lround (y / h);
    }
  }
  hc.n_obs = (int) nobs;
  hc.obs_x = ox.data (); hc.obs_y = oy.data ();
  hc.obs_i = oi.data (); hc.obs_j = oj.data ();

  /* ---- truth, data ------------------------------------------------ */
  Vec                 s_true, work;
  PetscCall (DMCreateGlobalVector (da, &s_true));
  PetscCall (DMCreateGlobalVector (da, &work));
  {
    PetscScalar       **sa;
    PetscInt            xs, ys, xm, ym;
    PetscCall (DMDAGetCorners (da, &xs, &ys, NULL, &xm, &ym, NULL));
    PetscCall (DMDAVecGetArray (da, s_true, &sa));
    for (PetscInt j = ys; j < ys + ym; j++)
      for (PetscInt i = xs; i < xs + xm; i++) {
        /* image row 0 is the TOP; grid j = 0 is the BOTTOM */
        const unsigned char g =
            px[(size_t) (ny - 1 - j) * (size_t) nx + (size_t) i];
        sa[j][i] = (double) g / 255.0;
      }
    PetscCall (DMDAVecRestoreArray (da, s_true, &sa));
  }

  std::vector<double> y_clean ((size_t) nobs), y ((size_t) nobs);
  PetscCall (VecCopy (s_true, work));
  PetscCall (heat_L (&hc, work));
  PetscCall (heat_B (&hc, work, y_clean.data ()));
  double              ymean = 0., yvar = 0., sigma_noise;
  for (int k = 0; k < nobs; k++) ymean += y_clean[(size_t) k];
  ymean /= (double) nobs;
  for (int k = 0; k < nobs; k++) {
    const double        dv = y_clean[(size_t) k] - ymean;
    yvar += dv * dv;
  }
  sigma_noise = noise_rel * sqrt (yvar / (double) nobs);
  for (int k = 0; k < nobs; k++)
    y[(size_t) k] = y_clean[(size_t) k]
      + sigma_noise * lgh::hashed_normal ((unsigned long) seed + 7,
                                          8000000L + k, 2);
  hc.inv_sigma2 = 1.0 / (sigma_noise * sigma_noise);
  PetscCall (PetscPrintf (PETSC_COMM_WORLD,
      "grid %d x %d (N = %d, h = %.4g), T = %.3e (sigma_heat = %.4g = "
      "%.1f px), n_obs = %d, sigma_noise = %.4g\n",
      (int) nx, (int) ny, (int) (nx * ny), h, T, sigma_heat,
      sigma_heat / h, (int) nobs, sigma_noise));

  /* ---- stage 1: the fit ------------------------------------------ */
  PetscInt            nloc;
  PetscInt            rstart;
  Vec                 vg;
  PetscCall (DMCreateGlobalVector (da, &vg));
  PetscCall (VecGetLocalSize (vg, &nloc));
  PetscCall (VecGetOwnershipRange (vg, &rstart, NULL));
  std::vector<double> coords ((size_t) nloc * 2), mass ((size_t) nloc);
  {
    PetscInt            xs, ys, xm, ym, l = 0;
    PetscCall (DMDAGetCorners (da, &xs, &ys, NULL, &xm, &ym, NULL));
    for (PetscInt j = ys; j < ys + ym; j++)
      for (PetscInt i = xs; i < xs + xm; i++, l++) {
        coords[(size_t) l * 2 + 0] = (double) i * h;
        coords[(size_t) l * 2 + 1] = (double) j * h;
        mass[(size_t) l] = hmass;
      }
  }
  lgh_fit_t          *fit = lgh_fit_create (PETSC_COMM_WORLD, 2, (int) nloc,
                                            (long) rstart, coords.data (),
                                            mass.data (), (int) threads);
  std::vector<double> ellf ((size_t) nloc, ell_lg);
  std::vector<double> sigma_lg ((size_t) nloc * 4);
  lgh_sigma_isotropic (2, (int) nloc, ellf.data (), sigma_lg.data ());

  /* probe pairs once (kmax of them), then fit at increasing k for the
   * QC-vs-k curve; the final fit (k = kmax) is THE operator */
  std::vector<double> V ((size_t) nloc * (size_t) kmax);
  std::vector<double> HV ((size_t) nloc * (size_t) kmax);
  {
    std::vector<double> zin ((size_t) nloc), zout ((size_t) nloc);
    for (PetscInt q = 0; q < kmax; q++) {
      for (PetscInt l = 0; l < nloc; l++)
        zin[(size_t) l] = lgh::hashed_normal ((unsigned long) seed,
                                              rstart + l, (long) q)
          / sqrt (mass[(size_t) l]);
      hd_apply (zin.data (), zout.data (), &hc);
      memcpy (V.data () + (size_t) q * (size_t) nloc, zin.data (),
              (size_t) nloc * sizeof (double));
      memcpy (HV.data () + (size_t) q * (size_t) nloc, zout.data (),
              (size_t) nloc * sizeof (double));
    }
  }
  PetscCall (PetscPrintf (PETSC_COMM_WORLD,
      "probes drawn: %d pairs (%d Hessian applies)\n", (int) kmax,
      hc.hd_applies));

  if (rank == 0) { mkdir ("out", 0775); }
  FILE               *fqc = NULL;
  if (rank == 0) {
    fqc = fopen ("out/qc_vs_k.txt", "w");
    fprintf (fqc, "# k  qc_energy  qc_rowmean\n");
  }
  lgh_fit_opts_t      fo = lgh_fit_opts_default ();
  fo.n_qc = (int) nqc;
  fo.verbose = 0;
  /* example tuning: the PSF here is an exact Gaussian, so a slightly
   * tighter window/assembly than the production defaults (5/6) loses
   * nothing and keeps the fit at minutes */
  fo.tau_window = 4.0;
  fo.tau_assemble = 4.0;
  lgh_fit_report_t    frep;
  for (PetscInt k = 20; k <= kmax; k += 20) {
    int                 rc = lgh_fit_probes (fit, (int) k, V.data (),
                                             HV.data (), sigma_lg.data (),
                                             &fo, &frep);
    PetscCheck (rc == 0, PETSC_COMM_WORLD, PETSC_ERR_LIB,
                "lgh_fit_probes rc = %d at k = %d", rc, (int) k);
    PetscCall (PetscPrintf (PETSC_COMM_WORLD,
        "  fit k = %2d: qcE = %.4f  qcM = %.4f  nnz = %ld  (%d fit / %d "
        "fallback rows)\n", (int) k, frep.qc_energy, frep.qc_rowmean,
        frep.nnz_global, frep.rows_fit, frep.rows_fallback));
    if (rank == 0)
      fprintf (fqc, "%d %.6e %.6e\n", (int) k, frep.qc_energy,
               frep.qc_rowmean);
  }
  if (rank == 0) fclose (fqc);

  Mat                 Hd_lg;
  PetscCall (lgh_fit_get_mat (fit, &Hd_lg));

  /* ---- stage 2: prior + GLR --------------------------------------- */
  /* Z = beta (len^2 K + M): correlation length len, overall strength
   * beta CALIBRATED so the pointwise prior std hits -prior_std (prior
   * draws are Z^{-1} M^{1/2} xi; a few probes estimate the std of the
   * unscaled draw, which scales as 1/beta).  -prior_b > 0 overrides. */
  Mat                 Z;
  Vec                 massv;
  PetscCall (MatDuplicate (K, MAT_COPY_VALUES, &Z));
  PetscCall (MatScale (Z, prior_len * prior_len));   /* len^2 K */
  PetscCall (MatShift (Z, hmass));                   /* + M      */
  PetscCall (DMCreateGlobalVector (da, &massv));
  PetscCall (VecSet (massv, hmass));
  {
    PetscReal           prior_std = 0.5;
    PetscCall (PetscOptionsGetReal (NULL, NULL, "-prior_std", &prior_std,
                                    NULL));
    if (prior_b <= 0.) {
      KSP                 kz;
      PC                  pcz;
      Vec                 zx, zb2, acc;
      double              s0;

      PetscCall (KSPCreate (PETSC_COMM_WORLD, &kz));
      PetscCall (KSPSetType (kz, KSPCG));
      PetscCall (KSPGetPC (kz, &pcz));
      PetscCall (PCSetType (pcz, PCGAMG));
      PetscCall (KSPSetOperators (kz, Z, Z));
      PetscCall (KSPSetTolerances (kz, 1e-10, 0., PETSC_DEFAULT, 1000));
      PetscCall (DMCreateGlobalVector (da, &zx));
      PetscCall (DMCreateGlobalVector (da, &zb2));
      PetscCall (DMCreateGlobalVector (da, &acc));
      PetscCall (VecZeroEntries (acc));
      for (int p = 0; p < 3; p++) {
        PetscScalar        *ba;
        PetscInt            lo3, hi3;
        PetscCall (VecGetOwnershipRange (zb2, &lo3, &hi3));
        PetscCall (VecGetArray (zb2, &ba));
        for (PetscInt l = lo3; l < hi3; l++)
          ba[l - lo3] = sqrt (hmass)
            * lgh::hashed_normal ((unsigned long) seed + 29, l, p);
        PetscCall (VecRestoreArray (zb2, &ba));
        PetscCall (VecZeroEntries (zx));
        PetscCall (KSPSolve (kz, zb2, zx));
        PetscCall (VecPointwiseMult (zx, zx, zx));
        PetscCall (VecAXPY (acc, 1.0 / 3.0, zx));
      }
      {
        PetscReal           mean;
        PetscInt            Ntot;
        PetscCall (VecGetSize (acc, &Ntot));
        PetscCall (VecSum (acc, &mean));
        s0 = sqrt ((double) mean / (double) Ntot);
      }
      prior_b = s0 / prior_std;
      PetscCall (PetscPrintf (PETSC_COMM_WORLD,
          "prior calibration: unscaled pointwise std %.4g -> beta = %.4g "
          "(target std %.3g)\n", s0, prior_b, (double) prior_std));
      PetscCall (KSPDestroy (&kz));
      PetscCall (VecDestroy (&zx));
      PetscCall (VecDestroy (&zb2));
      PetscCall (VecDestroy (&acc));
    }
  }
  PetscCall (MatScale (Z, prior_b));

  lgh_prior_mat_opts_t po = lgh_prior_mat_opts_default ();
  po.blocked_mode = 2;
  po.tile = 64;
  po.verbose = 1;
  lgh_prior_t        *prior;
  PetscCall ((PetscErrorCode) lgh_prior_create_mat (Z, massv, &po, &prior));

  lgh_glr_opts_t      go = lgh_glr_opts_default ();
  go.ell = (PetscInt) PetscMin ((PetscInt) ell_glr, nx * ny - 1);
  go.trunc_abs = trunc_abs;
  go.trunc_rel = 0.;
  go.seed = (unsigned long) seed ^ 0x5A17ED5EEDUL;
  go.backend = LGH_GLR_REPLICATED;   /* -ell + LGH_GLR_SCALAPACK builds
                                        scale this up */
  lgh_glr_t          *glr;
  lgh_glr_report_t    grep;
  PetscCall ((PetscErrorCode) lgh_glr_compute (Hd_lg, prior, &go, &glr,
                                               &grep));
  PetscCall (PetscPrintf (PETSC_COMM_WORLD,
      "GLR: kept %d / ell %d, |lam|max %.3e, next %.3e, negatives %d "
      "(t_op %.1fs, t_dense %.1fs)\n", grep.kept, (int) go.ell,
      grep.lam_abs_max, grep.next_abs, grep.n_negative_raw,
      grep.t_operator, grep.t_dense));
  PetscCall (PetscPrintf (PETSC_COMM_WORLD,
      "logdet piece sum log(1 + lam) = %.6e\n", lgh_glr_logdet (glr, 1.0)));
  if (rank == 0) {
    int                 nk;
    const double       *lam;
    FILE               *fe = fopen ("out/eigs.txt", "w");

    lgh_glr_eigs (glr, &nk, &lam);
    fprintf (fe, "# i  lambda_i (treated, |.|-descending)\n");
    for (int i = 0; i < nk; i++) fprintf (fe, "%d %.10e\n", i, lam[i]);
    fclose (fe);
  }

  /* ---- stage 3: reconstruction + posterior ------------------------ */
  Vec                 rhs, srec, xi, samp, smean, svar;
  PetscCall (DMCreateGlobalVector (da, &rhs));
  PetscCall (DMCreateGlobalVector (da, &srec));
  PetscCall (heat_LtBt (&hc, y.data (), rhs));
  PetscCall ((PetscErrorCode) lgh_glr_solve (glr, 1.0, rhs, srec));
  {
    PetscReal           smin, smax2;
    PetscCall (VecMin (srec, NULL, &smin));
    PetscCall (VecMax (srec, NULL, &smax2));
    PetscCall (PetscPrintf (PETSC_COMM_WORLD,
        "reconstruction range: [%.3f, %.3f] (truth is [0, 1])\n",
        (double) smin, (double) smax2));
  }

  /* report the data misfit of the reconstruction */
  {
    std::vector<double> yr ((size_t) nobs);
    double              m2 = 0.;
    PetscCall (VecCopy (srec, work));
    PetscCall (heat_L (&hc, work));
    PetscCall (heat_B (&hc, work, yr.data ()));
    for (int k = 0; k < nobs; k++) {
      const double        dv = yr[(size_t) k] - y[(size_t) k];
      m2 += dv * dv;
    }
    PetscCall (PetscPrintf (PETSC_COMM_WORLD,
        "reconstruction: rms data misfit %.4g (sigma_noise %.4g; Morozov "
        "ratio %.3f)\n", sqrt (m2 / (double) nobs), sigma_noise,
        sqrt (m2 / (double) nobs) / sigma_noise));
    if (rank == 0) {
      FILE               *fm = fopen ("out/data_misfit.txt", "w");
      fprintf (fm, "# rms_misfit sigma_noise\n%.6e %.6e\n",
               sqrt (m2 / (double) nobs), sigma_noise);
      fclose (fm);
    }
  }

  PetscCall (DMCreateGlobalVector (da, &xi));
  PetscCall (DMCreateGlobalVector (da, &samp));
  PetscCall (DMCreateGlobalVector (da, &smean));
  PetscCall (DMCreateGlobalVector (da, &svar));
  PetscCall (VecZeroEntries (smean));
  PetscCall (VecZeroEntries (svar));

  const double        img_lo = 0.0, img_hi = 1.0;
  PetscCall (write_pgm (da, s_true, "out/truth.pgm", img_lo, img_hi));
  PetscCall (write_pgm (da, srec, "out/recon.pgm", img_lo, img_hi));

  for (PetscInt m = 0; m < nstd; m++) {
    PetscScalar        *xa;
    PetscInt            lo2, hi2;
    PetscCall (VecGetOwnershipRange (xi, &lo2, &hi2));
    PetscCall (VecGetArray (xi, &xa));
    for (PetscInt l = lo2; l < hi2; l++)
      xa[l - lo2] = lgh::hashed_normal ((unsigned long) seed + 13, l,
                                        100 + (long) m);
    PetscCall (VecRestoreArray (xi, &xa));
    PetscCall ((PetscErrorCode) lgh_glr_sample (glr, 1.0, xi, samp));
    PetscCall (VecAXPY (samp, 1.0, srec));   /* s* + G^{-T} xi */
    if (m < nsamples) {
      char                fn[128];
      snprintf (fn, sizeof fn, "out/sample_%d.pgm", (int) m);
      PetscCall (write_pgm (da, samp, fn, img_lo, img_hi));
    }
    PetscCall (VecAXPY (smean, 1.0, samp));
    PetscCall (VecPointwiseMult (samp, samp, samp));
    PetscCall (VecAXPY (svar, 1.0, samp));
  }
  /* pointwise std: sqrt(E[s^2] - E[s]^2) */
  PetscCall (VecScale (smean, 1.0 / (double) nstd));
  PetscCall (VecScale (svar, 1.0 / (double) nstd));
  PetscCall (VecPointwiseMult (smean, smean, smean));
  PetscCall (VecAXPY (svar, -1.0, smean));
  PetscCall (VecSqrtAbs (svar));
  {
    PetscReal           smax;
    PetscCall (VecMax (svar, NULL, &smax));
    PetscCall (write_pgm (da, svar, "out/std.pgm", 0.0, (double) smax));
    PetscCall (PetscPrintf (PETSC_COMM_WORLD,
        "posterior pointwise std: max %.4g (out/std.pgm scaled to it)\n",
        (double) smax));
  }

  if (rank == 0) {   /* observation locations for the truth overlay */
    FILE               *fo2 = fopen ("out/obs_points.txt", "w");
    fprintf (fo2, "# x y  (domain: [0, %.6f] x [0, 1])\n", Lx);
    for (int k = 0; k < nobs; k++)
      fprintf (fo2, "%.6f %.6f\n", ox[(size_t) k], oy[(size_t) k]);
    fclose (fo2);
  }
  PetscCall (PetscPrintf (PETSC_COMM_WORLD,
      "done: images + data in out/ (render with plot.py); total Hessian "
      "applies %d\n", hc.hd_applies));

  /* ---- cleanup ---------------------------------------------------- */
  lgh_glr_destroy (glr);
  lgh_prior_destroy (prior);
  lgh_fit_destroy (fit);
  PetscCall (VecDestroy (&rhs));
  PetscCall (VecDestroy (&srec));
  PetscCall (VecDestroy (&xi));
  PetscCall (VecDestroy (&samp));
  PetscCall (VecDestroy (&smean));
  PetscCall (VecDestroy (&svar));
  PetscCall (VecDestroy (&s_true));
  PetscCall (VecDestroy (&work));
  PetscCall (VecDestroy (&vg));
  PetscCall (VecDestroy (&massv));
  PetscCall (MatDestroy (&Z));
  PetscCall (MatDestroy (&K));
  PetscCall (MatDestroy (&hc.Aheat));
  PetscCall (KSPDestroy (&hc.kheat));
  PetscCall (VecDestroy (&hc.utmp));
  PetscCall (VecDestroy (&hc.rtmp));
  PetscCall (DMDestroy (&da));
  }
  PetscCall (PetscFinalize ());
  return 0;
}

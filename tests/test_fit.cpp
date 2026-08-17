/* test_fit.cpp — gates for the distributed LG-PSF fit stage (lgh_fit_*),
 * runnable at any communicator size (registered at n=1,2,4).
 *
 * Operator: a Gaussian smoothing kernel plus a diagonal spike —
 * H_ij = mass_j exp(-|x_i - x_j|^2 / (2 h^2)) + a delta_ij — i.e. exactly
 * an operator with local point-spread structure, in d=2 and d=3.
 * The matvec callback allgathers the probe and computes owned rows from
 * the closed form (deterministic and partition-independent, so hashed
 * probes give partition-independent fits).
 *
 * Gates: ladder converges to qc_target; every row accounted for
 * (fit + fallback); B is symmetric (MatMult vs MatMultTranspose through
 * lgh_fit_get_mat); independent accuracy check on a fresh vector in the
 * dual norm; bitwise determinism across re-runs; lgh_fit_probes
 * (precomputed pairs) works; REFERENCE wsym convention at n=1;
 * lgh_fit_smooth_field preserves constants.
 */

#include <lgpsf_hessian/lgpsf_hessian.h>
#include <lgpsf_hessian/impl.hpp>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int          n_fail = 0;

static void
check (int ok, const char *what, double val)
{
  if (!ok) {
    n_fail++;
    PetscPrintf (PETSC_COMM_WORLD, "FAIL: %s (%.3e)\n", what, val);
  }
}

/* ---- the synthetic PSF operator -------------------------------------- */

typedef struct
{
  MPI_Comm            comm;
  int                 dim, nloc, nglob;
  long                gid0;
  double              h, spike;
  double             *coords;      /* local, interleaved */
  double             *mass;        /* local              */
  double             *all_coords;  /* global, gathered   */
  double             *all_mass;
  double             *gath;        /* nglob scratch      */
  int                *counts, *displs;
}
psf_op_t;

static void
psf_apply (const double *in_local, double *out_local, void *vctx)
{
  psf_op_t           *op = (psf_op_t *) vctx;

  MPI_Allgatherv ((void *) in_local, op->nloc, MPI_DOUBLE, op->gath,
                  op->counts, op->displs, MPI_DOUBLE, op->comm);
  for (int i = 0; i < op->nloc; i++)
  {
    const double       *xi = op->coords + (size_t) i * op->dim;
    double              s = op->spike * in_local[i];

    for (int j = 0; j < op->nglob; j++)
    {
      const double       *xj = op->all_coords + (size_t) j * op->dim;
      double              d2 = 0.;
      for (int a = 0; a < op->dim; a++)
      {
        const double        d = xi[a] - xj[a];
        d2 += d * d;
      }
      s += op->all_mass[j] * exp (-0.5 * d2 / (op->h * op->h))
        * op->gath[j];
    }
    out_local[i] = s;
  }
}

static double
mass_of (long g) { return 0.7 + 0.1 * (g % 5); }

/* grid coords for global node g */
static void
coords_of (int dim, int m, long g, double *x)
{
  if (dim == 2)
  {
    x[0] = (double) (g % m);
    x[1] = (double) (g / m);
  }
  else
  {
    x[0] = (double) (g % m);
    x[1] = (double) ((g / m) % m);
    x[2] = (double) (g / (m * m));
  }
}

static PetscErrorCode
setup_op (MPI_Comm comm, int dim, int m, double h, double spike,
          psf_op_t *op, long *gid0_out, int *nloc_out)
{
  int                 rank, size;
  const int           nglob = (dim == 2) ? m * m : m * m * m;

  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);
  const int           base = nglob / size, extra = nglob % size;
  const int           nloc = base + (rank < extra ? 1 : 0);
  const long          gid0 = (long) rank * base
    + (rank < extra ? rank : extra);

  op->comm = comm; op->dim = dim; op->nloc = nloc; op->nglob = nglob;
  op->gid0 = gid0; op->h = h; op->spike = spike;
  op->coords = (double *) malloc (sizeof (double) * (size_t) nloc * dim);
  op->mass = (double *) malloc (sizeof (double) * (size_t) nloc);
  op->all_coords = (double *) malloc (sizeof (double) * (size_t) nglob * dim);
  op->all_mass = (double *) malloc (sizeof (double) * (size_t) nglob);
  op->gath = (double *) malloc (sizeof (double) * (size_t) nglob);
  op->counts = (int *) malloc (sizeof (int) * (size_t) size);
  op->displs = (int *) malloc (sizeof (int) * (size_t) size);
  for (int r = 0; r < size; r++)
  {
    op->counts[r] = base + (r < extra ? 1 : 0);
    op->displs[r] = r * base + (r < extra ? r : extra);
  }
  for (int i = 0; i < nloc; i++)
  {
    coords_of (dim, m, gid0 + i, op->coords + (size_t) i * dim);
    op->mass[i] = mass_of (gid0 + i);
  }
  for (long g = 0; g < nglob; g++)
  {
    coords_of (dim, m, g, op->all_coords + (size_t) g * dim);
    op->all_mass[g] = mass_of (g);
  }
  *gid0_out = gid0;
  *nloc_out = nloc;
  return PETSC_SUCCESS;
}

static void
free_op (psf_op_t *op)
{
  free (op->coords); free (op->mass); free (op->all_coords);
  free (op->all_mass); free (op->gath); free (op->counts); free (op->displs);
}

/* ---- one scenario ---------------------------------------------------- */

static PetscErrorCode
run_scenario (int dim, int m, double qc_target)
{
  psf_op_t            op;
  long                gid0;
  int                 nloc;
  lgh_fit_t          *fit;
  lgh_fit_opts_t      fo = lgh_fit_opts_default ();
  lgh_fit_report_t    rep, rep2;
  double             *sigma, *ell;
  const double        h = 1.4;
  const int           nglob = (dim == 2) ? m * m : m * m * m;

  PetscCall (setup_op (PETSC_COMM_WORLD, dim, m, h, 0.8, &op, &gid0,
                       &nloc));
  fit = lgh_fit_create (PETSC_COMM_WORLD, dim, nloc, gid0, op.coords,
                        op.mass, 2);
  ell = (double *) malloc (sizeof (double) * (size_t) nloc);
  sigma = (double *) malloc (sizeof (double) * (size_t) nloc * dim * dim);
  for (int i = 0; i < nloc; i++) ell[i] = 1.5 * h;
  lgh_sigma_isotropic (dim, nloc, ell, sigma);

  fo.k0 = 15; fo.n_qc = 5; fo.k_max = 30;
  fo.qc_target = qc_target;
  fo.verbose = 0;
  if (dim == 3)
  {
    /* mode-set sizes grow with dimension: the probe-counting rule needs a
     * bigger pool per candidate — shrink the wedge, raise the pool */
    fo.wedge_order = 2;
    fo.k0 = 20;
  }

  {
    int                 rc = lgh_fit_hessian (fit, psf_apply, &op, sigma,
                                              &fo, &rep);
    check (rc == 0, "fit_hessian returns 0", (double) rc);
  }
  check (rep.qc_energy <= qc_target, "ladder reached qc_target",
         rep.qc_energy);
  check (rep.rows_fit + rep.rows_fallback == nglob,
         "every row accounted for",
         (double) (rep.rows_fit + rep.rows_fallback));
  check (rep.rows_fit > nglob / 2, "majority of rows searched-fit",
         (double) rep.rows_fit);

  /* symmetry + independent accuracy through the PETSc seam */
  {
    Mat                 B;
    Vec                 v, bv, btv, hv;
    PetscInt            rstart, rend;
    PetscReal           nd, nb;
    double              num = 0., den = 0., loc[2], glob[2];
    double             *vin = (double *) malloc (sizeof (double)
                                                 * (size_t) nloc);
    double             *hout = (double *) malloc (sizeof (double)
                                                  * (size_t) nloc);

    PetscCall ((PetscErrorCode) lgh_fit_get_mat (fit, &B));
    PetscCall (MatCreateVecs (B, &v, NULL));
    PetscCall (VecDuplicate (v, &bv));
    PetscCall (VecDuplicate (v, &btv));
    PetscCall (VecDuplicate (v, &hv));
    PetscCall (VecGetOwnershipRange (v, &rstart, &rend));
    {
      PetscScalar        *va;
      PetscCall (VecGetArray (v, &va));
      for (PetscInt i = rstart; i < rend; i++)
      {
        va[i - rstart] = cos (0.17 * (double) i + 0.5)
          / sqrt (mass_of ((long) i));
        vin[i - rstart] = va[i - rstart];
      }
      PetscCall (VecRestoreArray (v, &va));
    }
    PetscCall (MatMult (B, v, bv));
    PetscCall (MatMultTranspose (B, v, btv));
    PetscCall (VecAXPY (btv, -1.0, bv));
    PetscCall (VecNorm (btv, NORM_2, &nd));
    PetscCall (VecNorm (bv, NORM_2, &nb));
    check ((double) (nd / nb) < 1e-12, "B symmetric", (double) (nd / nb));

    psf_apply (vin, hout, &op);
    {
      const PetscScalar  *ba;
      PetscCall (VecGetArrayRead (bv, &ba));
      for (int i = 0; i < nloc; i++)
      {
        const double        r = ba[i] - hout[i];
        num += r * r / op.mass[i];
        den += hout[i] * hout[i] / op.mass[i];
      }
      PetscCall (VecRestoreArrayRead (bv, &ba));
    }
    loc[0] = num; loc[1] = den;
    PetscCallMPI (MPI_Allreduce (loc, glob, 2, MPI_DOUBLE, MPI_SUM,
                                 PETSC_COMM_WORLD));
    check (sqrt (glob[0] / glob[1]) < 2.0 * qc_target,
           "independent accuracy on a fresh vector",
           sqrt (glob[0] / glob[1]));
    free (vin); free (hout);
    PetscCall (VecDestroy (&v));
    PetscCall (VecDestroy (&bv));
    PetscCall (VecDestroy (&btv));
    PetscCall (VecDestroy (&hv));
  }

  /* bitwise determinism across re-runs (same inputs, same seed) */
  {
    int                 rc = lgh_fit_hessian (fit, psf_apply, &op, sigma,
                                              &fo, &rep2);
    check (rc == 0, "rerun returns 0", (double) rc);
    check (rep2.qc_energy == rep.qc_energy, "rerun qc bitwise identical",
           rep2.qc_energy - rep.qc_energy);
    check (rep2.nnz_global == rep.nnz_global, "rerun nnz identical",
           (double) (rep2.nnz_global - rep.nnz_global));
  }

  /* precomputed probe pairs */
  if (dim == 2)
  {
    const int           k = 20;
    double             *V = (double *) malloc (sizeof (double)
                                               * (size_t) nloc * k);
    double             *HV = (double *) malloc (sizeof (double)
                                                * (size_t) nloc * k);
    for (int j = 0; j < k; j++)
    {
      for (int i = 0; i < nloc; i++)
        V[(size_t) j * nloc + i] =
            lgh::hashed_normal (777UL, gid0 + i, j)
            / sqrt (op.mass[i]);
      psf_apply (V + (size_t) j * nloc, HV + (size_t) j * nloc, &op);
    }
    {
      int                 rc = lgh_fit_probes (fit, k, V, HV, sigma, &fo,
                                               &rep2);
      check (rc == 0, "fit_probes returns 0", (double) rc);
      check (rep2.qc_energy <= 2.0 * qc_target, "fit_probes qc sane",
             rep2.qc_energy);
    }
    free (V); free (HV);
  }

  /* REFERENCE wsym convention (serial-only scipy replica) */
  if (dim == 2 && op.nglob == nloc)
  {
    lgh_fit_opts_t      fr = fo;
    fr.wsym = LGH_WSYM_REFERENCE;
    {
      int                 rc = lgh_fit_hessian (fit, psf_apply, &op, sigma,
                                                &fr, &rep2);
      check (rc == 0, "REFERENCE wsym at n=1", (double) rc);
      check (fabs (rep2.qc_energy - rep.qc_energy) < 1e-6,
             "REFERENCE ~ WEIGHTED qc", rep2.qc_energy - rep.qc_energy);
    }
  }

  /* smoothing preserves constants (both spaces) */
  if (dim == 2)
  {
    double             *f = (double *) malloc (sizeof (double)
                                               * (size_t) nloc);
    double             *g = (double *) malloc (sizeof (double)
                                               * (size_t) nloc);
    double              worst = 0., gworst = 0.;

    for (int i = 0; i < nloc; i++) f[i] = 3.7;
    lgh_fit_smooth_field (fit, f, g, 2.0, 0);
    for (int i = 0; i < nloc; i++)
      worst = fmax (worst, fabs (g[i] - 3.7));
    lgh_fit_smooth_field (fit, f, g, 2.0, 1);
    for (int i = 0; i < nloc; i++)
      worst = fmax (worst, fabs (g[i] - 3.7));
    PetscCallMPI (MPI_Allreduce (&worst, &gworst, 1, MPI_DOUBLE, MPI_MAX,
                                 PETSC_COMM_WORLD));
    check (gworst < 1e-12, "smooth_field preserves constants", gworst);
    free (f); free (g);
  }

  free (ell); free (sigma);
  lgh_fit_destroy (fit);
  free_op (&op);
  return PETSC_SUCCESS;
}

int
main (int argc, char **argv)
{
  PetscCall (PetscInitialize (&argc, &argv, NULL, NULL));
  PetscCall (run_scenario (2, 12, 0.35));   /* d=2: N=144 */
  PetscCall (run_scenario (3, 5, 0.45));    /* d=3: N=125 */
  if (n_fail == 0)
    PetscCall (PetscPrintf (PETSC_COMM_WORLD, "PASS test_fit\n"));
  PetscCall (PetscFinalize ());
  return n_fail == 0 ? 0 : 1;
}

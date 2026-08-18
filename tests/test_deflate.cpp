/* test_deflate.cpp — dense-LAPACK oracle gates for the deflation
 * correction (lgh_glr_correct / lgh_glr_correct_probes), both backends,
 * runnable at any communicator size (registered at n = 1, 2, 4).
 *
 * Problem: N = 120, B = sum_k s_k a_k a_k^T SPD of exact rank 10 (the GLR
 * model of B is exact), diagonal Z and mass, and a TRUE Hessian
 * H_d = B + DH with a planted signed error DH of exact rank 4.  The
 * whitened error operator therefore has exactly 4 nonzero eigenvalues, so
 * a correction with enough probes/applies recovers F_true = whiten(H_d)
 * EXACTLY, and every claim gates against a redundant dense dsyevd oracle.
 *
 * Scenario A (value pass): basis/applies accounting, no clamps, corrected
 * SIGNED spectrum == dense oracle of F_true, qc_before >> qc_after ~ 0,
 * floor == oracle, the full downstream identity battery on the corrected
 * object, logdet vs oracle, and extend-after-correct refused.
 * Scenario A2 (fresh randomized): same recovery through the sketch path.
 * Scenario B (clamp): a large negative error mode making H_d(c0)
 * indefinite — exact d < -1 exists, the clamp engages, and the corrected
 * object remains internally consistent (identity battery still exact) with
 * floor < c0.
 */

#include <lgpsf_hessian/lgpsf_hessian.h>
#include <lgpsf_hessian/impl.hpp>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N_GLOBAL 120
#define RANK_B   10
#define RANK_C   4

static int          n_fail = 0;

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
factor_entry (int k, int i)      /* a_k[i] */
{
  return sin (0.1 * (i + 1) * (k + 3)) + 0.2 * cos (0.37 * i * k + k);
}

static double
b_entry (int i, int j)           /* SPD, exact rank RANK_B */
{
  double              s = 0.;
  for (int k = 0; k < RANK_B; k++)
    s += 2.0 / (k + 1.0) * factor_entry (k, i) * factor_entry (k, j);
  return s;
}

static double
perr_entry (int k, int i)        /* error factors p_k[i] */
{
  return cos (0.21 * (i + 2) * (k + 1)) + 0.3 * sin (0.4 * i - k);
}

/* err_scale: Scenario A uses a small signed error (H_d(c0) stays PD);
 * Scenario B blows up the leading negative mode to force clamping. */
static double
terr (int k, double err_scale)
{
  double              t = 0.08 / (k + 1.0) * ((k % 2) ? -1.0 : 1.0);
  if (k == 1) t *= err_scale;
  return t;
}

static double
hd_entry (int i, int j, double err_scale)
{
  double              s = b_entry (i, j);
  for (int k = 0; k < RANK_C; k++)
    s += terr (k, err_scale) * perr_entry (k, i) * perr_entry (k, j);
  return s;
}

static double
z_entry (int i) { return 1.0 + 0.01 * i; }

static double
m_entry (int i) { return 0.5 + 0.1 * (i % 7); }

static double
probe_col (int i, int j)         /* deterministic full-rank-ish probes */
{
  return sin (0.03 * (i + 1) * (j + 2)) + 0.1 * cos (0.7 * i - 1.3 * j);
}

/* prior callbacks: diagonal Z ----------------------------------------- */

typedef struct { Vec zdiag; } test_prior_ctx;

static void
cb_applyZ (Vec x, Vec y, void *ctx)
{
  test_prior_ctx     *c = (test_prior_ctx *) ctx;
  PetscErrorCode      ierr = VecPointwiseMult (y, x, c->zdiag);
  CHKERRABORT (PETSC_COMM_WORLD, ierr);
}

static void
cb_solveZ (Vec x, Vec y, void *ctx)
{
  test_prior_ctx     *c = (test_prior_ctx *) ctx;
  PetscErrorCode      ierr = VecPointwiseDivide (y, x, c->zdiag);
  CHKERRABORT (PETSC_COMM_WORLD, ierr);
}

/* the true-Hessian callback: H_d x on local arrays (allgather + dense) - */

typedef struct
{
  int                 nloc, rstart, size;
  int                *counts, *displs;
  double             *xglob;
  double              err_scale;
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
      s += hd_entry (c->rstart + i, j, c->err_scale) * c->xglob[j];
    out_local[i] = s;
  }
}

/* dense oracle: eigenvalues of F_true = whiten(H_d), SIGNED,
 * |lambda|-descending -------------------------------------------------- */

static void
oracle_true_eigs (double err_scale, double *w_absdesc)
{
  static double       F[N_GLOBAL * N_GLOBAL];
  double              w[N_GLOBAL];
  int                 n = N_GLOBAL, info = 0, lwork = -1, liwork = -1;
  int                 iwkopt = 0;
  double              wkopt;

  for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++)
      F[i + j * n] = sqrt (m_entry (i)) / z_entry (i)
        * hd_entry (i, j, err_scale) * sqrt (m_entry (j)) / z_entry (j);
  LGH_LAPACK_DSYEVD ("N", "U", &n, F, &n, w, &wkopt, &lwork, &iwkopt,
                     &liwork, &info);
  lwork = (int) wkopt; liwork = iwkopt;
  {
    double             *work = (double *) malloc (sizeof (double) * lwork);
    int                *iwork = (int *) malloc (sizeof (int) * liwork);
    LGH_LAPACK_DSYEVD ("N", "U", &n, F, &n, w, work, &lwork, iwork, &liwork,
                       &info);
    free (work); free (iwork);
  }
  if (info != 0) { n_fail++; return; }
  for (int i = 0; i < n; i++) w_absdesc[i] = w[i];
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
      if (fabs (w_absdesc[j]) > fabs (w_absdesc[i])) {
        double              t = w_absdesc[i];
        w_absdesc[i] = w_absdesc[j]; w_absdesc[j] = t;
      }
}

/* helpers ------------------------------------------------------------- */

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

static double
probe_entry (int i) { return cos (0.05 * i + 1.0); }

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

/* the downstream identity battery against the CORRECTED object: the
 * corrected apply, solve, factor, sample, pow, filter must all be mutually
 * consistent low-rank actions of the same (U', lam').                    */
static PetscErrorCode
identity_battery (lgh_glr_t *glr, double c, Vec v, Vec y, Vec x, Vec s1,
                  Vec s2, const char *tag)
{
  double              rd;
  char                what[128];

  PetscCall ((PetscErrorCode) lgh_glr_apply (glr, c, v, y));
  PetscCall ((PetscErrorCode) lgh_glr_solve (glr, c, y, x));
  PetscCall (rel_diff (x, v, s1, &rd));
  snprintf (what, sizeof (what), "%s: solve(apply(v)) == v", tag);
  check (rd < 1e-8, what, rd);

  PetscCall ((PetscErrorCode) lgh_glr_factor (glr, c, LGH_FACTOR_GT, v, x));
  PetscCall ((PetscErrorCode) lgh_glr_factor (glr, c, LGH_FACTOR_G, x, s2));
  PetscCall (rel_diff (s2, y, s1, &rd));
  snprintf (what, sizeof (what), "%s: G(GT(v)) == apply(v)", tag);
  check (rd < 1e-8, what, rd);

  PetscCall ((PetscErrorCode) lgh_glr_factor (glr, c, LGH_FACTOR_GT, v, y));
  PetscCall ((PetscErrorCode) lgh_glr_factor (glr, c, LGH_FACTOR_GINVT, y,
                                              x));
  PetscCall (rel_diff (x, v, s1, &rd));
  snprintf (what, sizeof (what), "%s: GINVT(GT(v)) == v", tag);
  check (rd < 1e-8, what, rd);
  PetscCall ((PetscErrorCode) lgh_glr_sample (glr, c, y, s2));
  PetscCall (rel_diff (s2, x, s1, &rd));
  snprintf (what, sizeof (what), "%s: sample == GINVT", tag);
  check (rd < 1e-14, what, rd);

  PetscCall ((PetscErrorCode) lgh_glr_pow (glr, c, -1.0, v, y));
  PetscCall ((PetscErrorCode) lgh_glr_pow (glr, c, 1.0, y, x));
  PetscCall (rel_diff (x, v, s1, &rd));
  snprintf (what, sizeof (what), "%s: pow(+1)(pow(-1)(v)) == v", tag);
  check (rd < 1e-10, what, rd);
  return PETSC_SUCCESS;
}

/* common construction -------------------------------------------------- */

typedef struct
{
  Mat                 B;
  Vec                 mass, zdiag, v, y, x, s1, s2;
  test_prior_ctx      pctx;
  lgh_prior_t        *prior;
  hd_ctx_t            hctx;
  PetscInt            rstart, rend;
}
setup_t;

static PetscErrorCode
setup_problem (setup_t *S, double err_scale)
{
  int                 P, rank;

  PetscCall (MatCreateDense (PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
                             N_GLOBAL, N_GLOBAL, NULL, &S->B));
  PetscCall (MatGetOwnershipRange (S->B, &S->rstart, &S->rend));
  {
    PetscScalar        *ba;
    PetscInt            lda;
    PetscCall (MatDenseGetLDA (S->B, &lda));
    PetscCall (MatDenseGetArrayWrite (S->B, &ba));
    for (PetscInt j = 0; j < N_GLOBAL; j++)
      for (PetscInt i = S->rstart; i < S->rend; i++)
        ba[(i - S->rstart) + j * lda] = b_entry ((int) i, (int) j);
    PetscCall (MatDenseRestoreArrayWrite (S->B, &ba));
  }
  PetscCall (MatAssemblyBegin (S->B, MAT_FINAL_ASSEMBLY));
  PetscCall (MatAssemblyEnd (S->B, MAT_FINAL_ASSEMBLY));

  PetscCall (MatCreateVecs (S->B, &S->mass, NULL));
  PetscCall (VecDuplicate (S->mass, &S->zdiag));
  PetscCall (VecDuplicate (S->mass, &S->v));
  PetscCall (VecDuplicate (S->mass, &S->y));
  PetscCall (VecDuplicate (S->mass, &S->x));
  PetscCall (VecDuplicate (S->mass, &S->s1));
  PetscCall (VecDuplicate (S->mass, &S->s2));
  PetscCall (fill_vec (S->mass, m_entry));
  PetscCall (fill_vec (S->zdiag, z_entry));
  PetscCall (fill_vec (S->v, probe_entry));

  S->pctx.zdiag = S->zdiag;
  {
    lgh_prior_callbacks_t cb;
    cb.applyZ = cb_applyZ; cb.applyZt = NULL;
    cb.solveZ = cb_solveZ; cb.solveZt = NULL;
    cb.solveZ_blocked = NULL; cb.solveZt_blocked = NULL;
    cb.ctx = &S->pctx;
    PetscCall (lgh_prior_create_callbacks (S->mass, &cb, &S->prior));
  }

  /* Hessian callback plumbing */
  PetscCallMPI (MPI_Comm_size (PETSC_COMM_WORLD, &P));
  PetscCallMPI (MPI_Comm_rank (PETSC_COMM_WORLD, &rank));
  S->hctx.nloc = (int) (S->rend - S->rstart);
  S->hctx.rstart = (int) S->rstart;
  S->hctx.size = P;
  S->hctx.err_scale = err_scale;
  S->hctx.counts = (int *) malloc (sizeof (int) * P);
  S->hctx.displs = (int *) malloc (sizeof (int) * P);
  S->hctx.xglob = (double *) malloc (sizeof (double) * N_GLOBAL);
  MPI_Allgather (&S->hctx.nloc, 1, MPI_INT, S->hctx.counts, 1, MPI_INT,
                 PETSC_COMM_WORLD);
  S->hctx.displs[0] = 0;
  for (int p = 1; p < P; p++)
    S->hctx.displs[p] = S->hctx.displs[p - 1] + S->hctx.counts[p - 1];
  return PETSC_SUCCESS;
}

static PetscErrorCode
teardown_problem (setup_t *S)
{
  free (S->hctx.counts);
  free (S->hctx.displs);
  free (S->hctx.xglob);
  lgh_prior_destroy (S->prior);
  PetscCall (VecDestroy (&S->mass));
  PetscCall (VecDestroy (&S->zdiag));
  PetscCall (VecDestroy (&S->v));
  PetscCall (VecDestroy (&S->y));
  PetscCall (VecDestroy (&S->x));
  PetscCall (VecDestroy (&S->s1));
  PetscCall (VecDestroy (&S->s2));
  PetscCall (MatDestroy (&S->B));
  return PETSC_SUCCESS;
}

static PetscErrorCode
build_glr (setup_t *S, lgh_glr_backend_t backend, lgh_glr_t **glr)
{
  lgh_glr_opts_t      go = lgh_glr_opts_default ();

  go.ell = 40;
  go.q_power = 1;
  go.trunc_abs = 0.;
  go.trunc_rel = 1e-8;
  go.backend = backend;
  go.nb = 8;
  go.panel = 16;
  go.check = 0;
  PetscCall (lgh_glr_compute (S->B, S->prior, &go, glr, NULL));
  return PETSC_SUCCESS;
}

/* fill caller-owned probe pairs: V deterministic, HV = H_d V computed
 * from closed forms on local rows (no communication needed).             */
static void
fill_probes (setup_t *S, int k, double err_scale, double *V, double *HV)
{
  const int           nloc = S->hctx.nloc, r0 = S->hctx.rstart;

  for (int l = 0; l < k; l++)
    for (int i = 0; i < nloc; i++) {
      double              s = 0.;
      V[(size_t) l * nloc + i] = probe_col (r0 + i, l);
      for (int j = 0; j < N_GLOBAL; j++)
        s += hd_entry (r0 + i, j, err_scale) * probe_col (j, l);
      HV[(size_t) l * nloc + i] = s;
    }
}

/* spectrum gate vs the SIGNED dense oracle                               */
static void
check_spectrum (lgh_glr_t *glr, const double *oracle, int nexpect,
                double tol, const char *tag)
{
  int                 nk;
  const double       *lam;
  double              worst = 0.;
  char                what[128];

  (void) lgh_glr_eigs (glr, &nk, &lam);
  snprintf (what, sizeof (what), "%s: kept == expected", tag);
  check (nk == nexpect, what, (double) nk);
  for (int i = 0; i < nk && i < nexpect; i++) {
    double              rd = fabs (lam[i] - oracle[i]) / fabs (oracle[0]);
    if (rd > worst) worst = rd;
  }
  snprintf (what, sizeof (what), "%s: corrected spectrum == oracle", tag);
  check (worst < tol, what, worst);
}

/* Scenario A: value pass ----------------------------------------------- */

static PetscErrorCode
run_valuepass (lgh_glr_backend_t backend)
{
  const double        c = 0.7, err_scale = 1.0;
  const int           k = 30, n_qc = 6;
  setup_t             S;
  lgh_glr_t          *glr;
  lgh_glr_correct_opts_t co = lgh_glr_correct_opts_default ();
  lgh_glr_correct_report_t crep;
  double             *V, *HV;
  double              oracle[N_GLOBAL];

  PetscCall (setup_problem (&S, err_scale));
  PetscCall (build_glr (&S, backend, &glr));

  V = (double *) malloc (sizeof (double) * (size_t) S.hctx.nloc * k + 8);
  HV = (double *) malloc (sizeof (double) * (size_t) S.hctx.nloc * k + 8);
  fill_probes (&S, k, err_scale, V, HV);

  co.c0 = 1.0;
  co.applies = 8;
  co.n_qc = n_qc;
  PetscCall ((PetscErrorCode) lgh_glr_correct_probes (glr, k, V, HV,
                                                      hd_apply, &S.hctx,
                                                      &co, &crep));

  check (crep.basis == RANK_C, "vp: basis == rank(DH)", (double) crep.basis);
  check (crep.hd_applies == RANK_C, "vp: applies == rank(DH)",
         (double) crep.hd_applies);
  check (crep.clamped == 0, "vp: no clamps", (double) crep.clamped);
  check (crep.qc_before > 1e-3, "vp: qc_before sees the error",
         crep.qc_before);
  check (crep.qc_after < 1e-7, "vp: qc_after ~ 0 (exact recovery)",
         crep.qc_after);

  oracle_true_eigs (err_scale, oracle);
  check_spectrum (glr, oracle, RANK_B + RANK_C, 1e-7, "vp");
  {
    double              lmin = 0.;
    for (int i = 0; i < RANK_B + RANK_C; i++)
      lmin = fmin (lmin, oracle[i]);
    check (fabs (crep.floor - fmax (0., -lmin)) < 1e-8, "vp: floor == oracle",
           crep.floor);
  }

  /* logdet vs oracle over the corrected (signed) spectrum */
  {
    double              ref = 0.;
    for (int i = 0; i < RANK_B + RANK_C; i++)
      ref += log (oracle[i] + c) - log (c);
    check (fabs (lgh_glr_logdet (glr, c) - ref) < 1e-8 * fabs (ref),
           "vp: logdet matches oracle", lgh_glr_logdet (glr, c) - ref);
  }

  PetscCall (identity_battery (glr, c, S.v, S.y, S.x, S.s1, S.s2, "vp"));

  /* extend after correct must be refused */
  {
    int                 ierr;
    PetscCall (PetscPushErrorHandler (PetscReturnErrorHandler, NULL));
    ierr = lgh_glr_extend (glr, 5, NULL);
    PetscCall (PetscPopErrorHandler ());
    check (ierr != 0, "vp: extend-after-correct refused", (double) ierr);
  }

  /* correcting twice must be refused */
  {
    int                 ierr;
    PetscCall (PetscPushErrorHandler (PetscReturnErrorHandler, NULL));
    ierr = lgh_glr_correct_probes (glr, k, V, HV, hd_apply, &S.hctx, &co,
                                   NULL);
    PetscCall (PetscPopErrorHandler ());
    check (ierr != 0, "vp: double correction refused", (double) ierr);
  }

  free (V); free (HV);
  lgh_glr_destroy (glr);
  PetscCall (teardown_problem (&S));
  return PETSC_SUCCESS;
}

/* Scenario A2: fresh randomized ---------------------------------------- */

static PetscErrorCode
run_fresh (lgh_glr_backend_t backend)
{
  const double        c = 0.7, err_scale = 1.0;
  setup_t             S;
  lgh_glr_t          *glr;
  lgh_glr_correct_opts_t co = lgh_glr_correct_opts_default ();
  lgh_glr_correct_report_t crep;
  double              oracle[N_GLOBAL];

  PetscCall (setup_problem (&S, err_scale));
  PetscCall (build_glr (&S, backend, &glr));

  co.c0 = 1.0;
  co.applies = 16;
  PetscCall ((PetscErrorCode) lgh_glr_correct (glr, hd_apply, &S.hctx, &co,
                                               &crep));
  check (crep.basis == RANK_C, "fresh: basis == rank(DH)",
         (double) crep.basis);
  check (crep.clamped == 0, "fresh: no clamps", (double) crep.clamped);

  oracle_true_eigs (err_scale, oracle);
  check_spectrum (glr, oracle, RANK_B + RANK_C, 1e-6, "fresh");
  PetscCall (identity_battery (glr, c, S.v, S.y, S.x, S.s1, S.s2, "fresh"));

  lgh_glr_destroy (glr);
  PetscCall (teardown_problem (&S));
  return PETSC_SUCCESS;
}

/* Scenario B: clamp ----------------------------------------------------- */

static PetscErrorCode
run_clamp (lgh_glr_backend_t backend)
{
  /* blow up the leading NEGATIVE error mode so H_d(c0) goes indefinite:
   * the exact whitened error then has d < -1 and the clamp must engage.  */
  const double        err_scale = 400.0, c0 = 1.0;
  setup_t             S;
  lgh_glr_t          *glr;
  lgh_glr_correct_opts_t co = lgh_glr_correct_opts_default ();
  lgh_glr_correct_report_t crep;
  const int           k = 30;
  double             *V, *HV;

  PetscCall (setup_problem (&S, err_scale));
  PetscCall (build_glr (&S, backend, &glr));

  V = (double *) malloc (sizeof (double) * (size_t) S.hctx.nloc * k + 8);
  HV = (double *) malloc (sizeof (double) * (size_t) S.hctx.nloc * k + 8);
  fill_probes (&S, k, err_scale, V, HV);

  co.c0 = c0;
  co.applies = 8;
  PetscCall ((PetscErrorCode) lgh_glr_correct_probes (glr, k, V, HV,
                                                      hd_apply, &S.hctx,
                                                      &co, &crep));
  check (crep.clamped >= 1, "clamp: engaged", (double) crep.clamped);
  check (crep.floor < c0, "clamp: floor < c0", crep.floor);
  check (crep.d_min >= -1.0 + co.clamp_eps - 1e-12, "clamp: d_min respected",
         crep.d_min);

  /* the corrected object remains internally consistent above its floor */
  PetscCall (identity_battery (glr, 2.0 * c0, S.v, S.y, S.x, S.s1, S.s2,
                               "clamp"));

  free (V); free (HV);
  lgh_glr_destroy (glr);
  PetscCall (teardown_problem (&S));
  return PETSC_SUCCESS;
}

int
main (int argc, char **argv)
{
  PetscCall (PetscInitialize (&argc, &argv, NULL, NULL));
  PetscCall (run_valuepass (LGH_GLR_REPLICATED));
  PetscCall (run_fresh (LGH_GLR_REPLICATED));
  PetscCall (run_clamp (LGH_GLR_REPLICATED));
#ifdef LGH_WITH_SCALAPACK
  PetscCall (run_valuepass (LGH_GLR_SCALAPACK));
  PetscCall (run_fresh (LGH_GLR_SCALAPACK));
  PetscCall (run_clamp (LGH_GLR_SCALAPACK));
#endif
  if (n_fail == 0)
    PetscCall (PetscPrintf (PETSC_COMM_WORLD, "PASS test_deflate\n"));
  PetscCall (PetscFinalize ());
  return n_fail == 0 ? 0 : 1;
}

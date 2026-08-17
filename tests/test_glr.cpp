/* test_glr.cpp — dense-LAPACK oracle gates for BOTH GLR backends
 * (replicated and, when compiled in, ScaLAPACK), runnable at any
 * communicator size (registered at n=1,2,4).
 *
 * Problem: N = 120, B = sum_k s_k a_k a_k^T with rank r = 10 (exact-rank
 * scenarios so the GLR model is exact and identities hold to solver
 * precision), diagonal Z and mass (solves exact => tight tolerances).
 * The oracle recomputes F = M^{1/2} Z^{-1} B Z^{-T} M^{1/2} densely and
 * redundantly on every rank from the same closed-form entries —
 * independent arithmetic, then LAPACK dsyevd.  Matching the oracle at
 * every rank count also proves cross-n invariance transitively.
 *
 * Scenario A (SPD B): spectrum, next_abs, residual gate, logdet, solve
 * o apply identity, factor identities (G G^T = H(c), G^{-T} G^T = I),
 * sample == GINVT, pow/filter consistency, extend == one-shot.
 * Scenario B (indefinite B): FLIP treatment — kept spectrum matches
 * |lambda| of the dense oracle, negative count, logdet over flipped
 * values.
 */

#include <lgpsf_hessian/lgpsf_hessian.h>
#include <lgpsf_hessian/impl.hpp>

#include <math.h>
#include <stdio.h>

#define N_GLOBAL 120
#define RANK_B   10

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
b_entry (int i, int j, int spd)
{
  double              s = 0.;
  for (int k = 0; k < RANK_B; k++) {
    double              sk = 2.0 / (k + 1.0);
    if (!spd && (k % 2)) sk = -sk;
    s += sk * factor_entry (k, i) * factor_entry (k, j);
  }
  return s;
}

static double
z_entry (int i) { return 1.0 + 0.01 * i; }

static double
m_entry (int i) { return 0.5 + 0.1 * (i % 7); }

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

/* dense oracle -------------------------------------------------------- */

/* eigenvalues of dense F, |lambda|-descending, into w[N]; every rank
 * redundantly (N is tiny). */
static void
oracle_eigs (int spd, double *w_absdesc)
{
  static double       F[N_GLOBAL * N_GLOBAL];
  double              w[N_GLOBAL];
  int                 n = N_GLOBAL, info = 0, lwork = -1, liwork = -1;
  int                 iwkopt = 0;
  double              wkopt;

  for (int j = 0; j < n; j++)
    for (int i = 0; i < n; i++)
      F[i + j * n] = sqrt (m_entry (i)) / z_entry (i) * b_entry (i, j, spd)
        * sqrt (m_entry (j)) / z_entry (j);
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
  /* sort |.| descending (selection sort, N tiny) */
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

/* one scenario -------------------------------------------------------- */

static PetscErrorCode
run_scenario (int spd, lgh_glr_backend_t backend)
{
  const double        c = 0.7;
  Mat                 B;
  Vec                 mass, zdiag, v, y, x, s1, s2;
  test_prior_ctx      pctx;
  lgh_prior_t        *prior;
  lgh_prior_callbacks_t cb;
  lgh_glr_t          *glr, *glr2;
  lgh_glr_opts_t      go = lgh_glr_opts_default ();
  lgh_glr_report_t    rep, rep2;
  double              dense_lam[N_GLOBAL];
  PetscInt            rstart, rend;

  /* B as MPIDENSE (the engine only ever calls MatMult) */
  PetscCall (MatCreateDense (PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
                             N_GLOBAL, N_GLOBAL, NULL, &B));
  PetscCall (MatGetOwnershipRange (B, &rstart, &rend));
  {
    PetscScalar        *ba;
    PetscInt            lda;
    PetscCall (MatDenseGetLDA (B, &lda));
    PetscCall (MatDenseGetArrayWrite (B, &ba));
    for (PetscInt j = 0; j < N_GLOBAL; j++)
      for (PetscInt i = rstart; i < rend; i++)
        ba[(i - rstart) + j * lda] = b_entry ((int) i, (int) j, spd);
    PetscCall (MatDenseRestoreArrayWrite (B, &ba));
  }
  PetscCall (MatAssemblyBegin (B, MAT_FINAL_ASSEMBLY));
  PetscCall (MatAssemblyEnd (B, MAT_FINAL_ASSEMBLY));

  PetscCall (MatCreateVecs (B, &mass, NULL));
  PetscCall (VecDuplicate (mass, &zdiag));
  PetscCall (VecDuplicate (mass, &v));
  PetscCall (VecDuplicate (mass, &y));
  PetscCall (VecDuplicate (mass, &x));
  PetscCall (VecDuplicate (mass, &s1));
  PetscCall (VecDuplicate (mass, &s2));
  PetscCall (fill_vec (mass, m_entry));
  PetscCall (fill_vec (zdiag, z_entry));
  PetscCall (fill_vec (v, probe_entry));

  pctx.zdiag = zdiag;
  cb.applyZ = cb_applyZ; cb.applyZt = NULL;
  cb.solveZ = cb_solveZ; cb.solveZt = NULL;
  cb.solveZ_blocked = NULL; cb.solveZt_blocked = NULL;
  cb.ctx = &pctx;
  PetscCall (lgh_prior_create_callbacks (mass, &cb, &prior));

  go.ell = 40;
  go.q_power = 1;
  go.trunc_abs = 0.;
  go.trunc_rel = 1e-8;
  go.backend = backend;
  go.nb = 8;         /* small blocks: exercise real block-cyclic layout */
  go.panel = 16;
  go.check = 1;
  PetscCall (lgh_glr_compute (B, prior, &go, &glr, &rep));

  oracle_eigs (spd, dense_lam);

  /* spectrum vs dense oracle (FLIP: |lambda|) */
  check (rep.kept == RANK_B, "kept == rank(B)", (double) rep.kept);
  {
    int                 nk; const double *lam;
    double              worst = 0.;
    PetscCall ((PetscErrorCode) lgh_glr_eigs (glr, &nk, &lam));
    for (int i = 0; i < nk && i < RANK_B; i++) {
      double              rd = fabs (lam[i] - fabs (dense_lam[i]))
        / fabs (dense_lam[i]);
      if (rd > worst) worst = rd;
    }
    check (worst < 1e-8, "kept spectrum matches dense oracle", worst);
  }
  check (rep.next_abs < 1e-6 * rep.lam_abs_max, "next_abs ~ 0 at exact rank",
         rep.next_abs);
  if (backend == LGH_GLR_REPLICATED)   /* eigen-residual check */
    check (rep.resid_max < 1e-7 * rep.lam_abs_max, "residual gate",
           rep.resid_max);
  else                                 /* pdsyevd-vs-dsyevd rel |dlam| */
    check (rep.resid_max < 1e-12, "pdsyevd cross-check gate",
           rep.resid_max);
  if (!spd)
    check (rep.n_negative_raw >= RANK_B / 2, "negatives present pre-FLIP",
           (double) rep.n_negative_raw);

  /* logdet vs dense (flipped values; truncated modes contribute 0) */
  {
    double              ref = 0.;
    for (int i = 0; i < RANK_B; i++)
      ref += log (fabs (dense_lam[i]) + c) - log (c);
    check (fabs (lgh_glr_logdet (glr, c) - ref) < 1e-9 * fabs (ref),
           "logdet matches dense oracle", lgh_glr_logdet (glr, c) - ref);
  }

  if (spd) {
    double              rd;

    /* solve o apply == identity (exact-rank SPD: model == exact) */
    PetscCall ((PetscErrorCode) lgh_glr_apply (glr, c, v, y));
    PetscCall ((PetscErrorCode) lgh_glr_solve (glr, c, y, x));
    PetscCall (rel_diff (x, v, s1, &rd));
    check (rd < 1e-8, "solve(apply(v)) == v", rd);

    /* G G^T == H(c) */
    PetscCall ((PetscErrorCode) lgh_glr_factor (glr, c, LGH_FACTOR_GT, v, x));
    PetscCall ((PetscErrorCode) lgh_glr_factor (glr, c, LGH_FACTOR_G, x, s2));
    PetscCall (rel_diff (s2, y, s1, &rd));
    check (rd < 1e-8, "G(GT(v)) == apply(v)", rd);

    /* G^{-T} G^T == identity; sample == GINVT */
    PetscCall ((PetscErrorCode) lgh_glr_factor (glr, c, LGH_FACTOR_GT, v, y));
    PetscCall ((PetscErrorCode) lgh_glr_factor (glr, c, LGH_FACTOR_GINVT, y,
                                                x));
    PetscCall (rel_diff (x, v, s1, &rd));
    check (rd < 1e-8, "GINVT(GT(v)) == v", rd);
    PetscCall ((PetscErrorCode) lgh_glr_sample (glr, c, y, s2));
    PetscCall (rel_diff (s2, x, s1, &rd));
    check (rd < 1e-14, "sample == GINVT", rd);

    /* pow(+1) o pow(-1) == identity (prior-weighted frame) */
    PetscCall ((PetscErrorCode) lgh_glr_pow (glr, c, -1.0, v, y));
    PetscCall ((PetscErrorCode) lgh_glr_pow (glr, c, 1.0, y, x));
    PetscCall (rel_diff (x, v, s1, &rd));
    check (rd < 1e-10, "pow(+1)(pow(-1)(v)) == v", rd);

    /* filter == pow at f = (lam + c)^{-1/2} */
    {
      int                 nk; const double *lam;
      double              fl[N_GLOBAL];
      PetscCall ((PetscErrorCode) lgh_glr_eigs (glr, &nk, &lam));
      for (int i = 0; i < nk; i++) fl[i] = 1.0 / sqrt (lam[i] + c);
      PetscCall ((PetscErrorCode) lgh_glr_filter (glr, fl, 1.0 / sqrt (c),
                                                  v, y));
      PetscCall ((PetscErrorCode) lgh_glr_pow (glr, c, -0.5, v, x));
      PetscCall (rel_diff (y, x, s1, &rd));
      check (rd < 1e-12, "filter == pow(-1/2)", rd);
    }
  }

  /* extend(10) from ell=30 == one-shot ell=40 (deterministic sketch) */
  {
    lgh_glr_opts_t      go2 = go;
    int                 nk1, nk2;
    const double       *l1, *l2;
    double              worst = 0.;

    go2.ell = 30;
    PetscCall (lgh_glr_compute (B, prior, &go2, &glr2, &rep2));
    PetscCall (lgh_glr_extend (glr2, 10, &rep2));
    PetscCall ((PetscErrorCode) lgh_glr_eigs (glr, &nk1, &l1));
    PetscCall ((PetscErrorCode) lgh_glr_eigs (glr2, &nk2, &l2));
    check (nk1 == nk2, "extend: kept matches one-shot", (double) nk2);
    for (int i = 0; i < nk1 && i < nk2; i++)
      worst = fmax (worst, fabs (l1[i] - l2[i]) / fabs (l1[i]));
    check (worst < 1e-10, "extend: spectrum matches one-shot", worst);
    lgh_glr_destroy (glr2);
  }

  lgh_glr_destroy (glr);
  lgh_prior_destroy (prior);
  PetscCall (VecDestroy (&mass));
  PetscCall (VecDestroy (&zdiag));
  PetscCall (VecDestroy (&v));
  PetscCall (VecDestroy (&y));
  PetscCall (VecDestroy (&x));
  PetscCall (VecDestroy (&s1));
  PetscCall (VecDestroy (&s2));
  PetscCall (MatDestroy (&B));
  return PETSC_SUCCESS;
}

/* cross-backend invariants: same problem, both engines, compare solve
 * outputs directly (spectra are compared through the shared oracle).    */
#ifdef LGH_WITH_SCALAPACK
static PetscErrorCode
run_cross_backend (void)
{
  const double        c = 0.7;
  Mat                 B;
  Vec                 mass, zdiag, v, x1, x2, s1;
  test_prior_ctx      pctx;
  lgh_prior_t        *prior;
  lgh_prior_callbacks_t cb;
  lgh_glr_t          *g1, *g2;
  lgh_glr_opts_t      go = lgh_glr_opts_default ();
  PetscInt            rstart, rend;
  double              rd;

  PetscCall (MatCreateDense (PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
                             N_GLOBAL, N_GLOBAL, NULL, &B));
  PetscCall (MatGetOwnershipRange (B, &rstart, &rend));
  {
    PetscScalar        *ba;
    PetscInt            lda;
    PetscCall (MatDenseGetLDA (B, &lda));
    PetscCall (MatDenseGetArrayWrite (B, &ba));
    for (PetscInt j = 0; j < N_GLOBAL; j++)
      for (PetscInt i = rstart; i < rend; i++)
        ba[(i - rstart) + j * lda] = b_entry ((int) i, (int) j, 1);
    PetscCall (MatDenseRestoreArrayWrite (B, &ba));
  }
  PetscCall (MatAssemblyBegin (B, MAT_FINAL_ASSEMBLY));
  PetscCall (MatAssemblyEnd (B, MAT_FINAL_ASSEMBLY));
  PetscCall (MatCreateVecs (B, &mass, NULL));
  PetscCall (VecDuplicate (mass, &zdiag));
  PetscCall (VecDuplicate (mass, &v));
  PetscCall (VecDuplicate (mass, &x1));
  PetscCall (VecDuplicate (mass, &x2));
  PetscCall (VecDuplicate (mass, &s1));
  PetscCall (fill_vec (mass, m_entry));
  PetscCall (fill_vec (zdiag, z_entry));
  PetscCall (fill_vec (v, probe_entry));
  pctx.zdiag = zdiag;
  cb.applyZ = cb_applyZ; cb.applyZt = NULL;
  cb.solveZ = cb_solveZ; cb.solveZt = NULL;
  cb.solveZ_blocked = NULL; cb.solveZt_blocked = NULL;
  cb.ctx = &pctx;
  PetscCall (lgh_prior_create_callbacks (mass, &cb, &prior));

  go.ell = 40; go.trunc_rel = 1e-8; go.nb = 8; go.panel = 16; go.check = 0;
  go.backend = LGH_GLR_REPLICATED;
  PetscCall (lgh_glr_compute (B, prior, &go, &g1, NULL));
  go.backend = LGH_GLR_SCALAPACK;
  PetscCall (lgh_glr_compute (B, prior, &go, &g2, NULL));

  PetscCall ((PetscErrorCode) lgh_glr_solve (g1, c, v, x1));
  PetscCall ((PetscErrorCode) lgh_glr_solve (g2, c, v, x2));
  PetscCall (rel_diff (x1, x2, s1, &rd));
  check (rd < 1e-8, "cross-backend solve agreement", rd);
  check (fabs (lgh_glr_logdet (g1, c) - lgh_glr_logdet (g2, c))
         < 1e-9 * fabs (lgh_glr_logdet (g1, c)),
         "cross-backend logdet agreement",
         lgh_glr_logdet (g1, c) - lgh_glr_logdet (g2, c));

  lgh_glr_destroy (g1);
  lgh_glr_destroy (g2);
  lgh_prior_destroy (prior);
  PetscCall (VecDestroy (&mass));
  PetscCall (VecDestroy (&zdiag));
  PetscCall (VecDestroy (&v));
  PetscCall (VecDestroy (&x1));
  PetscCall (VecDestroy (&x2));
  PetscCall (VecDestroy (&s1));
  PetscCall (MatDestroy (&B));
  return PETSC_SUCCESS;
}
#endif

int
main (int argc, char **argv)
{
  PetscCall (PetscInitialize (&argc, &argv, NULL, NULL));
  PetscCall (run_scenario (1, LGH_GLR_REPLICATED));
  PetscCall (run_scenario (0, LGH_GLR_REPLICATED));
#ifdef LGH_WITH_SCALAPACK
  PetscCall (run_scenario (1, LGH_GLR_SCALAPACK));
  PetscCall (run_scenario (0, LGH_GLR_SCALAPACK));
  PetscCall (run_cross_backend ());
#endif
  if (n_fail == 0)
    PetscCall (PetscPrintf (PETSC_COMM_WORLD, "PASS test_glr\n"));
  PetscCall (PetscFinalize ());
  return n_fail == 0 ? 0 : 1;
}

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
 * sample == GINVT, pow/filter consistency, extend == one-shot (both
 * extension modes; the deflated one's old x new block of Q^T Q).
 * Scenario B (indefinite B): FLIP treatment — kept spectrum matches
 * |lambda| of the dense oracle, negative count, logdet over flipped
 * values.
 * Both scenarios, ScaLAPACK backend: opts.eigsolver = 1 (pdsyev) against the
 * pdsyevd build (kept, treated spectrum, a solve), once through an
 * extension; lgh_glr_check_basis's Gershgorin positivity fields (2026-10-16).
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
  {
    /* the applied operator is what the formulas say (lgh_glr_check_basis, 2026-10-14):
     * every kept column orthonormal, the quadratic forms along the kept directions
     * 1/(lam + c) through the whitened filter and through the full apply, random
     * vectors consistent between the two and positive, the apply symmetric */
    lgh_glr_basis_report_t brep;

    PetscCall (lgh_glr_check_basis (glr, c, -1, 4, 2, 4, &brep));
    check (brep.orth_cols == rep.kept, "basis check: every kept column", (double) brep.orth_cols);
    check (brep.orth_diag_max < 1e-10 && brep.orth_off_max < 1e-10,
           "basis check: U^T U = I", PetscMax (brep.orth_diag_max, brep.orth_off_max));
    check (brep.n_dir == 6 && brep.dir_dev_white_max < 1e-9,
           "basis check: whitened form = 1/(lam+c)", brep.dir_dev_white_max);
    check (brep.n_dir_full == brep.n_dir && brep.dir_dev_full_max < 1e-9 && brep.dir_q_full_min > 0.,
           "basis check: full apply = 1/(lam+c), positive", brep.dir_dev_full_max);
    check (brep.n_rand == 4 && brep.rand_rel_max < 1e-9 && brep.rand_q1_min > 0.,
           "basis check: r'H^-1 r = whitened form, positive", brep.rand_rel_max);
    check (brep.n_sym == 2 && brep.sym_max < 1e-9, "basis check: symmetric", brep.sym_max);
    /* the Gershgorin positivity sums (glr.h, gersh_*): finite, their columns
     * in range, and ~0 for an orthonormal basis (O(eps lam_max / c)) */
    check (isfinite (brep.gersh_max) && isfinite (brep.gersh_diag_max)
           && brep.gersh_arg >= 0 && brep.gersh_arg < rep.kept
           && brep.gersh_diag_arg >= 0 && brep.gersh_diag_arg < rep.kept,
           "basis check: Gershgorin fields filled", brep.gersh_max);
    check (brep.gersh_max < 1e-8 && brep.gersh_diag_max < 1e-8,
           "basis check: Gershgorin sums ~ 0 for an orthonormal basis",
           PetscMax (brep.gersh_max, brep.gersh_diag_max));
  }

  /* opts.eigsolver = 1 (pdsyev) against the pdsyevd build above: the same
   * kept count, the same treated spectrum to 1e-10, the same solve on a
   * hashed random vector to 1e-8, and the pdsyev-vs-dsyevd cross-check
   * (rep.resid_max, opts.check); then an extension 30 -> 40 under pdsyev ==
   * the one-shot build.  The replicated backend ignores the option: it must
   * still accept it and give the same build.  diag_orth = 1 exercises the
   * [GLR-DIAG] eigensolver lines for pdsyev (prints only).  2026-10-16.  */
  {
    lgh_glr_opts_t      go_ev = go;
    lgh_glr_t          *gev;
    lgh_glr_report_t    rev;
    int                 nk1, nk2;
    const double       *l1, *l2;
    double              worst = 0., rd;
    const int           sca = (backend == LGH_GLR_SCALAPACK);
    Vec                 rnd;

    go_ev.eigsolver = 1;
    go_ev.diag_orth = sca;
    PetscCall (lgh_glr_compute (B, prior, &go_ev, &gev, &rev));
    check (rev.kept == rep.kept, sca ? "pdsyev: kept == pdsyevd's" : "eigsolver 1, replicated: kept unchanged",
           (double) rev.kept);
    PetscCall ((PetscErrorCode) lgh_glr_eigs (glr, &nk1, &l1));
    PetscCall ((PetscErrorCode) lgh_glr_eigs (gev, &nk2, &l2));
    for (int i = 0; i < nk1 && i < nk2; i++)
      worst = fmax (worst, fabs (l1[i] - l2[i]) / fabs (l1[i]));
    check (worst < 1e-10, sca ? "pdsyev: treated spectrum == pdsyevd's" : "eigsolver 1, replicated: spectrum unchanged",
           worst);
    if (sca)
      check (rev.resid_max < 1e-12, "pdsyev cross-check gate (vs dsyevd)", rev.resid_max);
    PetscCall (VecDuplicate (v, &rnd));
    {
      PetscInt            rs, re;
      PetscScalar        *a;
      PetscCall (VecGetOwnershipRange (rnd, &rs, &re));
      PetscCall (VecGetArray (rnd, &a));
      for (PetscInt i = rs; i < re; i++) a[i - rs] = lgh_randn_at (0x5EEDUL, i, 0);
      PetscCall (VecRestoreArray (rnd, &a));
    }
    PetscCall ((PetscErrorCode) lgh_glr_solve (glr, c, rnd, x));
    PetscCall ((PetscErrorCode) lgh_glr_solve (gev, c, rnd, y));
    PetscCall (rel_diff (y, x, s1, &rd));
    check (rd < 1e-8, sca ? "pdsyev: solve == pdsyevd's on a random vector" : "eigsolver 1, replicated: solve unchanged",
           rd);
    PetscCall (VecDestroy (&rnd));
    lgh_glr_destroy (gev);

    if (sca) {   /* an extension's re-eigensolve under pdsyev */
      go_ev.ell = 30;
      go_ev.diag_orth = 0;
      PetscCall (lgh_glr_compute (B, prior, &go_ev, &gev, &rev));
      PetscCall (lgh_glr_extend (gev, 10, &rev));
      PetscCall ((PetscErrorCode) lgh_glr_eigs (gev, &nk2, &l2));
      check (nk1 == nk2, "pdsyev: extend (30 -> 40) kept matches one-shot", (double) nk2);
      worst = 0.;
      for (int i = 0; i < nk1 && i < nk2; i++)
        worst = fmax (worst, fabs (l1[i] - l2[i]) / fabs (l1[i]));
      check (worst < 1e-10, "pdsyev: extend (30 -> 40) spectrum matches one-shot", worst);
      lgh_glr_destroy (gev);
    }
  }
  /* the default eigensolver: ELPA (2) in an ELPA build, pdsyevd (0) otherwise (2026-10-19) */
#ifdef LGH_WITH_ELPA
  check (lgh_glr_opts_default ().eigsolver == 2, "default eigsolver is 2 (ELPA) in an ELPA build", (double) lgh_glr_opts_default ().eigsolver);
#else
  check (lgh_glr_opts_default ().eigsolver == 0, "default eigsolver is 0 (pdsyevd) without ELPA", (double) lgh_glr_opts_default ().eigsolver);
#endif
#ifdef LGH_WITH_ELPA
  /* opts.eigsolver = 2 (ELPA, two-stage then one-stage) against the pdsyevd build: the same kept
   * count, the same treated spectrum to 1e-10, the same solve to 1e-8; grid2d left at -1, so ELPA
   * runs on the near-square grid; then an extension 30 -> 40 under ELPA == the one-shot build; then
   * the 1 x P grid forced (grid2d 0) at a small ell so that some process columns own no block of T
   * and the sub-communicator path runs.  2026-10-18 (the ELPA adoption candidate).               */
  if (backend == LGH_GLR_SCALAPACK) {
    int                 nk1, nk2;
    const double       *l1, *l2;
    Vec                 rnd;
    lgh_glr_opts_t      go0 = go;        /* the pdsyevd reference, explicit (ELPA is the default of an ELPA build) */
    lgh_glr_t          *g0;
    lgh_glr_report_t    rep0;

    go0.eigsolver = 0;
    PetscCall (lgh_glr_compute (B, prior, &go0, &g0, &rep0));
    PetscCall ((PetscErrorCode) lgh_glr_eigs (g0, &nk1, &l1));
    PetscCall (VecDuplicate (v, &rnd));
    {
      PetscInt            rs, re;
      PetscScalar        *a;
      PetscCall (VecGetOwnershipRange (rnd, &rs, &re));
      PetscCall (VecGetArray (rnd, &a));
      for (PetscInt i = rs; i < re; i++) a[i - rs] = lgh_randn_at (0x5EEDUL, i, 0);
      PetscCall (VecRestoreArray (rnd, &a));
    }
    PetscCall ((PetscErrorCode) lgh_glr_solve (g0, c, rnd, x));
    for (int es = 2; es >= 1; es--) {
      lgh_glr_opts_t      go_el = go;
      lgh_glr_t          *gel;
      lgh_glr_report_t    rel;
      double              worst = 0., rd;
      char                what[128];

      go_el.eigsolver = 2; go_el.elpa_solver = es; go_el.diag_orth = (es == 2);
      PetscCall (lgh_glr_compute (B, prior, &go_el, &gel, &rel));
      snprintf (what, sizeof what, "ELPA %d-stage: kept == pdsyevd's", es);
      check (rel.kept == rep0.kept, what, (double) rel.kept);
      PetscCall ((PetscErrorCode) lgh_glr_eigs (gel, &nk2, &l2));
      for (int i = 0; i < nk1 && i < nk2; i++) worst = fmax (worst, fabs (l1[i] - l2[i]) / fabs (l1[i]));
      snprintf (what, sizeof what, "ELPA %d-stage: treated spectrum == pdsyevd's", es);
      check (worst < 1e-10, what, worst);
      PetscCall ((PetscErrorCode) lgh_glr_solve (gel, c, rnd, y));
      PetscCall (rel_diff (y, x, s1, &rd));
      snprintf (what, sizeof what, "ELPA %d-stage: solve == pdsyevd's on a random vector", es);
      check (rd < 1e-8, what, rd);
      lgh_glr_destroy (gel);
      if (es == 2) {   /* an extension's re-eigensolve under ELPA */
        go_el.ell = 30; go_el.diag_orth = 0;
        PetscCall (lgh_glr_compute (B, prior, &go_el, &gel, &rel));
        PetscCall (lgh_glr_extend (gel, 10, &rel));
        PetscCall ((PetscErrorCode) lgh_glr_eigs (gel, &nk2, &l2));
        check (nk1 == nk2, "ELPA: extend (30 -> 40) kept matches one-shot", (double) nk2);
        worst = 0.;
        for (int i = 0; i < nk1 && i < nk2; i++) worst = fmax (worst, fabs (l1[i] - l2[i]) / fabs (l1[i]));
        check (worst < 1e-10, "ELPA: extend (30 -> 40) spectrum matches one-shot", worst);
        lgh_glr_destroy (gel);
      }
    }
    {   /* the sub-communicator path: 1 x P forced, ell 10 with nb 8 = two block columns (idle ranks at P > 2) */
      lgh_glr_opts_t      go_a = go, go_b = go;
      lgh_glr_t          *ga, *gb;
      lgh_glr_report_t    ra, rb;
      double              worst = 0.;

      go_a.ell = 10; go_a.grid2d = 0; go_a.eigsolver = 0;
      go_b.ell = 10; go_b.grid2d = 0; go_b.eigsolver = 2; go_b.diag_orth = 1;
      PetscCall (lgh_glr_compute (B, prior, &go_a, &ga, &ra));
      PetscCall (lgh_glr_compute (B, prior, &go_b, &gb, &rb));
      check (ra.kept == rb.kept, "ELPA on the owning sub-communicator (1 x P, ell 10): kept == pdsyevd's", (double) rb.kept);
      PetscCall ((PetscErrorCode) lgh_glr_eigs (ga, &nk1, &l1));
      PetscCall ((PetscErrorCode) lgh_glr_eigs (gb, &nk2, &l2));
      for (int i = 0; i < nk1 && i < nk2; i++) worst = fmax (worst, fabs (l1[i] - l2[i]) / fabs (l1[i]));
      check (worst < 1e-10, "ELPA on the owning sub-communicator: spectrum == pdsyevd's", worst);
      lgh_glr_destroy (ga); lgh_glr_destroy (gb);
    }
    lgh_glr_destroy (g0);
    PetscCall (VecDestroy (&rnd));
  }
#endif
  if (backend == LGH_GLR_REPLICATED)   /* eigen-residual check */
    check (rep.resid_max < 1e-7 * rep.lam_abs_max, "residual gate",
           rep.resid_max);
  else                                 /* pdsyevd-vs-dsyevd rel |dlam| */
    check (rep.resid_max < 1e-12, "pdsyevd cross-check gate",
           rep.resid_max);
  if (!spd)
    check (rep.n_negative_raw >= RANK_B / 2, "negatives present pre-FLIP",
           (double) rep.n_negative_raw);

  /* build-time breakdown: exact counts, and the top-level rows partition
   * the build (ScaLAPACK backend; the replicated one fills F rows only) */
  {
    PetscMPIInt         P;
    double              top = 0.;
    PetscCallMPI (MPI_Comm_size (PETSC_COMM_WORLD, &P));
    /* the replicated backend's residual gate applies F to the kept modes */
    check (rep.calls_max[LGH_GT_MATVEC]
             == 3. * go.ell + (backend == LGH_GLR_REPLICATED ? rep.kept : 0),
           "breakdown: 3 ell MatMults at q = 1", rep.calls_max[LGH_GT_MATVEC]);
    check (fabs (rep.nloc_mean * P - N_GLOBAL) < 1e-9,
           "breakdown: rows per rank add up", rep.nloc_mean * P);
    check (rep.nnzB_max == -1., "breakdown: nnz(B) -1 for a dense B",
           rep.nnzB_max);
    check (rep.t_total_max >= rep.t_total_mean && rep.t_total_mean > 0.,
           "breakdown: total max >= mean > 0", rep.t_total_max);
    if (backend == LGH_GLR_SCALAPACK) {
      for (int k = 0; k < LGH_GT_NTOP; k++) top += rep.t_mean[k];
      check (top <= 1.0001 * rep.t_total_mean + 1e-6 && top >= 0.5 * rep.t_total_mean,
             "breakdown: top-level rows partition the build",
             top / rep.t_total_mean);
      check (rep.calls_max[LGH_GT_OMEGA] == go.ell && rep.calls_max[LGH_GT_ORTH_RED] > 0.
             && rep.calls_max[LGH_GT_T_RED] == (double) ((go.ell + go.panel - 1) / go.panel),
             "breakdown: omega columns, orth and T allreduce counts",
             rep.calls_max[LGH_GT_T_RED]);
      check (rep.bytes_max[LGH_GT_T_RED] == 8. * go.ell * go.ell,
             "breakdown: T allreduce moves ell^2 doubles", rep.bytes_max[LGH_GT_T_RED]);
      check (rep.mem_mb_eig > 0. && rep.t_max[LGH_GT_EIG] > 0.,
             "breakdown: memory and eigensolve recorded", rep.mem_mb_eig);
    }
  }

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

  /* extend(10) from ell=30 == one-shot ell=40 (deterministic sketch).
   * extend_deflate = 0: the one-shot build exactly (span-equivalent).
   * extend_deflate = 1 (opt-in): a deflated block-Krylov extension,
   * not span-equivalent in general -- but B has exact rank 10 < 30, so the
   * old basis already holds range(F) and both must reproduce the one-shot
   * spectrum; the deflated one must also leave the old x new block of
   * Q^T Q at roundoff (check_extend).  2026-10-15.                        */
  for (int defl = 0; defl < 2; defl++) {
    lgh_glr_opts_t      go2 = go;
    int                 nk1, nk2;
    const double       *l1, *l2;
    double              worst = 0.;
    char                what[128];

    go2.ell = 30;
    go2.extend_deflate = defl;
    go2.check_extend = 1;
    go2.diag_orth = (backend == LGH_GLR_SCALAPACK);   /* exercise the [GLR-DIAG] path (prints only) */
    PetscCall (lgh_glr_compute (B, prior, &go2, &glr2, &rep2));
    PetscCall (lgh_glr_extend (glr2, 10, &rep2));
    if (backend == LGH_GLR_SCALAPACK) {  /* replicated: full rebuild */
      snprintf (what, sizeof what, "extend (deflate %d): 3 k_new MatMults (the new columns only)", defl);
      check (rep2.calls_max[LGH_GT_MATVEC] == 30., what, rep2.calls_max[LGH_GT_MATVEC]);
      snprintf (what, sizeof what, "extend (deflate %d): old x new block of Q^T Q measured", defl);
      check (rep2.ext_orth_max >= 0., what, rep2.ext_orth_max);
      if (defl) {
        snprintf (what, sizeof what, "extend (deflate %d): old x new block of Q^T Q at roundoff", defl);
        check (rep2.ext_orth_max < 1e-12, what, rep2.ext_orth_max);
      }
    }
    else {
      snprintf (what, sizeof what, "extend (deflate %d, replicated): no extension check", defl);
      check (rep2.ext_orth_max == -1., what, rep2.ext_orth_max);
    }
    PetscCall ((PetscErrorCode) lgh_glr_eigs (glr, &nk1, &l1));
    PetscCall ((PetscErrorCode) lgh_glr_eigs (glr2, &nk2, &l2));
    snprintf (what, sizeof what, "extend (deflate %d): kept matches one-shot", defl);
    check (nk1 == nk2, what, (double) nk2);
    for (int i = 0; i < nk1 && i < nk2; i++)
      worst = fmax (worst, fabs (l1[i] - l2[i]) / fabs (l1[i]));
    snprintf (what, sizeof what, "extend (deflate %d): spectrum matches one-shot", defl);
    check (worst < 1e-10, what, worst);
    lgh_glr_destroy (glr2);
  }

  /* extend(20) from ell=20 == one-shot ell=40: k_new = 20 > panel = 16, so the
   * border Q_all^T (F Q_new) is formed in TWO panels (16 + 4): the panelled
   * border of 2026-09-30 (before it, one replicated ell2 x k_new block)      */
  {
    lgh_glr_opts_t      go3 = go;
    int                 nk1, nk3;
    const double       *l1, *l3;
    double              worst = 0.;

    go3.ell = 20;
    go3.extend_deflate = 0;   /* the span-equivalent extension (the default) */
    PetscCall (lgh_glr_compute (B, prior, &go3, &glr2, &rep2));
    PetscCall (lgh_glr_extend (glr2, 20, &rep2));
    if (backend == LGH_GLR_SCALAPACK)
      check (rep2.calls_max[LGH_GT_MATVEC] == 60.,
             "extend (2 panels): 3 k_new MatMults (the new columns only)",
             rep2.calls_max[LGH_GT_MATVEC]);
    PetscCall ((PetscErrorCode) lgh_glr_eigs (glr, &nk1, &l1));
    PetscCall ((PetscErrorCode) lgh_glr_eigs (glr2, &nk3, &l3));
    check (nk1 == nk3, "extend (2 panels): kept matches one-shot", (double) nk3);
    for (int i = 0; i < nk1 && i < nk3; i++)
      worst = fmax (worst, fabs (l1[i] - l3[i]) / fabs (l1[i]));
    check (worst < 1e-10, "extend (2 panels): spectrum matches one-shot", worst);
    lgh_glr_destroy (glr2);
  }

  /* adaptive sketch (exact rank 10): from ell0 = 4 the sketch is saturated
   * at 4 and 8, doubles to 16, and stops there by the oversampling rule
   * (10 <= 0.7 * 16); == one-shot at 16.  From ell0 = 12 the FIXED ladder
   * doubles to 24 (fixed_ladder = 1, the default); the older rule
   * (fixed_ladder = 0) jumps to ceil(1.1 * 10 / 0.7) = 16.  ell_max = 12
   * clamps the last step (8 -> 12) and stops CAPPED; ell_max = 8 likewise.
   * (These small widths also exercise the pdsyevd workspace padding: the
   * reference ScaLAPACK's query under-reported here, 2026-09-28.)       */
  {
    lgh_glr_opts_t      go2 = go;
    lgh_glr_adapt_t     ad = lgh_glr_adapt_default ();
    lgh_glr_adapt_report_t ar;
    lgh_glr_t          *ga, *g1;
    lgh_glr_report_t    ra, r1;
    int                 na, n1;
    const double       *la, *l1;
    double              worst = 0.;

    ad.ell0 = 4;
    PetscCall (lgh_glr_compute_adaptive (B, prior, &go, &ad, &ga, &ra, &ar));
    check (ar.ell_final == 16 && ar.steps == 3 && !ar.capped,
           "adaptive: 4 -> 8 -> 16, stop by rho", (double) ar.ell_final);
    check (ra.kept == RANK_B, "adaptive: kept == rank(B)", (double) ra.kept);
    go2.ell = 16;
    PetscCall (lgh_glr_compute (B, prior, &go2, &g1, &r1));
    PetscCall ((PetscErrorCode) lgh_glr_eigs (ga, &na, &la));
    PetscCall ((PetscErrorCode) lgh_glr_eigs (g1, &n1, &l1));
    check (na == n1, "adaptive: kept matches one-shot at the final ell", (double) na);
    for (int i = 0; i < na && i < n1; i++)
      worst = fmax (worst, fabs (la[i] - l1[i]) / fabs (la[i]));
    check (worst < 1e-10, "adaptive: spectrum matches one-shot at the final ell", worst);
    lgh_glr_destroy (g1);
    lgh_glr_destroy (ga);

    ad.ell0 = 12;
    PetscCall (lgh_glr_compute_adaptive (B, prior, &go, &ad, &ga, &ra, &ar));
    check (ar.ell_final == 24 && ar.steps == 2 && !ar.capped && ra.kept == RANK_B,
           "adaptive (fixed ladder): 12 -> 24, stop by rho", (double) ar.ell_final);
    lgh_glr_destroy (ga);

    ad.ell0 = 12; ad.fixed_ladder = 0;
    PetscCall (lgh_glr_compute_adaptive (B, prior, &go, &ad, &ga, &ra, &ar));
    check (ar.ell_final == 16 && ar.steps == 2 && !ar.capped && ra.kept == RANK_B,
           "adaptive (kept-based rule): 12 -> 16 in one jump", (double) ar.ell_final);
    lgh_glr_destroy (ga);
    ad.fixed_ladder = 1;

    ad.ell0 = 4; ad.ell_max = 12;
    PetscCall (lgh_glr_compute_adaptive (B, prior, &go, &ad, &ga, &ra, &ar));
    check (ar.ell_final == 12 && ar.steps == 3 && ar.capped == 1 && ra.kept == RANK_B,
           "adaptive: 4 -> 8 -> 12 (last step clamped to the ceiling), capped", (double) ar.ell_final);
    lgh_glr_destroy (ga);

    ad.ell0 = 4; ad.ell_max = 8;
    PetscCall (lgh_glr_compute_adaptive (B, prior, &go, &ad, &ga, &ra, &ar));
    check (ar.ell_final == 8 && ar.capped == 1 && ra.kept == 8,
           "adaptive: capped at ell_max", (double) ar.ell_final);
    lgh_glr_destroy (ga);
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

/* test_balance.cpp — the GLR build's balanced / blocked sparse matvec
 * (impl/balance_impl.h), run at n = 1, 2, 4:
 *   1. the split: optimal bottleneck against a brute-force DP, deterministic,
 *      empty parts and a row heavier than the mean;
 *   2. the column-block move between two contiguous layouts: every entry
 *      lands at its global row, and the round trip is bitwise;
 *   3. GLR builds on a sparse B whose heavy rows cluster on rank 0, in all
 *      four matvec modes (plain, balance, block, balance + block): spectra
 *      agree, MatMult counts, the balanced layout's nnz max / mean improves,
 *      the report says what ran; extend reuses the balanced view.          */

#include <lgpsf_hessian/lgpsf_hessian.h>
#include <lgpsf_hessian/impl.hpp>

#include <math.h>
#include <stdio.h>

static int          n_fail = 0;

static void
check (int ok, const char *what, double val)
{
  if (!ok) {
    n_fail++;
    PetscPrintf (PETSC_COMM_WORLD, "FAIL: %s (%.3e)\n", what, val);
  }
}

/* ---- 1. the split ---------------------------------------------------- */

static long long
brute_cap (const long long *w, int N, int P)
{
  /* best[p][i]: minimal bottleneck for rows 0..i-1 in at most p parts */
  static long long    best[8][32];
  for (int i = 0; i <= N; i++) {
    long long           s = 0;
    for (int t = 0; t < i; t++) s += w[t];
    best[1][i] = s;
  }
  for (int p = 2; p <= P; p++)
    for (int i = 0; i <= N; i++) {
      long long           b = best[p - 1][i];
      for (int j = 0; j < i; j++) {
        long long           s = 0;
        for (int t = j; t < i; t++) s += w[t];
        long long           c = PetscMax (best[p - 1][j], s);
        if (c < b) b = c;
      }
      best[p][i] = b;
    }
  return best[P][N];
}

static PetscErrorCode
test_split (void)
{
  unsigned long       st = 12345UL;
  int                 worst_ok = 1, cover_ok = 1, cap_ok = 1;

  for (int trial = 0; trial < 400; trial++) {
    long long           w[16], cap, cap2;
    PetscInt            starts[9], starts2[9];
    int                 N, P, fl, fl2;
    st = st * 6364136223846793005UL + 1442695040888963407UL;
    N = 1 + (int) ((st >> 33) % 14);
    P = 1 + (int) ((st >> 13) % 7);
    for (int i = 0; i < N; i++) {
      st = st * 6364136223846793005UL + 1442695040888963407UL;
      w[i] = (long long) ((st >> 40) % 20);
      if (trial % 5 == 0 && i == N / 2) w[i] = 200;    /* one row heavier than the mean */
      if (trial % 7 == 0) w[i] = 0;                    /* all zero */
    }
    PetscCall (lgh_bal_split (w, N, P, starts, &cap, &fl));
    PetscCall (lgh_bal_split (w, N, P, starts2, &cap2, &fl2));
    for (int r = 0; r <= P; r++) if (starts[r] != starts2[r]) worst_ok = 0;   /* deterministic */
    if (starts[0] != 0 || starts[P] != N) cover_ok = 0;
    for (int r = 0; r < P; r++) {
      long long           s = 0;
      if (starts[r + 1] < starts[r]) cover_ok = 0;
      for (PetscInt i = starts[r]; i < starts[r + 1]; i++) s += w[i];
      if (s > cap) cap_ok = 0;
    }
    {
      long long           bc = brute_cap (w, N, P);
      if (PetscMax (bc, 1) != cap && !(bc == 0 && cap <= 1)) {
        check (0, "split: optimal bottleneck (brute force)", (double) (cap - bc));
      }
    }
  }
  check (worst_ok, "split: deterministic", 0.);
  check (cover_ok, "split: covers the rows, monotone starts", 0.);
  check (cap_ok, "split: every part within the cap", 0.);
  return PETSC_SUCCESS;
}

/* ---- 2. the move ----------------------------------------------------- */

static PetscErrorCode
test_move (void)
{
  const PetscInt      N = 97, k = 5;
  lgh_glr_t          *g;
  struct lgh_mv      *m;
  PetscMPIInt         P, rank;
  long long           w[97], cap;
  int                 fl;
  PetscInt            i, j, nH, nB;
  double             *X, *Y, *Z;
  int                 ok_land = 1, ok_trip = 1;

  PetscCallMPI (MPI_Comm_size (PETSC_COMM_WORLD, &P));
  PetscCallMPI (MPI_Comm_rank (PETSC_COMM_WORLD, &rank));
  PetscCall (PetscNew (&g));
  PetscCall (PetscNew (&m));
  g->comm = PETSC_COMM_WORLD; g->mv = m; m->P = P; m->rank = rank;
  PetscCall (PetscMalloc1 (P + 1, &m->sH));
  PetscCall (PetscMalloc1 (P + 1, &m->sB));
  PetscCall (PetscMalloc1 (2 * (size_t) P, &m->req));
  for (int r = 0; r <= P; r++) m->sH[r] = (N * r) / P;          /* even rows */
  for (i = 0; i < N; i++) w[i] = (i < 15) ? 50 : 1;             /* skewed    */
  PetscCall (lgh_bal_split (w, N, P, m->sB, &cap, &fl));
  nH = m->sH[rank + 1] - m->sH[rank];
  nB = m->sB[rank + 1] - m->sB[rank];
  PetscCall (PetscMalloc1 (PetscMax (nH, 1) * k, &X));
  PetscCall (PetscMalloc1 (PetscMax (nB, 1) * k, &Y));
  PetscCall (PetscMalloc1 (PetscMax (nH, 1) * k, &Z));
  for (j = 0; j < k; j++)
    for (i = 0; i < nH; i++) X[i + j * nH] = 1000. * (m->sH[rank] + i) + j + 0.25;
  PetscCall (lgh_mv_move (g, m->sH, m->sB, X, nH, Y, nB, k));
  for (j = 0; j < k; j++)
    for (i = 0; i < nB; i++)
      if (Y[i + j * nB] != 1000. * (m->sB[rank] + i) + j + 0.25) ok_land = 0;
  PetscCall (lgh_mv_move (g, m->sB, m->sH, Y, nB, Z, nH, k));
  for (j = 0; j < k; j++)
    for (i = 0; i < nH; i++)
      if (Z[i + j * nH] != X[i + j * nH]) ok_trip = 0;
  check (ok_land, "move: every entry lands at its global row", 0.);
  check (ok_trip, "move: round trip bitwise", 0.);
  PetscCall (PetscFree (X)); PetscCall (PetscFree (Y)); PetscCall (PetscFree (Z));
  lgh_mv_destroy (g);
  PetscCall (PetscFree (g));
  return PETSC_SUCCESS;
}

/* ---- 3. GLR builds in the four modes ---------------------------------- */

#define NB 240

typedef struct { Vec zdiag; } test_prior_ctx;
static void
cb_applyZ (Vec x, Vec y, void *ctx)
{
  PetscErrorCode ierr = VecPointwiseMult (y, x, ((test_prior_ctx *) ctx)->zdiag);
  CHKERRABORT (PETSC_COMM_WORLD, ierr);
}
static void
cb_solveZ (Vec x, Vec y, void *ctx)
{
  PetscErrorCode ierr = VecPointwiseDivide (y, x, ((test_prior_ctx *) ctx)->zdiag);
  CHKERRABORT (PETSC_COMM_WORLD, ierr);
}

static int
in_pattern (int i, int j)          /* heavy rows 0..9: a wide band */
{
  int                 d = abs (i - j);
  return d <= 2 || (PetscMin (i, j) < 10 && d <= 90);
}

static PetscErrorCode
build_B (Mat *B)
{
  PetscInt            rs, re;
  PetscCall (MatCreateAIJ (PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, NB, NB, 100, NULL, 100, NULL, B));
  PetscCall (MatGetOwnershipRange (*B, &rs, &re));
  for (PetscInt i = rs; i < re; i++)
    for (PetscInt j = 0; j < NB; j++)
      if (in_pattern ((int) i, (int) j)) {
        PetscScalar v = (i == j) ? 3.0 + 0.01 * i : exp (-fabs ((double) (i - j)) / 7.0) * cos (0.3 * (i + j));
        PetscCall (MatSetValue (*B, i, j, v, INSERT_VALUES));
      }
  PetscCall (MatAssemblyBegin (*B, MAT_FINAL_ASSEMBLY));
  PetscCall (MatAssemblyEnd (*B, MAT_FINAL_ASSEMBLY));
  return PETSC_SUCCESS;
}

static PetscErrorCode
test_modes (lgh_glr_backend_t backend)
{
  Mat                 B;
  Vec                 mass, zdiag;
  test_prior_ctx      pctx;
  lgh_prior_callbacks_t cb;
  lgh_prior_t        *prior;
  lgh_glr_opts_t      go = lgh_glr_opts_default ();
  lgh_glr_t          *g[4], *ge;
  lgh_glr_report_t    rep[4], rx;
  PetscMPIInt         P;
  const char         *name[4] = { "plain", "balance", "block", "balance+block" };

  PetscCallMPI (MPI_Comm_size (PETSC_COMM_WORLD, &P));
  PetscCall (build_B (&B));
  PetscCall (MatCreateVecs (B, &mass, NULL));
  PetscCall (VecDuplicate (mass, &zdiag));
  {
    PetscInt rs, re; PetscScalar *a;
    PetscCall (VecGetOwnershipRange (mass, &rs, &re));
    PetscCall (VecGetArray (mass, &a));
    for (PetscInt i = rs; i < re; i++) a[i - rs] = 0.5 + 0.1 * (i % 7);
    PetscCall (VecRestoreArray (mass, &a));
    PetscCall (VecGetArray (zdiag, &a));
    for (PetscInt i = rs; i < re; i++) a[i - rs] = 1.0 + 0.01 * i;
    PetscCall (VecRestoreArray (zdiag, &a));
  }
  pctx.zdiag = zdiag;
  cb.applyZ = cb_applyZ; cb.applyZt = NULL; cb.solveZ = cb_solveZ; cb.solveZt = NULL;
  cb.solveZ_blocked = NULL; cb.solveZt_blocked = NULL; cb.ctx = &pctx;
  PetscCall (lgh_prior_create_callbacks (mass, &cb, &prior));

  go.ell = 40; go.q_power = 1; go.trunc_abs = 0.; go.trunc_rel = 1e-12;
  go.backend = backend; go.nb = 8; go.panel = 16; go.check = 0;
  go.matvec_tile = 16;                 /* 40 columns: 2 full tiles + 1 partial */
  for (int md = 0; md < 4; md++) {
    lgh_glr_opts_t      o = go;
    o.balance = (md & 1); o.matvec_block = (md >> 1) & 1;
    PetscCall (lgh_glr_compute (B, prior, &o, &g[md], &rep[md]));
  }
  {
    int n0; const double *l0;
    PetscCall ((PetscErrorCode) lgh_glr_eigs (g[0], &n0, &l0));
    for (int md = 1; md < 4; md++) {
      int n1; const double *l1; double worst = 0.; char what[128];
      PetscCall ((PetscErrorCode) lgh_glr_eigs (g[md], &n1, &l1));
      snprintf (what, sizeof (what), "modes: kept equal (%s vs plain)", name[md]);
      check (n1 == n0, what, (double) n1);
      for (int i = 0; i < n0 && i < n1; i++) worst = fmax (worst, fabs (l0[i] - l1[i]) / fabs (l0[0]));
      snprintf (what, sizeof (what), "modes: spectrum equal to rounding (%s vs plain)", name[md]);
      check (worst < 1e-12, what, worst);
      snprintf (what, sizeof (what), "modes: 3 ell MatMult columns (%s)", name[md]);
      check (rep[md].calls_max[LGH_GT_MATVEC] == 3. * go.ell, what, rep[md].calls_max[LGH_GT_MATVEC]);
    }
  }
  check (rep[0].balanced == 0 && rep[0].blocked == 0, "report: plain", 0.);
  check (rep[1].balanced == (P > 1) && rep[1].blocked == 0, "report: balance", (double) rep[1].balanced);
  check (rep[2].balanced == 0 && rep[2].blocked == 1, "report: block", 0.);
  check (rep[3].balanced == (P > 1) && rep[3].blocked == 1, "report: balance+block", 0.);
  if (P > 1) {
    const double        before = rep[0].nnzB_max / rep[0].nnzB_mean;
    const double        after = rep[1].nnzBal_max / rep[1].nnzBal_mean;
    check (after < before, "balance: nnz max/mean improves", after);
    check (after <= 1.0001 * rep[1].bal_predicted + 0.05, "balance: achieved about as predicted", after - rep[1].bal_predicted);
    check (rep[1].bytes_max[LGH_GT_BAL_EXCH] > 0., "balance: the moves carried data", rep[1].bytes_max[LGH_GT_BAL_EXCH]);
  }
  if (backend == LGH_GLR_SCALAPACK) {   /* the replicated backend times F only */
    double top = 0.;
    for (int k = 0; k < LGH_GT_NTOP; k++) top += rep[3].t_mean[k];
    check (top <= 1.0001 * rep[3].t_total_mean + 1e-6 && top >= 0.5 * rep[3].t_total_mean,
           "breakdown: top-level rows still partition the build", top / rep[3].t_total_mean);
  }
  /* extend reuses the balanced view: 30 + 10 == one-shot 40 */
  {
    lgh_glr_opts_t      o = go;
    int n1, n2; const double *l1, *l2; double worst = 0.;
    o.balance = 1; o.matvec_block = 1; o.ell = 30;
    PetscCall (lgh_glr_compute (B, prior, &o, &ge, &rx));
    PetscCall (lgh_glr_extend (ge, 10, &rx));
    if (backend == LGH_GLR_SCALAPACK)
      check (rx.calls_max[LGH_GT_MATVEC] == 30., "extend: 3 k_new MatMult columns (balanced)", rx.calls_max[LGH_GT_MATVEC]);
    PetscCall ((PetscErrorCode) lgh_glr_eigs (g[3], &n1, &l1));
    PetscCall ((PetscErrorCode) lgh_glr_eigs (ge, &n2, &l2));
    check (n1 == n2, "extend (balanced): kept matches one-shot", (double) n2);
    for (int i = 0; i < n1 && i < n2; i++) worst = fmax (worst, fabs (l1[i] - l2[i]) / fabs (l1[0]));
    check (worst < 1e-10, "extend (balanced): spectrum matches one-shot", worst);
    lgh_glr_destroy (ge);
  }
  for (int md = 0; md < 4; md++) lgh_glr_destroy (g[md]);
  lgh_prior_destroy (prior);
  PetscCall (VecDestroy (&mass)); PetscCall (VecDestroy (&zdiag)); PetscCall (MatDestroy (&B));
  return PETSC_SUCCESS;
}

int
main (int argc, char **argv)
{
  PetscCall (PetscInitialize (&argc, &argv, NULL, NULL));
  PetscCall (test_split ());
  PetscCall (test_move ());
  PetscCall (test_modes (LGH_GLR_REPLICATED));
#ifdef LGH_WITH_SCALAPACK
  PetscCall (test_modes (LGH_GLR_SCALAPACK));
#endif
  if (n_fail == 0) PetscCall (PetscPrintf (PETSC_COMM_WORLD, "PASS test_balance\n"));
  PetscCall (PetscFinalize ());
  return n_fail == 0 ? 0 : 1;
}

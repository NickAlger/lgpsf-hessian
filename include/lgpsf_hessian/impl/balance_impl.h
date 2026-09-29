/* balance_impl.h — the sparse matvec inside F, balanced and/or blocked
 * (GLR-BALANCE-DESIGN.md in the ice-sheet paper repo; summary here).
 *
 * B's rows are distributed like the caller's vectors: balanced in ROWS, but
 * the nonzeros per row span two orders of magnitude and cluster in space, so
 * one rank can hold 25-100x the mean nnz and the matvec waits for it.  With
 * opts.balance the matvec runs in a second, contiguous layout L_bal whose cut
 * points balance the row weights w_i = nnz_i + row_cost (same global order;
 * only ownership boundaries move).  B_bal is a view of B in L_bal for rows
 * AND columns, so the input moves into L_bal too — leaving the input in the
 * home layout would make the ranks owning the heavy region a send hotspot.
 *
 * The memory rule: full-width (ell-column) blocks stay in the home layout.
 * Balancing nnz unbalances rows (a rank may own 6-8x the mean rows in
 * L_bal), so only B_bal and column TILES ever live there: each F apply moves
 * one tile of the Z^{-T}-solved block into L_bal, multiplies, and moves the
 * product back.  opts.matvec_block multiplies a tile with one MatMatMult
 * (one ghost exchange per tile, B read once per tile) instead of one
 * MatMult per column; with balance off it runs on B itself.
 *
 * The split is a pure function of the gathered weights and the rank count,
 * computed identically on every rank (no broadcast): the smallest bottleneck
 * cap with a greedy contiguous fill, by binary search.  Its predicted max /
 * mean is reported next to the achieved nnz max / mean.
 *
 * Compiled inside the single consumer TU (see impl.hpp), after the GLR
 * object and the timers of glr_common_impl.h.                            */

#ifndef LGPSF_HESSIAN_BALANCE_IMPL_H
#define LGPSF_HESSIAN_BALANCE_IMPL_H

struct lgh_mv
{
  Mat                 Bmv;          /* B_bal, or B (referenced)          */
  int                 balanced, blocked;
  PetscInt            tile;
  PetscInt           *sH, *sB;      /* layout starts, P + 1 each: home   *
                                     * (B's rows) and the matvec's       */
  Mat                 Yb, Wb;       /* nloc_bal x tile scratch           */
  double             *sbuf, *rbuf;
  size_t              sbuf_len, rbuf_len;
  MPI_Request        *req;
  PetscMPIInt         P, rank;
  /* report */
  double              nnz_max, nnz_mean, rows_max, predicted;
  int                 floor_binds;
};

/* ------------------------------------------------------------------ */
/* the split: pure, serial, deterministic                              */

/* largest b in [a, N] with cs[b] - cs[a] <= cap (cs nondecreasing)    */
static PetscInt
lgh_bal_reach (const long long *cs, PetscInt N, PetscInt a, long long cap)
{
  PetscInt            lo = a, hi = N;

  while (lo < hi) {
    PetscInt            m = lo + (hi - lo + 1) / 2;
    if (cs[m] - cs[a] <= cap) lo = m;
    else hi = m - 1;
  }
  return lo;
}

/* number of greedy contiguous parts at cap; P + 1 if more than P      */
static int
lgh_bal_parts (const long long *cs, PetscInt N, int P, long long cap)
{
  PetscInt            a = 0;
  int                 k = 0;

  while (a < N) {
    PetscInt            b = lgh_bal_reach (cs, N, a, cap);
    if (b == a || ++k > P) return P + 1;
    a = b;
  }
  return k;
}

/* Contiguous split of rows 0..N-1 (weights w_i >= 0) into P parts that
 * minimizes the heaviest part: binary search on the cap, greedy fill.
 * starts[0..P]: part r owns rows [starts[r], starts[r+1]); trailing parts
 * may be empty.  *cap_out = the optimal bottleneck; *floor_out = 1 when it
 * equals the heaviest single row (the granularity floor binds).          */
static PetscErrorCode
lgh_bal_split (const long long *w, PetscInt N, int P, PetscInt *starts,
               long long *cap_out, int *floor_out)
{
  long long          *cs, lo, hi, wmax = 0;
  PetscInt            i, a;
  int                 r;

  PetscCall (PetscMalloc1 (N + 1, &cs));
  cs[0] = 0;
  for (i = 0; i < N; i++) {
    cs[i + 1] = cs[i] + w[i];
    if (w[i] > wmax) wmax = w[i];
  }
  /* lo infeasible, hi feasible */
  hi = PetscMax (cs[N], 1);
  lo = PetscMax (wmax, (cs[N] + P - 1) / P) - 1;
  if (lo < 0) lo = 0;
  if (lgh_bal_parts (cs, N, P, lo) <= P) hi = lo;     /* e.g. all zero */
  while (hi - lo > 1) {
    long long           mid = lo + (hi - lo) / 2;
    if (lgh_bal_parts (cs, N, P, mid) <= P) hi = mid;
    else lo = mid;
  }
  a = 0;
  for (r = 0; r < P; r++) {
    starts[r] = a;
    if (a < N) a = lgh_bal_reach (cs, N, a, hi);
  }
  starts[P] = N;
  PetscCheck (a == N, PETSC_COMM_SELF, PETSC_ERR_PLIB,
              "lgh_bal_split: greedy fill did not cover the rows");
  *cap_out = hi;
  *floor_out = (hi == wmax);
  PetscCall (PetscFree (cs));
  return PETSC_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* moving a column block between two contiguous layouts                */

/* X (rows sF[me]..sF[me+1], leading dim ldx) -> Y (rows sT[me]..sT[me+1],
 * leading dim ldy), k columns: every global row lands at the same global
 * row.  Point-to-point with the ranks whose ranges overlap (contiguous
 * layouts: a handful), self by copy.  Collective over g->comm.           */
static PetscErrorCode
lgh_mv_move (lgh_glr_t *g, const PetscInt *sF, const PetscInt *sT,
             const double *X, PetscInt ldx, double *Y, PetscInt ldy,
             PetscInt k)
{
  struct lgh_mv      *m = g->mv;
  const PetscMPIInt   P = m->P, me = m->rank;
  const PetscInt      f0 = sF[me], f1 = sF[me + 1], t0 = sT[me], t1 = sT[me + 1];
  size_t              ns = 0, nr = 0;
  int                 nreq = 0, q;
  PetscInt            i, j;
  double              bytes = 0.;

  if (k == 0) return PETSC_SUCCESS;
  /* sizes */
  for (q = 0; q < P; q++) {
    PetscInt            a = PetscMax (f0, sT[q]), b = PetscMin (f1, sT[q + 1]);
    if (q != me && b > a) ns += (size_t) (b - a) * (size_t) k;
    a = PetscMax (t0, sF[q]); b = PetscMin (t1, sF[q + 1]);
    if (q != me && b > a) nr += (size_t) (b - a) * (size_t) k;
  }
  if (m->sbuf_len < ns) {
    PetscCall (PetscFree (m->sbuf));
    PetscCall (PetscMalloc1 (ns, &m->sbuf)); m->sbuf_len = ns;
  }
  if (m->rbuf_len < nr) {
    PetscCall (PetscFree (m->rbuf));
    PetscCall (PetscMalloc1 (nr, &m->rbuf)); m->rbuf_len = nr;
  }
  /* post receives */
  nr = 0;
  for (q = 0; q < P; q++) {
    PetscInt            a = PetscMax (t0, sF[q]), b = PetscMin (t1, sF[q + 1]);
    size_t              c;
    if (q == me || b <= a) continue;
    c = (size_t) (b - a) * (size_t) k;
    PetscCheck (c <= INT_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP, "lgh_mv_move: message too large");
    PetscCallMPI (MPI_Irecv (m->rbuf + nr, (int) c, MPI_DOUBLE, q, 7301, g->comm, &m->req[nreq++]));
    nr += c;
  }
  /* pack and send */
  ns = 0;
  for (q = 0; q < P; q++) {
    PetscInt            a = PetscMax (f0, sT[q]), b = PetscMin (f1, sT[q + 1]);
    size_t              c;
    if (q == me || b <= a) continue;
    c = (size_t) (b - a) * (size_t) k;
    for (j = 0; j < k; j++)
      PetscCall (PetscArraycpy (m->sbuf + ns + (size_t) j * (size_t) (b - a),
                                X + (size_t) j * ldx + (a - f0), (size_t) (b - a)));
    PetscCallMPI (MPI_Isend (m->sbuf + ns, (int) c, MPI_DOUBLE, q, 7301, g->comm, &m->req[nreq++]));
    ns += c;
    bytes += 8. * (double) c;
  }
  /* self */
  {
    PetscInt            a = PetscMax (f0, t0), b = PetscMin (f1, t1);
    if (b > a)
      for (j = 0; j < k; j++)
        PetscCall (PetscArraycpy (Y + (size_t) j * ldy + (a - t0),
                                  X + (size_t) j * ldx + (a - f0), (size_t) (b - a)));
  }
  PetscCallMPI (MPI_Waitall (nreq, m->req, MPI_STATUSES_IGNORE));
  /* unpack */
  nr = 0;
  for (q = 0; q < P; q++) {
    PetscInt            a = PetscMax (t0, sF[q]), b = PetscMin (t1, sF[q + 1]);
    if (q == me || b <= a) continue;
    for (j = 0; j < k; j++)
      for (i = a; i < b; i++)
        Y[(size_t) j * ldy + (i - t0)] = m->rbuf[nr + (size_t) j * (size_t) (b - a) + (size_t) (i - a)];
    nr += (size_t) (b - a) * (size_t) k;
  }
  lgh_gt_count (g, LGH_GT_BAL_EXCH, bytes, (double) nreq);
  return PETSC_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* setup (once per build; extensions reuse it) and teardown           */

static void
lgh_mv_destroy (lgh_glr_t *g)
{
  struct lgh_mv      *m = g->mv;
  if (m == NULL) return;
  (void) MatDestroy (&m->Bmv);
  (void) MatDestroy (&m->Yb);
  (void) MatDestroy (&m->Wb);
  (void) PetscFree (m->sH);
  (void) PetscFree (m->sB);
  (void) PetscFree (m->sbuf);
  (void) PetscFree (m->rbuf);
  (void) PetscFree (m->req);
  (void) PetscFree (g->mv);
}

static PetscErrorCode
lgh_mv_setup (lgh_glr_t *g)
{
  struct lgh_mv      *m;
  PetscBool           isaij = PETSC_FALSE;
  PetscMPIInt         P, rank;
  const PetscInt     *rng;
  PetscInt            nlocb, r;
  double              loc[2], mx[2], sm[2];

  if (!(g->opts.balance || g->opts.matvec_block)) return PETSC_SUCCESS;
  PetscCall (PetscObjectTypeCompareAny ((PetscObject) g->B, &isaij,
                                        MATSEQAIJ, MATMPIAIJ, ""));
  if (!isaij) return PETSC_SUCCESS;   /* a shell B: the column loop */
  PetscCall (lgh_gt_begin (g, LGH_GT_BAL_SETUP));
  PetscCallMPI (MPI_Comm_size (g->comm, &P));
  PetscCallMPI (MPI_Comm_rank (g->comm, &rank));
  PetscCall (PetscNew (&m));
  g->mv = m;
  m->P = P; m->rank = rank;
  m->tile = PetscMax (g->opts.matvec_tile, 1);
  m->blocked = g->opts.matvec_block ? 1 : 0;
  m->balanced = (g->opts.balance && P > 1) ? 1 : 0;
  PetscCall (PetscMalloc1 (P + 1, &m->sH));
  PetscCall (PetscMalloc1 (P + 1, &m->sB));
  PetscCall (PetscMalloc1 (2 * (size_t) P, &m->req));
  PetscCall (MatGetOwnershipRanges (g->B, &rng));
  for (r = 0; r <= P; r++) m->sH[r] = rng[r];

  if (m->balanced) {
    long long          *wl, *w, cap = 0, tot = 0, wmax = 0;
    PetscMPIInt        *cnt, *dsp;
    PetscInt            i, nloc = m->sH[rank + 1] - m->sH[rank];
    IS                  isr, isc;

    PetscCall (PetscMalloc1 (PetscMax (nloc, 1), &wl));
    PetscCall (PetscMalloc1 (g->Nglob, &w));
    PetscCall (PetscMalloc2 (P, &cnt, P, &dsp));
    for (i = 0; i < nloc; i++) {
      PetscInt            nc;
      PetscCall (MatGetRow (g->B, m->sH[rank] + i, &nc, NULL, NULL));
      wl[i] = (long long) nc + (long long) PetscMax (g->opts.balance_row_cost, 0);
      PetscCall (MatRestoreRow (g->B, m->sH[rank] + i, &nc, NULL, NULL));
    }
    for (r = 0; r < P; r++) {
      cnt[r] = (PetscMPIInt) (m->sH[r + 1] - m->sH[r]);
      dsp[r] = (PetscMPIInt) m->sH[r];
    }
    PetscCallMPI (MPI_Allgatherv (wl, (PetscMPIInt) nloc, MPI_LONG_LONG, w, cnt, dsp,
                                  MPI_LONG_LONG, g->comm));
    PetscCall (lgh_bal_split (w, g->Nglob, P, m->sB, &cap, &m->floor_binds));
    for (i = 0; i < g->Nglob; i++) { tot += w[i]; if (w[i] > wmax) wmax = w[i]; }
    m->predicted = (tot > 0) ? (double) cap / ((double) tot / P) : 1.;
    PetscCall (PetscFree (wl));
    PetscCall (PetscFree (w));
    PetscCall (PetscFree2 (cnt, dsp));
    nlocb = m->sB[rank + 1] - m->sB[rank];
    PetscCall (ISCreateStride (g->comm, nlocb, m->sB[rank], 1, &isr));
    PetscCall (ISCreateStride (g->comm, nlocb, m->sB[rank], 1, &isc));
    PetscCall (MatCreateSubMatrix (g->B, isr, isc, MAT_INITIAL_MATRIX, &m->Bmv));
    PetscCall (ISDestroy (&isr));
    PetscCall (ISDestroy (&isc));
  }
  else {
    for (r = 0; r <= P; r++) m->sB[r] = m->sH[r];
    m->Bmv = g->B;
    PetscCall (PetscObjectReference ((PetscObject) g->B));
    m->predicted = 0.;
    nlocb = m->sB[rank + 1] - m->sB[rank];
  }
  PetscCall (MatCreateDense (g->comm, nlocb, PETSC_DECIDE, g->Nglob, m->tile, NULL, &m->Yb));
  PetscCall (MatZeroEntries (m->Yb));
  if (m->blocked) {   /* symbolic product once; numeric per tile (MAT_REUSE_MATRIX) */
    PetscCall (MatMatMult (m->Bmv, m->Yb, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &m->Wb));
  }
  else {
    PetscCall (MatCreateDense (g->comm, nlocb, PETSC_DECIDE, g->Nglob, m->tile, NULL, &m->Wb));
  }
  /* the matvec layout's nnz and rows per rank */
  {
    MatInfo             info;
    PetscCall (MatGetInfo (m->Bmv, MAT_LOCAL, &info));
    loc[0] = info.nz_used; loc[1] = (double) nlocb;
    PetscCallMPI (MPI_Allreduce (loc, mx, 2, MPI_DOUBLE, MPI_MAX, g->comm));
    PetscCallMPI (MPI_Allreduce (loc, sm, 2, MPI_DOUBLE, MPI_SUM, g->comm));
    m->nnz_max = mx[0]; m->nnz_mean = sm[0] / P; m->rows_max = mx[1];
  }
  PetscCall (lgh_gt_end (g, LGH_GT_BAL_SETUP));
  return PETSC_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* W = B Y (home layout, ncols columns), tile by tile                  */

static PetscErrorCode
lgh_mv_apply (lgh_glr_t *g, Mat Y, Mat W)
{
  struct lgh_mv      *m = g->mv;
  const PetscInt      tile = m->tile;
  PetscInt            ncols, c0, ldy, ldw, ldyb, ldwb, j;
  const PetscScalar  *ya;
  PetscScalar        *wa;

  PetscCall (MatGetSize (Y, NULL, &ncols));
  PetscCall (MatDenseGetLDA (Y, &ldy));
  PetscCall (MatDenseGetLDA (W, &ldw));
  PetscCall (MatDenseGetLDA (m->Yb, &ldyb));
  PetscCall (MatDenseGetLDA (m->Wb, &ldwb));
  for (c0 = 0; c0 < ncols; c0 += tile) {
    const PetscInt      k = PetscMin (tile, ncols - c0);
    PetscScalar        *yb;

    /* in: tile of Y -> Yb (zero-padded to the tile width when blocked) */
    PetscCall (lgh_gt_begin (g, LGH_GT_BAL_EXCH));
    PetscCall (MatDenseGetArrayRead (Y, &ya));
    PetscCall (MatDenseGetArray (m->Yb, &yb));
    PetscCall (lgh_mv_move (g, m->sH, m->sB, ya + (size_t) c0 * ldy, ldy, yb, ldyb, k));
    if (m->blocked && k < tile) {
      const PetscInt      nb = m->sB[m->rank + 1] - m->sB[m->rank];
      for (j = k; j < tile; j++)
        PetscCall (PetscArrayzero (yb + (size_t) j * ldyb, (size_t) nb));
    }
    PetscCall (MatDenseRestoreArray (m->Yb, &yb));
    PetscCall (MatDenseRestoreArrayRead (Y, &ya));
    PetscCall (lgh_gt_end (g, LGH_GT_BAL_EXCH));

    /* multiply */
    PetscCall (lgh_gt_begin (g, LGH_GT_MATVEC));
    if (m->blocked) {
      PetscCall (MatMatMult (m->Bmv, m->Yb, MAT_REUSE_MATRIX, PETSC_DEFAULT, &m->Wb));
    }
    else {
      for (j = 0; j < k; j++) {
        Vec                 yj, wj;
        PetscCall (MatDenseGetColumnVecRead (m->Yb, j, &yj));
        PetscCall (MatDenseGetColumnVecWrite (m->Wb, j, &wj));
        PetscCall (MatMult (m->Bmv, yj, wj));
        PetscCall (MatDenseRestoreColumnVecWrite (m->Wb, j, &wj));
        PetscCall (MatDenseRestoreColumnVecRead (m->Yb, j, &yj));
      }
    }
    PetscCall (lgh_gt_end (g, LGH_GT_MATVEC));
    lgh_gt_count (g, LGH_GT_MATVEC, 0., (double) k);

    /* out: Wb -> tile of W */
    PetscCall (lgh_gt_begin (g, LGH_GT_BAL_EXCH));
    {
      const PetscScalar  *wb;
      PetscCall (MatDenseGetArrayRead (m->Wb, &wb));
      PetscCall (MatDenseGetArray (W, &wa));
      PetscCall (lgh_mv_move (g, m->sB, m->sH, wb, ldwb, wa + (size_t) c0 * ldw, ldw, k));
      PetscCall (MatDenseRestoreArray (W, &wa));
      PetscCall (MatDenseRestoreArrayRead (m->Wb, &wb));
    }
    PetscCall (lgh_gt_end (g, LGH_GT_BAL_EXCH));
  }
  return PETSC_SUCCESS;
}

#endif /* LGPSF_HESSIAN_BALANCE_IMPL_H */

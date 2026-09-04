/* fit_impl.hpp — Stage 1: the distributed LG-PSF fit (lgh_fit_*).
 * C++; needs MPI + the lgpsf library (which brings ellipsoid_tree and
 * Eigen).  Self-contained: does NOT depend on the PETSc-side impl files,
 * so a fit-only consumer may include just this file in its C++ TU.  The
 * PETSc seam (lgh_fit_get_mat) is defined at the bottom, only when the
 * GLR impl is present in the same TU (impl.hpp arranges this).
 *
 * Provenance: the production SPMD (Tier-B) distributed bridge of a
 * large-scale ice-sheet inversion, generalized for the public library:
 *   - per-node kernel covariances sigma are CALLER-SUPPLIED (the
 *     glaciological sigma recipe stayed with the ice code); the generic
 *     halo-plan mass-weighted field smoothing it used is exposed as
 *     lgh_fit_smooth_field;
 *   - dimension-generic coordinates (the lgpsf fit machinery is
 *     dimension-generic by construction, "1 to 4");
 *   - the per-row fit configuration comes from lgh_fit_opts_t; its
 *     defaults are the validated production configuration;
 *   - both weighted-symmetrization conventions ship: WEIGHTED is the
 *     production one (lgpsf::mpi::dist_weighted_symmetrize, zeros kept,
 *     gated bitwise vs serial in lgpsf/tests/mpi); REFERENCE is the
 *     bit-exact replica of the original scipy pipeline (serial only),
 *     kept so frozen validation artifacts remain checkable.
 *
 * SPMD at every communicator size.  With hashed probes (default) and
 * exact Hessian responses the fit is partition-independent and bitwise
 * identical across rank counts (gate G-L2 discipline).
 */

#ifndef LGPSF_HESSIAN_FIT_IMPL_HPP
#define LGPSF_HESSIAN_FIT_IMPL_HPP

#include "lgpsf_hessian/fit.h"

#include "lgpsf/operator_fit.hpp"
#include "lgpsf/lg_operator.hpp"
#include "lgpsf/whitening.hpp"
#include "lgpsf/mode_policy.hpp"
#include "lgpsf/mpi/dist_fit.hpp"
#include "lgpsf/mpi/dist_wsym.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace lgh {

using Clock = std::chrono::steady_clock;

/* splitmix64 -> xorshift64* -> one Box-Muller draw: one standard normal
 * as a pure function of (seed, a, b) — the partition-independent probe
 * generator (same construction as the GLR sketch). */
inline double
hashed_normal (unsigned long seed, long a, long b)
{
  unsigned long z = seed + 0x9E3779B97F4A7C15UL * (unsigned long) (a + 1)
                    + 0xC2B2AE3D27D4EB4FUL * (unsigned long) (b + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
  z = z ^ (z >> 31);
  unsigned long s = z ? z : 0x853C49E6748FEA9BUL;
  for (;;)
  {
    unsigned long x = s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27; s = x;
    const double u1 =
        (double) ((x * 2685821657736338717UL) >> 11) / 9007199254740992.0;
    x = s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; s = x;
    const double u2 =
        (double) ((x * 2685821657736338717UL) >> 11) / 9007199254740992.0;
    const double v1 = 2.0 * u1 - 1.0, v2 = 2.0 * u2 - 1.0;
    const double r2 = v1 * v1 + v2 * v2;
    if (r2 > 0.0 && r2 < 1.0)
    {
      return v1 * std::sqrt (-2.0 * std::log (r2) / r2);
    }
  }
}

/* Pull values of a per-column field at arbitrary off-rank gids (owner =
 * contiguous-range lookup).  needed: sorted, mine excluded.  Returns
 * (nneed, m) aligned with `needed`. */
inline Eigen::MatrixXd
gid_pull (MPI_Comm comm, const std::vector<long> &row_ranges,
          const std::vector<long> &needed,
          const Eigen::Ref<const Eigen::MatrixXd> &local_vals, long gid0)
{
  int rank = 0, size = 1;
  MPI_Comm_rank (comm, &rank);
  MPI_Comm_size (comm, &size);
  const int m = (int) local_vals.cols ();
  const auto owner = [&] (long g) {
    return (int) (std::upper_bound (row_ranges.begin (), row_ranges.end (), g)
                  - row_ranges.begin ()) - 1;
  };
  std::vector<int> req_cnt ((size_t) size, 0);
  for (long g : needed) req_cnt[(size_t) owner (g)]++;
  std::vector<int> srv_cnt ((size_t) size, 0);
  MPI_Alltoall (req_cnt.data (), 1, MPI_INT, srv_cnt.data (), 1, MPI_INT,
                comm);
  std::vector<MPI_Request> reqs;
  std::vector<std::vector<long>> srv_gids ((size_t) size);
  for (int r = 0; r < size; r++)
  {
    if (srv_cnt[(size_t) r] > 0)
    {
      srv_gids[(size_t) r].resize ((size_t) srv_cnt[(size_t) r]);
      reqs.emplace_back ();
      MPI_Irecv (srv_gids[(size_t) r].data (), srv_cnt[(size_t) r], MPI_LONG,
                 r, 91, comm, &reqs.back ());
    }
  }
  {
    size_t off = 0;
    for (int r = 0; r < size; r++)
    {
      if (req_cnt[(size_t) r] > 0)
      {
        reqs.emplace_back ();
        MPI_Isend (needed.data () + off, req_cnt[(size_t) r], MPI_LONG, r,
                   91, comm, &reqs.back ());
        off += (size_t) req_cnt[(size_t) r];
      }
    }
  }
  MPI_Waitall ((int) reqs.size (), reqs.data (), MPI_STATUSES_IGNORE);
  reqs.clear ();
  Eigen::MatrixXd out ((Eigen::Index) needed.size (), m);
  std::vector<std::vector<double>> reply ((size_t) size);
  std::vector<std::vector<double>> take ((size_t) size);
  for (int r = 0; r < size; r++)
  {
    if (req_cnt[(size_t) r] > 0)
    {
      take[(size_t) r].resize ((size_t) req_cnt[(size_t) r] * (size_t) m);
      reqs.emplace_back ();
      MPI_Irecv (take[(size_t) r].data (), req_cnt[(size_t) r] * m,
                 MPI_DOUBLE, r, 92, comm, &reqs.back ());
    }
  }
  for (int r = 0; r < size; r++)
  {
    if (srv_cnt[(size_t) r] > 0)
    {
      std::vector<double> &bb = reply[(size_t) r];
      bb.resize ((size_t) srv_cnt[(size_t) r] * (size_t) m);
      for (int k = 0; k < srv_cnt[(size_t) r]; k++)
      {
        const long li = srv_gids[(size_t) r][(size_t) k] - gid0;
        for (int j = 0; j < m; j++)
          bb[(size_t) k * (size_t) m + (size_t) j] = local_vals (li, j);
      }
      reqs.emplace_back ();
      MPI_Isend (bb.data (), (int) bb.size (), MPI_DOUBLE, r, 92, comm,
                 &reqs.back ());
    }
  }
  MPI_Waitall ((int) reqs.size (), reqs.data (), MPI_STATUSES_IGNORE);
  {
    std::vector<size_t> off ((size_t) size, 0);
    size_t pos = 0;
    for (long g : needed)
    {
      const int r = owner (g);
      for (int j = 0; j < m; j++)
        out ((Eigen::Index) pos, j) =
            take[(size_t) r][off[(size_t) r] * (size_t) m + (size_t) j];
      off[(size_t) r]++;
      pos++;
    }
  }
  return out;
}

/* The REFERENCE weighted symmetrization: a bit-exact replica of the
 * original scipy pipeline that every frozen validation artifact lives in.
 * Two deliberate wrinkles preserved: (1) row norms go through sqrt and
 * are re-squared (scipy's double rounding); (2) exact-zero entries are
 * PRUNED, as scipy's matmuls do silently.  The WEIGHTED convention uses
 * row energy directly and keeps the zeros — same mathematics, values
 * differing ~1e-12. */
inline Eigen::SparseMatrix<double>
weighted_symmetrize_ref (const Eigen::SparseMatrix<double> &A_in)
{
  Eigen::SparseMatrix<double> A = A_in;
  A.makeCompressed ();
  const Eigen::Index  n = A.rows ();

  Eigen::VectorXd energy = Eigen::VectorXd::Zero (n);
  for (int outer = 0; outer < A.outerSize (); ++outer)
  {
    for (Eigen::SparseMatrix<double>::InnerIterator it (A, outer); it; ++it)
    {
      energy (it.row ()) += it.value () * it.value ();
    }
  }
  Eigen::VectorXd rn = energy.array ().sqrt ();

  std::vector<double> positive;
  for (Eigen::Index i = 0; i < n; ++i)
  {
    if (rn (i) > 0.0) { positive.push_back (rn (i)); }
  }
  if (positive.empty ())
  {
    return A;   /* nothing to symmetrize (callers gate on empty fits) */
  }
  std::sort (positive.begin (), positive.end ());
  const std::size_t   half = positive.size () / 2;
  const double        median =
      (positive.size () % 2 == 1)
          ? positive[half]
          : 0.5 * (positive[half - 1] + positive[half]);
  const double        delta = 1e-2 * median;

  Eigen::VectorXd w2 (n);
  for (Eigen::Index i = 0; i < n; ++i)
  {
    w2 (i) = 1.0 / (rn (i) * rn (i) + delta * delta);
  }

  std::vector<Eigen::Triplet<double>> numerator;
  numerator.reserve (2 * static_cast<std::size_t> (A.nonZeros ()));
  for (int outer = 0; outer < A.outerSize (); ++outer)
  {
    for (Eigen::SparseMatrix<double>::InnerIterator it (A, outer); it; ++it)
    {
      if (it.value () == 0.0) { continue; }   /* scipy matmul prune */
      const double        product = w2 (it.row ()) * it.value ();
      numerator.emplace_back (static_cast<int> (it.row ()),
                              static_cast<int> (it.col ()), product);
      numerator.emplace_back (static_cast<int> (it.col ()),
                              static_cast<int> (it.row ()), product);
    }
  }
  Eigen::SparseMatrix<double> B (n, n);
  B.setFromTriplets (numerator.begin (), numerator.end ());
  for (int outer = 0; outer < B.outerSize (); ++outer)
  {
    for (Eigen::SparseMatrix<double>::InnerIterator it (B, outer); it; ++it)
    {
      it.valueRef () /= w2 (it.row ()) + w2 (it.col ());
    }
  }
  B.makeCompressed ();
  return B;
}

} /* namespace lgh */

/* ------------------------------------------------------------------ */

struct lgh_fit
{
  MPI_Comm            comm = MPI_COMM_NULL;
  int                 rank = 0, size = 1;
  int                 dim = 2;
  int                 nloc = 0;
  long                gid0 = 0, nglob = 0;
  int                 num_threads = 0;
  Eigen::MatrixXd     x;        /* (nloc, dim)  */
  Eigen::VectorXd     mass;     /* (nloc)       */
  std::vector<long>   gids;     /* [gid0, gid0+nloc) */
  std::vector<long>   row_ranges;   /* (size+1) */

  /* last successful fit: local rows of B, global cols */
  std::vector<int>    b_rowptr;
  std::vector<long>   b_colgids;
  std::vector<double> b_vals;
  bool                ready = false;
  void               *Bmat = NULL;  /* cached PETSc Mat (lgh_fit_get_mat) */
};

lgh_fit_opts_t
lgh_fit_opts_default (void)
{
  lgh_fit_opts_t      o;

  /* ladder */
  o.k0 = 15;
  o.n_qc = 5;
  o.k_max = 60;
  o.qc_target = 0.3;
  o.seed = 20260817UL;
  o.probe_mode = LGH_PROBES_HASHED;
  o.whitened_fit_probes = 1;
  /* per-row fit: the validated production configuration */
  o.tau_window = 5.0;
  o.window_aspect_cap = 1.0;    /* ball windows */
  o.spike = 1;
  o.wedge_order = 10;
  o.wedge_step = 2;
  o.mu_pinned = 1;
  o.tau_assemble = 6.0;
  o.wsym = LGH_WSYM_WEIGHTED;
  o.verbose = 1;
  return o;
}

lgh_fit_t *
lgh_fit_create (MPI_Comm comm, int dim, int nloc, long gid0,
                const double *coords, const double *mass_lumps,
                int num_threads)
{
  lgh_fit_t          *b = new lgh_fit ();

  b->comm = comm;
  MPI_Comm_rank (comm, &b->rank);
  MPI_Comm_size (comm, &b->size);
  b->dim = dim;
  b->nloc = nloc;
  b->gid0 = gid0;
  b->num_threads = num_threads;
  b->x.resize (nloc, dim);
  b->mass.resize (nloc);
  b->gids.resize ((size_t) nloc);
  for (int i = 0; i < nloc; i++)
  {
    for (int a = 0; a < dim; a++) b->x (i, a) = coords[i * dim + a];
    b->mass (i) = mass_lumps[i];
    b->gids[(size_t) i] = gid0 + i;
  }
  b->row_ranges.assign ((size_t) b->size + 1, 0);
  {
    long hi = gid0 + nloc;
    std::vector<long> his ((size_t) b->size);
    MPI_Allgather (&hi, 1, MPI_LONG, his.data (), 1, MPI_LONG, comm);
    for (int r = 0; r < b->size; r++)
      b->row_ranges[(size_t) r + 1] = his[(size_t) r];
  }
  b->nglob = b->row_ranges[(size_t) b->size];
  return b;
}

/* defined at the bottom when the PETSc side is in the TU */
static void lgh_fit_drop_mat (lgh_fit_t *b);

void
lgh_fit_destroy (lgh_fit_t *b)
{
  if (b == NULL) return;
  lgh_fit_drop_mat (b);
  delete b;
}

void
lgh_sigma_isotropic (int dim, int nloc, const double *ell_field,
                     double *sigma_out)
{
  for (int i = 0; i < nloc; i++)
  {
    const double        e2 = ell_field[i] * ell_field[i];
    for (int a = 0; a < dim; a++)
      for (int c = 0; c < dim; c++)
        sigma_out[i * dim * dim + a * dim + c] = (a == c) ? e2 : 0.0;
  }
}

/* Mass-weighted Gaussian smoothing of a per-node scalar field over the
 * distributed point cloud: candidates within 3*radius via a ball-footprint
 * halo plan; per-point sum over own + halo in ASCENDING GLOBAL ID with the
 * cutoff test — bitwise identical to the serial O(n^2) loop at comm size 1
 * and partition-independent above. */
int
lgh_fit_smooth_field (lgh_fit_t *b, const double *field_in,
                      double *field_out, double radius, int log_space)
{
  const int           nloc = b->nloc;
  const double        s2 = radius * radius;
  const double        cut2 = 9.0 * s2;
  Eigen::VectorXd     val (nloc);

  for (int i = 0; i < nloc; i++)
  {
    val (i) = log_space ? std::log (std::max (field_in[i], 1e-300))
                        : field_in[i];
  }

  std::vector<ellipsoid_tree::Ellipsoid> balls ((size_t) nloc);
  for (int i = 0; i < nloc; i++)
  {
    balls[(size_t) i].mu = b->x.row (i).transpose ();
    balls[(size_t) i].Sigma =
        cut2 * Eigen::MatrixXd::Identity (b->dim, b->dim);
  }
  const lgpsf::mpi::HaloPlan plan =
      lgpsf::mpi::halo_plan (b->comm, balls, b->x, b->gids, 32);
  Eigen::MatrixXd     payload (nloc, 2);
  payload.col (0) = b->mass;
  payload.col (1) = val;
  const Eigen::MatrixXd halo = lgpsf::mpi::halo_push (plan, payload);
  const int           nhalo = plan.nhalo ();

  const int           ncomb = nloc + nhalo;
  Eigen::MatrixXd     cx (ncomb, b->dim);
  Eigen::VectorXd     cmass (ncomb), cval (ncomb);
  {
    int io = 0, ih = 0;
    for (int c = 0; c < ncomb; c++)
    {
      const bool own = ih >= nhalo
          || (io < nloc
              && b->gids[(size_t) io] < plan.halo_gids[(size_t) ih]);
      if (own)
      {
        cx.row (c) = b->x.row (io);
        cmass (c) = b->mass (io);
        cval (c) = val (io);
        io++;
      }
      else
      {
        cx.row (c) = plan.halo_x.row (ih);
        cmass (c) = halo (ih, 0);
        cval (c) = halo (ih, 1);
        ih++;
      }
    }
  }
  for (int i = 0; i < nloc; i++)
  {
    double              wsum = 0.0, vsum = 0.0;
    for (int j = 0; j < ncomb; j++)
    {
      double              d2 = 0.0;
      for (int a = 0; a < b->dim; a++)
      {
        const double        d = cx (j, a) - b->x (i, a);
        d2 += d * d;
      }
      if (d2 > cut2)
      {
        continue;
      }
      const double        w = cmass (j) * std::exp (-0.5 * d2 / s2);
      wsum += w;
      vsum += w * cval (j);
    }
    const double        sm = vsum / wsum;
    field_out[i] = log_space ? std::exp (sm) : sm;
  }
  return 0;
}

/* ---- the fit core ---------------------------------------------------- */

namespace lgh {

inline lgpsf::OperatorFitConfig
make_config (const lgh_fit_opts_t &o, int num_threads)
{
  lgpsf::OperatorFitConfig config;

  config.tau_window = o.tau_window;
  config.window_aspect_cap = o.window_aspect_cap;
  config.spike = (o.spike != 0);
  config.row.mode_policy =
      std::make_shared<lgpsf::WedgeLadder> (o.wedge_order, o.wedge_step);
  if (o.mu_pinned)
  {
    config.row.mu = lgpsf::MuPolicy::Pinned;
  }
  config.num_threads = num_threads;
  return config;
}

/* One fit + symmetrization over the leading kfit probe columns; fills the
 * QC numbers over the trailing n_qc columns.  Returns the symmetrized
 * global triplets (sorted by row, col for WEIGHTED; REFERENCE/NONE follow
 * their conventions). */
inline int
fit_once (lgh_fit_t *b, const lgh_fit_opts_t &o,
          const lgpsf::OperatorFitConfig &config,
          const std::vector<ellipsoid_tree::Ellipsoid> &windows,
          const lgpsf::mpi::HaloPlan &plan,
          const std::vector<Eigen::MatrixXd> &sigma,
          const Eigen::MatrixXd &Z, const Eigen::MatrixXd &Y, int kfit,
          int n_qc, lgpsf::mpi::DistFitResult &fit,
          std::vector<lgpsf::mpi::GlobalTriplet> &bsym,
          lgh_fit_report_t *rep)
{
  const int           nloc = b->nloc;
  const auto          t1 = Clock::now ();

  lgpsf::mpi::DistFitInput in;
  in.x_local = b->x;
  in.col_gids = b->gids;
  in.m2_local = b->mass;
  in.V_local = Z.leftCols (kfit);
  in.x_rows = b->x;
  in.m1_local = b->mass;
  in.sigma = sigma;
  in.row_own_gid = b->gids;
  in.HV_local = Y.leftCols (kfit);
  fit = lgpsf::mpi::dist_fit (plan, in, windows, config, o.tau_assemble);
  if (b->rank == 0 && std::getenv ("LGH_QC_DEBUG") != NULL)
  {
    std::fprintf (stderr, "[LGH-FIT-DEBUG] kfit=%d B_local nnz=%ld rows=%ld"
                  " cols=%ld nfail=%ld\n", kfit,
                  (long) fit.B_local.nonZeros (),
                  (long) fit.B_local.rows (), (long) fit.B_local.cols (),
                  (long) fit.fit.diagnostics.failures.size ());
    if (!fit.fit.diagnostics.failures.empty ())
    {
      const auto         &f = *fit.fit.diagnostics.failures.begin ();
      std::fprintf (stderr, "[LGH-FIT-DEBUG] first failure (row %d): %s\n",
                    f.first, f.second.c_str ());
    }
  }

  /* local block -> global triplets INCLUDING stored zeros (the WEIGHTED
   * convention's input; zeros kept) */
  std::vector<lgpsf::mpi::GlobalTriplet> rows_local;
  rows_local.reserve ((size_t) fit.B_local.nonZeros ());
  for (int outer = 0; outer < fit.B_local.outerSize (); ++outer)
  {
    for (Eigen::SparseMatrix<double>::InnerIterator it (fit.B_local, outer);
         it; ++it)
    {
      rows_local.push_back (lgpsf::mpi::GlobalTriplet{
          b->gid0 + (long) it.row (),
          fit.col_gids[(size_t) it.col ()], it.value ()});
    }
  }
  /* per-rank fit wall (fit + triplet conversion), allreduced, BEFORE the
   * barrier: the load-imbalance meter (see fit.h, t_fit_rank_*) */
  {
    double              tl[3], tmax, tmin, tsum;

    tl[0] = std::chrono::duration<double> (Clock::now () - t1).count ();
    MPI_Allreduce (&tl[0], &tmax, 1, MPI_DOUBLE, MPI_MAX, b->comm);
    MPI_Allreduce (&tl[0], &tmin, 1, MPI_DOUBLE, MPI_MIN, b->comm);
    MPI_Allreduce (&tl[0], &tsum, 1, MPI_DOUBLE, MPI_SUM, b->comm);
    rep->t_fit_rank_max += tmax;
    rep->t_fit_rank_min += tmin;
    rep->t_fit_rank_mean += tsum / (double) b->size;
  }
  MPI_Barrier (b->comm);   /* so t_fit_symmetrize times the symmetrize only */
  const auto          t_presym = Clock::now ();
  switch (o.wsym)
  {
  case LGH_WSYM_WEIGHTED:
    bsym = lgpsf::mpi::dist_weighted_symmetrize (b->comm, rows_local,
                                                 b->row_ranges);
    break;
  case LGH_WSYM_REFERENCE:
    {
      if (b->size != 1)
      {
        return 4;   /* the scipy-replica convention is serial-only */
      }
      std::vector<Eigen::Triplet<double>> trips;
      trips.reserve (rows_local.size ());
      for (const auto &t : rows_local)
        trips.emplace_back ((int) (t.row - b->gid0),
                            (int) (t.col - b->gid0), t.value);
      Eigen::SparseMatrix<double> A (nloc, nloc);
      A.setFromTriplets (trips.begin (), trips.end ());
      const Eigen::SparseMatrix<double> B = weighted_symmetrize_ref (A);
      bsym.clear ();
      for (int outer = 0; outer < B.outerSize (); ++outer)
        for (Eigen::SparseMatrix<double>::InnerIterator it (B, outer); it;
             ++it)
          bsym.push_back (lgpsf::mpi::GlobalTriplet{
              b->gid0 + (long) it.row (), b->gid0 + (long) it.col (),
              it.value ()});
      std::sort (bsym.begin (), bsym.end (),
                 [] (const lgpsf::mpi::GlobalTriplet &p,
                     const lgpsf::mpi::GlobalTriplet &q) {
                   return p.row != q.row ? p.row < q.row : p.col < q.col;
                 });
    }
    break;
  default:                      /* LGH_WSYM_NONE: as-fitted */
    bsym = rows_local;
    std::sort (bsym.begin (), bsym.end (),
               [] (const lgpsf::mpi::GlobalTriplet &p,
                   const lgpsf::mpi::GlobalTriplet &q) {
                 return p.row != q.row ? p.row < q.row : p.col < q.col;
               });
    break;
  }
  rep->t_fit += std::chrono::duration<double> (Clock::now () - t1).count ();
  rep->t_fit_symmetrize +=
      std::chrono::duration<double> (Clock::now () - t_presym).count ();
  rep->ladder_rungs++;

  /* Guard: a globally EMPTY fit (every row dead) means the probe pool is
   * below what the per-row fitter can work with — surface it as an error
   * instead of a meaningless qc = 1.  Production pools start at k0 ~ 15. */
  {
    long                loc = (long) bsym.size (), glob = 0;
    MPI_Allreduce (&loc, &glob, 1, MPI_LONG, MPI_SUM, b->comm);
    if (b->rank == 0 && std::getenv ("LGH_QC_DEBUG") != NULL)
    {
      double              vmax = 0.;
      for (const auto &t : bsym) vmax = std::max (vmax, std::abs (t.value));
      std::fprintf (stderr, "[LGH-FIT-DEBUG] bsym local=%ld global=%ld "
                    "|v|max=%.3e\n", loc, glob, vmax);
    }
    if (glob == 0)
    {
      if (b->rank == 0)
      {
        std::fprintf (stderr, "lgh_fit: the fit came back globally empty "
                      "(%ld/%d local rows raised)%s%s\n",
                      (long) fit.fit.diagnostics.failures.size (), nloc,
                      fit.fit.diagnostics.failures.empty () ? "" : ": ",
                      fit.fit.diagnostics.failures.empty ()
                          ? ""
                          : fit.fit.diagnostics.failures.begin ()
                                ->second.c_str ());
        std::fprintf (stderr, "lgh_fit: probe demand grows with dimension "
                      "and wedge_order — raise k0/k_max or lower "
                      "wedge_order\n");
      }
      return 6;
    }
  }

  /* -- held-out QC on the symmetrized B (Allreduced) -------------------- */
  if (n_qc > 0)
  {
    const Eigen::MatrixXd Zq = Z.rightCols (n_qc);
    const Eigen::MatrixXd Yq = Y.rightCols (n_qc);
    std::vector<long>   need;
    for (const auto &t : bsym)
    {
      if (t.col < b->gid0 || t.col >= b->gid0 + nloc)
      {
        need.push_back (t.col);
      }
    }
    std::sort (need.begin (), need.end ());
    need.erase (std::unique (need.begin (), need.end ()), need.end ());
    const Eigen::MatrixXd zfar =
        gid_pull (b->comm, b->row_ranges, need, Zq, b->gid0);
    const auto zval = [&] (long g, int q) {
      if (g >= b->gid0 && g < b->gid0 + nloc)
      {
        return Zq ((Eigen::Index) (g - b->gid0), q);
      }
      const size_t p = (size_t) (std::lower_bound (need.begin (),
                                                   need.end (), g)
                                 - need.begin ());
      return zfar ((Eigen::Index) p, q);
    };
    /* legacy MEAN of per-probe ratios (heavy-tailed; diagnostic) + the
     * ENERGY ratio that DECIDES the ladder — residual and response in the
     * dual norm |v|^2_{M^-1} (lgpsf::dual_norm2; whitening notes,
     * operator-level QC section) */
    double              total_rel = 0.0, num2 = 0.0, den2 = 0.0;
    for (int q = 0; q < n_qc; q++)
    {
      Eigen::VectorXd     r = -Yq.col (q);
      for (const auto &t : bsym)
      {
        r ((Eigen::Index) (t.row - b->gid0)) += t.value * zval (t.col, q);
      }
      double              loc[2] = { lgpsf::dual_norm2 (r, b->mass),
                                     lgpsf::dual_norm2 (Yq.col (q),
                                                        b->mass) };
      double              glob[2];
      MPI_Allreduce (loc, glob, 2, MPI_DOUBLE, MPI_SUM, b->comm);
      if (b->rank == 0 && std::getenv ("LGH_QC_DEBUG") != NULL)
      {
        std::fprintf (stderr,
                      "[LGH-QC] probe %d: |Bz-y|dual = %.6e  |y|dual = "
                      "%.6e  rel = %.6f\n",
                      q, std::sqrt (glob[0]), std::sqrt (glob[1]),
                      std::sqrt (glob[0]) / std::sqrt (glob[1]));
      }
      total_rel += std::sqrt (glob[0]) / std::sqrt (glob[1]);
      num2 += glob[0];
      den2 += glob[1];
    }
    rep->qc_rowmean = total_rel / (double) n_qc;
    rep->qc_energy = std::sqrt (num2 / den2);
  }
  return 0;
}

/* telemetry + retained CSR after the final rung */
inline void
finish_fit (lgh_fit_t *b, const lgpsf::mpi::DistFitResult &fit,
            const std::vector<lgpsf::mpi::GlobalTriplet> &bsym,
            lgh_fit_report_t *rep)
{
  const int           nloc = b->nloc;

  {
    const Eigen::VectorXd sm = lgpsf::spike_measure (fit.fit.model);
    double              loc[2] = { 0.0, 0.0 };
    for (int i = 0; i < sm.size (); i++)
    {
      if (std::isfinite (sm[i]))
      {
        loc[0] += std::abs (sm[i]);
        loc[1] = std::max (loc[1], std::abs (sm[i]));
      }
    }
    double              sum = 0.0, mx = 0.0;
    MPI_Allreduce (&loc[0], &sum, 1, MPI_DOUBLE, MPI_SUM, b->comm);
    MPI_Allreduce (&loc[1], &mx, 1, MPI_DOUBLE, MPI_MAX, b->comm);
    rep->spike_mass = sum;
    rep->spike_max = mx;
  }
  {
    long                counts[3] = { (long) bsym.size (), 0, 0 };
    for (const auto st : fit.fit.diagnostics.status)
    {
      if (st == lgpsf::RowStatus::Fit) counts[1]++;
      else if (st == lgpsf::RowStatus::FallbackBaseline) counts[2]++;
    }
    long                glob[3];
    MPI_Allreduce (counts, glob, 3, MPI_LONG, MPI_SUM, b->comm);
    rep->nnz_local = (long) bsym.size ();
    rep->nnz_global = glob[0];
    rep->rows_fit = (int) glob[1];
    rep->rows_fallback = (int) glob[2];
  }

  b->b_rowptr.assign ((size_t) nloc + 1, 0);
  b->b_colgids.resize (bsym.size ());
  b->b_vals.resize (bsym.size ());
  for (size_t k = 0; k < bsym.size (); k++)
  {
    b->b_rowptr[(size_t) (bsym[k].row - b->gid0) + 1]++;
    b->b_colgids[k] = bsym[k].col;   /* bsym sorted by (row, col) */
    b->b_vals[k] = bsym[k].value;
  }
  for (int i = 0; i < nloc; i++)
  {
    b->b_rowptr[(size_t) i + 1] += b->b_rowptr[(size_t) i];
  }
  lgh_fit_drop_mat (b);   /* refit invalidates the cached Mat */
  b->ready = true;
}

inline std::vector<Eigen::MatrixXd>
sigma_to_vec (const double *sigma, int nloc, int dim)
{
  std::vector<Eigen::MatrixXd> S ((size_t) nloc);

  for (int i = 0; i < nloc; i++)
  {
    S[(size_t) i] = Eigen::Map<const Eigen::MatrixXd> (
        sigma + (size_t) i * dim * dim, dim, dim);
  }
  return S;
}

} /* namespace lgh */

int
lgh_fit_hessian (lgh_fit_t *b, lgh_hessian_fn hessian_apply, void *ctx,
                 const double *sigma_in, const lgh_fit_opts_t *opts,
                 lgh_fit_report_t *rep)
{
  using lgh::Clock;
  const lgh_fit_opts_t o = (opts != NULL) ? *opts : lgh_fit_opts_default ();
  const int           nloc = b->nloc;

  std::memset (rep, 0, sizeof (*rep));
  if (o.probe_mode == LGH_PROBES_LEGACY && b->size != 1)
  {
    return 2;   /* legacy generator is a serial replica: comm size 1 only */
  }
  if (!o.whitened_fit_probes && o.k_max > o.k0)
  {
    return 3;   /* EXPERIMENT mode: single-rung ladders only (see fit.h) */
  }

  const std::vector<Eigen::MatrixXd> sigma =
      lgh::sigma_to_vec (sigma_in, nloc, b->dim);
  const lgpsf::OperatorFitConfig config =
      lgh::make_config (o, b->num_threads);
  const std::vector<ellipsoid_tree::Ellipsoid> windows =
      lgpsf::mpi::make_window_ellipsoids (b->x, sigma, config);
  const lgpsf::mpi::HaloPlan plan =
      lgpsf::mpi::halo_plan (b->comm, windows, b->x, b->gids, 64);

  Eigen::MatrixXd     Z (nloc, 0), Y (nloc, 0);
  std::mt19937_64     legacy_gen (o.seed);
  std::normal_distribution<double> legacy_normal (0.0, 1.0);
  std::vector<double> in_buf ((size_t) nloc), out_buf ((size_t) nloc);
  int                 drawn = 0;
  const auto          draw = [&] (int count) {
    const auto          tp = Clock::now ();
    const Eigen::Index  base = Z.cols ();
    Z.conservativeResize (Eigen::NoChange, base + count);
    Y.conservativeResize (Eigen::NoChange, base + count);
    for (int j = 0; j < count; ++j)
    {
      if (o.probe_mode == LGH_PROBES_LEGACY)
      {
        for (int i = 0; i < nloc; i++)
          in_buf[(size_t) i] = legacy_normal (legacy_gen);
      }
      else
      {
        for (int i = 0; i < nloc; i++)
          in_buf[(size_t) i] =
              lgh::hashed_normal (o.seed, b->gids[(size_t) i], drawn + j);
      }
      /* z = M^{-1/2} randn: L^2-white-noise probes, the fixed metric
       * family of the energy-ratio QC (whitening notes).  Held-out QC
       * columns are ALWAYS scaled; fit columns may be left coordinate-iid
       * in the experimental unwhitened mode. */
      if (o.whitened_fit_probes || drawn + j >= o.k0)
      {
        for (int i = 0; i < nloc; i++)
          in_buf[(size_t) i] /= std::sqrt (b->mass (i));
      }
      hessian_apply (in_buf.data (), out_buf.data (), ctx);
      for (int i = 0; i < nloc; i++)
      {
        Z (i, base + j) = in_buf[(size_t) i];
        Y (i, base + j) = out_buf[(size_t) i];
      }
    }
    drawn += count;
    rep->hessian_applies += count;
    rep->t_probes +=
        std::chrono::duration<double> (Clock::now () - tp).count ();
  };

  draw (o.k0 + o.n_qc);
  int                 kfit = o.k0;
  lgpsf::mpi::DistFitResult fit;
  std::vector<lgpsf::mpi::GlobalTriplet> bsym;
  for (;;)
  {
    const int           rc = lgh::fit_once (b, o, config, windows, plan,
                                            sigma, Z, Y, kfit, o.n_qc, fit,
                                            bsym, rep);
    if (rc != 0)
    {
      return rc;
    }
    if (b->rank == 0 && o.verbose)
    {
      std::fprintf (stderr,
                    "[LGH-LADDER] rung %d: k=%d qcE=%.4f qcM=%.4f%s\n",
                    rep->ladder_rungs, kfit, rep->qc_energy,
                    rep->qc_rowmean,
                    (rep->qc_energy <= o.qc_target) ? " (target met)" : "");
    }
    if (rep->qc_energy <= o.qc_target || kfit + o.n_qc > o.k_max)
    {
      break;
    }
    /* fold the QC probes into the fit pool, draw fresh QC probes */
    draw (o.n_qc);
    kfit += o.n_qc;
  }
  rep->ladder_k = kfit;
  lgh::finish_fit (b, fit, bsym, rep);
  return 0;
}

int
lgh_fit_probes (lgh_fit_t *b, int k, const double *V, const double *HV,
                const double *sigma_in, const lgh_fit_opts_t *opts,
                lgh_fit_report_t *rep)
{
  const lgh_fit_opts_t o = (opts != NULL) ? *opts : lgh_fit_opts_default ();
  const int           nloc = b->nloc;
  const int           n_qc = (o.n_qc < k) ? o.n_qc : 0;
  const int           kfit = k - n_qc;

  std::memset (rep, 0, sizeof (*rep));
  if (kfit <= 0)
  {
    return 5;
  }

  const std::vector<Eigen::MatrixXd> sigma =
      lgh::sigma_to_vec (sigma_in, nloc, b->dim);
  const lgpsf::OperatorFitConfig config =
      lgh::make_config (o, b->num_threads);
  const std::vector<ellipsoid_tree::Ellipsoid> windows =
      lgpsf::mpi::make_window_ellipsoids (b->x, sigma, config);
  const lgpsf::mpi::HaloPlan plan =
      lgpsf::mpi::halo_plan (b->comm, windows, b->x, b->gids, 64);

  /* supplied probe pairs, used AS-IS (probe covariance is the caller's) */
  Eigen::MatrixXd     Z = Eigen::Map<const Eigen::MatrixXd> (V, nloc, k);
  Eigen::MatrixXd     Y = Eigen::Map<const Eigen::MatrixXd> (HV, nloc, k);

  lgpsf::mpi::DistFitResult fit;
  std::vector<lgpsf::mpi::GlobalTriplet> bsym;
  const int           rc = lgh::fit_once (b, o, config, windows, plan,
                                          sigma, Z, Y, kfit, n_qc, fit,
                                          bsym, rep);
  if (rc != 0)
  {
    return rc;
  }
  rep->ladder_k = kfit;
  lgh::finish_fit (b, fit, bsym, rep);
  return 0;
}

long
lgh_fit_get_rows (const lgh_fit_t *b, const int **rowptr,
                  const long **colgids, const double **vals)
{
  if (!b->ready)
  {
    return -1;
  }
  *rowptr = b->b_rowptr.data ();
  *colgids = b->b_colgids.data ();
  *vals = b->b_vals.data ();
  return (long) b->b_vals.size ();
}

/* ---- the PETSc seam --------------------------------------------------- */

#ifdef LGPSF_HESSIAN_GLR_COMMON_IMPL_H

static void
lgh_fit_drop_mat (lgh_fit_t *b)
{
  if (b->Bmat != NULL)
  {
    Mat                 M = (Mat) b->Bmat;
    (void) MatDestroy (&M);
    b->Bmat = NULL;
  }
}

int
lgh_fit_get_mat (lgh_fit_t *b, Mat *B_out)
{
  PetscCheck (b->ready, b->comm, PETSC_ERR_ORDER,
              "lgh_fit_get_mat: no successful fit yet");
  if (b->Bmat == NULL)
  {
    Mat                 B;
    const int           nloc = b->nloc;
    std::vector<PetscInt> nnz_d ((size_t) nloc, 0), nnz_o ((size_t) nloc, 0);

    for (int i = 0; i < nloc; i++)
    {
      for (int p = b->b_rowptr[(size_t) i]; p < b->b_rowptr[(size_t) i + 1];
           p++)
      {
        const long          c = b->b_colgids[(size_t) p];
        if (c >= b->gid0 && c < b->gid0 + nloc) nnz_d[(size_t) i]++;
        else nnz_o[(size_t) i]++;
      }
    }
    PetscCall (MatCreateAIJ (b->comm, nloc, nloc, PETSC_DETERMINE,
                             PETSC_DETERMINE, 0, nnz_d.data (), 0,
                             nnz_o.data (), &B));
    {
      std::vector<PetscInt> cols;
      for (int i = 0; i < nloc; i++)
      {
        const int           p0 = b->b_rowptr[(size_t) i];
        const int           p1 = b->b_rowptr[(size_t) i + 1];
        const PetscInt      row = (PetscInt) (b->gid0 + i);

        cols.resize ((size_t) (p1 - p0));
        for (int p = p0; p < p1; p++)
          cols[(size_t) (p - p0)] = (PetscInt) b->b_colgids[(size_t) p];
        PetscCall (MatSetValues (B, 1, &row, (PetscInt) (p1 - p0),
                                 cols.data (), b->b_vals.data () + p0,
                                 INSERT_VALUES));
      }
    }
    PetscCall (MatAssemblyBegin (B, MAT_FINAL_ASSEMBLY));
    PetscCall (MatAssemblyEnd (B, MAT_FINAL_ASSEMBLY));
    b->Bmat = (void *) B;
  }
  *B_out = (Mat) b->Bmat;
  return PETSC_SUCCESS;
}

#else /* fit-only TU: no cached Mat to drop */

static void
lgh_fit_drop_mat (lgh_fit_t *)
{
}

#endif /* LGPSF_HESSIAN_GLR_COMMON_IMPL_H */

#endif /* LGPSF_HESSIAN_FIT_IMPL_HPP */

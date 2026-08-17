/* fit.h — Stage 1: sparse LG-PSF approximation of a misfit Hessian.
 *
 * You bring: node coordinates, lumped mass, per-node kernel covariances
 * sigma, and a matvec callback for your misfit Hessian.  The fit draws
 * random probes, applies your Hessian to them, fits every row with a
 * Laguerre-Gaussian point spread function (lgpsf library), and assembles a
 * weighted-symmetrized sparse approximation B ~= H_misfit.  Accuracy is
 * controlled by one knob (opts.qc_target): probes are added in rungs until
 * the held-out energy-ratio QC — an estimate of the relative error
 * |B - H|_F / |H|_F in mass-whitened variables — falls below the target.
 *
 * This header is PETSc-free (MPI + plain arrays) so the fit stage can be
 * used from any framework; the PETSc seam (lgh_fit_get_mat, giving B as a
 * ready MPIAIJ Mat) is declared in glr.h.
 *
 * SPMD at every communicator size, including 1.  Deterministic: with hashed
 * probes (the default) the fit is independent of the partition, and bitwise
 * identical across rank counts for exact Hessian responses.
 */

#ifndef LGPSF_HESSIAN_FIT_H
#define LGPSF_HESSIAN_FIT_H

#ifndef MPI_SUCCESS /* probe instead of plain #include: survives headers that
                       fake-and-#undef MPI_INCLUDED (e.g. netcdf wrappers) */
#include <mpi.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/* y = H x on LOCAL arrays (this rank's owned dofs, ascending global order).
 * Collective: every rank is called at the same point in the ladder. */
typedef void (*lgh_hessian_fn) (const double *in_local, double *out_local,
                                void *ctx);

typedef struct lgh_fit lgh_fit_t; /* opaque */

/* Probe generators.  HASHED (default): each probe entry is a pure function
 * of (seed, global id, probe index) — partition-independent, extendable.
 * LEGACY: full-vector mt19937_64 draws, valid ONLY at comm size 1; exists
 * for bitwise reproduction of historical serial runs and tests. */
typedef enum
{
  LGH_PROBES_HASHED = 0,
  LGH_PROBES_LEGACY = 1
}
lgh_probe_mode_t;

/* Weighted-symmetrization convention for the assembled B.
 * WEIGHTED (default, production): rows that resolve an entry own it —
 *   scale-aware symmetrization; entries stored even when exactly zero.
 * REFERENCE: bit-exact replica of the original scipy reference pipeline
 *   (sqrt-re-squared row weights, exact zeros pruned); same mathematics,
 *   values differ ~1e-12.  Exists for validation against frozen artifacts.
 * NONE: as-fitted, unsymmetrized (rectangular/expert use). */
typedef enum
{
  LGH_WSYM_WEIGHTED  = 0,
  LGH_WSYM_REFERENCE = 1,
  LGH_WSYM_NONE      = 2
}
lgh_wsym_t;

typedef struct lgh_fit_opts
{
  /* -- accuracy ladder ------------------------------------------------- */
  int              k0;          /* initial fit-pool size (probes)          */
  int              n_qc;        /* held-out QC probes per rung             */
  int              k_max;       /* probe budget; ladder stops at k_max     */
  double           qc_target;   /* stop when energy-ratio QC <= this       */
  unsigned long    seed;        /* probe seed (deterministic)              */
  lgh_probe_mode_t probe_mode;  /* HASHED unless reproducing serial runs   */
  int  whitened_fit_probes;     /* 1 (default): fit probes are M^{-1/2}-
                                   scaled white noise, matching the QC
                                   metric family.  0 is experimental and
                                   valid only for single-rung ladders.     */
  /* -- per-row fit configuration (defaults = the validated production
   *    configuration; change only with cause) -------------------------- */
  double     tau_window;        /* row window radius, in sigma units       */
  double     window_aspect_cap; /* 1.0 = ball windows                      */
  int        spike;             /* 1: include a Dirac-spike term per row   */
  int        wedge_order;       /* LG mode ladder: wedge order             */
  int        wedge_step;        /*                 wedge step              */
  int        mu_pinned;         /* 1: pin the PSF center to the node       */
  double     tau_assemble;      /* sparsity truncation radius (sigma units)*/
  lgh_wsym_t wsym;              /* symmetrization convention               */
  int        verbose;           /* 1 (default): print ladder rungs to
                                   stderr on rank 0                        */
}
lgh_fit_opts_t;

/* All defaults; returns by value so initialization cannot be forgotten. */
lgh_fit_opts_t lgh_fit_opts_default (void);

typedef struct lgh_fit_report
{
  /* fit */
  int    rows_fit;        /* rows with a searched fit                      */
  int    rows_fallback;   /* rows on the a-priori baseline                 */
  double qc_energy;       /* held-out energy-ratio QC (decides the ladder):
                             sqrt(sum|Bz - Hz|^2 / sum|Hz|^2) over QC
                             probes ~ |B - H|_F / |H|_F, whitened          */
  double qc_rowmean;      /* legacy mean-of-relative-residuals QC
                             (heavy-tailed; diagnostic only)               */
  long   nnz_local;       /* sparse nonzeros, this rank                    */
  long   nnz_global;      /* sparse nonzeros, total                        */
  /* costs */
  int    hessian_applies; /* probe + QC Hessian applications              */
  int    ladder_k;        /* final fit-pool size                          */
  int    ladder_rungs;    /* number of fits performed                     */
  double t_probes, t_fit; /* seconds                                      */
  /* spike diagnostics (mass-weighted Dirac content; mesh-independent —
   * the resolution meter) */
  double spike_mass;      /* sum over fitted rows of m|s|                 */
  double spike_max;       /* max over fitted rows of m|s|                 */
}
lgh_fit_report_t;

/* Create a fit object: nloc owned dofs with contiguous global ids
 * [gid0, gid0 + nloc); coords interleaved (coords[i*dim + a] is coordinate
 * a of local node i); mass_lumps = diagonal of the lumped mass matrix.
 * num_threads: per-rank shared-memory fit threads (0 = library chooses).
 * Collective on comm; all arrays are copied. */
lgh_fit_t *lgh_fit_create (MPI_Comm comm, int dim, int nloc, long gid0,
                           const double *coords, const double *mass_lumps,
                           int num_threads);
void       lgh_fit_destroy (lgh_fit_t *fit);

/* THE main entry point: probe the Hessian through the callback and run the
 * accuracy ladder until qc_target or k_max.  sigma: per-node kernel
 * covariances, sigma[i*dim*dim + a*dim + b], symmetric positive definite,
 * squared length units — your a-priori estimate of each row's PSF extent
 * (see lgh_sigma_isotropic below for the simplest construction).
 * Fills *report; returns 0 on success.  Nonzero codes: 2 = LEGACY probes
 * at comm size > 1; 3 = unwhitened fit probes with a multi-rung ladder;
 * 4 = REFERENCE symmetrization at comm size > 1; 6 = the fit came back
 * globally empty (probe pool below what the per-row fitter needs —
 * production pools start at k0 ~ 15). */
int lgh_fit_hessian (lgh_fit_t *fit, lgh_hessian_fn hessian_apply, void *ctx,
                     const double *sigma, const lgh_fit_opts_t *opts,
                     lgh_fit_report_t *report);

/* Alternative entry point for PRECOMPUTED probe pairs (offline Hessian
 * samples): V and HV hold k probes and responses in columns, column-major
 * with each probe contiguous (V[j*nloc + i] = entry i of probe j).  Fits
 * once (no ladder); QC fields are filled from a held-out split of the
 * supplied pairs per opts->n_qc.  Returns 0 on success. */
int lgh_fit_probes (lgh_fit_t *fit, int k, const double *V, const double *HV,
                    const double *sigma, const lgh_fit_opts_t *opts,
                    lgh_fit_report_t *report);

/* This rank's rows of B: CSR over local rows with GLOBAL column ids.
 * Zero-copy; pointers valid until the next fit/destroy.  Returns local nnz,
 * or -1 before a successful fit.  (For B as a PETSc Mat, see
 * lgh_fit_get_mat in glr.h.) */
long lgh_fit_get_rows (const lgh_fit_t *fit, const int **rowptr,
                       const long **colgids, const double **vals);

/* -- helpers ------------------------------------------------------------ */

/* Isotropic sigma from a length-scale field: sigma_i = ell_i^2 * I.
 * Local, not collective. */
void lgh_sigma_isotropic (int dim, int nloc, const double *ell_field,
                          double *sigma_out);

/* Mass-weighted smoothing of a per-node scalar field over the distributed
 * point cloud (radius in coordinate units), using the fit object's halo
 * machinery; log_space = 1 smooths log(field) (for positive, ratio-like
 * fields such as length scales).  Bitwise equal to the serial O(n^2) loop
 * at comm size 1.  Collective. */
int lgh_fit_smooth_field (lgh_fit_t *fit, const double *field_in,
                          double *field_out, double radius, int log_space);

#ifdef __cplusplus
}
#endif

#endif /* LGPSF_HESSIAN_FIT_H */

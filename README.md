# lgpsf-hessian

Distributed-memory approximation of operators with local point-spread structure —
"Hessians" — using Laguerre-Gaussian point spread function (LG-PSF) row fits and a
prior-preconditioned global low rank (GLR) eigendecomposition, on a PETSc backend.

The intended user has a **misfit Hessian** available only through matvecs (each one a
forward + adjoint PDE solve), a **lumped mass matrix**, and a **prior / regularization**
with the standard square-root structure `R = Z M⁻¹ Z` (bilaplacian-like). The library
gives you, in three stages:

1. **Fit** — probe the Hessian with random vectors and fit every row with an LG-PSF
   (via the [lgpsf](https://github.com/NickAlger/lgpsf) library), assembling a sparse
   approximation `B ≈ H_misfit`. Accuracy is one knob: probes are added until a held-out
   energy-ratio QC (≈ relative Frobenius error in mass-whitened variables) meets your
   target.
2. **GLR** — randomized eigendecomposition of the prior-preconditioned approximation
   `F = M^{1/2} Z⁻¹ B Z⁻¹ M^{1/2} ≈ U Λ Uᵀ` (never formed), with a deterministic
   partition-independent sketch and first-class incremental rank growth.
3. **Downstream** — operations with `H(c) = B + c R`, one build serving every shift `c`:
   solves (Newton-CG preconditioning), sampling with covariance `H(c)⁻¹` (MCMC
   proposals), truncation-consistent log-determinants and matrix functions in
   prior-weighted coordinates.

Requires a diagonal (lumped) mass matrix. Dimension-generic (2D and 3D).

## Quickstart

```c
#include <lgpsf_hessian/lgpsf_hessian.h>
/* ...and put  #include <lgpsf_hessian/impl.hpp>  in ONE C++ TU of your build. */

/* Stage 1 — sparse LG-PSF approximation of your misfit Hessian */
lgh_fit_t *fit = lgh_fit_create(comm, dim, nloc, gid0, coords, mass_lumps, nthreads);
lgh_fit_opts_t fo = lgh_fit_opts_default();
lgh_fit_report_t frep;
lgh_fit_hessian(fit, my_misfit_hessian_apply, my_ctx, sigma, &fo, &frep);
Mat B;  lgh_fit_get_mat(fit, &B);                 /* symmetric MPIAIJ */

/* Stage 2 — global low rank in the prior-preconditioned frame (R = Z M^{-1} Z) */
lgh_prior_t *prior;  lgh_prior_create_mat(Z, mass_vec, NULL, &prior);
lgh_glr_opts_t go = lgh_glr_opts_default();
lgh_glr_t *glr;  lgh_glr_report_t grep;
lgh_glr_compute(B, prior, &go, &glr, &grep);
lgh_glr_extend(glr, 200, &grep);                  /* rank not converged? grow it */

/* Stage 3 — downstream with H(c) = B + c R */
lgh_glr_solve (glr, c, rhs, x);                   /* x = H(c)^{-1} rhs           */
lgh_glr_sample(glr, c, xi,  x);                   /* draw with covariance H(c)^{-1} */
double ld = lgh_glr_logdet(glr, c);               /* prior-weighted, truncation-consistent */
```

See `include/lgpsf_hessian/{fit,prior,glr}.h` for the full API — the headers are the
reference documentation. `examples/` will contain a complete heat-equation
source-inversion example with scattered point observations.

## Status

**API-review stage.** The public headers are complete; the implementation (validated in
a large-scale ice-sheet inversion) is being migrated in.

## Building

Header-only. CMake targets:

| target | contents | needs |
|---|---|---|
| `lgpsf_hessian::glr` | stages 2–3 (GLR + prior) | PETSc, MPI, LAPACK; ScaLAPACK for the distributed backend |
| `lgpsf_hessian::fit` | stage 1 (LG-PSF fit) | MPI, [lgpsf](https://github.com/NickAlger/lgpsf) (brings ellipsoid_tree + Eigen) |
| `lgpsf_hessian::lgpsf_hessian` | everything | both |

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2        # NOTE: -j 2, see below
ctest --test-dir build
```

**Build memory warning:** translation units that include `impl.hpp` are Eigen-heavy and
peak around **3 GB of memory each** at `-O2`. Use a low parallel job count — `-j 2` is
safe on a 16 GB machine; scale by available RAM, not cores.

ScaLAPACK is usually available from PETSc's `--download-scalapack`. Two platform
gotchas (details in `docs/platform-notes.md` once migrated): reference ScaLAPACK built
without `-DZeroByteTypeBug` SIGFPEs on 2D process grids under MPICH — the library
defaults to a safe 1×P grid (`opts.grid2d = 0`); and floating-point bit-identity holds
within a build, not across compilers/flags.

## License

MIT. See `LICENSE`.

/* lgpsf_hessian.h — umbrella header for lgpsf-hessian.
 *
 * Distributed-memory approximation of operators with local point-spread
 * structure ("Hessians"), in three stages:
 *
 *   Stage 1  (fit.h)    Probe the misfit Hessian with random vectors and fit
 *                       each row with a Laguerre-Gaussian point spread
 *                       function (via the lgpsf library), producing a sparse
 *                       matrix approximation B.
 *   Stage 2  (glr.h)    Randomized eigendecomposition of the
 *                       prior-preconditioned approximation
 *                       F = M^{1/2} Z^{-1} B Z^{-1} M^{1/2}  ~=  U diag(lam) U^T,
 *                       where the prior precision is R = Z M^{-1} Z (prior.h)
 *                       and M is the (diagonal, lumped) mass matrix.
 *   Stage 3  (glr.h)    Downstream operations with H(c) = B + c R: solves,
 *                       sampling (covariance H(c)^{-1}), log-determinant and
 *                       matrix functions in prior-weighted coordinates.
 *
 * Conventions used throughout:
 *   - SPMD: every function is collective on the object's communicator unless
 *     its documentation says otherwise.  All arrays are LOCAL (this rank's
 *     owned entries) and global ids are contiguous per rank:
 *     [gid0, gid0 + nloc).
 *   - Coordinates are in any consistent length unit; per-node covariances
 *     sigma are in that unit squared.  The mass matrix must be diagonal
 *     (lumped); mass_lumps holds its diagonal.
 *   - Functions return 0 on success, nonzero on failure, unless documented
 *     otherwise.  In/out Vec arguments must not alias.
 *   - The library is header-only.  Exactly ONE C++ translation unit in your
 *     program must contain:  #include <lgpsf_hessian/impl.hpp>
 *     Every other file just includes this header (plain C is fine).
 *
 * Dependencies: MPI and PETSc always; the lgpsf library (and Eigen, via
 * lgpsf) for stage 1; ScaLAPACK for the distributed eigensolver backend
 * (optional — the replicated backend needs only LAPACK).
 */

#ifndef LGPSF_HESSIAN_H
#define LGPSF_HESSIAN_H

#define LGPSF_HESSIAN_VERSION_MAJOR 0
#define LGPSF_HESSIAN_VERSION_MINOR 1
#define LGPSF_HESSIAN_VERSION_PATCH 0

#include "lgpsf_hessian/fit.h"
#include "lgpsf_hessian/prior.h"
#include "lgpsf_hessian/glr.h"

#endif /* LGPSF_HESSIAN_H */

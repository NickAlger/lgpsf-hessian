/* impl.hpp — the implementation of lgpsf-hessian.
 *
 * Include this in exactly ONE C++ translation unit of your program; include
 * lgpsf_hessian.h (plain C is fine) everywhere else.
 *
 * The GLR/prior implementation needs PETSc (+ ScaLAPACK when
 * LGH_WITH_SCALAPACK is defined); the fit implementation needs the lgpsf
 * library (header-only, brings Eigen).  NOTE: once the fit stage is
 * included, TUs containing this file are Eigen-heavy and can peak around
 * ~3 GB of memory each at -O2 — build with a low parallel job count
 * (-j 2 or less is safe).
 */

#ifndef LGPSF_HESSIAN_IMPL_HPP
#define LGPSF_HESSIAN_IMPL_HPP

/* The full public API first, in canonical order: every definition below
 * must see its extern "C" declaration (notably the lgh_fit_get_mat seam,
 * whose declaration in glr.h activates only after fit.h). */
#include "lgpsf_hessian/lgpsf_hessian.h"

#include "lgpsf_hessian/impl/glr_common_impl.h"
#ifdef LGH_WITH_SCALAPACK
#include "lgpsf_hessian/impl/glr_scalapack_impl.h"
#endif
#include "lgpsf_hessian/impl/zsolve_impl.h"
#ifdef LGH_HAVE_LGPSF /* defined by the lgpsf_hessian::fit CMake target;
                         fit-only consumers may instead include
                         impl/fit_impl.hpp directly (it is self-contained,
                         MPI + lgpsf only) */
#include "lgpsf_hessian/impl/fit_impl.hpp"
#endif

#endif /* LGPSF_HESSIAN_IMPL_HPP */

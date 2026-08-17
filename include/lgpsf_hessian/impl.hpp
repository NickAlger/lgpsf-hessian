/* impl.hpp — the implementation of lgpsf-hessian.
 *
 * Include this in exactly ONE C++ translation unit of your program; include
 * lgpsf_hessian.h (plain C is fine) everywhere else.
 *
 * The GLR/prior implementation needs PETSc (+ ScaLAPACK when
 * LGH_WITH_SCALAPACK is defined); the fit implementation needs the lgpsf
 * library (header-only, brings Eigen).  NOTE: TUs that include this file
 * are Eigen-heavy and can peak around ~3 GB of memory per TU at -O2 —
 * build with a low parallel job count (-j 2 or less is safe).
 */

#ifndef LGPSF_HESSIAN_IMPL_HPP
#define LGPSF_HESSIAN_IMPL_HPP

#error "lgpsf-hessian is at the API-review stage: the implementation lands \
with migration slices B-E (see docs/architecture.md once written). The \
public API in lgpsf_hessian.h is the artifact under review."

/* Will become:
 * #include "lgpsf_hessian/impl/glr_common_impl.h"      (kit + replicated)
 * #include "lgpsf_hessian/impl/glr_scalapack_impl.h"   (LGH_WITH_SCALAPACK)
 * #include "lgpsf_hessian/impl/zsolve_impl.h"          (prior Mat path)
 * #include "lgpsf_hessian/impl/fit_impl.hpp"           (lgpsf fit stage)
 */

#endif /* LGPSF_HESSIAN_IMPL_HPP */

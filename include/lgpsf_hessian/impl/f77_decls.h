/* f77_decls.h — self-contained BLAS/LAPACK (and, with the ScaLAPACK
 * backend, BLACS/ScaLAPACK) Fortran declarations.
 *
 * Deliberately NOT petscblaslapack.h: consumers (notably TUs that also
 * include the p4est/sc bundle) may already declare these symbols, and
 * identical plain extern declarations coexist where macro-heavy wrappers
 * collide.  Override LGH_F77_FUNC if your Fortran mangles differently than
 * lowercase-with-underscore.
 *
 * Integer convention: 32-bit (LP64) BLAS/LAPACK, the default everywhere we
 * build; block sizes here are ell-scale, far below 2^31. */

#ifndef LGPSF_HESSIAN_F77_DECLS_H
#define LGPSF_HESSIAN_F77_DECLS_H

#ifndef LGH_F77_FUNC
#define LGH_F77_FUNC(name, NAME) name##_
#endif

#define LGH_BLAS_DGEMM    LGH_F77_FUNC (dgemm, DGEMM)
#define LGH_BLAS_DGEMV    LGH_F77_FUNC (dgemv, DGEMV)
#define LGH_BLAS_DTRSM    LGH_F77_FUNC (dtrsm, DTRSM)
#define LGH_LAPACK_DSYEVD LGH_F77_FUNC (dsyevd, DSYEVD)
#define LGH_LAPACK_DPOTRF LGH_F77_FUNC (dpotrf, DPOTRF)

#ifdef __cplusplus
extern "C"
{
#endif

extern void LGH_BLAS_DGEMM (const char *transa, const char *transb,
                            const int *m, const int *n, const int *k,
                            const double *alpha, const double *a,
                            const int *lda, const double *b, const int *ldb,
                            const double *beta, double *c, const int *ldc);
extern void LGH_BLAS_DGEMV (const char *trans, const int *m, const int *n,
                            const double *alpha, const double *a,
                            const int *lda, const double *x, const int *incx,
                            const double *beta, double *y, const int *incy);
extern void LGH_BLAS_DTRSM (const char *side, const char *uplo,
                            const char *transa, const char *diag,
                            const int *m, const int *n, const double *alpha,
                            const double *a, const int *lda, double *b,
                            const int *ldb);
/* divide-and-conquer symmetric eig: ~6-7x faster than dsyev at ell ~ 2000+ */
extern void LGH_LAPACK_DSYEVD (const char *jobz, const char *uplo,
                               const int *n, double *a, const int *lda,
                               double *w, double *work, const int *lwork,
                               int *iwork, const int *liwork, int *info);
extern void LGH_LAPACK_DPOTRF (const char *uplo, const int *n, double *a,
                               const int *lda, int *info);

#ifdef __cplusplus
}
#endif

#endif /* LGPSF_HESSIAN_F77_DECLS_H */

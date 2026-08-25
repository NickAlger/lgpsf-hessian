# Build warnings seen in consumers

Warnings this library emits when compiled into a downstream project, with the analysis so the
next person does not have to redo it. Add to this file rather than re-investigating.

---

## `'Ue' may be used uninitialized` — `correct_impl.h:344`

**Status: false positive. Harmless, but worth silencing.**

Seen 2026-08-25 building `ymir-uqice/example/antarctica/antarctica_brlr` (gcc 9.4, `-O2 -Wall`,
`-DLGH_WITH_SCALAPACK` via `-DANTARCTICA_WITH_SCALAPACK`):

```
include/lgpsf_hessian/impl/correct_impl.h: In function
  'PetscErrorCode lgh_correct_price_and_graft(lgh_glr_t*, const lgh_glr_correct_opts_t*,
   lgh_glr_correct_report_t*, Mat, Mat)':
include/lgpsf_hessian/impl/correct_impl.h:344:10: warning: 'Ue' may be used uninitialized
  in this function [-Wmaybe-uninitialized]
```

### The code

`lgh_glr_graft`, `correct_impl.h:338-346` (attributed to `lgh_correct_price_and_graft` in the
diagnostic because `lgh_glr_graft` is inlined into it):

```c
#ifdef LGH_WITH_SCALAPACK
  if (g->U == NULL) {   /* ScaLAPACK representation: make U explicit, drop
                         * the sketch state (extension is over anyway).    */
    Mat                 Ue;
    PetscCall (lgh_glrd_materialize_U (g, &Ue));
    lgh_glrd_destroy_state (g);
    g->U = Ue;                                    /* <- line 344 */
  }
#endif
```

### Why it is safe

`lgh_glrd_materialize_U` (`glr_scalapack_impl.h:1029`) has exactly one success path, and it
assigns the out-parameter on it:

```c
  ...
  PetscCall (PetscFree2 (cl, cf));
  *Uout = U;
  return PETSC_SUCCESS;
```

Every earlier exit from that function is a `PetscCall` / `PetscCallMPI`, which returns a nonzero
error code. The caller wraps the call in `PetscCall` too, so on any such failure the caller
returns before reaching line 344. Therefore `Ue` is always written before it is read.

GCC cannot see this because the relationship it would have to prove — "`PetscCall` propagates
the error, and the callee assigns `*Uout` on every path that reaches `PETSC_SUCCESS`" — crosses
an early-return macro and several opaque calls (`MPI_Allreduce`, the BLAS `dgemm`). This is the
same shape as the classic out-parameter-plus-error-macro false positive.

### Suggested fix

One line, zero cost, and it makes the invariant explicit:

```c
    Mat                 Ue = NULL;
```

Worth doing because the warning fires on **every** build of a ScaLAPACK-enabled consumer, and a
recurring known-benign warning is exactly the cover under which a real one goes unnoticed.

### Why nobody hit this

`lgh_glr_graft` is reached only through the stage-2.5 correction layer
(`lgh_glr_correct_probes`). As of 2026-08-25 that layer is **not wired up** in the ymir
consumer — the driver calls `lgh_fit_*`, `lgh_prior_*`, `lgh_glr_compute` and `lgh_glr_solve`,
but never the correction entry points. So the code compiles but does not run there.

Two consequences: this path has less runtime exposure than the rest of the library, and if
stage 2.5 is ever turned on it should get a careful read first rather than being trusted by
association.

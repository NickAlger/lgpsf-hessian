# Platform notes

Hard-won operational knowledge. Each item cost real debugging time in the
engine's originating production deployment; read this before filing a bug.

## ScaLAPACK on 2D process grids: the ZeroByteTypeBug SIGFPE

**Symptom**: SIGFPE (integer divide-by-zero) inside MPICH's
`MPIR_Datatype_get_density`, during `pdsyevd` on a 2D BLACS process grid.

**Root cause** (diagnosed 2026-08-16): on 2D grids, `pdlarfb`'s row-scope BLACS
broadcast of the trapezoidal `T` factor can produce an EMPTY piece on some rank.
BLACS then sends a zero-byte derived datatype, and MPICH (ch3 device; observed
on 4.2.1) hits an unguarded division `size / num_contig_blocks = 0/0`. BLACS
has carried a workaround since the MPICH-1 era — compile ScaLAPACK with
`-DZeroByteTypeBug`, which substitutes a zero-count `MPI_BYTE` — but neither
the reference CMake build nor PETSc's `--download-scalapack` enables it.

**What the library does**: the BLACS grid defaults to `1 × P`
(`opts.grid2d = 0`), which never triggers the bug. Enable the near-square 2D
grid (`opts.grid2d = 1`, faster at scale) only with a ScaLAPACK known to be
healthy: MKL's, or a reference build recompiled with `-DZeroByteTypeBug`
(edit `CMakeCache` of the ScaLAPACK build; redo after any PETSc reconfigure).

## Floating-point bit-identity: within a build, not across builds

With hashed probes/sketches and exact operator responses, fits and builds are
bitwise identical **across rank counts** within one binary. They are NOT stable
across compilers, optimization flags, or code layout: reordered FP reductions
shift results by an ULP, and on a small fraction of rows one ULP sends the
per-row Levenberg–Marquardt into a different local minimum (observed rate
~0.08% of rows with `-march=native`). Validation discipline: compare
invariants and QC metrics across builds, exact bits only within one.

## Build memory

Translation units that include `impl.hpp` with the fit stage enabled are
Eigen-heavy: expect **~3 GB peak memory per TU** at `-O2`. Use a low parallel
job count — `-j 2` is safe on a 16 GB machine; scale by available RAM, not
cores. (CI builds with `-j 2`.)

Also build the library optimized: the per-row fit is 10–50× slower at `-O0`.

## Probe demand grows with dimension and wedge order

The per-row fitter's mode-counting rule requires enough probes per candidate
mode set. In 2D with the default `wedge_order = 10`, pools of `k0 ≥ ~12` work
and the production default is `k0 = 15`. In 3D the mode sets are larger: either
raise `k0`/`k_max` or lower `wedge_order` (the library's 3D tests use
`wedge_order = 2`, `k0 = 20`). If every row fails the counting rule the fit
returns error 6 and prints the row-level reason.

## MPI flavor mismatches

Link everything — PETSc, ScaLAPACK, this library's consumers — against the SAME
MPI. PETSc built with `--download-mpich` reuses one MPICH across arches; point
CMake at it explicitly when the system also has OpenMPI:

```sh
cmake -B build \
  -DMPI_C_COMPILER=$PETSC_DIR/$PETSC_ARCH/bin/mpicc \
  -DMPI_CXX_COMPILER=$PETSC_DIR/$PETSC_ARCH/bin/mpicxx \
  -DMPIEXEC_EXECUTABLE=$PETSC_DIR/$PETSC_ARCH/bin/mpiexec
```

The symptom of a mismatch is a clean `#error` from `petscsys.h` if you are
lucky, and silent corruption if you are not.

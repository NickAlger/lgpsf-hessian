# Tests land with migration slices B-E:
#   - GLR engine gates on synthetic known-spectrum operators (BCGS2
#     orthonormality; pdsyevd vs replicated dsyevd; backend cross-checks by
#     invariants; spectral-function/factor identities; extend-vs-oneshot;
#     FLIP on an indefinite operator; logdet vs dense reference)
#   - fit gates in d=2 and d=3 (+ the n=1 legacy-probes bitwise gate)
#   - prior: Chebyshev vs KSP reference; blocked == loop
# MPI variants run at n=1,2,4 via ${MPIEXEC_EXECUTABLE}.
#
# Build memory: each test TU that includes impl.hpp peaks ~3 GB; keep -j <= 2.

# GLR engine gates (dense-LAPACK oracles, run at n = 1, 2, 4).  Fit and
# prior/zsolve tests arrive with slices D-E.
#
# Build memory: each test TU that includes impl.hpp will become Eigen-heavy
# once the fit stage lands (~3 GB peak); keep -j <= 2.

add_executable(test_glr_replicated test_glr_replicated.cpp)
target_link_libraries(test_glr_replicated PRIVATE lgpsf_hessian::glr)

add_test(NAME glr_replicated_n1 COMMAND test_glr_replicated)
foreach(np 2 4)
  add_test(NAME glr_replicated_n${np}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                   $<TARGET_FILE:test_glr_replicated>)
endforeach()

# GLR engine gates (dense-LAPACK oracles, both backends, run at n = 1, 2, 4).
# Fit and prior/zsolve tests arrive with slices D-E.
#
# Build memory: each test TU that includes impl.hpp will become Eigen-heavy
# once the fit stage lands (~3 GB peak); keep -j <= 2.

add_executable(test_glr test_glr.cpp)
target_link_libraries(test_glr PRIVATE lgpsf_hessian::glr)

add_test(NAME glr_n1 COMMAND test_glr)
foreach(np 2 4)
  add_test(NAME glr_n${np}
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                   $<TARGET_FILE:test_glr>)
endforeach()

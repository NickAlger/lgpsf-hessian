# GLR engine gates (dense-LAPACK oracles, both backends, run at n = 1, 2, 4).
# Fit and prior/zsolve tests arrive with slices D-E.
#
# Build memory: each test TU that includes impl.hpp will become Eigen-heavy
# once the fit stage lands (~3 GB peak); keep -j <= 2.

set(_lgh_tests glr prior_mat deflate)

foreach(t test_glr test_prior_mat test_deflate)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE lgpsf_hessian::glr)
endforeach()

if(TARGET lgpsf_hessian::lgpsf_hessian)
  add_executable(test_fit test_fit.cpp)
  target_link_libraries(test_fit PRIVATE lgpsf_hessian::lgpsf_hessian)
  list(APPEND _lgh_tests fit)
endif()

foreach(t IN LISTS _lgh_tests)
  add_test(NAME ${t}_n1 COMMAND test_${t})
  foreach(np 2 4)
    add_test(NAME ${t}_n${np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                     $<TARGET_FILE:test_${t}>)
  endforeach()
endforeach()

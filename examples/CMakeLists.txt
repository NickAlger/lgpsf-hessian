if(TARGET lgpsf_hessian::lgpsf_hessian)
  add_executable(heat_source_inversion heat/heat_source_inversion.cpp)
  target_link_libraries(heat_source_inversion
    PRIVATE lgpsf_hessian::lgpsf_hessian)
  # run from examples/heat so the committed source_*.pgm files are found
  set_target_properties(heat_source_inversion PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/heat)
else()
  message(STATUS "examples: lgpsf not available, skipping heat example")
endif()

# Blocked prior Z-solve benchmark (PETSc only): hierarchy quality vs grid size,
# domain size, near-null shift, rank count and hypre's coarsening knobs.
add_executable(prior_zsolve_bench prior_bench/prior_zsolve_bench.cpp)
target_link_libraries(prior_zsolve_bench PRIVATE lgpsf_hessian::glr)

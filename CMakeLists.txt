cmake_minimum_required(VERSION 3.16)

# Version is single-sourced in include/lgpsf_hessian/lgpsf_hessian.h.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/lgpsf_hessian/lgpsf_hessian.h"
     _lgh_header)
string(REGEX MATCH "LGPSF_HESSIAN_VERSION_MAJOR ([0-9]+)" _ "${_lgh_header}")
set(_lgh_major "${CMAKE_MATCH_1}")
string(REGEX MATCH "LGPSF_HESSIAN_VERSION_MINOR ([0-9]+)" _ "${_lgh_header}")
set(_lgh_minor "${CMAKE_MATCH_1}")
string(REGEX MATCH "LGPSF_HESSIAN_VERSION_PATCH ([0-9]+)" _ "${_lgh_header}")
set(_lgh_patch "${CMAKE_MATCH_1}")

project(lgpsf_hessian
        VERSION "${_lgh_major}.${_lgh_minor}.${_lgh_patch}"
        LANGUAGES C CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(_lgh_standalone ON)
else()
  set(_lgh_standalone OFF)
endif()

option(LGH_BUILD_TESTS "Build lgpsf-hessian tests" ${_lgh_standalone})
option(LGH_BUILD_EXAMPLES "Build lgpsf-hessian examples" ${_lgh_standalone})
option(LGH_WITH_SCALAPACK
       "Enable the distributed (ScaLAPACK) eigensolver backend" ON)

# ---- dependencies -----------------------------------------------------------

find_package(MPI REQUIRED COMPONENTS C CXX)

# PETSc ships pkg-config metadata, not a CMake config.
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

# ScaLAPACK: commonly built by PETSc (--download-scalapack) into its lib dir.
if(LGH_WITH_SCALAPACK)
  find_library(LGH_SCALAPACK_LIBRARY
               NAMES scalapack scalapack-openmpi scalapack-mpich
               HINTS ${PETSC_LIBRARY_DIRS})
  if(NOT LGH_SCALAPACK_LIBRARY)
    message(FATAL_ERROR
      "ScaLAPACK not found. Point LGH_SCALAPACK_LIBRARY at it, or configure "
      "with -DLGH_WITH_SCALAPACK=OFF (the replicated backend needs only "
      "LAPACK).")
  endif()
endif()

# lgpsf (header-only; brings ellipsoid_tree and Eigen) — needed by the fit
# component only.  find_package first; FetchContent pin to follow with CI.
find_package(lgpsf 0.1 QUIET CONFIG)
if(NOT lgpsf_FOUND)
  message(STATUS
    "lgpsf not found: the fit component (lgpsf_hessian::fit) is unavailable. "
    "Set lgpsf_DIR (or CMAKE_PREFIX_PATH) to an installed lgpsf to enable it.")
endif()

# ---- targets ----------------------------------------------------------------

# Stage 2/3: the GLR engines + prior machinery.  PETSc to the bone; no lgpsf.
add_library(lgpsf_hessian_glr INTERFACE)
add_library(lgpsf_hessian::glr ALIAS lgpsf_hessian_glr)
target_include_directories(lgpsf_hessian_glr INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(lgpsf_hessian_glr INTERFACE
  PkgConfig::PETSC MPI::MPI_C)
target_compile_features(lgpsf_hessian_glr INTERFACE c_std_99)
if(LGH_WITH_SCALAPACK)
  target_link_libraries(lgpsf_hessian_glr INTERFACE ${LGH_SCALAPACK_LIBRARY})
  target_compile_definitions(lgpsf_hessian_glr INTERFACE LGH_WITH_SCALAPACK)
endif()

# Stage 1: the fit.  MPI + lgpsf (Eigen transitively); PETSc only at the
# lgh_fit_get_mat seam, which lives in glr.h.
if(lgpsf_FOUND)
  add_library(lgpsf_hessian_fit INTERFACE)
  add_library(lgpsf_hessian::fit ALIAS lgpsf_hessian_fit)
  target_include_directories(lgpsf_hessian_fit INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(lgpsf_hessian_fit INTERFACE
    lgpsf::lgpsf MPI::MPI_CXX)
  target_compile_features(lgpsf_hessian_fit INTERFACE cxx_std_17)

  # The umbrella: the whole pipeline.
  add_library(lgpsf_hessian INTERFACE)
  add_library(lgpsf_hessian::lgpsf_hessian ALIAS lgpsf_hessian)
  target_link_libraries(lgpsf_hessian INTERFACE
    lgpsf_hessian_glr lgpsf_hessian_fit)
endif()

# ---- tests / examples -------------------------------------------------------
# NOTE (build memory): TUs including impl.hpp are Eigen-heavy, ~3 GB peak
# each at -O2.  Build with -j 2 or less; CI must do the same.

if(LGH_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(LGH_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
  add_subdirectory(examples)
endif()

# ---- install / export -------------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/lgpsf_hessian
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS lgpsf_hessian_glr EXPORT lgpsf_hessianTargets)
if(lgpsf_FOUND)
  install(TARGETS lgpsf_hessian_fit lgpsf_hessian
          EXPORT lgpsf_hessianTargets)
endif()
install(EXPORT lgpsf_hessianTargets
        NAMESPACE lgpsf_hessian::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lgpsf_hessian)
configure_package_config_file(
  cmake/lgpsf_hessianConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lgpsf_hessianConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lgpsf_hessian)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/lgpsf_hessianConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion
  ARCH_INDEPENDENT)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/lgpsf_hessianConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/lgpsf_hessianConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lgpsf_hessian)

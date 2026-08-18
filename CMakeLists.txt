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
option(LGH_FETCH_LGPSF
       "FetchContent lgpsf when find_package misses" OFF)

# ---- dependencies -----------------------------------------------------------

find_package(MPI REQUIRED COMPONENTS C CXX)

# PETSc ships pkg-config metadata, not a CMake config.
find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)

# The GLR engines call BLAS/LAPACK directly (dgemm/dtrsm/dsyevd/dpotrf).
find_package(LAPACK REQUIRED)

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
# component only.  find_package first, then a pinned FetchContent fallback;
# local checkouts via -DFETCHCONTENT_SOURCE_DIR_LGPSF=/path/to/lgpsf (and
# -DFETCHCONTENT_SOURCE_DIR_ELLIPSOID_TREE for its dependency).
find_package(lgpsf 0.1 QUIET CONFIG)
set(_lgh_lgpsf_vendored OFF)
if(NOT lgpsf_FOUND AND LGH_FETCH_LGPSF)
  include(FetchContent)
  # The lgpsf commit pinned below still pins ellipsoid_tree v0.2.0, which
  # predates box_forest.hpp — a header lgpsf's MPI layer includes (lgpsf's
  # own CI builds without MPI and never notices; we compile that layer).
  # First-declaration-wins: declaring ellipsoid_tree here overrides lgpsf's
  # pin.  Drop this once the lgpsf pin below moves past its v0.3.0 bump.
  FetchContent_Declare(ellipsoid_tree
    URL https://github.com/NickAlger/ellipsoid_tree/archive/refs/tags/v0.3.0.tar.gz
    URL_HASH SHA256=3ff1756630faf9897317900c066e747354bdba5223fb1cdeb468aed19cfde8b9)
  FetchContent_Declare(lgpsf
    GIT_REPOSITORY https://github.com/NickAlger/lgpsf.git
    # pinned commit (lgpsf main as validated with this repo's test suite);
    # move to a release tag when lgpsf cuts one
    GIT_TAG        572c935258644ab2d5c980278932d5271bada745)
  set(LGPSF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LGPSF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(lgpsf)
  set(lgpsf_FOUND ON)
  set(_lgh_lgpsf_vendored ON)   # vendored targets cannot be re-exported:
                                # the fit component is usable in-tree but
                                # not installed (lgpsf's own policy)
endif()
if(NOT lgpsf_FOUND)
  message(STATUS
    "lgpsf not found: the fit component (lgpsf_hessian::fit) is unavailable. "
    "Set lgpsf_DIR/CMAKE_PREFIX_PATH to an installed lgpsf, or configure "
    "with -DLGH_FETCH_LGPSF=ON (optionally with FETCHCONTENT_SOURCE_DIR_"
    "LGPSF pointing at a local checkout).")
endif()

# ---- targets ----------------------------------------------------------------

# Stage 2/3: the GLR engines + prior machinery.  PETSc to the bone; no lgpsf.
add_library(lgpsf_hessian_glr INTERFACE)
add_library(lgpsf_hessian::glr ALIAS lgpsf_hessian_glr)
target_include_directories(lgpsf_hessian_glr INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(lgpsf_hessian_glr INTERFACE
  PkgConfig::PETSC MPI::MPI_C ${LAPACK_LIBRARIES})
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
  # impl.hpp includes the (Eigen-heavy) fit implementation only when this
  # macro is present, so glr-only TUs stay light to compile.
  target_compile_definitions(lgpsf_hessian_fit INTERFACE LGH_HAVE_LGPSF)

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
if(lgpsf_FOUND AND NOT _lgh_lgpsf_vendored)
  install(TARGETS lgpsf_hessian_fit lgpsf_hessian
          EXPORT lgpsf_hessianTargets)
elseif(_lgh_lgpsf_vendored)
  message(STATUS
    "lgpsf is vendored (FetchContent): installing the glr component only; "
    "install lgpsf properly to install the fit component.")
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

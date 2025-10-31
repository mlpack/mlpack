# - Find mlpack
# Find the mlpack C++ library
#
# author Omar Shrit

##===================================================
##  FUNCTION DOCUMENTATION
##===================================================
#
# find_mlpack()
#----------------------
#
# Call this macro to find mlpack and its dependencies (Armadillo, ensmallen,
# cereal, and any Armadillo dependencies).
#
# This function will not automatically download any missing dependencies, and
# will instead throw errors if dependencies are not found.  For a version that
# automatically downloads dependencies, see `fetch_mlpack()`.
#
# Configuration options:
#
#   MLPACK_DISABLE_OPENMP: if set, parallelism via OpenMP will be disabled.
#   MLPACK_USE_SYSTEM_STB: if set, STB will be searched for on the system,
#       instead of using the version bundled with mlpack.
#   MLPACK_DONT_FIND_MLPACK: if set, mlpack itself will not be searched
#       for---only its dependencies.
#
# If mlpack is successfully found, the `MLPACK_FOUND` variable will be set to
# `TRUE`; otherwise, it will be set to `FALSE`.
#
# This macro will set the following variables:
#
# MLPACK_INCLUDE_DIRS: list of all include directories for mlpack and its
#                      dependencies (Armadillo, cereal, ensmallen)
# MLPACK_LIBRARIES: list of all dependency libraries to link against (typically
#                   just OpenBLAS)
#
#
# fetch_mlpack(COMPILE_OPENBLAS)
#-----------------------
#
# This macro downloads the mlpack library and its dependencies.  Call this
# function to find mlpack and its dependencies (Armadillo, ensmallen, cereal) on
# a system where mlpack or those dependencies may not be available.
#
# fetch_mlpack() accepts one parameter, `COMPILE_OPENBLAS`.  When this is set to
# `TRUE`, then OpenBLAS will be downloaded and compiled as a dependency of
# Armadillo.  If `COMPILE_OPENBLAS` is set to `FALSE`, then it is expected that
# OpenBLAS or a BLAS/LAPACK library is already available on the system.  When
# CMAKE_CROSSCOMPILING is set, then OpenBLAS is always compiled for the target
# architecture.
#
# Other dependencies of mlpack do not need compilation, as they are all
# header-only.
#
# If mlpack is not found on the system, `fetch_mlpack()` will download the
# latest stable version of mlpack.
#
# Configuration options:
#
#   MLPACK_DISABLE_OPENMP: if set, parallelism via OpenMP will be disabled.
#   MLPACK_USE_SYSTEM_STB: if set, STB will be searched for on the system,
#       instead of using the version bundled with mlpack.
#
# After all libraries are downloaded and set up, the macro will set the
# following variables:
#
# MLPACK_INCLUDE_DIRS: list of all include directories for mlpack and its
#                      dependencies (Armadillo, cereal, ensmallen)
# MLPACK_LIBRARIES: list of all dependency libraries to link against (typically
#                   just OpenBLAS)
#
##===================================================
##  INTERNAL FUNCTION DOCUMENTATION
##===================================================
#
# get_deps(LINK DEPS_NAME PACKAGE)
#-------------------
#
# This macro allows to download dependenices from the link that is provided to
# them. You need to pass the LINK to download from, the name of
# the dependency, and the filename to store the downloaded package to such
# as armadillo.tar.gz and they are downloaded into
# ${CMAKE_BINARY_DIR}/deps/${PACKAGE}
# At each download, this module sets a GENERIC_INCLUDE_DIR path,
# which means that you need to set the main path for the include
# directories for each package.
# Note that, the package should be compressed only as .tar.gz
#
#
# find_armadillo()
#------------------
#
# This macro finds armadillo library, and sets the necessary paths to each
# one of the parameters. If the library is not found this macro will set
# ARMADILLO_FOUND to false.
#
# This macro sets the following variables:
#  ARMADILLO_FOUND - set to true if the library is found
#  ARMADILLO_INCLUDE_DIRS - list of required include directories
#  ARMADILLO_LIBRARIES - list of libraries to be linked
#  ARMADILLO_VERSION_MAJOR - major version number
#  ARMADILLO_VERSION_MINOR - minor version number
#  ARMADILLO_VERSION_PATCH - patch version number
#  ARMADILLO_VERSION_STRING - version number as a string (ex: "1.0.4")
#  ARMADILLO_VERSION_NAME - name of the version (ex: "Antipodean Antileech")
#
#
# find_ensmallen
#------------------
#
# This macro finds ensmallen library and sets the necessary paths to each
# one of the parameters. If the library is not found this function will set
# ENSMALLEN_FOUND to false.
#
# This module sets the following variables:
#  ENSMALLEN_FOUND - set to true if the library is found
#  ENSMALLEN_INCLUDE_DIR - list of required include directories
#  ENSMALLEN_VERSION_MAJOR - major version number
#  ENSMALLEN_VERSION_MINOR - minor version number
#  ENSMALLEN_VERSION_PATCH - patch version number
#  ENSMALLEN_VERSION_STRING - version number as a string (ex: "1.0.4")
#  ENSMALLEN_VERSION_NAME - name of the version (ex: "Antipodean Antileech")
#
#
# find_cereal()
#------------------
#
# This macro finds cereal library and sets the necessary paths to each
# one of the parameters. If the library is not found this macro will set
# CEREAL_FOUND to false.

# This module sets the following variables:
#  CEREAL_FOUND - set to true if the library is found
#  CEREAL_INCLUDE_DIR - list of required include directories
#  CEREAL_VERSION_MAJOR - major version number
#  CEREAL_VERSION_MINOR - minor version number
#  CEREAL_VERSION_PATCH - patch version number
#  CEREAL_VERSION_STRING - version number as a string (ex: "1.0.4")
#
#
# find stb()
#------------------
#
# This macro finds STB library and sets the necessary paths to each
# one of the parameters. If the library is not found this macro will set
# STB_FOUND to false.
#
# - Find STB_IMAGE
#
# This module sets the following variables:
#  STB_IMAGE_FOUND - set to true if the library is found
#  STB_IMAGE_INCLUDE_DIR - list of required include directories
#  STB_INCLUDE_NEEDS_STB_SUFFIX - whether or not the include files are under an
#     stb/ directory; if "YES", then includes must be done as, e.g.,
#     stb/stb_image.h.
#
#
# find_openmp()
#-----------------------
#
# This macro finds if OpenMP library is installed and supported by the
# compiler. if the library found then it sets the following parameters:
#
# OpenMP_FOUND - set to true if the library is found
#
#
# find_mlpack_internal()
#-----------------------
#
# This macro finds the mlpack library and sets the necessary paths to each one
# of the parameters. If the library is not found this macro will set
# MLPACK_FOUND to false.
#
# This module sets the following variables:
#  MLPACK_FOUND - set to true if the library is found
#  MLPACK_VERSION_MAJOR - major version number
#  MLPACK_VERSION_MINOR - minor version number
#  MLPACK_VERSION_PATCH - patch version number
#  MLPACK_VERSION_STRING - version number as a string (ex: "1.0.4")
#  MLPACK_INCLUDE_DIR - list of mlpack include directories
#
#
##===================================================
##  MLPACK DEPENDENCIES SETTINGS.
##===================================================

# Set minimum library versions required by mlpack.
#
# For Armadillo, try to keep the minimum required version less than or equal to
# what's available on the current Ubuntu LTS or most recent stable RHEL release.
# See https://github.com/mlpack/mlpack/issues/3033 for some more discussion.
set(ARMADILLO_VERSION "10.8.2")
set(ENSMALLEN_VERSION "2.10.0")
set(CEREAL_VERSION "1.1.2")
set(OPENBLAS_VERSION "0.3.29")

# Set library version to be used when fetching them from the source.
set(ARMADILLO_FETCH_VERSION "12.6.5")
set(ENSMALLEN_FETCH_VERSION "latest")
set(CEREAL_FETCH_VERSION "1.3.2")
set(MLPACK_FETCH_VERSION "latest")

# Set required standard to C++17, if it's not already set.
if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()

if (NOT MLPACK_DISABLE_OPENMP)
  set(MLPACK_DISABLE_OPENMP OFF)
endif ()

##===================================================
##  MLPACK AUTODOWNLOADER DEPENDENCIES FUNCTIONS
##===================================================

# This function auto-downloads mlpack dependencies.
macro(get_deps LINK DEPS_NAME PACKAGE)
  if (NOT EXISTS "${CMAKE_BINARY_DIR}/deps/${PACKAGE}")
    file(DOWNLOAD ${LINK}
           "${CMAKE_BINARY_DIR}/deps/${PACKAGE}"
            STATUS DOWNLOAD_STATUS_LIST LOG DOWNLOAD_LOG
            SHOW_PROGRESS)
    list(GET DOWNLOAD_STATUS_LIST 0 DOWNLOAD_STATUS)
    if (DOWNLOAD_STATUS EQUAL 0)
      execute_process(COMMAND ${CMAKE_COMMAND} -E
          tar xf "${CMAKE_BINARY_DIR}/deps/${PACKAGE}"
          WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/deps/")
    else ()
      list(GET DOWNLOAD_STATUS_LIST 1 DOWNLOAD_ERROR)
      message(FATAL_ERROR
          "Could not download ${DEPS_NAME}! Error code ${DOWNLOAD_STATUS}: ${DOWNLOAD_ERROR}!  Error log: ${DOWNLOAD_LOG}")
    endif()
  endif()
  # Get the name of the directory.
  file (GLOB DIRECTORIES RELATIVE "${CMAKE_BINARY_DIR}/deps/"
      "${CMAKE_BINARY_DIR}/deps/${DEPS_NAME}*.*")
  if(${DEPS_NAME} MATCHES "stb")
    file (GLOB DIRECTORIES RELATIVE "${CMAKE_BINARY_DIR}/deps/"
        "${CMAKE_BINARY_DIR}/deps/${DEPS_NAME}")
  endif()
  # list(FILTER) is not available on 3.5 or older, but try to keep
  # configuring without filtering the list anyway
  # (it works only if the file is present as .tar.gz).
  list(FILTER DIRECTORIES EXCLUDE REGEX ".*\.tar\.gz")
  list(LENGTH DIRECTORIES DIRECTORIES_LEN)
  if (DIRECTORIES_LEN GREATER 0)
    list(GET DIRECTORIES 0 DEPENDENCY_DIR)
    set(GENERIC_INCLUDE_DIR "${CMAKE_BINARY_DIR}/deps/${DEPENDENCY_DIR}/include")
    install(DIRECTORY "${GENERIC_INCLUDE_DIR}/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
  else ()
    message(FATAL_ERROR
            "Problem unpacking ${DEPS_NAME}! Expected only one directory "
            "${DEPS_NAME};. Try to remove the directory ${CMAKE_BINARY_DIR}/deps and reconfigure.")
  endif ()
endmacro()

macro(find_armadillo)
  cmake_policy(PUSH)
  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.29")
    cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>
  endif()

  set(CURRENT_PATH ${ARGN})
  if (CURRENT_PATH)
    find_path(ARMADILLO_INCLUDE_DIR
      NAMES armadillo
      PATHS "${CURRENT_PATH}/deps/Armadillo/include"
      NO_DEFAULT_PATH)
  else()
    find_path(ARMADILLO_INCLUDE_DIR
      NAMES armadillo
      PATHS "$ENV{ProgramFiles}/Armadillo/include")
  endif()
  if (ARMADILLO_INCLUDE_DIR)
    # ------------------------------------------------------------------------
    #  Extract version information from <armadillo>
    # ------------------------------------------------------------------------

    # WARNING: Early releases of Armadillo didn't have the arma_version.hpp file.
    # (e.g. v.0.9.8-1 in ubuntu maverick packages (2001-03-15))
    # If the file is missing, set all values to 0
    set(ARMADILLO_VERSION_MAJOR 0)
    set(ARMADILLO_VERSION_MINOR 0)
    set(ARMADILLO_VERSION_PATCH 0)
    set(ARMADILLO_VERSION_NAME "EARLY RELEASE")

    if (EXISTS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/arma_version.hpp")

      # Read and parse armdillo version header file for version number
      file(STRINGS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/arma_version.hpp" _ARMA_HEADER_CONTENTS REGEX "#define ARMA_VERSION_[A-Z]+ ")
      string(REGEX REPLACE ".*#define ARMA_VERSION_MAJOR ([0-9]+).*" "\\1" ARMADILLO_VERSION_MAJOR "${_ARMA_HEADER_CONTENTS}")
      string(REGEX REPLACE ".*#define ARMA_VERSION_MINOR ([0-9]+).*" "\\1" ARMADILLO_VERSION_MINOR "${_ARMA_HEADER_CONTENTS}")
      string(REGEX REPLACE ".*#define ARMA_VERSION_PATCH ([0-9]+).*" "\\1" ARMADILLO_VERSION_PATCH "${_ARMA_HEADER_CONTENTS}")

      # WARNING: The number of spaces before the version name is not one.
      string(REGEX REPLACE ".*#define ARMA_VERSION_NAME\ +\"([0-9a-zA-Z\ _-]+)\".*" "\\1" ARMADILLO_VERSION_NAME "${_ARMA_HEADER_CONTENTS}")
      set(ARMADILLO_FOUND YES)
    endif()

    set(ARMADILLO_VERSION_STRING "${ARMADILLO_VERSION_MAJOR}.${ARMADILLO_VERSION_MINOR}.${ARMADILLO_VERSION_PATCH}")
  endif()

  if (EXISTS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp")
    file(STRINGS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp" _ARMA_CONFIG_CONTENTS REGEX "^#define ARMA_USE_[A-Z]+")
    string(REGEX MATCH "ARMA_USE_WRAPPER" _ARMA_USE_WRAPPER "${_ARMA_CONFIG_CONTENTS}")
    string(REGEX MATCH "ARMA_USE_LAPACK" _ARMA_USE_LAPACK "${_ARMA_CONFIG_CONTENTS}")
    string(REGEX MATCH "ARMA_USE_BLAS" _ARMA_USE_BLAS "${_ARMA_CONFIG_CONTENTS}")
    string(REGEX MATCH "ARMA_USE_ARPACK" _ARMA_USE_ARPACK "${_ARMA_CONFIG_CONTENTS}")
    string(REGEX MATCH "ARMA_USE_HDF5" _ARMA_USE_HDF5 "${_ARMA_CONFIG_CONTENTS}")
  endif()

  # If _ARMA_USE_WRAPPER is set, then we just link to armadillo, but if it's not then we need support libraries instead
  set(_ARMA_SUPPORT_LIBRARIES)

  if(_ARMA_USE_WRAPPER)
    # Link to the armadillo wrapper library.
    find_library(ARMADILLO_LIBRARY
      NAMES armadillo
      NAMES_PER_DIR
      PATHS
        "$ENV{ProgramFiles}/Armadillo/lib"
        "$ENV{ProgramFiles}/Armadillo/lib64"
        "$ENV{ProgramFiles}/Armadillo"
      )
    set(_ARMA_REQUIRED_VARS ARMADILLO_LIBRARY)
  else()
    set(ARMADILLO_LIBRARY "")
  endif()

  # Transitive linking with the wrapper does not work with MSVC,
  # so we must *also* link against Armadillo's dependencies.
  if(NOT _ARMA_USE_WRAPPER OR MSVC)
    # Link directly to individual components.
    foreach(pkg
        LAPACK
        BLAS
        ARPACK
        HDF5
        )
      if(_ARMA_USE_${pkg})
        find_package(${pkg})
        list(APPEND _ARMA_REQUIRED_VARS "${pkg}_FOUND")
        if(${pkg}_FOUND)
          list(APPEND _ARMA_SUPPORT_LIBRARIES ${${pkg}_LIBRARIES})
        endif()
      endif()
    endforeach()
  endif()
  if (ARMADILLO_FOUND)
    set(ARMADILLO_INCLUDE_DIRS ${ARMADILLO_INCLUDE_DIR})
    set(ARMADILLO_LIBRARIES ${ARMADILLO_LIBRARY} ${_ARMA_SUPPORT_LIBRARIES})
  endif()
  # Clean up internal variables
  unset(_ARMA_REQUIRED_VARS)
  unset(_ARMA_SUPPORT_LIBRARIES)
  unset(_ARMA_USE_WRAPPER)
  unset(_ARMA_USE_LAPACK)
  unset(_ARMA_USE_BLAS)
  unset(_ARMA_USE_ARPACK)
  unset(_ARMA_USE_HDF5)
  unset(_ARMA_CONFIG_CONTENTS)
  unset(_ARMA_HEADER_CONTENTS)

  cmake_policy(POP)
endmacro()

# Findcereal.cmake
macro(find_cereal)

  set(CURRENT_PATH ${ARGN})
  if (CURRENT_PATH)
    find_path(CEREAL_INCLUDE_DIR
      NAMES cereal
      PATHS "${CURRENT_PATH}/deps/cereal/include"
      NO_DEFAULT_PATH)
  else()
    find_path(CEREAL_INCLUDE_DIR
      NAMES cereal
      PATHS "$ENV{ProgramFiles}/cereal/include")
  endif()

  if (CEREAL_INCLUDE_DIR)
    # ------------------------------------------------------------------------
    #  Extract version information from <CEREAL>
    # ------------------------------------------------------------------------
    set(CEREAL_FOUND YES)
    set(CEREAL_VERSION_MAJOR 0)
    set(CEREAL_VERSION_MINOR 0)
    set(CEREAL_VERSION_PATCH 0)

    if (EXISTS "${CEREAL_INCLUDE_DIR}/cereal/version.hpp")

      # Read and parse cereal version header file for version number
      file(READ "${CEREAL_INCLUDE_DIR}/cereal/version.hpp"
          _CEREAL_HEADER_CONTENTS)
      string(REGEX REPLACE ".*#define CEREAL_VERSION_MAJOR ([0-9]+).*" "\\1"
          CEREAL_VERSION_MAJOR "${_CEREAL_HEADER_CONTENTS}")
      string(REGEX REPLACE ".*#define CEREAL_VERSION_MINOR ([0-9]+).*" "\\1"
          CEREAL_VERSION_MINOR "${_CEREAL_HEADER_CONTENTS}")
      string(REGEX REPLACE ".*#define CEREAL_VERSION_PATCH ([0-9]+).*" "\\1"
          CEREAL_VERSION_PATCH "${_CEREAL_HEADER_CONTENTS}")

    elseif (EXISTS "${CEREAL_INCLUDE_DIR}/cereal/details/polymorphic_impl_fwd.hpp")

      set(CEREAL_VERSION_MAJOR 1)
      set(CEREAL_VERSION_MINOR 2)
      set(CEREAL_VERSION_PATCH 0)
    elseif (EXISTS "${CEREAL_INCLUDE_DIR}/cereal/types/valarray.hpp")

      set(CEREAL_VERSION_MAJOR 1)
      set(CEREAL_VERSION_MINOR 1)
      set(CEREAL_VERSION_PATCH 2)
    elseif (EXISTS "${CEREAL_INCLUDE_DIR}/cereal/cereal.hpp")

    set(CEREAL_VERSION_MAJOR 1)
    set(CEREAL_VERSION_MINOR 1)
    set(CEREAL_VERSION_PATCH 1)
  else()

    set(CEREAL_FOUND NO)
    endif()
    set(CEREAL_VERSION_STRING "${CEREAL_VERSION_MAJOR}.${CEREAL_VERSION_MINOR}.${CEREAL_VERSION_PATCH}")
  endif ()

endmacro()

macro(find_ensmallen)

  set(CURRENT_PATH ${ARGN})
  if (CURRENT_PATH)
    find_path(ENSMALLEN_INCLUDE_DIR
      NAMES ensmallen.hpp
      PATHS "${CURRENT_PATH}/deps/ensmallen/include"
      NO_DEFAULT_PATH)
  else()
    file(GLOB ENSMALLEN_SEARCH_PATHS
        ${CMAKE_BINARY_DIR}/deps/ensmallen-[0-9]*.[0-9]*.[0-9]*)
    find_path(ENSMALLEN_INCLUDE_DIR
      NAMES ensmallen.hpp
      PATHS ${ENSMALLEN_SEARCH_PATHS}/include)
  endif()

  if (ENSMALLEN_INCLUDE_DIR)
    # ------------------------------------------------------------------------
    #  Extract version information from <ensmallen>
    # ------------------------------------------------------------------------

    set(ENSMALLEN_VERSION_MAJOR 0)
    set(ENSMALLEN_VERSION_MINOR 0)
    set(ENSMALLEN_VERSION_PATCH 0)
    set(ENSMALLEN_VERSION_NAME "unknown")

    if(EXISTS "${ENSMALLEN_INCLUDE_DIR}/ensmallen_bits/ens_version.hpp")

      set(ENSMALLEN_FOUND YES)

      # Read and parse Ensmallen version header file for version number
      file(READ "${ENSMALLEN_INCLUDE_DIR}/ensmallen_bits/ens_version.hpp"
          _ensmallen_HEADER_CONTENTS)
      string(REGEX REPLACE ".*#define ENS_VERSION_MAJOR ([0-9]+).*" "\\1"
          ENSMALLEN_VERSION_MAJOR "${_ensmallen_HEADER_CONTENTS}")
      string(REGEX REPLACE ".*#define ENS_VERSION_MINOR ([0-9]+).*" "\\1"
          ENSMALLEN_VERSION_MINOR "${_ensmallen_HEADER_CONTENTS}")
      string(REGEX REPLACE ".*#define ENS_VERSION_PATCH ([0-9]+).*" "\\1"
          ENSMALLEN_VERSION_PATCH "${_ensmallen_HEADER_CONTENTS}")

      # WARNING: The number of spaces before the version name is not one.
      string(REGEX REPLACE
          ".*#define ENS_VERSION_NAME\ +\"([0-9a-zA-Z\ _-]+)\".*" "\\1"
          ENSMALLEN_VERSION_NAME "${_ensmallen_HEADER_CONTENTS}")

    endif()

    set(ENSMALLEN_VERSION_STRING "${ENSMALLEN_VERSION_MAJOR}.${ENSMALLEN_VERSION_MINOR}.${ENSMALLEN_VERSION_PATCH}")
  endif ()
endmacro()

macro(find_stb)
  file(GLOB STB_IMAGE_SEARCH_PATHS
      ${CMAKE_BINARY_DIR}/deps/
      ${CMAKE_BINARY_DIR}/deps/stb)
  find_path(STB_IMAGE_INCLUDE_DIR_1
      NAMES stb_image.h stb_image_write.h stb_image_resize2.h
      PATHS ${STB_IMAGE_SEARCH_PATHS} ${STB_IMAGE_INCLUDE_DIR})

  if(STB_IMAGE_INCLUDE_DIR_1)
    set(STB_IMAGE_INCLUDE_DIR "${STB_IMAGE_INCLUDE_DIR_1}" CACHE PATH
        "stb_image include directory")

    # Either we found /usr/include/stb_image.h (or similar), or the user passed
    # a directory in STB_IMAGE_SEARCH_PATHS that directly contains stb_image.h.
    # In either of those cases, we want to include <stb_image.h>, not
    # <stb/stb_image.h>.
    set(STB_INCLUDE_NEEDS_STB_SUFFIX "NO")
  else ()
    find_path(STB_IMAGE_INCLUDE_DIR_2
          NAMES stb_image.h stb_image_write.h stb_image_resize2.h
          PATHS ${STB_IMAGE_SEARCH_PATHS} ${STB_IMAGE_INCLUDE_DIR}
          PATH_SUFFIXES stb/)

    if (STB_IMAGE_INCLUDE_DIR_2)
      set(STB_IMAGE_INCLUDE_DIR "${STB_IMAGE_INCLUDE_DIR_2}" CACHE PATH
          "stb_image include directory")

      # Since we searched the same paths but allowed an stb/ suffix this time,
      # then there is definitely a suffix.
      set(STB_INCLUDE_NEEDS_STB_SUFFIX "YES")
      # Strip the suffix.
      string(REGEX REPLACE ".*stb[/]?$" "" STB_IMAGE_INCLUDE_DIR
          "${STB_IMAGE_INCLUDE_DIR}")
    endif ()
  endif ()

endmacro()

macro(find_openmp)
  find_package(OpenMP)

  if (OpenMP_FOUND AND OpenMP_CXX_VERSION VERSION_GREATER_EQUAL 3.0.0)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
  else ()
    # Disable warnings for all the unknown OpenMP pragmas.
    if (NOT MSVC)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    else ()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")
    endif ()
    set(OpenMP_CXX_FLAGS "")
  endif ()
endmacro()

macro(find_mlpack_internal)

  set(CURRENT_PATH ${ARGN})

  if (CURRENT_PATH)
    file(GLOB MLPACK_SEARCH_PATHS
      ${CURRENT_PATH}/deps/mlpack-[0-9]*.[0-9]*.[0-9]*/src/)

    list(POP_BACK MLPACK_SEARCH_PATHS MLPACK_SEARCH_PATH)
    if (EXISTS ${MLPACK_SEARCH_PATH}/mlpack.hpp)
      set(MLPACK_INCLUDE_DIR ${MLPACK_SEARCH_PATH})
    endif()

  else()
    file(GLOB MLPACK_SEARCH_PATHS
      ${CMAKE_BINARY_DIR}/deps/mlpack-[0-9]*.[0-9]*.[0-9]*)

    # This will be executed if mlpack is installed already.
    find_path(MLPACK_INCLUDE_DIR
      NAMES mlpack.hpp
      PATHS ${MLPACK_SEARCH_PATHS}/include)

    # This will be executed when compiling mlpack bindings and tests.
    if (NOT MLPACK_INCLUDE_DIR)
      find_path(MLPACK_INCLUDE_DIR
        NAMES mlpack.hpp
        PATHS "${CMAKE_CURRENT_SOURCE_DIR}/src/")
    endif()
  endif()

  if (MLPACK_INCLUDE_DIR)
    # ------------------------------------------------------------------------
    #  Extract version information from <mlpack>
    # ------------------------------------------------------------------------

    set(MLPACK_VERSION_MAJOR 0)
    set(MLPACK_VERSION_MINOR 0)
    set(MLPACK_VERSION_PATCH 0)

    if (EXISTS "${MLPACK_INCLUDE_DIR}/mlpack/core/util/version.hpp")

      set(MLPACK_FOUND YES)

      # Read and parse mlpack version header file for version number
      file(READ "${MLPACK_INCLUDE_DIR}/mlpack/core/util/version.hpp"
          _mlpack_HEADER_CONTENTS)
      string(REGEX REPLACE ".*#define MLPACK_VERSION_MAJOR ([0-9]+).*" "\\1"
          MLPACK_VERSION_MAJOR "${_mlpack_HEADER_CONTENTS}")
      string(REGEX REPLACE ".*#define MLPACK_VERSION_MINOR ([0-9]+).*" "\\1"
          MLPACK_VERSION_MINOR "${_mlpack_HEADER_CONTENTS}")
      string(REGEX REPLACE ".*#define MLPACK_VERSION_PATCH ([0-9]+).*" "\\1"
          MLPACK_VERSION_PATCH "${_mlpack_HEADER_CONTENTS}")

    endif()

    set(MLPACK_VERSION_STRING "${MLPACK_VERSION_MAJOR}.${MLPACK_VERSION_MINOR}.${MLPACK_VERSION_PATCH}")
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(mlpack
    REQUIRED_VARS MLPACK_INCLUDE_DIR
    VERSION_VAR MLPACK_VERSION_STRING)

endmacro()

macro(compile_OpenBLAS)
  if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(OPENBLAS_SRC_DIR ${CMAKE_BINARY_DIR}/deps/OpenBLAS-${OPENBLAS_VERSION})
    set(OPENBLAS_BUILD_DIR ${OPENBLAS_SRC_DIR}/build)
    set(OPENBLAS_OUTPUT_LIB_DIR ${OPENBLAS_BUILD_DIR}/lib/Release)
    # always compile BLAS as release.
    set(BLASS_BUILD_TYPE "Release")

    if (NOT EXISTS "${OPENBLAS_OUTPUT_LIB_DIR}/openblas.lib")
      message(STATUS "Compiling OpenBLAS")
      file(MAKE_DIRECTORY ${OPENBLAS_BUILD_DIR})
      # -G -A -T to pass settings from current cmake command.
      execute_process(
              COMMAND ${CMAKE_COMMAND}
              -G "${CMAKE_GENERATOR}"
              -A "${CMAKE_GENERATOR_PLATFORM}"
              -T "${CMAKE_GENERATOR_TOOLSET}"
              "-DCMAKE_BUILD_TYPE=${BLASS_BUILD_TYPE}"
              "-DBUILD_SHARED_LIBS=OFF"
              -S ${OPENBLAS_SRC_DIR} -B ${OPENBLAS_BUILD_DIR}

              WORKING_DIRECTORY ${OPENBLAS_SRC_DIR}
      )
      execute_process(
              COMMAND ${CMAKE_COMMAND} --build ${OPENBLAS_BUILD_DIR}
              --config ${BLASS_BUILD_TYPE} --parallel
              WORKING_DIRECTORY ${OPENBLAS_SRC_DIR}
      )
    else()
      message(STATUS "OpenBLAS is already compiled")
    endif()
    file(GLOB OPENBLAS_LIBRARIES ${OPENBLAS_OUTPUT_LIB_DIR}/openblas.lib)
  else()
    if (NOT EXISTS "${OPENBLAS_OUTPUT_LIB_DIR}/libopenblas.a")
      # Set any extra variables for make that the user specified.
      # First, turn OPENBLAS_EXTRA_ARGS into a list.
      separate_arguments(ARG_LIST NATIVE_COMMAND ${OPENBLAS_EXTRA_ARGS})
      execute_process(
          COMMAND make NO_SHARED=1 ${ARG_LIST}
          WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/deps/OpenBLAS-${OPENBLAS_VERSION})

      file(GLOB OPENBLAS_LIBRARIES "${CMAKE_BINARY_DIR}/deps/OpenBLAS-${OPENBLAS_VERSION}/libopenblas.a")
    endif ()
  endif()
  set(BLAS_openblas_LIBRARY ${OPENBLAS_LIBRARIES})
  set(LAPACK_openblas_LIBRARY ${OPENBLAS_LIBRARIES})
  set(BLAS_FOUND ON)
endmacro()

macro(fetch_mlpack COMPILE_OPENBLAS)

  if (CMAKE_CROSSCOMPILING)
    search_openblas(${OPENBLAS_VERSION})
    # Set to cross compile openblas if the user forgot to do so.
    set(COMPILE_OPENBLAS ON)
  endif()

  if (NOT CMAKE_CROSSCOMPILING)
    # Only search for system BLAS if we know we can use it (e.g. if OpenBLAS
    # doesn't need to be cross-compiled).
    find_package(BLAS QUIET)
  endif ()

  if (NOT BLAS_FOUND)
    # Also search in case we already downloaded it.
    find_package(BLAS PATHS ${CMAKE_BINARY_DIR} QUIET)
  endif()

  if (NOT BLAS_FOUND OR (NOT BLAS_LIBRARIES))
    get_deps(https://github.com/xianyi/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz
        OpenBLAS OpenBLAS-${OPENBLAS_VERSION}.tar.gz)
    if (NOT COMPILE_OPENBLAS)
      message(WARNING "OpenBLAS is downloaded but not compiled. Please compile
      OpenBLAS before compiling mlpack")
    else()
      compile_OpenBLAS()
    endif()
  endif()

  find_armadillo(${CMAKE_BINARY_DIR})
  if (NOT ARMADILLO_FOUND)
    if (NOT CMAKE_CROSSCOMPILING)
      find_package(BLAS QUIET)
      find_package(LAPACK QUIET)
    endif()
    get_deps(https://files.mlpack.org/armadillo-${ARMADILLO_FETCH_VERSION}.tar.gz armadillo armadillo-${ARMADILLO_FETCH_VERSION}.tar.gz)
    set(ARMADILLO_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
    find_armadillo(${CMAKE_BINARY_DIR})
  endif()
  if (ARMADILLO_FOUND)
    # Include directories for the previous dependencies.
    set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
    set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})
  endif()

  find_ensmallen(${CMAKE_BINARY_DIR})
  if (NOT ENSMALLEN_FOUND)
    get_deps(https://www.ensmallen.org/files/ensmallen-${ENSMALLEN_FETCH_VERSION}.tar.gz ensmallen ensmallen-${ENSMALLEN_FETCH_VERSION}.tar.gz)
    set(ENSMALLEN_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
    find_ensmallen(${CMAKE_BINARY_DIR})
  endif()
  if (ENSMALLEN_FOUND)
    set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${ENSMALLEN_INCLUDE_DIR})
  endif()

  find_cereal(${CMAKE_BINARY_DIR})
  if (NOT CEREAL_FOUND)
    get_deps(https://github.com/USCiLab/cereal/archive/refs/tags/v${CEREAL_FETCH_VERSION}.tar.gz cereal cereal-${CEREAL_FETCH_VERSION}.tar.gz)
    set(CEREAL_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
    find_cereal(${CMAKE_BINARY_DIR})
  endif()
  if (CEREAL_FOUND)
    set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${CEREAL_INCLUDE_DIR})
  endif()

  if (NOT MLPACK_DONT_FIND_MLPACK)
    find_mlpack_internal(${CMAKE_BINARY_DIR})
    if (NOT MLPACK_FOUND)
      get_deps(https://www.mlpack.org/files/mlpack-${MLPACK_FETCH_VERSION}.tar.gz mlpack mlpack-${MLPACK_FETCH_VERSION}.tar.gz)
      set(MLPACK_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
      find_mlpack_internal(${CMAKE_BINARY_DIR})
    endif()
    if (MLPACK_FOUND)
      set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${MLPACK_INCLUDE_DIR})
    endif()
  endif()

  if (NOT MLPACK_DISABLE_OPENMP)
    find_openmp()
  endif ()

endmacro()

##===================================================
##  MLPACK MAIN FUNCTIONS CALL.
##===================================================

macro(find_mlpack)
  # If we're using gcc, then we need to link against pthreads to use std::thread,
  # which we do in the tests.
  if (CMAKE_COMPILER_IS_GNUCC)
    find_package(Threads)
    set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
  endif()

  if (NOT MLPACK_DISABLE_OPENMP)
    find_openmp()
  endif ()

  find_armadillo()
  if (ARMADILLO_FOUND)
    set(MLPACK_INCLUDE_DIRS ${ARMADILLO_INCLUDE_DIRS})
    set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})
  else()
    message(FATAL_ERROR "Armadillo not found, (required dependency of mlpack).")
  endif ()

  find_ensmallen()
  if (ENSMALLEN_FOUND)
    set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${ENSMALLEN_INCLUDE_DIR})
  else()
    message(FATAL_ERROR "Ensmallen not found, (required dependency of mlpack).")
  endif()

  find_cereal()
  if (CEREAL_FOUND)
    set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${CEREAL_INCLUDE_DIR})
  else()
    message(FATAL_ERROR "Cereal not found, (required dependency of mlpack).")
  endif()

  if (MLPACK_USE_SYSTEM_STB)
    find_stb()
  endif()
  if (StbImage_FOUND)
    set(STB_AVAILABLE "1")
    add_definitions(-DMLPACK_USE_SYSTEM_STB)
    set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${STB_IMAGE_INCLUDE_DIR})
  endif()

  find_mlpack_internal()
  if (MLPACK_FOUND)
    set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${MLPACK_INCLUDE_DIR})
  else()
    message(FATAL_ERROR "mlpack not found!")
  endif()

  mark_as_advanced(MLPACK_INCLUDE_DIR)
  mark_as_advanced(MLPACK_INCLUDE_DIRS)
  mark_as_advanced(MLPACK_LIBRARIES)

endmacro()

# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindArmadillo
-------------

Find the Armadillo C++ library.
Armadillo is a library for linear algebra & scientific computing.

Using Armadillo:

::

  find_package(Armadillo REQUIRED)
  include_directories(${ARMADILLO_INCLUDE_DIRS})
  add_executable(foo foo.cc)
  target_link_libraries(foo ${ARMADILLO_LIBRARIES})

This module sets the following variables:

::

  ARMADILLO_FOUND - set to true if the library is found
  ARMADILLO_INCLUDE_DIRS - list of required include directories
  ARMADILLO_LIBRARIES - list of libraries to be linked
  ARMADILLO_VERSION_MAJOR - major version number
  ARMADILLO_VERSION_MINOR - minor version number
  ARMADILLO_VERSION_PATCH - patch version number
  ARMADILLO_VERSION_STRING - version number as a string (ex: "1.0.4")
  ARMADILLO_VERSION_NAME - name of the version (ex: "Antipodean Antileech")
#]=======================================================================]

find_path(ARMADILLO_INCLUDE_DIR
  NAMES armadillo
  PATHS "$ENV{ProgramFiles}/Armadillo/include"
  )

if(ARMADILLO_INCLUDE_DIR)
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

  if(EXISTS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/arma_version.hpp")

    # Read and parse armdillo version header file for version number
    file(STRINGS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/arma_version.hpp" _armadillo_HEADER_CONTENTS REGEX "#define ARMA_VERSION_[A-Z]+ ")
    string(REGEX REPLACE ".*#define ARMA_VERSION_MAJOR ([0-9]+).*" "\\1" ARMADILLO_VERSION_MAJOR "${_armadillo_HEADER_CONTENTS}")
    string(REGEX REPLACE ".*#define ARMA_VERSION_MINOR ([0-9]+).*" "\\1" ARMADILLO_VERSION_MINOR "${_armadillo_HEADER_CONTENTS}")
    string(REGEX REPLACE ".*#define ARMA_VERSION_PATCH ([0-9]+).*" "\\1" ARMADILLO_VERSION_PATCH "${_armadillo_HEADER_CONTENTS}")

    # WARNING: The number of spaces before the version name is not one.
    string(REGEX REPLACE ".*#define ARMA_VERSION_NAME\ +\"([0-9a-zA-Z\ _-]+)\".*" "\\1" ARMADILLO_VERSION_NAME "${_armadillo_HEADER_CONTENTS}")

  endif()

  set(ARMADILLO_VERSION_STRING "${ARMADILLO_VERSION_MAJOR}.${ARMADILLO_VERSION_MINOR}.${ARMADILLO_VERSION_PATCH}")
endif ()

# Read relevant variables from header
if(EXISTS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp")
  file(STRINGS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp" _armadillo_CONFIG_CONTENTS REGEX "#define ARMA_USE_[A-Z]+ ")
  # _ARMA_USE_WRAPPER
  string(REGEX MATCH "\r?\n[\t ]*#define[ \t]+ARMA_USE_WRAPPER[ \t]*\r?\n" _ARMA_USE_WRAPPER "${_armadillo_CONFIG_CONTENTS}")
  # _ARMA_USE_LAPACK
  string(REGEX MATCH "\r?\n[\t ]*#if[\t ]+!defined[(]ARMA_USE_LAPACK[)][\t ]*\r?\n[\t ]*#define[ \t]+_ARMA_USE_LAPACK[ \t]*\r?\n" _ARMA_USE_LAPACK "${_armadillo_CONFIG_CONTENTS}")
  # _ARMA_USE_BLAS
  string(REGEX MATCH "\r?\n[\t ]*#if[\t ]+!defined[(]ARMA_USE_BLAS[)][\t ]*\r?\n[\t ]*#define[ \t]+_ARMA_USE_BLAS[ \t]*\r?\n" _ARMA_USE_BLAS "${_armadillo_CONFIG_CONTENTS}")
  # _ARMA_USE_ARPACK
  string(REGEX MATCH "\r?\n[\t ]*#if[\t ]+!defined[(]ARMA_USE_ARPACK[)][\t ]*\r?\n[\t ]*#define[ \t]+_ARMA_USE_ARPACK[ \t]*\r?\n" _ARMA_USE_ARPACK "${_armadillo_CONFIG_CONTENTS}")
  # _ARMA_USE_HDF5
  string(REGEX MATCH "\r?\n[\t ]*#if[\t ]+!defined[(]ARMA_USE_HDF5[)][\t ]*\r?\n[\t ]*#define[ \t]+_ARMA_USE_HDF5[ \t]*\r?\n" _ARMA_USE_HDF5 "${_armadillo_CONFIG_CONTENTS}")
  unset(_armadillo_HEADER_CONTENTS)
endif()

include(FindPackageHandleStandardArgs)

# If _ARMA_USE_WRAPPER is set, then we just link to armadillo, but if it's not then we need support libraries instead
set(_ARMA_SUPPORT_LIBRARIES)

if(_ARMA_USE_WRAPPER)
  # UNIX paths are standard, no need to write.
  find_library(ARMADILLO_LIBRARY
    NAMES armadillo
    PATHS "$ENV{ProgramFiles}/Armadillo/lib"  "$ENV{ProgramFiles}/Armadillo/lib64" "$ENV{ProgramFiles}/Armadillo"
    )
  find_package_handle_standard_args(Armadillo
    REQUIRED_VARS ARMADILLO_LIBRARY ARMADILLO_INCLUDE_DIR
    VERSION_VAR ARMADILLO_VERSION_STRING)
else(_ARMA_USE_WRAPPER)
  # don't link to armadillo in this case
  set(ARMADILLO_LIBRARY "")
  if(_ARMA_USE_LAPACK)
    find_package(LAPACK REQUIRED)
    set(_ARMA_SUPPORT_LIBRARIES "${_ARMA_SUPPORT_LIBRARIES}" "${LAPACK_LIBRARIES}")
  endif(_ARMA_USE_LAPACK)
  if(_ARMA_USE_BLAS)
    find_package(BLAS REQUIRED)
    set(_ARMA_SUPPORT_LIBRARIES "${_ARMA_SUPPORT_LIBRARIES}" "${BLAS_LIBRARIES}")
  endif(_ARMA_USE_BLAS)
  if(_ARMA_USE_ARPACK)
    find_package(ARPACK REQUIRED)
    set(_ARMA_SUPPORT_LIBRARIES "${_ARMA_SUPPORT_LIBRARIES}" "${ARPACK_LIBRARIES}")
  endif(_ARMA_USE_ARPACK)
  if(_ARMA_USE_HDF5)
    find_package(HDF5 REQUIRED)
    set(_ARMA_SUPPORT_LIBRARIES "${_ARMA_SUPPORT_LIBRARIES}" "${HDF5_LIBRARIES}")
  endif(_ARMA_USE_HDF5)
  set(ARMADILLO_FOUND true)
  find_package_handle_standard_args(Armadillo
    REQUIRED_VARS ARMADILLO_INCLUDE_DIR
    VERSION_VAR ARMADILLO_VERSION_STRING)
endif(_ARMA_USE_WRAPPER)

if (ARMADILLO_FOUND)
  set(ARMADILLO_INCLUDE_DIRS ${ARMADILLO_INCLUDE_DIR})
  set(ARMADILLO_LIBRARIES ${ARMADILLO_LIBRARY} ${_ARMA_SUPPORT_LIBRARIES})
endif ()

# Clean up internal variables
unset(_ARMA_SUPPORT_LIBRARIES)
unset(_ARMA_USE_WRAPPER)
unset(_ARMA_USE_LAPACK)
unset(_ARMA_USE_BLAS)
unset(_ARMA_USE_ARPACK)
unset(_ARMA_USE_HDF5)

# Hide internal variables
mark_as_advanced(
  _ARMA_SUPPORT_LIBRARIES
  ARMADILLO_INCLUDE_DIR
  ARMADILLO_LIBRARY)

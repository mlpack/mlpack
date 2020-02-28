# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindArmadillo
-------------

Find the Armadillo C++ library.
Armadillo is library for linear algebra & scientific computing.

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
  ARMADILLO_LIBRARIES - list of libraries to be linked, if required
  ARMADILLO_VERSION_MAJOR - major version number
  ARMADILLO_VERSION_MINOR - minor version number
  ARMADILLO_VERSION_PATCH - patch version number
  ARMADILLO_VERSION_STRING - version number as a string (ex: "1.0.4")
  ARMADILLO_VERSION_NAME - name of the version (ex: "Antipodean Antileech")
  ARMA_USE_WRAPPER - set to true if armadillo has been configured as a wrapper
  ARMA_USE_LAPACK - set to true if armadillo has been configured to use LAPACK
  ARMA_USE_BLAS - set to true if armadillo has been configured to use BLAS
  ARMA_USE_ARPACK - set to true if armadillo has been configured to use ARPACK
  ARMA_USE_HDF5 - set to true if armadillo has been configured to use HDF5
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
    file(READ "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/arma_version.hpp" _armadillo_HEADER_CONTENTS)
    string(REGEX REPLACE ".*#define ARMA_VERSION_MAJOR ([0-9]+).*" "\\1" ARMADILLO_VERSION_MAJOR "${_armadillo_HEADER_CONTENTS}")
    string(REGEX REPLACE ".*#define ARMA_VERSION_MINOR ([0-9]+).*" "\\1" ARMADILLO_VERSION_MINOR "${_armadillo_HEADER_CONTENTS}")
    string(REGEX REPLACE ".*#define ARMA_VERSION_PATCH ([0-9]+).*" "\\1" ARMADILLO_VERSION_PATCH "${_armadillo_HEADER_CONTENTS}")

    # WARNING: The number of spaces before the version name is not one.
    string(REGEX REPLACE ".*#define ARMA_VERSION_NAME\ +\"([0-9a-zA-Z\ _-]+)\".*" "\\1" ARMADILLO_VERSION_NAME "${_armadillo_HEADER_CONTENTS}")

  endif()

  set(ARMADILLO_VERSION_STRING "${ARMADILLO_VERSION_MAJOR}.${ARMADILLO_VERSION_MINOR}.${ARMADILLO_VERSION_PATCH}")
endif ()

# read relevant variables from header
if(EXISTS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp")
  file(READ "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp" _armadillo_CONFIG_CONTENTS)
  # ARMA_USE_WRAPPER
  string(REGEX MATCH "\r?\n[\t ]*#define[ \t]+ARMA_USE_WRAPPER[ \t]*\r?\n" ARMA_USE_WRAPPER "${_armadillo_CONFIG_CONTENTS}")
  # ARMA_USE_LAPACK
  string(REGEX MATCH "\r?\n[\t ]*#if[\t ]+!defined[(]ARMA_USE_LAPACK[)][\t ]*\r?\n[\t ]*#define[ \t]+ARMA_USE_LAPACK[ \t]*\r?\n" ARMA_USE_LAPACK "${_armadillo_CONFIG_CONTENTS}")
  # ARMA_USE_BLAS
  string(REGEX MATCH "\r?\n[\t ]*#if[\t ]+!defined[(]ARMA_USE_BLAS[)][\t ]*\r?\n[\t ]*#define[ \t]+ARMA_USE_BLAS[ \t]*\r?\n" ARMA_USE_BLAS "${_armadillo_CONFIG_CONTENTS}")
  # ARMA_USE_ARPACK
  string(REGEX MATCH "\r?\n[\t ]*#if[\t ]+!defined[(]ARMA_USE_ARPACK[)][\t ]*\r?\n[\t ]*#define[ \t]+ARMA_USE_ARPACK[ \t]*\r?\n" ARMA_USE_ARPACK "${_armadillo_CONFIG_CONTENTS}")
  # ARMA_USE_HDF5
  string(REGEX MATCH "\r?\n[\t ]*#if[\t ]+!defined[(]ARMA_USE_HDF5[)][\t ]*\r?\n[\t ]*#define[ \t]+ARMA_USE_HDF5[ \t]*\r?\n" ARMA_USE_HDF5 "${_armadillo_CONFIG_CONTENTS}")
endif()

if (ARMA_USE_WRAPPER)
  message(STATUS "Using ARMA_USE_WRAPPER")
  # UNIX paths are standard, no need to write.
  find_library(ARMADILLO_LIBRARY
    NAMES armadillo
    PATHS "$ENV{ProgramFiles}/Armadillo/lib"  "$ENV{ProgramFiles}/Armadillo/lib64" "$ENV{ProgramFiles}/Armadillo"
    )
else()
  message(STATUS "Not using ARMA_USE_WRAPPER")
  # std args complains if this isnt' set to something
  set(ARMADILLO_LIBRARY "TEMP")
  set(ARMADILLO_FOUND true)
endif(ARMA_USE_WRAPPER)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Armadillo
  REQUIRED_VARS ARMADILLO_LIBRARY ARMADILLO_INCLUDE_DIR
  VERSION_VAR ARMADILLO_VERSION_STRING)
# version_var fails with cmake < 2.8.4.

if (NOT ARMA_USE_WRAPPER)
  # unset since we don't use this
  set(ARMADILLO_LIBRARY "")
endif()

if (ARMADILLO_FOUND)
  set(ARMADILLO_INCLUDE_DIRS ${ARMADILLO_INCLUDE_DIR})
  set(ARMADILLO_LIBRARIES ${ARMADILLO_LIBRARY})
endif ()


# Hide internal variables
mark_as_advanced(
  ARMADILLO_INCLUDE_DIR
  ARMADILLO_LIBRARY)

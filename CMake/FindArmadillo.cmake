# - Find Armadillo
# Find the Armadillo C++ library
#
# Using Armadillo:
#  find_package(Armadillo REQUIRED)
#  include_directories(${ARMADILLO_INCLUDE_DIRS})
#  add_executable(foo foo.cc)
#  target_link_libraries(foo ${ARMADILLO_LIBRARIES})
# This module sets the following variables:
#  ARMADILLO_FOUND - set to true if the library is found
#  ARMADILLO_INCLUDE_DIRS - list of required include directories
#  ARMADILLO_LIBRARIES - list of libraries to be linked
#  ARMADILLO_VERSION_MAJOR - major version number
#  ARMADILLO_VERSION_MINOR - minor version number
#  ARMADILLO_VERSION_PATCH - patch version number
#  ARMADILLO_VERSION_STRING - version number as a string (ex: "1.0.4")
#  ARMADILLO_VERSION_NAME - name of the version (ex: "Antipodean Antileech")

#=============================================================================
# Copyright 2011 Clement Creusot <creusot@cs.york.ac.uk>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)


# UNIX paths are standard, no need to write.
find_library(ARMADILLO_LIBRARY
  NAMES armadillo
  PATHS "$ENV{ProgramFiles}/Armadillo/lib"  "$ENV{ProgramFiles}/Armadillo/lib64" "$ENV{ProgramFiles}/Armadillo"
  )
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

  endif(EXISTS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/arma_version.hpp")

  set(ARMADILLO_VERSION_STRING "${ARMADILLO_VERSION_MAJOR}.${ARMADILLO_VERSION_MINOR}.${ARMADILLO_VERSION_PATCH}")
endif (ARMADILLO_INCLUDE_DIR)


#======================

# Determine whether or not we need to link against HDF5.  We need to look in
# config.hpp.
if(EXISTS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp")
  # Look for #define ARMA_USE_HDF5.
  file(READ "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp" _armadillo_CONFIG_CONTENTS)
  string(REGEX MATCH "[\r\n][\t ]*#define[ \t]+ARMA_USE_HDF5[ \t\r\n]" ARMA_USE_HDF5 "${_armadillo_CONFIG_CONTENTS}")

  if(NOT "${ARMA_USE_HDF5}" STREQUAL "")
    message(STATUS "Armadillo HDF5 support is enabled.")
    # We have HDF5 support and need to link against HDF5.
    find_package(HDF5 REQUIRED)
  endif(NOT "${ARMA_USE_HDF5}" STREQUAL "")
endif(EXISTS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp")

#======================


# Checks 'REQUIRED', 'QUIET' and versions.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Armadillo
  REQUIRED_VARS ARMADILLO_LIBRARY ARMADILLO_INCLUDE_DIR
  VERSION_VAR ARMADILLO_VERSION_STRING)
# version_var fails with cmake < 2.8.4.

if (ARMADILLO_FOUND)
  set(ARMADILLO_INCLUDE_DIRS ${ARMADILLO_INCLUDE_DIR})
  # HDF5 libraries are stored in HDF5_LIBRARIES, if they were necessary.
  set(ARMADILLO_LIBRARIES ${ARMADILLO_LIBRARY} ${HDF5_LIBRARIES})
endif (ARMADILLO_FOUND)


# Hide internal variables
mark_as_advanced(
  ARMADILLO_INCLUDE_DIR
  ARMADILLO_LIBRARY)

#======================

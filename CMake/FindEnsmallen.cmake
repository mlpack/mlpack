# - Find Ensmallen
# Find the Ensmallen C++ library
#
# This module sets the following variables:
#  ENSMALLEN_FOUND - set to true if the library is found
#  ENSMALLEN_INCLUDE_DIR - list of required include directories
#  ENSMALLEN_VERSION_MAJOR - major version number
#  ENSMALLEN_VERSION_MINOR - minor version number
#  ENSMALLEN_VERSION_PATCH - patch version number
#  ENSMALLEN_VERSION_STRING - version number as a string (ex: "1.0.4")
#  ENSMALLEN_VERSION_NAME - name of the version (ex: "Antipodean Antileech")

file(GLOB ENSMALLEN_SEARCH_PATHS
    ${CMAKE_BINARY_DIR}/deps/ensmallen-[0-9]*.[0-9]*.[0-9]*)
find_path(ENSMALLEN_INCLUDE_DIR
  NAMES ensmallen.hpp
  PATHS ${ENSMALLEN_SEARCH_PATHS}/include)

if(ENSMALLEN_INCLUDE_DIR)
  # ------------------------------------------------------------------------
  #  Extract version information from <ensmallen>
  # ------------------------------------------------------------------------

  set(ENSMALLEN_VERSION_MAJOR 0)
  set(ENSMALLEN_VERSION_MINOR 0)
  set(ENSMALLEN_VERSION_PATCH 0)
  set(ENSMALLEN_VERSION_NAME "unknown")

  if(EXISTS "${ENSMALLEN_INCLUDE_DIR}/ensmallen_bits/ens_version.hpp")

    set(ENSMALLEN_FOUND YES)

    # Read and parse armdillo version header file for version number
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

# Checks 'REQUIRED', 'QUIET' and versions.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Ensmallen
    REQUIRED_VARS ENSMALLEN_INCLUDE_DIR
    VERSION_VAR ENSMALLEN_VERSION_STRING)

mark_as_advanced(ENSMALLEN_INCLUDE_DIR)

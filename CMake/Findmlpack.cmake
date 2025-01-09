# - Find mlpack
# Find the mlpack C++ library
#
# This module sets the following variable:
#  MLPACK_FOUND - set to true if the library is found
#  MLPACK_INCLUDE_DIR - list of required include directories
#  MLPACK_VERSION_MAJOR - major version number
#  MLPACK_VERSION_MINOR - minor version number
#  MLPACK_VERSION_PATCH - patch version number
#  MLPACK_VERSION_STRING - version number as a string (ex: "1.0.4")

file(GLOB MLPACK_SEARCH_PATHS
    ${CMAKE_BINARY_DIR}/deps/mlpack-[0-9]*.[0-9]*.[0-9]*)
find_path(MLPACK_INCLUDE_DIR
  NAMES mlpack.hpp
  PATHS ${MLPACK_SEARCH_PATHS}/include)

if(MLPACK_INCLUDE_DIR)
  # ------------------------------------------------------------------------
  #  Extract version information from <mlpack>
  # ------------------------------------------------------------------------

  set(MLPACK_VERSION_MAJOR 0)
  set(MLPACK_VERSION_MINOR 0)
  set(MLPACK_VERSION_PATCH 0)

  if(EXISTS "${MLPACK_INCLUDE_DIR}/mlpack/core/util/version.hpp")

    set(MLPACK_FOUND YES)

    # Read and parse armdillo version header file for version number
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
endif ()


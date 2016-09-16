# As input the following variables should be set:
#
#   MLPACK_SOURCE_DIR: directory containing mlpack sources.
#
# And our goal in this file is to generate/configure mlpack.pc.

# First, we need to extract the version string.
if (NOT EXISTS "${CMAKE_BINARY_DIR}/include/mlpack/core/util/version.hpp")
  message(FATAL_ERROR "Cannot open "
      "${CNAKE_BINARY_DIR}/include/mlpack/core/util/version.hpp to extract "
      "version!")
endif ()

file(READ "${CMAKE_BINARY_DIR}/include/mlpack/core/util/version.hpp"
    VERSION_HPP_CONTENTS)
string(REGEX REPLACE ".*#define MLPACK_VERSION_MAJOR ([0-9]+).*" "\\1"
    MLPACK_VERSION_MAJOR "${VERSION_HPP_CONTENTS}")
string(REGEX REPLACE ".*#define MLPACK_VERSION_MINOR ([0-9]+).*" "\\1"
    MLPACK_VERSION_MINOR "${VERSION_HPP_CONTENTS}")
string(REGEX REPLACE ".*#define MLPACK_VERSION_PATCH [\"]?([0-9x]+)[\"]?.*" "\\1"
    MLPACK_VERSION_PATCH "${VERSION_HPP_CONTENTS}")

set(MLPACK_VERSION_STRING
    "${MLPACK_VERSION_MAJOR}.${MLPACK_VERSION_MINOR}.${MLPACK_VERSION_PATCH}")

configure_file(
    ${CMAKE_BINARY_DIR}/CMake/mlpack.pc.in.partial
    ${CMAKE_BINARY_DIR}/lib/pkgconfig/mlpack.pc @ONLY)

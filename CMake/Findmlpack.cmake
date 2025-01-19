# - Find mlpack
# Find the mlpack C++ library
#
# This module will look for mlpack dependencies and then look for mlpack itself
# if it is here or not.
# 
# We are basically inlining the previous ensmallen and Cereal into this file.
#

# 1. Let us find Armadillo
find_package(Armadillo "${ARMADILLO_VERSION}" REQUIRED)
if (ARMADILLO_FOUND)
  # Include directories for the previous dependencies.
  set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
  set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})
else ()
  message(FATAL_ERROR
    "Armadillo library is not found, please make sure to install Armadillo,"
    " or use the autodownloader. mlpack can not be installed without Armadillo"
  )
endif()

# Findcereal.cmake
find_path(CEREAL_INCLUDE_DIR
  NAMES cereal
  PATHS "$ENV{ProgramFiles}/cereal/include"
  )

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

mark_as_advanced(CEREAL_INCLUDE_DIR)

file(GLOB MLPACK_SEARCH_PATHS
    ${CMAKE_BINARY_DIR}/deps/mlpack-[0-9]*.[0-9]*.[0-9]*)

if (CEREAL_FOUND)
  set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${CEREAL_INCLUDE_DIR})
else()
  message(FATAL_ERROR
    "Cereal library is not found, please make sure to install cereal, or use"
    " the autodownloader. mlpack can not be installed without cereal")
endif()

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

mark_as_advanced(ENSMALLEN_INCLUDE_DIR)

if (ENSMALLEN_FOUND)
    set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} "${ENSMALLEN_INCLUDE_DIR}")
else()
  message(FATAL_ERROR 
    "Ensmallen library is not found, please make sure to install ensmallen,"
    " or use the autodownloader. mlpack can not be installed without cereal")
endif()

if (USE_OPENMP)
  find_package(OpenMP)
endif ()

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

# This module sets the following variable:
#  MLPACK_FOUND - set to true if the library is found
#  MLPACK_INCLUDE_DIR - list of required include directories
#  MLPACK_VERSION_MAJOR - major version number
#  MLPACK_VERSION_MINOR - minor version number
#  MLPACK_VERSION_PATCH - patch version number
#  MLPACK_VERSION_STRING - version number as a string (ex: "1.0.4")

# This will be executed if mlpack is installed already.
find_path(MLPACK_INCLUDE_DIR
  NAMES mlpack.hpp
  PATHS ${MLPACK_SEARCH_PATHS}/include)

# This will be executed when compiling mlpack bindings and tests.
if (NOT MLPACK_INCLUDE_DIR)
  find_path(MLPACK_INCLUDE_DIR
    NAMES mlpack.hpp
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/src/)
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

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mlpack
  REQUIRED_VARS MLPACK_INCLUDE_DIR
  VERSION_VAR MLPACK_VERSION_STRING)

mark_as_advanced(MLPACK_INCLUDE_DIR)

set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} "${MLPACK_INCLUDE_DIR}")

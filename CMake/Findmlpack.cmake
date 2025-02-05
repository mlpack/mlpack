# - Find mlpack
# Find the mlpack C++ library
#
# author Omar Shrit
#
# This module will look for mlpack dependencies and then look for mlpack itself
#
# This module can also fetch mlpack dependencies from online repositories.
#
# This module sets the following variables:
#  MLPACK_FOUND - set to true if the library is found
#  MLPACK_VERSION_MAJOR - major version number
#  MLPACK_VERSION_MINOR - minor version number
#  MLPACK_VERSION_PATCH - patch version number
#  MLPACK_VERSION_STRING - version number as a string (ex: "1.0.4")
#  MLPACK_INCLUDE_DIRS - list of required include directories
#  MLPACK_LIBRARIES - list of required libraries



## What needs to be done in this PR:
# 1. Create a macro / function to each one of them
# 2. Call each one of them at the end of this file
# 3. If found make it set a list of one that are not found
# 4. Use the list in the case we need to call the autodownloader
# 5. Simplyfy the Autodownloader call and make it only one with for loop if
# necessary, or even better, make all the link recorded already inside the
# autodownload function and then iterate over the current ones.
# This will simplfy the autodownloader cmake file
# STB locally and no need to search for it in the case of autdownload.
# Be sure to have Autdownload in the same file, maybe even move autodownload
# into this one.
# Add another function fetch_mlpack() from source code using autdownloader
# this should happen inside this file, the user does not need to call
# autdownload or anyting else
# A simple comment above this function call should be enough
# another function can be called fetch_mlpack_and_crosscompile()
# move minimum version to this file since it is not required anyware else.

# The purpose of this file is to allow the user to use CMake to find mlpack in
# any usecase. Therefore, some of the parameters are moved from inside
# CMakeLists to here, since they will be required to be set by the user at
# somepoint.

# Set minimum library versions required by mlpack.
#
# For Armadillo, try to keep the minimum required version less than or equal to
# what's available on the current Ubuntu LTS or most recent stable RHEL release.
# See https://github.com/mlpack/mlpack/issues/3033 for some more discussion.
set(ARMADILLO_VERSION "10.8")
set(ENSMALLEN_VERSION "2.10.0")
set(CEREAL_VERSION "1.1.2")

# Set required standard to C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MLPACK_DISABLE_OPENMP OFF)

if (NOT USE_OPENMP)
  set(MLPACK_DISABLE_OPENMP ON)
endif()

# If we're using gcc, then we need to link against pthreads to use std::thread,
# which we do in the tests.
if (CMAKE_COMPILER_IS_GNUCC)
  find_package(Threads)
  set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif()

#[=======================================================================[.rst:

The following code is ported from FindArmdillo from CMake 4.0
contains @rcurtin fixes regarding Windows trasitive linking.
-------------
FindArmadillo
-------------

Find the Armadillo C++ library.
Armadillo is a library for linear algebra & scientific computing.

.. versionadded:: 3.18
  Support for linking wrapped libraries directly (``ARMA_DONT_USE_WRAPPER``).

Using Armadillo:

.. code-block:: cmake

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

macro(find_Armadillo)
  cmake_policy(PUSH)
  cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

  find_path(ARMADILLO_INCLUDE_DIR
    NAMES armadillo
    PATHS "$ENV{ProgramFiles}/Armadillo/include"
    )
  mark_as_advanced(ARMADILLO_INCLUDE_DIR)

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
      file(STRINGS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/arma_version.hpp" _ARMA_HEADER_CONTENTS REGEX "#define ARMA_VERSION_[A-Z]+ ")
      string(REGEX REPLACE ".*#define ARMA_VERSION_MAJOR ([0-9]+).*" "\\1" ARMADILLO_VERSION_MAJOR "${_ARMA_HEADER_CONTENTS}")
      string(REGEX REPLACE ".*#define ARMA_VERSION_MINOR ([0-9]+).*" "\\1" ARMADILLO_VERSION_MINOR "${_ARMA_HEADER_CONTENTS}")
      string(REGEX REPLACE ".*#define ARMA_VERSION_PATCH ([0-9]+).*" "\\1" ARMADILLO_VERSION_PATCH "${_ARMA_HEADER_CONTENTS}")

      # WARNING: The number of spaces before the version name is not one.
      string(REGEX REPLACE ".*#define ARMA_VERSION_NAME\ +\"([0-9a-zA-Z\ _-]+)\".*" "\\1" ARMADILLO_VERSION_NAME "${_ARMA_HEADER_CONTENTS}")

    endif()

    set(ARMADILLO_VERSION_STRING "${ARMADILLO_VERSION_MAJOR}.${ARMADILLO_VERSION_MINOR}.${ARMADILLO_VERSION_PATCH}")
  endif ()

  if(EXISTS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp")
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
    mark_as_advanced(ARMADILLO_LIBRARY)
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
        find_package(${pkg} QUIET)
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
  else()
    message(FATAL_ERROR "Armadillo not found, (required dependency of mlpack).")
  endif ()

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
  message(FATAL_ERROR "Cereal not found, (required dependency of mlpack).")
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

macro(find_Ensmallen)
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

endmacro()



# - Find STB_IMAGE
# Find the STB_IMAGE C++ library
#
# This module sets the following variables:
#  STB_IMAGE_FOUND - set to true if the library is found
#  STB_IMAGE_INCLUDE_DIR - list of required include directories
#  STB_INCLUDE_NEEDS_STB_SUFFIX - whether or not the include files are under an
#     stb/ directory; if "YES", then includes must be done as, e.g.,
#     stb/stb_image.h.
macro(find_STB)

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

    if(STB_IMAGE_INCLUDE_DIR_2)
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

  # Checks 'REQUIRED'.
  mark_as_advanced(STB_IMAGE_INCLUDE_DIR)

endmacro()


## Find OpenMP
if (NOT MLPACK_DISABLE_OPENMP)
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
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mlpack
  REQUIRED_VARS MLPACK_INCLUDE_DIR
  VERSION_VAR MLPACK_VERSION_STRING)

mark_as_advanced(MLPACK_INCLUDE_DIR)

set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} "${MLPACK_INCLUDE_DIR}")

mark_as_advanced(MLPACK_INCLUDE_DIRS)
mark_as_advanced(MLPACK_LIBRARIES)


if (ENSMALLEN_FOUND)
  set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} "${ENSMALLEN_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "Ensmallen not found, (required dependency of mlpack).")
endif()






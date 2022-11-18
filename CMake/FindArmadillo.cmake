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
  # Read and parse armdillo version header file for version number
  file(STRINGS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/arma_version.hpp" _ARMA_HEADER_CONTENTS REGEX "#define ARMA_VERSION_[A-Z]+ ")
  string(REGEX REPLACE ".*#define ARMA_VERSION_MAJOR ([0-9]+).*" "\\1" ARMADILLO_VERSION_MAJOR "${_ARMA_HEADER_CONTENTS}")
  string(REGEX REPLACE ".*#define ARMA_VERSION_MINOR ([0-9]+).*" "\\1" ARMADILLO_VERSION_MINOR "${_ARMA_HEADER_CONTENTS}")
  string(REGEX REPLACE ".*#define ARMA_VERSION_PATCH ([0-9]+).*" "\\1" ARMADILLO_VERSION_PATCH "${_ARMA_HEADER_CONTENTS}")

  # WARNING: The number of spaces before the version name is not one.
  string(REGEX REPLACE ".*#define ARMA_VERSION_NAME\ +\"([0-9a-zA-Z\ _-]+)\".*" "\\1" ARMADILLO_VERSION_NAME "${_ARMA_HEADER_CONTENTS}")

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

include(FindPackageHandleStandardArgs)

# If _ARMA_USE_WRAPPER is set, then we just link to armadillo, but if it's not then we need support libraries instead
set(_ARMA_SUPPORT_LIBRARIES)

if(_ARMA_USE_WRAPPER)
  # UNIX paths are standard, no need to write.
  find_library(ARMADILLO_LIBRARY
    NAMES armadillo
    PATHS "$ENV{ProgramFiles}/Armadillo/lib"  "$ENV{ProgramFiles}/Armadillo/lib64" "$ENV{ProgramFiles}/Armadillo"
    )
    set(_ARMA_REQUIRED_VARS ARMADILLO_LIBRARY ARMADILLO_INCLUDE_DIR VERSION_VAR ARMADILLO_VERSION_STRING)
else()
  # don't link to armadillo in this case
  set(ARMADILLO_LIBRARY "")
endif()

# Link to support libraries in either case on MSVC.
if(NOT _ARMA_USE_WRAPPER OR MSVC)
  if(_ARMA_USE_LAPACK)
    if(APPLE)
      # Use -framework Accelerate to link against the Accelerate framework on
      # MacOS; ignore OpenBLAS or other variants.
      set(LAPACK_LIBRARIES "-framework Accelerate")
      set(LAPACK_FOUND YES)
    else()
      if(ARMADILLO_FIND_QUIETLY OR NOT ARMADILLO_FIND_REQUIRED)
        find_package(LAPACK QUIET)
      else()
        find_package(LAPACK REQUIRED)
      endif()
    endif()

    if(LAPACK_FOUND)
      set(_ARMA_SUPPORT_LIBRARIES "${_ARMA_SUPPORT_LIBRARIES}" "${LAPACK_LIBRARIES}")
    endif()
  endif()
  if(_ARMA_USE_BLAS)
    if(APPLE)
      # Use -framework Accelerate to link against the Accelerate framework on
      # MacOS; ignore OpenBLAS or other variants.
      set(BLAS_LIBRARIES "-framework Accelerate")
      set(BLAS_FOUND YES)
    else()
      if(ARMADILLO_FIND_QUIETLY OR NOT ARMADILLO_FIND_REQUIRED)
        find_package(BLAS QUIET)
      else()
        find_package(BLAS REQUIRED)
      endif()
    endif()

    if(BLAS_FOUND)
      # Avoid doubly linking (not that it makes much difference other than a
      # nicer command-line).
      if (NOT BLAS_LIBRARIES EQUAL LAPACK_LIBRARIES)
        set(_ARMA_SUPPORT_LIBRARIES "${_ARMA_SUPPORT_LIBRARIES}" "${BLAS_LIBRARIES}")
      endif ()
    endif()
  endif()
  if(_ARMA_USE_ARPACK)
    if(ARMADILLO_FIND_QUIETLY OR NOT ARMADILLO_FIND_REQUIRED)
      find_package(ARPACK QUIET)
    else()
      find_package(ARPACK REQUIRED)
    endif()
    if(ARPACK_FOUND)
      set(_ARMA_SUPPORT_LIBRARIES "${_ARMA_SUPPORT_LIBRARIES}" "${ARPACK_LIBRARIES}")
    endif()
  endif()
  if(_ARMA_USE_HDF5)
    find_package(HDF5 QUIET)
    if(NOT HDF5_FOUND)
      # On Debian systems, the HDF5 package has been split into multiple
      # packages so that it is co-installable.  But this may mean that the
      # include files are hidden somewhere very odd that FindHDF5.cmake  will
      # not find.  Thus, we'll also quickly check pkgconfig to see if there is
      # information on what to use there.
      message(WARNING "HDF5 required but not found; using PkgConfig")
      find_package(PkgConfig)
      if (PKG_CONFIG_FOUND)
        pkg_check_modules(HDF5 REQUIRED hdf5)
        link_directories("${HDF5_LIBRARY_DIRS}")
      else()
        message(FATAL_ERROR "PkgConfig (Used to help find HDF5) was not found")
      endif()
    endif()
    set(_ARMA_SUPPORT_INCLUDE_DIRS "${HDF5_INCLUDE_DIRS}")
    set(_ARMA_SUPPORT_LIBRARIES "${_ARMA_SUPPORT_LIBRARIES}" "${HDF5_LIBRARIES}")
  endif()
  set(ARMADILLO_FOUND true)
  set(_ARMA_REQUIRED_VARS ARMADILLO_INCLUDE_DIR VERSION_VAR ARMADILLO_VERSION_STRING)
endif()

find_package_handle_standard_args(Armadillo REQUIRED_VARS ${_ARMA_REQUIRED_VARS})

if (ARMADILLO_FOUND)
  set(ARMADILLO_INCLUDE_DIRS ${ARMADILLO_INCLUDE_DIR})
  set(ARMADILLO_LIBRARIES ${ARMADILLO_LIBRARY} ${_ARMA_SUPPORT_LIBRARIES})
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
unset(__ARMA_SUPPORT_INCLUDE_DIRS)

# Hide internal variables
mark_as_advanced(
    ARMADILLO_INCLUDE_DIR
    ARMADILLO_LIBRARY
    ARMADILLO_LIBRARIES)

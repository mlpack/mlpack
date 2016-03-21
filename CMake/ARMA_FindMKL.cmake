# - Find the MKL libraries (no includes)
# This module defines
#  MKL_LIBRARIES, the libraries needed to use Intel's implementation of BLAS & LAPACK.
#  MKL_FOUND, If false, do not try to use MKL.

set(MKL_NAMES ${MKL_NAMES} mkl_lapack)
set(MKL_NAMES ${MKL_NAMES} mkl_intel_thread)
set(MKL_NAMES ${MKL_NAMES} mkl_core)
set(MKL_NAMES ${MKL_NAMES} guide)
set(MKL_NAMES ${MKL_NAMES} mkl)
set(MKL_NAMES ${MKL_NAMES} iomp5)
#set(MKL_NAMES ${MKL_NAMES} pthread)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(MKL_NAMES ${MKL_NAMES} mkl_intel_lp64)
else()
  set(MKL_NAMES ${MKL_NAMES} mkl_intel)
endif()

foreach (MKL_NAME ${MKL_NAMES})
  find_library(${MKL_NAME}_LIBRARY
    NAMES ${MKL_NAME}
    PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib /opt/intel/lib/intel64 /opt/intel/lib/ia32 /opt/intel/mkl/lib/lib64 /opt/intel/mkl/lib/intel64 /opt/intel/mkl/lib/ia32 /opt/intel/mkl/lib /opt/intel/*/mkl/lib/intel64 /opt/intel/*/mkl/lib/ia32/ /opt/mkl/*/lib/em64t /opt/mkl/*/lib/32 /opt/intel/mkl/*/lib/em64t /opt/intel/mkl/*/lib/32
    )

  set(TMP_LIBRARY ${${MKL_NAME}_LIBRARY})

  if(TMP_LIBRARY)
    set(MKL_LIBRARIES ${MKL_LIBRARIES} ${TMP_LIBRARY})
  endif()
endforeach()

if (MKL_LIBRARIES)
  set(MKL_FOUND "YES")
else ()
  set(MKL_FOUND "NO")
endif ()

if (MKL_FOUND)
  if (NOT MKL_FIND_QUIETLY)
    message(STATUS "Found MKL libraries: ${MKL_LIBRARIES}")
  endif ()
else ()
  if (MKL_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find MKL libraries")
  endif ()
endif ()

# mark_as_advanced(MKL_LIBRARY)

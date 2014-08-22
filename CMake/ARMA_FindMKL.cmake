# - Find the MKL libraries (no includes)
# This module defines
#  MKL_LIBRARIES, the libraries needed to use Intel's implementation of BLAS & LAPACK.
#  MKL_FOUND, If false, do not try to use MKL.

SET(MKL_NAMES ${MKL_NAMES} mkl_lapack)
SET(MKL_NAMES ${MKL_NAMES} mkl_intel_thread)
SET(MKL_NAMES ${MKL_NAMES} mkl_core)
SET(MKL_NAMES ${MKL_NAMES} guide)
SET(MKL_NAMES ${MKL_NAMES} mkl)
SET(MKL_NAMES ${MKL_NAMES} iomp5)
#SET(MKL_NAMES ${MKL_NAMES} pthread)

IF(CMAKE_SIZEOF_VOID_P EQUAL 8)
  SET(MKL_NAMES ${MKL_NAMES} mkl_intel_lp64)
ELSE(CMAKE_SIZEOF_VOID_P EQUAL 8)
  SET(MKL_NAMES ${MKL_NAMES} mkl_intel)
ENDIF(CMAKE_SIZEOF_VOID_P EQUAL 8)

FOREACH (MKL_NAME ${MKL_NAMES})
  FIND_LIBRARY(${MKL_NAME}_LIBRARY
    NAMES ${MKL_NAME}
    PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib /opt/intel/lib/intel64 /opt/intel/lib/ia32 /opt/intel/mkl/lib/lib64 /opt/intel/mkl/lib/intel64 /opt/intel/mkl/lib/ia32 /opt/intel/mkl/lib /opt/intel/*/mkl/lib/intel64 /opt/intel/*/mkl/lib/ia32/ /opt/mkl/*/lib/em64t /opt/mkl/*/lib/32 /opt/intel/mkl/*/lib/em64t /opt/intel/mkl/*/lib/32
    )

  SET(TMP_LIBRARY ${${MKL_NAME}_LIBRARY})

  IF(TMP_LIBRARY)
    SET(MKL_LIBRARIES ${MKL_LIBRARIES} ${TMP_LIBRARY})
  ENDIF(TMP_LIBRARY)
ENDFOREACH(MKL_NAME)

IF (MKL_LIBRARIES)
  SET(MKL_FOUND "YES")
ELSE (MKL_LIBRARIES)
  SET(MKL_FOUND "NO")
ENDIF (MKL_LIBRARIES)

IF (MKL_FOUND)
  IF (NOT MKL_FIND_QUIETLY)
    MESSAGE(STATUS "Found MKL libraries: ${MKL_LIBRARIES}")
  ENDIF (NOT MKL_FIND_QUIETLY)
ELSE (MKL_FOUND)
  IF (MKL_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find MKL libraries")
  ENDIF (MKL_FIND_REQUIRED)
ENDIF (MKL_FOUND)

# MARK_AS_ADVANCED(MKL_LIBRARY)

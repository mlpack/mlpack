# Searches for an installation of the ARPACK library. On success, it sets the following variables:
#
#   ARPACK_FOUND      Set to true to indicate the library was found
#   ARPACK_LIBRARIES  All libraries needed to use ARPACK (with full path)
#
# To specify an additional directory to search, set ARPACK_ROOT.
#
# TODO: Do we need to explicitly search for BLAS and LAPACK as well? The source distribution statically links these to
# libarpack. Are there any installations that don't do this or the equivalent?
#
# Author: Siddhartha Chaudhuri, 2009
#

SET(ARPACK_FOUND FALSE)

# First look in user-provided root directory, then look in system locations
FIND_LIBRARY(ARPACK_LIBRARIES NAMES arpack libarpack ARPACK libARPACK PATHS "${ARPACK_ROOT}" "${ARPACK_ROOT}/lib"
             NO_DEFAULT_PATH)
IF(NOT ARPACK_LIBRARIES)
  FIND_LIBRARY(ARPACK_LIBRARIES NAMES arpack libarpack ARPACK libARPACK)
ENDIF(NOT ARPACK_LIBRARIES)

IF(ARPACK_LIBRARIES)
  # On OS X we probably also need gfortran and BLAS and LAPACK libraries
  IF(APPLE)
    FIND_LIBRARY(ARPACK_LAPACK_LIBRARY NAMES lapack LAPACK PATHS "${ARPACK_ROOT}" "${ARPACK_ROOT}/lib")
    FIND_LIBRARY(ARPACK_BLAS_LIBRARY NAMES blas BLAS PATHS "${ARPACK_ROOT}" "${ARPACK_ROOT}/lib")
    FIND_LIBRARY(ARPACK_GFORTRAN_LIBRARY NAMES gfortran PATHS "${ARPACK_ROOT}" "${ARPACK_ROOT}/lib"
                                                        PATH_SUFFIXES "" "gfortran/lib" "../gfortran/lib")

    IF(ARPACK_BLAS_LIBRARY)
      SET(ARPACK_LIBRARIES ${ARPACK_LIBRARIES} ${ARPACK_BLAS_LIBRARY})
    ENDIF(ARPACK_BLAS_LIBRARY)

    IF(ARPACK_LAPACK_LIBRARY)
      SET(ARPACK_LIBRARIES ${ARPACK_LIBRARIES} ${ARPACK_LAPACK_LIBRARY})
    ENDIF(ARPACK_LAPACK_LIBRARY)

    IF(ARPACK_GFORTRAN_LIBRARY)
      SET(ARPACK_LIBRARIES ${ARPACK_LIBRARIES} ${ARPACK_GFORTRAN_LIBRARY})
    ENDIF(ARPACK_GFORTRAN_LIBRARY)
  ENDIF(APPLE)

  SET(ARPACK_FOUND TRUE)
ENDIF(ARPACK_LIBRARIES)

IF(ARPACK_FOUND)
  IF(NOT ARPACK_FIND_QUIETLY)
    MESSAGE(STATUS "Found ARPACK: libraries at ${ARPACK_LIBRARIES}")
  ENDIF(NOT ARPACK_FIND_QUIETLY)
ELSE(ARPACK_FOUND)
  IF(ARPACK_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "ARPACK not found")
  ENDIF(ARPACK_FIND_REQUIRED)
ENDIF(ARPACK_FOUND)

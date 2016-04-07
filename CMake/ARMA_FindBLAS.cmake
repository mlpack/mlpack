# - Find a BLAS library (no includes)
# This module defines
#  BLAS_LIBRARIES, the libraries needed to use BLAS.
#  BLAS_FOUND, If false, do not try to use BLAS.
# also defined, but not for general use are
#  BLAS_LIBRARY, where to find the BLAS library.

set(BLAS_NAMES ${BLAS_NAMES} blas)

# Find the ATLAS version preferentially.
find_library(BLAS_LIBRARY
  NAMES ${BLAS_NAMES}
  PATHS /usr/lib64/atlas /usr/lib/atlas /usr/local/lib64/atlas /usr/local/lib/atlas
  NO_DEFAULT_PATH)

find_library(BLAS_LIBRARY
  NAMES ${BLAS_NAMES}
  PATHS /usr/lib64/atlas /usr/lib/atlas /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
  )

if (BLAS_LIBRARY)
  set(BLAS_LIBRARIES ${BLAS_LIBRARY})
  set(BLAS_FOUND "YES")
else ()
  set(BLAS_FOUND "NO")
endif ()


if (BLAS_FOUND)
   if (NOT BLAS_FIND_QUIETLY)
      message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
   endif ()
else ()
   if (BLAS_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find BLAS")
   endif ()
endif ()

# Deprecated declarations.
get_filename_component (NATIVE_BLAS_LIB_PATH ${BLAS_LIBRARY} PATH)

mark_as_advanced(
  BLAS_LIBRARY
  )

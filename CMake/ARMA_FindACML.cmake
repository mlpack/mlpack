# - Find AMD's ACML library (no includes) which provides optimised BLAS and LAPACK functions
# This module defines
#  ACML_LIBRARIES, the libraries needed to use ACML.
#  ACML_FOUND, If false, do not try to use ACML.
# also defined, but not for general use are
#  ACML_LIBRARY, where to find the ACML library.

set(ACML_NAMES ${ACML_NAMES} acml)
find_library(ACML_LIBRARY
  NAMES ${ACML_NAMES}
  PATHS /usr/lib64 /usr/lib /usr/*/lib64 /usr/*/lib /usr/*/gfortran64/lib/ /usr/*/gfortran32/lib/ /usr/local/lib64 /usr/local/lib /opt/lib64 /opt/lib /opt/*/lib64 /opt/*/lib /opt/*/gfortran64/lib/ /opt/*/gfortran32/lib/
  )

if (ACML_LIBRARY)
  set(ACML_LIBRARIES ${ACML_LIBRARY})
  set(ACML_FOUND "YES")
else ()
  set(ACML_FOUND "NO")
endif ()


if (ACML_FOUND)
   if (NOT ACML_FIND_QUIETLY)
      message(STATUS "Found the ACML library: ${ACML_LIBRARIES}")
   endif ()
else ()
   if (ACML_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find the ACML library")
   endif ()
endif ()

# Deprecated declarations.
get_filename_component (NATIVE_ACML_LIB_PATH ${ACML_LIBRARY} PATH)

mark_as_advanced(
  ACML_LIBRARY
  )

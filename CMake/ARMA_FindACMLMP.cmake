# - Find AMD's ACMLMP library (no includes) which provides optimised and parallelised BLAS and LAPACK functions
# This module defines
#  ACMLMP_LIBRARIES, the libraries needed to use ACMLMP.
#  ACMLMP_FOUND, If false, do not try to use ACMLMP.
# also defined, but not for general use are
#  ACMLMP_LIBRARY, where to find the ACMLMP library.

set(ACMLMP_NAMES ${ACMLMP_NAMES} acml_mp)
find_library(ACMLMP_LIBRARY
  NAMES ${ACMLMP_NAMES}
  PATHS /usr/lib64 /usr/lib /usr/*/lib64 /usr/*/lib /usr/*/gfortran64_mp/lib/ /usr/*/gfortran32_mp/lib/ /usr/local/lib64 /usr/local/lib /opt/lib64 /opt/lib /opt/*/lib64 /opt/*/lib /opt/*/gfortran64_mp/lib/ /opt/*/gfortran32_mp/lib/
  )

if (ACMLMP_LIBRARY)
  set(ACMLMP_LIBRARIES ${ACMLMP_LIBRARY})
  set(ACMLMP_FOUND "YES")
else ()
  set(ACMLMP_FOUND "NO")
endif ()


if (ACMLMP_FOUND)
   if (NOT ACMLMP_FIND_QUIETLY)
      message(STATUS "Found the ACMLMP library: ${ACMLMP_LIBRARIES}")
   endif ()
else ()
   if (ACMLMP_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find the ACMLMP library")
   endif ()
endif ()

# Deprecated declarations.
get_filename_component (NATIVE_ACMLMP_LIB_PATH ${ACMLMP_LIBRARY} PATH)

mark_as_advanced(
  ACMLMP_LIBRARY
  )

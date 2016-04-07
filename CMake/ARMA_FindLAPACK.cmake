# - Find a LAPACK library (no includes)
# This module defines
#  LAPACK_LIBRARIES, the libraries needed to use LAPACK.
#  LAPACK_FOUND, If false, do not try to use LAPACK.
# also defined, but not for general use are
#  LAPACK_LIBRARY, where to find the LAPACK library.

set(LAPACK_NAMES ${LAPACK_NAMES} lapack)

# Check ATLAS paths preferentially, using this necessary hack (I love CMake).
find_library(LAPACK_LIBRARY
  NAMES ${LAPACK_NAMES}
  PATHS /usr/lib64/atlas /usr/lib/atlas /usr/local/lib64/atlas /usr/local/lib/atlas
  NO_DEFAULT_PATH)

find_library(LAPACK_LIBRARY
  NAMES ${LAPACK_NAMES}
  PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
  )

if (LAPACK_LIBRARY)
  set(LAPACK_LIBRARIES ${LAPACK_LIBRARY})
  set(LAPACK_FOUND "YES")
else ()
  set(LAPACK_FOUND "NO")
endif ()


if (LAPACK_FOUND)
   if (NOT LAPACK_FIND_QUIETLY)
      message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
   endif ()
else ()
   if (LAPACK_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find LAPACK")
   endif ()
endif ()

# Deprecated declarations.
get_filename_component (NATIVE_LAPACK_LIB_PATH ${LAPACK_LIBRARY} PATH)

mark_as_advanced(
  LAPACK_LIBRARY
  )

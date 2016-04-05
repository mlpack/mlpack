# - Find CBLAS (includes and library)
# This module defines
#  CBLAS_INCLUDE_DIR
#  CBLAS_LIBRARIES
#  CBLAS_FOUND
# also defined, but not for general use are
#  CBLAS_LIBRARY, where to find the library.

find_path(CBLAS_INCLUDE_DIR cblas.h
/usr/include/atlas/
/usr/local/include/atlas/
/usr/include/
/usr/local/include/
)

set(CBLAS_NAMES ${CBLAS_NAMES} cblas)
find_library(CBLAS_LIBRARY
  NAMES ${CBLAS_NAMES}
  PATHS /usr/lib64/atlas-sse3 /usr/lib64/atlas /usr/lib64 /usr/local/lib64/atlas /usr/local/lib64 /usr/lib/atlas-sse3 /usr/lib/atlas-sse2 /usr/lib/atlas-sse /usr/lib/atlas-3dnow /usr/lib/atlas /usr/lib /usr/local/lib/atlas /usr/local/lib
  )

if (CBLAS_LIBRARY AND CBLAS_INCLUDE_DIR)
    set(CBLAS_LIBRARIES ${CBLAS_LIBRARY})
    set(CBLAS_FOUND "YES")
else ()
  set(CBLAS_FOUND "NO")
endif ()


if (CBLAS_FOUND)
   if (NOT CBLAS_FIND_QUIETLY)
      message(STATUS "Found a CBLAS library: ${CBLAS_LIBRARIES}")
   endif ()
else ()
   if (CBLAS_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find a CBLAS library")
   endif ()
endif ()

# Deprecated declarations.
set (NATIVE_CBLAS_INCLUDE_PATH ${CBLAS_INCLUDE_DIR} )
get_filename_component (NATIVE_CBLAS_LIB_PATH ${CBLAS_LIBRARY} PATH)

mark_as_advanced(
  CBLAS_LIBRARY
  CBLAS_INCLUDE_DIR
  )

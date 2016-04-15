# - Find a version of CLAPACK (includes and library)
# This module defines
#  CLAPACK_INCLUDE_DIR
#  CLAPACK_LIBRARIES
#  CLAPACK_FOUND
# also defined, but not for general use are
#  CLAPACK_LIBRARY, where to find the library.

find_path(CLAPACK_INCLUDE_DIR clapack.h
/usr/include/atlas/
/usr/local/include/atlas/
/usr/include/
/usr/local/include/
)

set(CLAPACK_NAMES ${CLAPACK_NAMES} lapack_atlas)
set(CLAPACK_NAMES ${CLAPACK_NAMES} clapack)
find_library(CLAPACK_LIBRARY
  NAMES ${CLAPACK_NAMES}
  PATHS /usr/lib64/atlas-sse3 /usr/lib64/atlas /usr/lib64 /usr/local/lib64/atlas /usr/local/lib64 /usr/lib/atlas-sse3 /usr/lib/atlas-sse2 /usr/lib/atlas-sse /usr/lib/atlas-3dnow /usr/lib/atlas /usr/lib /usr/local/lib/atlas /usr/local/lib
  )

if (CLAPACK_LIBRARY AND CLAPACK_INCLUDE_DIR)
    set(CLAPACK_LIBRARIES ${CLAPACK_LIBRARY})
    set(CLAPACK_FOUND "YES")
else ()
  set(CLAPACK_FOUND "NO")
endif ()


if (CLAPACK_FOUND)
   if (NOT CLAPACK_FIND_QUIETLY)
      message(STATUS "Found a CLAPACK library: ${CLAPACK_LIBRARIES}")
   endif ()
else ()
   if (CLAPACK_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find a CLAPACK library")
   endif ()
endif ()

# Deprecated declarations.
set (NATIVE_CLAPACK_INCLUDE_PATH ${CLAPACK_INCLUDE_DIR} )
get_filename_component (NATIVE_CLAPACK_LIB_PATH ${CLAPACK_LIBRARY} PATH)

mark_as_advanced(
  CLAPACK_LIBRARY
  CLAPACK_INCLUDE_DIR
  )

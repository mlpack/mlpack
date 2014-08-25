# - Find a version of CLAPACK (includes and library)
# This module defines
#  CLAPACK_INCLUDE_DIR
#  CLAPACK_LIBRARIES
#  CLAPACK_FOUND
# also defined, but not for general use are
#  CLAPACK_LIBRARY, where to find the library.

FIND_PATH(CLAPACK_INCLUDE_DIR clapack.h
/usr/include/atlas/
/usr/local/include/atlas/
/usr/include/
/usr/local/include/
)

SET(CLAPACK_NAMES ${CLAPACK_NAMES} lapack_atlas)
SET(CLAPACK_NAMES ${CLAPACK_NAMES} clapack)
FIND_LIBRARY(CLAPACK_LIBRARY
  NAMES ${CLAPACK_NAMES}
  PATHS /usr/lib64/atlas-sse3 /usr/lib64/atlas /usr/lib64 /usr/local/lib64/atlas /usr/local/lib64 /usr/lib/atlas-sse3 /usr/lib/atlas-sse2 /usr/lib/atlas-sse /usr/lib/atlas-3dnow /usr/lib/atlas /usr/lib /usr/local/lib/atlas /usr/local/lib
  )

IF (CLAPACK_LIBRARY AND CLAPACK_INCLUDE_DIR)
    SET(CLAPACK_LIBRARIES ${CLAPACK_LIBRARY})
    SET(CLAPACK_FOUND "YES")
ELSE (CLAPACK_LIBRARY AND CLAPACK_INCLUDE_DIR)
  SET(CLAPACK_FOUND "NO")
ENDIF (CLAPACK_LIBRARY AND CLAPACK_INCLUDE_DIR)


IF (CLAPACK_FOUND)
   IF (NOT CLAPACK_FIND_QUIETLY)
      MESSAGE(STATUS "Found a CLAPACK library: ${CLAPACK_LIBRARIES}")
   ENDIF (NOT CLAPACK_FIND_QUIETLY)
ELSE (CLAPACK_FOUND)
   IF (CLAPACK_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find a CLAPACK library")
   ENDIF (CLAPACK_FIND_REQUIRED)
ENDIF (CLAPACK_FOUND)

# Deprecated declarations.
SET (NATIVE_CLAPACK_INCLUDE_PATH ${CLAPACK_INCLUDE_DIR} )
GET_FILENAME_COMPONENT (NATIVE_CLAPACK_LIB_PATH ${CLAPACK_LIBRARY} PATH)

MARK_AS_ADVANCED(
  CLAPACK_LIBRARY
  CLAPACK_INCLUDE_DIR
  )

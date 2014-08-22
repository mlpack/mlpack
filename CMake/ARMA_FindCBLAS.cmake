# - Find CBLAS (includes and library)
# This module defines
#  CBLAS_INCLUDE_DIR
#  CBLAS_LIBRARIES
#  CBLAS_FOUND
# also defined, but not for general use are
#  CBLAS_LIBRARY, where to find the library.

FIND_PATH(CBLAS_INCLUDE_DIR cblas.h
/usr/include/atlas/
/usr/local/include/atlas/
/usr/include/
/usr/local/include/
)

SET(CBLAS_NAMES ${CBLAS_NAMES} cblas)
FIND_LIBRARY(CBLAS_LIBRARY
  NAMES ${CBLAS_NAMES}
  PATHS /usr/lib64/atlas-sse3 /usr/lib64/atlas /usr/lib64 /usr/local/lib64/atlas /usr/local/lib64 /usr/lib/atlas-sse3 /usr/lib/atlas-sse2 /usr/lib/atlas-sse /usr/lib/atlas-3dnow /usr/lib/atlas /usr/lib /usr/local/lib/atlas /usr/local/lib
  )

IF (CBLAS_LIBRARY AND CBLAS_INCLUDE_DIR)
    SET(CBLAS_LIBRARIES ${CBLAS_LIBRARY})
    SET(CBLAS_FOUND "YES")
ELSE (CBLAS_LIBRARY AND CBLAS_INCLUDE_DIR)
  SET(CBLAS_FOUND "NO")
ENDIF (CBLAS_LIBRARY AND CBLAS_INCLUDE_DIR)


IF (CBLAS_FOUND)
   IF (NOT CBLAS_FIND_QUIETLY)
      MESSAGE(STATUS "Found a CBLAS library: ${CBLAS_LIBRARIES}")
   ENDIF (NOT CBLAS_FIND_QUIETLY)
ELSE (CBLAS_FOUND)
   IF (CBLAS_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find a CBLAS library")
   ENDIF (CBLAS_FIND_REQUIRED)
ENDIF (CBLAS_FOUND)

# Deprecated declarations.
SET (NATIVE_CBLAS_INCLUDE_PATH ${CBLAS_INCLUDE_DIR} )
GET_FILENAME_COMPONENT (NATIVE_CBLAS_LIB_PATH ${CBLAS_LIBRARY} PATH)

MARK_AS_ADVANCED(
  CBLAS_LIBRARY
  CBLAS_INCLUDE_DIR
  )

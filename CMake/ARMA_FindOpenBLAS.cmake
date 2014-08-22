# - Find the OpenBLAS library (no includes)
# This module defines
#  OpenBLAS_LIBRARIES, the libraries needed to use OpenBLAS.
#  OpenBLAS_FOUND, If false, do not try to use OpenBLAS.
# also defined, but not for general use are
#  OpenBLAS_LIBRARY, where to find the OpenBLAS library.

SET(OpenBLAS_NAMES ${OpenBLAS_NAMES} openblas)
FIND_LIBRARY(OpenBLAS_LIBRARY
  NAMES ${OpenBLAS_NAMES}
  PATHS /lib64 /lib /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
  )

IF (OpenBLAS_LIBRARY)
  SET(OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARY})
  SET(OpenBLAS_FOUND "YES")
ELSE (OpenBLAS_LIBRARY)
  SET(OpenBLAS_FOUND "NO")
ENDIF (OpenBLAS_LIBRARY)


IF (OpenBLAS_FOUND)
   IF (NOT OpenBLAS_FIND_QUIETLY)
      MESSAGE(STATUS "Found the OpenBLAS library: ${OpenBLAS_LIBRARIES}")
   ENDIF (NOT OpenBLAS_FIND_QUIETLY)
ELSE (OpenBLAS_FOUND)
   IF (OpenBLAS_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could not find the OpenBLAS library")
   ENDIF (OpenBLAS_FIND_REQUIRED)
ENDIF (OpenBLAS_FOUND)

# Deprecated declarations.
GET_FILENAME_COMPONENT (NATIVE_OpenBLAS_LIB_PATH ${OpenBLAS_LIBRARY} PATH)

MARK_AS_ADVANCED(
  OpenBLAS_LIBRARY
  )

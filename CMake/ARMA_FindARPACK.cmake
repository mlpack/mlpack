# - Try to find ARPACK
# Once done this will define
#
#  ARPACK_FOUND        - system has ARPACK
#  ARPACK_LIBRARY      - Link this to use ARPACK


find_library(ARPACK_LIBRARY
  NAMES arpack
  PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
  )


IF (ARPACK_LIBRARY)
  SET(ARPACK_FOUND YES)
ELSE ()
  # Search for PARPACK.
  find_library(ARPACK_LIBRARY
    NAMES parpack
    PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
  )

  IF (ARPACK_LIBRARY)
    SET(ARPACK_FOUND YES)
  ELSE ()
    SET(ARPACK_FOUND NO)
  ENDIF ()
ENDIF ()


IF (ARPACK_FOUND)
  IF (NOT ARPACK_FIND_QUIETLY)
     MESSAGE(STATUS "Found an ARPACK library: ${ARPACK_LIBRARY}")
  ENDIF (NOT ARPACK_FIND_QUIETLY)
ELSE (ARPACK_FOUND)
  IF (ARPACK_FIND_REQUIRED)
     MESSAGE(FATAL_ERROR "Could not find an ARPACK library")
  ENDIF (ARPACK_FIND_REQUIRED)
ENDIF (ARPACK_FOUND)

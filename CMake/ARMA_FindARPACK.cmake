# - Try to find ARPACK
# Once done this will define
#
#  ARPACK_FOUND        - system has ARPACK
#  ARPACK_LIBRARY      - Link this to use ARPACK


find_library(ARPACK_LIBRARY
  NAMES arpack
  PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
  )


if (ARPACK_LIBRARY)
  set(ARPACK_FOUND YES)
else ()
  # Search for PARPACK.
  find_library(ARPACK_LIBRARY
    NAMES parpack
    PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
  )

  if (ARPACK_LIBRARY)
    set(ARPACK_FOUND YES)
  else ()
    set(ARPACK_FOUND NO)
  endif ()
endif ()


if (ARPACK_FOUND)
  if (NOT ARPACK_FIND_QUIETLY)
     message(STATUS "Found an ARPACK library: ${ARPACK_LIBRARY}")
  endif ()
else ()
  if (ARPACK_FIND_REQUIRED)
     message(FATAL_ERROR "Could not find an ARPACK library")
  endif ()
endif ()

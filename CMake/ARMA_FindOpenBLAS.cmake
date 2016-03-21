# - Find the OpenBLAS library (no includes)
# This module defines
#  OpenBLAS_LIBRARIES, the libraries needed to use OpenBLAS.
#  OpenBLAS_FOUND, If false, do not try to use OpenBLAS.
# also defined, but not for general use are
#  OpenBLAS_LIBRARY, where to find the OpenBLAS library.

set(OpenBLAS_NAMES ${OpenBLAS_NAMES} openblas)
find_library(OpenBLAS_LIBRARY
  NAMES ${OpenBLAS_NAMES}
  PATHS /lib64 /lib /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
  )

if (OpenBLAS_LIBRARY)
  set(OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARY})
  set(OpenBLAS_FOUND "YES")
else ()
  set(OpenBLAS_FOUND "NO")
endif ()


if (OpenBLAS_FOUND)
   if (NOT OpenBLAS_FIND_QUIETLY)
      message(STATUS "Found the OpenBLAS library: ${OpenBLAS_LIBRARIES}")
   endif ()
else ()
   if (OpenBLAS_FIND_REQUIRED)
      message(FATAL_ERROR "Could not find the OpenBLAS library")
   endif ()
endif ()

# Deprecated declarations.
get_filename_component (NATIVE_OpenBLAS_LIB_PATH ${OpenBLAS_LIBRARY} PATH)

mark_as_advanced(
  OpenBLAS_LIBRARY
  )

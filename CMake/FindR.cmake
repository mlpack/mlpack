#
# - This module locates an installed R distribution.
#
# Defines the following:
#
# R_INCLUDE_DIR      - Path to R include directory
# R_LIBRARIES        - Path to R library
# R_LIBRARY_BASE     -
# R_EXECUTABLE       - Path to R command
# RSCRIPT_EXECUTABLE - Path to Rscript command
#
# Make sure find package macros are included
set(TEMP_CMAKE_FIND_APPBUNDLE ${CMAKE_FIND_APPBUNDLE})
set(CMAKE_FIND_APPBUNDLE "NEVER")

# Find R.
find_program(R_EXECUTABLE R DOC "R executable.")

if(R_EXECUTABLE)
  # Get the location of R.
  execute_process(
      WORKING_DIRECTORY .
      COMMAND ${R_EXECUTABLE} RHOME
      OUTPUT_VARIABLE R_BASE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Get the R version.
  execute_process(
      COMMAND ${R_EXECUTABLE} --version
      OUTPUT_VARIABLE R_VERSION_STRING
      RESULT_VARIABLE RESULT
  )
  if (RESULT EQUAL 0)
    string(REGEX REPLACE ".*([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1"
        R_VERSION_STRING ${R_VERSION_STRING})
  endif ()

  set(R_HOME ${R_BASE_DIR} CACHE PATH "R home directory obtained from R RHOME")
  mark_as_advanced(R_HOME)
endif()

# Find the Rscript program.
find_program(RSCRIPT_EXECUTABLE Rscript DOC "Rscript executable.")

set(CMAKE_FIND_APPBUNDLE ${TEMP_CMAKE_FIND_APPBUNDLE})

# R.h gets installed in all sorts of places -
# ubuntu: /usr/share/R/include, RHEL/Fedora: /usr/include/R/R.h
find_path(
    R_INCLUDE_DIR R.h
    PATHS ${R_INCLUDE_DIR_HINT} /usr/local/lib /usr/local/lib64 /usr/share /usr/include ${R_BASE_DIR}
    PATH_SUFFIXES include R R/include
    DOC "Path to file R.h"
)
find_library(
    R_LIBRARY_BASE R
    PATHS ${R_BASE_DIR}
    PATH_SUFFIXES /lib
    DOC "R library (example libR.a, libR.dylib, etc.)."
)

set(R_LIBRARIES ${R_LIBRARY_BASE})
mark_as_advanced(RSCRIPT_EXECUTABLE R_LIBRARIES R_INCLUDE_DIR R_EXECUTABLE R_LIBRARY_BASE)
set( _REQUIRED_R_VARIABLES R_INCLUDE_DIR R_EXECUTABLE )

if( APPLE )
  # On linux platform some times the libR.so is not available, however
  # on apple a link error results if the library is linked.
  list(  APPEND _REQUIRED_R_VARIABLES R_LIBRARIES R_LIBRARY_BASE )
endif()

include( FindPackageHandleStandardArgs )
find_package_handle_standard_args(
    R
    REQUIRED_VARS ${_REQUIRED_R_VARIABLES}
    VERSION_VAR R_VERSION_STRING
    FAIL_MESSAGE "R not found"
)

# This module looks for mex, the MATLAB compiler.
# The following variables are defined when the script completes:
#   MATLAB_MEX: location of mex compiler
#   MATLAB_ROOT: root of MATLAB installation
#   MATLABMEX_FOUND: 0 if not found, 1 if found

SET(MATLABMEX_FOUND 0)

IF(WIN32)
  # This is untested but taken from the older FindMatlab.cmake script as well as
  # the modifications by Ramon Casero and Tom Doel for Gerardus.

  # Search for a version of Matlab available, starting from the most modern one
  # to older versions.
  FOREACH(MATVER "7.20" "7.19" "7.18" "7.17" "7.16" "7.15" "7.14" "7.13" "7.12"
"7.11" "7.10" "7.9" "7.8" "7.7" "7.6" "7.5" "7.4")
    IF((NOT DEFINED MATLAB_ROOT)
        OR ("${MATLAB_ROOT}" STREQUAL "")
        OR ("${MATLAB_ROOT}" STREQUAL "/registry"))
      GET_FILENAME_COMPONENT(MATLAB_ROOT
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\${MATVER};MATLABROOT]"
        ABSOLUTE)
      SET(MATLAB_VERSION ${MATVER})
    ENDIF((NOT DEFINED MATLAB_ROOT)
      OR ("${MATLAB_ROOT}" STREQUAL "")
      OR ("${MATLAB_ROOT}" STREQUAL "/registry"))
  ENDFOREACH(MATVER)

  FIND_PROGRAM(MATLAB_MEX
    mex
    ${MATLAB_ROOT}/bin
    )
ELSE(WIN32)
  # Check if this is a Mac.
  IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # This code is untested but taken from the older FindMatlab.cmake script as
    # well as the modifications by Ramon Casero and Tom Doel for Gerardus.

   SET(LIBRARY_EXTENSION .dylib)

    # If this is a Mac and the attempts to find MATLAB_ROOT have so far failed,~
    # we look in the applications folder
    IF((NOT DEFINED MATLAB_ROOT) OR ("${MATLAB_ROOT}" STREQUAL ""))

    # Search for a version of Matlab available, starting from the most modern
    # one to older versions
      FOREACH(MATVER "R2013b" "R2013a" "R2012b" "R2012a" "R2011b" "R2011a"
"R2010b" "R2010a" "R2009b" "R2009a" "R2008b")
        IF((NOT DEFINED MATLAB_ROOT) OR ("${MATLAB_ROOT}" STREQUAL ""))
          IF(EXISTS /Applications/MATLAB_${MATVER}.app)
            SET(MATLAB_ROOT /Applications/MATLAB_${MATVER}.app)

          ENDIF(EXISTS /Applications/MATLAB_${MATVER}.app)
        ENDIF((NOT DEFINED MATLAB_ROOT) OR ("${MATLAB_ROOT}" STREQUAL ""))
      ENDFOREACH(MATVER)

    ENDIF((NOT DEFINED MATLAB_ROOT) OR ("${MATLAB_ROOT}" STREQUAL ""))

    FIND_PROGRAM(MATLAB_MEX
      mex
      PATHS
      ${MATLAB_ROOT}/bin
    )

  ELSE(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # On a Linux system.  The goal is to find MATLAB_ROOT.
    SET(LIBRARY_EXTENSION .so)

    FIND_PROGRAM(MATLAB_MEX_POSSIBLE_LINK
      mex
      PATHS
      ${MATLAB_ROOT}/bin
      /opt/matlab/bin
      /usr/local/matlab/bin
      $ENV{HOME}/matlab/bin
      # Now all the versions
      /opt/matlab/[rR]20[0-9][0-9][abAB]/bin
      /usr/local/matlab/[rR]20[0-9][0-9][abAB]/bin
      /opt/matlab-[rR]20[0-9][0-9][abAB]/bin
      /opt/matlab_[rR]20[0-9][0-9][abAB]/bin
      /usr/local/matlab-[rR]20[0-9][0-9][abAB]/bin
      /usr/local/matlab_[rR]20[0-9][0-9][abAB]/bin
      $ENV{HOME}/matlab/[rR]20[0-9][0-9][abAB]/bin
      $ENV{HOME}/matlab-[rR]20[0-9][0-9][abAB]/bin
      $ENV{HOME}/matlab_[rR]20[0-9][0-9][abAB]/bin
    )

    GET_FILENAME_COMPONENT(MATLAB_MEX "${MATLAB_MEX_POSSIBLE_LINK}" REALPATH)
    GET_FILENAME_COMPONENT(MATLAB_BIN_ROOT "${MATLAB_MEX}" PATH)
    # Strip ./bin/.
    GET_FILENAME_COMPONENT(MATLAB_ROOT "${MATLAB_BIN_ROOT}" PATH)
  ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
ENDIF(WIN32)

IF(NOT EXISTS "${MATLAB_MEX}" AND "${MatlabMex_FIND_REQUIRED}")
  MESSAGE(FATAL_ERROR "Could not find MATLAB mex compiler; try specifying MATLAB_ROOT.")
ELSE(NOT EXISTS "${MATLAB_MEX}" AND "${MatlabMex_FIND_REQUIRED}")
  IF(EXISTS "${MATLAB_MEX}")
    MESSAGE(STATUS "Found MATLAB mex compiler: ${MATLAB_MEX}")
    MESSAGE(STATUS "MATLAB root: ${MATLAB_ROOT}")
    SET(MATLABMEX_FOUND 1)
  ENDIF(EXISTS "${MATLAB_MEX}")
ENDIF(NOT EXISTS "${MATLAB_MEX}" AND "${MatlabMex_FIND_REQUIRED}")

MARK_AS_ADVANCED(
  MATLAB_MEX
  MATLABMEX_FOUND
  MATLAB_ROOT
)


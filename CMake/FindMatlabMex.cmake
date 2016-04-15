# This module looks for mex, the MATLAB compiler.
# The following variables are defined when the script completes:
#   MATLAB_MEX: location of mex compiler
#   MATLAB_ROOT: root of MATLAB installation
#   MATLABMEX_FOUND: 0 if not found, 1 if found

set(MATLABMEX_FOUND 0)

if(WIN32)
  # This is untested but taken from the older FindMatlab.cmake script as well as
  # the modifications by Ramon Casero and Tom Doel for Gerardus.

  # Search for a version of Matlab available, starting from the most modern one
  # to older versions.
  foreach(MATVER "7.20" "7.19" "7.18" "7.17" "7.16" "7.15" "7.14" "7.13" "7.12"
"7.11" "7.10" "7.9" "7.8" "7.7" "7.6" "7.5" "7.4")
    if((NOT DEFINED MATLAB_ROOT)
        OR ("${MATLAB_ROOT}" STREQUAL "")
        OR ("${MATLAB_ROOT}" STREQUAL "/registry"))
      get_filename_component(MATLAB_ROOT
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\${MATVER};MATLABROOT]"
        ABSOLUTE)
      set(MATLAB_VERSION ${MATVER})
    endif()
      OR ("${MATLAB_ROOT}" STREQUAL "")
      OR ("${MATLAB_ROOT}" STREQUAL "/registry"))
  endforeach()

  find_program(MATLAB_MEX
    mex
    ${MATLAB_ROOT}/bin
    )
else()
  # Check if this is a Mac.
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # This code is untested but taken from the older FindMatlab.cmake script as
    # well as the modifications by Ramon Casero and Tom Doel for Gerardus.

   set(LIBRARY_EXTENSION .dylib)

    # If this is a Mac and the attempts to find MATLAB_ROOT have so far failed,~
    # we look in the applications folder
    if((NOT DEFINED MATLAB_ROOT) OR ("${MATLAB_ROOT}" STREQUAL ""))

    # Search for a version of Matlab available, starting from the most modern
    # one to older versions
      foreach(MATVER "R2013b" "R2013a" "R2012b" "R2012a" "R2011b" "R2011a"
"R2010b" "R2010a" "R2009b" "R2009a" "R2008b")
        if((NOT DEFINED MATLAB_ROOT) OR ("${MATLAB_ROOT}" STREQUAL ""))
          if(EXISTS /Applications/MATLAB_${MATVER}.app)
            set(MATLAB_ROOT /Applications/MATLAB_${MATVER}.app)

          endif()
        endif()
      endforeach()

    endif()

    find_program(MATLAB_MEX
      mex
      PATHS
      ${MATLAB_ROOT}/bin
    )

  else()
    # On a Linux system.  The goal is to find MATLAB_ROOT.
    set(LIBRARY_EXTENSION .so)

    find_program(MATLAB_MEX_POSSIBLE_LINK
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

    get_filename_component(MATLAB_MEX "${MATLAB_MEX_POSSIBLE_LINK}" REALPATH)
    get_filename_component(MATLAB_BIN_ROOT "${MATLAB_MEX}" PATH)
    # Strip ./bin/.
    get_filename_component(MATLAB_ROOT "${MATLAB_BIN_ROOT}" PATH)
  endif()
endif()

if(NOT EXISTS "${MATLAB_MEX}" AND "${MatlabMex_FIND_REQUIRED}")
  message(FATAL_ERROR "Could not find MATLAB mex compiler; try specifying MATLAB_ROOT.")
else()
  if(EXISTS "${MATLAB_MEX}")
    message(STATUS "Found MATLAB mex compiler: ${MATLAB_MEX}")
    message(STATUS "MATLAB root: ${MATLAB_ROOT}")
    set(MATLABMEX_FOUND 1)
  endif()
endif()

mark_as_advanced(
  MATLAB_MEX
  MATLABMEX_FOUND
  MATLAB_ROOT
)


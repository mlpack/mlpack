# Our goal is to make sure that the toolbox/mlpack directory is in the MATLAB
# default path.  It is possible that that directory is already in the path, so
# we must consider that possibility too.
#
# This script assumes that ${MATLAB_ROOT} is set and writes the (potentially)
# modified file to ${PATHDEF_OUTPUT_FILE}.

# This could potentially be incorrect for older versions of MATLAB.
file(READ "${MATLAB_ROOT}/toolbox/local/pathdef.m" PATHDEF)

string(REGEX MATCH "matlabroot,'/toolbox/mlpack:',[ ]*..." MLPACK_PATHDEF
    "${PATHDEF}")

if("${MLPACK_PATHDEF}" STREQUAL "")
  # The MLPACK toolbox does not exist in the path.  Therefore we have to modify
  # the file.
  string(REPLACE "%%% END ENTRIES %%%"
      "matlabroot,'/toolbox/mlpack:', ...\n%%% END ENTRIES %%%" MOD_PATHDEF
      "${PATHDEF}")

  file(WRITE "${PATHDEF_OUTPUT_FILE}" "${MOD_PATHDEF}")
else()
  # Write unmodified file.
  file(WRITE "${PATHDEF_OUTPUT_FILE}" "${PATHDEF}")
endif()

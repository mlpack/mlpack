# Validate md5 hash given md5file url and the file directory.

# This module does the following:
# HASH_CHECK_FAIL set to true if the hash check fails.
# Remove the downloaded files.

macro (check_hash MD5_URL DIR HASH_CHECK_FAIL)
  file(DOWNLOAD MD5_URL "${DIR}/hash.md5")
  execute_process(COMMAND ${CMAKE_COMMAND} -E md5sum -c "${DIR}/hash.md5"
      RESULT_VARIABLE HASH_CHECK_FAIL)
  if (HASH_CHECK_FAIL)
    file(REMOVE_RECURSE DIR)
    message(WARNING
        "Could not verify md5 hash! Error code ${HASH_CHECK_FAIL}")
  endif ()
endmacro (check_hash)

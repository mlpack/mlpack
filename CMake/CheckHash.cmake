# Validate md5 hash given md5file url and the file directory.

# This module does the following on hash failure:
# Set HASH_CHECK_FAIL to 1.
# Remove the downloaded files.

macro (check_hash MD5_URL DIR HASH_CHECK_FAIL)
  set(HASH_CHECK_FAIL 0)
  file(DOWNLOAD ${MD5_URL}
      "${DIR}/hash.md5"
      STATUS MD5_DOWNLOAD_STATUS_LIST)
  list(GET MD5_DOWNLOAD_STATUS_LIST 0 MD5_DOWNLOAD_STATUS)
  if (MD5_DOWNLOAD_STATUS EQUAL 0)
    file(STRINGS "${DIR}/hash.md5" HASH_DATA NEWLINE_CONSUME)
    string(REGEX REPLACE "\n" ";" HASH_LIST "${HASH_DATA}")
    foreach(item ${HASH_LIST})
      string(SUBSTRING ${item} 0 32 EXPECTED_HASH)
      string(SUBSTRING ${item} 34 -1 FILE_NAME)
      file(MD5 "${DIR}/${FILE_NAME}" LOCAL_HASH)
      if (NOT LOCAL_HASH STREQUAL EXPECTED_HASH)
        set(HASH_CHECK_FAIL 1)
        file(REMOVE_RECURSE ${DIR})
        message(WARNING
            "md5sum verification error for ${item}!  Got ${LOCAL_HASH}, expected ${EXPECTED_HASH}.")
        break()
      endif()
    endforeach()
  else ()
    set(HASH_CHECK_FAIL 1)
    file(REMOVE_RECURSE ${DIR})
    list(GET MD5_DOWNLOAD_STATUS_LIST 1 MD5_DOWNLOAD_ERROR)
    message(WARNING
        "Could not download the md5 for hash verification! Error code ${MD5_DOWNLOAD_STATUS}: ${MD5_DOWNLOAD_ERROR}!")
  endif ()
endmacro (check_hash)

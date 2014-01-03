# Using the CMake subversion tools, create svnversion.hpp, which just contains
# the implementation of GetVersion() assuming that we are working inside of a
# subversion repository.
include(FindSubversion)
Subversion_WC_INFO(${CMAKE_SOURCE_DIR} MLPACK)

# Get the current version, if it exists.
if(EXISTS ${CMAKE_SOURCE_DIR}/src/mlpack/core/util/svnversion.hpp)
  file(READ ${CMAKE_SOURCE_DIR}/src/mlpack/core/util/svnversion.hpp
      _OLD_SVNVERSION_CONTENTS)
  string(REGEX REPLACE ".*return \"mlpack trunk-r([0-9]+)\".*" "\\1"
      OLD_SVN_REVISION ${_OLD_SVNVERSION_CONTENTS})
else(EXISTS ${CMAKE_SOURCE_DIR}/src/mlpack/core/util/svnversion.hpp)
  set(OLD_SVN_REVISION "notfound")
endif(EXISTS ${CMAKE_SOURCE_DIR}/src/mlpack/core/util/svnversion.hpp)

if("${OLD_SVN_REVISION}" STREQUAL "${MLPACK_WC_REVISION}")
  message(STATUS "svnversion.hpp is already up to date.")
else("${OLD_SVN_REVISION}" STREQUAL "${MLPACK_WC_REVISION}")
  # Remove the old version.
  file(REMOVE ${CMAKE_SOURCE_DIR}/src/mlpack/core/util/svnversion.hpp)
  file(WRITE ${CMAKE_SOURCE_DIR}/src/mlpack/core/util/svnversion.hpp
      "return \"mlpack trunk-r${MLPACK_WC_REVISION}\";\n")
  message(STATUS "Updated svnversion.hpp.")
endif("${OLD_SVN_REVISION}" STREQUAL "${MLPACK_WC_REVISION}")

# Using the CMake tools to create the gitversion.hpp, which just contains
# the implementation of GetVersion() assuming that we are working inside of a
# git repository.
find_package(Git)

execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE NEW_GIT_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

# Get the current version, if it exists.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/mlpack/core/util/gitversion.hpp)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/mlpack/core/util/gitversion.hpp
      _OLD_GITVERSION_CONTENTS)
  string(REGEX REPLACE ".*return \"mlpack git-([0-9a-f]+)\".*" "\\1"
      OLD_GIT_REVISION ${_OLD_GITVERSION_CONTENTS})
else()
  set(OLD_GIT_REVISION "notfound")
endif()

if("${OLD_GIT_REVISION}" STREQUAL "${NEW_GIT_REVISION}")
  message(STATUS "gitversion.hpp is already up to date.")
else()
  # Remove the old version.
  file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/src/mlpack/core/util/gitversion.hpp)
  file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/src/mlpack/core/util/gitversion.hpp
      "return \"mlpack git-${NEW_GIT_REVISION}\";\n")
  message(STATUS "Updated gitversion.hpp.")
endif()

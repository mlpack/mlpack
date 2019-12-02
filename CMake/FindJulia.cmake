# FindJulia.cmake
#
# Based on Jay's implementation:
# https://gist.github.com/JayKickliter/06d0e7c4f84ef7ccc7a9
if (JULIA_FOUND)
  return()
endif()

# Find the Julia program.
find_program(JULIA_EXECUTABLE julia DOC "Julia executable")

# Get the Julia version.
if (JULIA_EXECUTABLE)
  execute_process(
      COMMAND ${JULIA_EXECUTABLE} --version
      OUTPUT_VARIABLE JULIA_VERSION_STRING
      RESULT_VARIABLE RESULT
  )
  if (RESULT EQUAL 0)
    string(REGEX REPLACE ".*([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1"
        JULIA_VERSION_STRING ${JULIA_VERSION_STRING})
  endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Julia
    REQUIRED_VARS JULIA_EXECUTABLE
    VERSION_VAR JULIA_VERSION_STRING
    FAIL_MESSAGE "Julia not found"
)

# FindGo.cmake
if (GO_FOUND)
  return()
endif()

# Find the Go program.
find_program(GO_EXECUTABLE go DOC "Go executable")

# Get the Go version.
if (GO_EXECUTABLE)
  execute_process(
       COMMAND ${GO_EXECUTABLE} version 
       OUTPUT_VARIABLE GO_VERSION_STRING
       RESULT_VARIABLE RESULT
  )
  if (RESULT EQUAL 0)
    string(REGEX REPLACE ".*([0-9]+\\.[0-9]+\(\\.[0-9]+\)?).*" "\\1"
        GO_VERSION_STRING ${GO_VERSION_STRING})
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Go
    REQUIRED_VARS GO_EXECUTABLE
    VERSION_VAR GO_VERSION_STRING
    FAIL_MESSAGE "Go not found"
)

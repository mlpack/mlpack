# FindGonum.cmake
#
# Find the Gonum program.
if (GO_EXECUTABLE)
  execute_process(
    COMMAND ${GO_EXECUTABLE} env -w GO111MODULE=auto
  )
  execute_process(
     COMMAND ${GO_EXECUTABLE} list -m gonum.org/v1/gonum
     OUTPUT_VARIABLE GONUM_VERSION_STRING
     RESULT_VARIABLE RESULT
  )
  if (RESULT EQUAL 0)
    string(REGEX REPLACE ".*([0-9]+\\.[0-9]+\\.[0-9]+[\n]+).*" "\\1"
        GONUM_VERSION_STRING ${GONUM_VERSION_STRING})
    string(REGEX REPLACE "\n$" ""
        GONUM_VERSION_STRING ${GONUM_VERSION_STRING})
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Gonum
  REQUIRED_VARS GONUM_VERSION_STRING
  FAIL_MESSAGE "Gonum not found"
)

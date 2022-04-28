# FindGonum.cmake
#
# Find the Gonum program.
if (GO_EXECUTABLE)
  execute_process(
     COMMAND ${GO_EXECUTABLE} list gonum.org/v1/gonum/mat
     OUTPUT_VARIABLE GONUM_RAW_STRING
     RESULT_VARIABLE RESULT
  )
  if (RESULT EQUAL 0)
    string(REGEX REPLACE "\n$" ""
    GONUM_RAW_STRING ${GONUM_RAW_STRING})
    if ("${GONUM_RAW_STRING}" STREQUAL "gonum.org/v1/gonum/mat")
      set(GONUM_VERSION_STRING "0.0.1")
    # string(REGEX MATCH "([0-9]+\\.[0-9]+\\.[0-9]+[\n]+)"
    #     GONUM_VERSION_STRING "${GONUM_VERSION_STRING}")
    # string(REGEX REPLACE "\n$" ""
    #     GONUM_VERSION_STRING ${GONUM_VERSION_STRING})
    endif()
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Gonum
  REQUIRED_VARS GONUM_VERSION_STRING
  FAIL_MESSAGE "Gonum not found"
)

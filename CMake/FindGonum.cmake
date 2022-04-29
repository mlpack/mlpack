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
      set(GONUM_FOUND 1)
    endif()
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Gonum
  REQUIRED_VARS GONUM_FOUND
  FAIL_MESSAGE "Gonum not found"
)

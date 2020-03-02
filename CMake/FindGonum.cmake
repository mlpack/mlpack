# The module defines the following variables:
#   GONUM_FOUND - true if the Gonum was found
#   GONUM - Gonum version number

if(GO_EXECUTABLE)
  execute_process(
     COMMAND ${GO_EXECUTABLE} list gonum.org/v1/gonum/mat
     OUTPUT_VARIABLE GONUM_VERSION_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(GONUM_VERSION_OUTPUT MATCHES "gonum.org/v1/gonum/mat")
      set(GONUM ${GONUM_VERSION_OUTPUT})
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Gonum
  REQUIRED_VARS GONUM
  FAIL_MESSAGE "Gonum not found"
)

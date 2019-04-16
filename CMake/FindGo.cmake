# Find the Go package.

# The module defines the following variables:
#   GO_FOUND - true if the Go was found
#   GO_EXECUTABLE - path to the executable
#   GO_VERSION - Go version number
#   GO_PLATFORM - i.e. linux
#   GO_ARCH - i.e. amd64

if (GO_FOUND)
  return()
endif()

find_program(GO_EXECUTABLE go PATHS $ENV{HOME}/go ENV GOROOT GOPATH PATH_SUFFIXES bin)
if(GO_EXECUTABLE)
  if (DEFINED ENV{GOROOT})
    set(GO_ROOT "$ENV{GOROOT}")
  else()
    set(GO_ROOT "/usr/lib/go")
  endif()
endif()

# Get the Go version.
if (GO_EXECUTABLE)
  execute_process(
       COMMAND ${GO_EXECUTABLE} version 
       OUTPUT_VARIABLE GO_VERSION_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE
  )
    if(GO_VERSION_OUTPUT MATCHES "go([0-9]+\\.[0-9]+\\.?[0-9]*)[a-zA-Z0-9]* ([^/]+)/(.*)")
        set(GO_VERSION ${CMAKE_MATCH_1})
        set(GO_PLATFORM ${CMAKE_MATCH_2})
        set(GO_ARCH ${CMAKE_MATCH_3})
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Go
    REQUIRED_VARS GO_EXECUTABLE
    VERSION_VAR GO_VERSION
    FAIL_MESSAGE "Go not found"
)


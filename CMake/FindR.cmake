# FindR.cmake
if (R_FOUND)
  return()
endif()

# Find the R and Rscript program.
find_program ( 
    RSCRIPT_EXECUTABLE
    NAMES Rscript Rscript.exe
    DOC "Path to the Rscript command interpreter"
)

find_program (
    R_EXECUTABLE
    NAMES R R.exe
    DOC "Path to the R command interpreter"
)

# Get the R version.
if (R_EXECUTABLE AND RSCRIPT_EXECUTABLE)
  execute_process(
      COMMAND ${R_EXECUTABLE} --version
      OUTPUT_VARIABLE R_VERSION_STRING
      RESULT_VARIABLE RESULT
  )
  if (RESULT EQUAL 0)
    string(REGEX REPLACE ".*([0-9]+\\.[0-9]+\\.[0-9]+).*" "\\1"
        R_VERSION_STRING ${R_VERSION_STRING})
  endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    R
    REQUIRED_VARS R_EXECUTABLE RSCRIPT_EXECUTABLE
    VERSION_VAR R_VERSION_STRING
    FAIL_MESSAGE "R not found"
)

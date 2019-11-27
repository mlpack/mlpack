# RunProgram.cmake: a CMake script that actually runs the given program to
# generate a file, which is output into the given directory.
#
# This script depends on the following arguments:
#
#   PROGRAM: the program to run to.
#   OUTPUT_FILE: the file to store the output in.
execute_process(COMMAND ${PROGRAM} OUTPUT_FILE ${OUTPUT_FILE}
    ERROR_VARIABLE err)

if (err)
  message(FATAL_ERROR "Fatal error running ${PROGRAM}: ${err}!")
endif ()

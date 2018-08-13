# GenerateH.cmake: a CMake script that actually runs the given program to
# generate a .h file.
#
# This script depends on the following arguments:
#
#   GENERATE_H_PROGRAM: the program to run to generate the .h file.
#   H_OUTPUT_FILE: the file to store the output in.
execute_process(COMMAND ${GENERATE_H_PROGRAM} OUTPUT_FILE ${H_OUTPUT_FILE})

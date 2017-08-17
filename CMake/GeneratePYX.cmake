# GeneratePYX.cmake: a CMake script that actually runs the given program to
# generate a .pyx file.
#
# This script depends on the following arguments:
#
#   GENERATE_PYX_PROGRAM: the program to run to generate the .pyx file.
#   PYX_OUTPUT_FILE: the file to store the output in.
execute_process(COMMAND ${GENERATE_PYX_PROGRAM} OUTPUT_FILE ${PYX_OUTPUT_FILE})

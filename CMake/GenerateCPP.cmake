# GenerateCPP.cmake: a CMake script that actually runs the given program to
# generate a .cpp file.
#
# This script depends on the following arguments:
#
#   GENERATE_CPP_PROGRAM: the program to run to generate the .cpp file.
#   CPP_OUTPUT_FILE: the file to store the output in.
execute_process(COMMAND ${GENERATE_CPP_PROGRAM} OUTPUT_FILE ${CPP_OUTPUT_FILE})

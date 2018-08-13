# GenerateGo.cmake: a CMake script that actually runs the given program to
# generate a .go file.
#
# This script depends on the following arguments:
#
#   GENERATE_GO_PROGRAM: the program to run to generate the .go file.
#   GO_OUTPUT_FILE: the file to store the output in.
execute_process(COMMAND ${GENERATE_GO_PROGRAM} OUTPUT_FILE ${GO_OUTPUT_FILE})

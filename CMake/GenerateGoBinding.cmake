# GenerateGoBinding.cmake: a CMake script that actually runs the given program to
# generate an mlpack binding file.
#
# This script depends on the following arguments:
#
#   GENERATE_H_PROGRAM: the program to run to generate the .h file.
#   H_OUTPUT_FILE: the file to store the output in.
#   GENERATE_GO_PROGRAM: the program to run to generate the .go file.
#   GO_OUTPUT_FILE: the file to store the output in.
#   GENERATE_CPP_PROGRAM: the program to run to generate the .cpp file.
#   CPP_OUTPUT_FILE: the file to store the output in.
execute_process(COMMAND ${GENERATE_BINDING_PROGRAM}
                OUTPUT_FILE ${BINDING_OUTPUT_FILE})

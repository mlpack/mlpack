# GenerateBinding.cmake: a CMake script that actually runs the given program to
# generate an mlpack binding file.
#
# This script depends on the following arguments:
#
#   GENERATE_BINDING_PROGRAM: the program to run to generate the binding file.
#   BINDING_OUTPUT_FILE: the file to store the output in.
execute_process(COMMAND ${GENERATE_BINDING_PROGRAM}
                OUTPUT_FILE ${BINDING_OUTPUT_FILE})

# ConfigureGoHCPP.cmake: generate an mlpack .h file for a Go binding given
# input arguments.
#
# This file depends on the following variables being set:
#
#  * PROGRAM_NAME: name of the binding
#  * PROGRAM_MAIN_FILE: the file containing the mlpackMain() function.
#  * GENERATE_GO_IN: path of the generate_go.cpp.in file.
#  * GENERATE_GO_OUT: name of the output .go file.
#  * GENERATE_CPP_IN: path of the generate_cpp.cpp.in file.
#  * GENERATE_CPP_OUT: name of the output .cpp file.
#  * GENERATE_H_IN: path of the generate_h.cpp.in file.
#  * GENERATE_H_OUT: name of the output .h file.
configure_file("${GENERATE_BINDING_IN}" "${GENERATE_BINDING_OUT}")

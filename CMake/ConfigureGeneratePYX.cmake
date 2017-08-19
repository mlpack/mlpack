# ConfigureGeneratePYX.cmake: generate an mlpack .pyx file given input
# arguments.
#
# This file depends on the following variables being set:
#
#  * GENERATE_CPP_IN: the .cpp.in file to configure.
#  * GENERATE_CPP_OUT: the .cpp file we'll generate.
#  * PROGRAM_MAIN_FILE: the file containing the main() function.
#  * PROGRAM_NAME: the name of the program (i.e. "pca").
configure_file(${GENERATE_CPP_IN} ${GENERATE_CPP_OUT})

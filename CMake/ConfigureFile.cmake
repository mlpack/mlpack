# ConfigureFile.cmake: generate an mlpack binding file given input
# arguments.
#
# This file depends on the following variables being set:
#
#  * GENERATE_CPP_IN: the .cpp.in file to configure.
#  * GENERATE_CPP_OUT: the .cpp file we'll generate.
#
# Any other defined variables will be passed on to the file that is being
# generated.
configure_file(${GENERATE_CPP_IN} ${GENERATE_CPP_OUT})

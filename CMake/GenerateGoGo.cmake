# GenerateGoInclude.cmake: generate an mlpack .go include file for a Go
# binding.
#
# This file depends on the following variables being set:
#
#  * GENERATE_GO_IN: the .go in file to configure.
#  * GENERATE_GO_OUT: the .go file we'll generate.
#  * PROGRAM_MAIN_FILE: the file containing the main() function.
#  * PROGRAM_NAME: the name of the program (i.e. "pca").
configure_file(${GENERATE_GO_IN} ${GENERATE_GO_OUT})

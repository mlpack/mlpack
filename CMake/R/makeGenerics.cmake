# makeGenerics.cmake: make a generic for S3 dispatch of predict()
# on the given pointer class.
#
# This file is called wit
# - NAME: the binding name
# - DIRECTORY: the methods directory name
# - R_GENERICS_IN: the name of the base file
# - R_GENERICS_OUT: the name of the created file
#message(STATUS "Called Generics")

# Now configure the file.
configure_file("${R_GENERICS_IN}" "${R_GENERICS_OUT}")

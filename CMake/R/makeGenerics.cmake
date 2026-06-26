# makeGenerics.cmake: make a generic for S3 dispatch of predict()
# on the given pointer class.
#
# This file is called with
# - NAME: the binding name
# - R_GENERICS_IN: the name of the base file
# - R_GENERICS_OUT: the name of the created file
# - VALID_METHODS: the (supplied) method list, eg "train;predict;probabilities"
#           which is a driver of differentiation in the created code
# - TYPES: a string with the types of supported results, usually
#          'c("predict", "probabilities")
# - ELSE: the 'else' branch of code (as string) in case of probabilities
# - ACTION: the 'action' generally one of 'predict' or 'classify'

# Now configure the file.
configure_file("${R_GENERICS_IN}" "${R_GENERICS_OUT}")

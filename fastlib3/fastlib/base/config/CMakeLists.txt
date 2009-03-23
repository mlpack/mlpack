cmake_minimum_required(VERSION 2.6)

ADD_EXECUTABLE ( template_types template_types.c )

# Create the sub-directory called base under the current build directory.
MAKE_DIRECTORY( ${CMAKE_CURRENT_BINARY_DIR}/base
)

# Compile template_types.c and generate basic_types.h from it.
ADD_CUSTOM_COMMAND(
  OUTPUT basic_types.h
  COMMAND template_types > ${CMAKE_CURRENT_BINARY_DIR}/base/basic_types.h
  DEPENDS template_types
)

ADD_CUSTOM_TARGET( template_types_detect ALL DEPENDS
  basic_types.h
)

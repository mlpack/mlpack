cmake_minimum_required(VERSION 2.6)

add_executable( template_types template_types.c )

# # Create the sub-directory called base under the current build directory.
# make_directory( ${CMAKE_CURRENT_BINARY_DIR}/base
# )

# Compile template_types.c and generate basic_types.h from it.
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/fastlib/base/basic_types.h 
  #  COMMAND template_types > ${CMAKE_CURRENT_BINARY_DIR}/base/basic_types.h
  #${CMAKE_CURRENT_BINARY_DIR} = fastlib/build/base/config
  ##!COMMAND template_types > ${CMAKE_CURRENT_BINARY_DIR}/../basic_types.h
  COMMAND template_types > ${CMAKE_BINARY_DIR}/fastlib/base/basic_types.h
  DEPENDS template_types
)

add_custom_target( template_types_detect ALL DEPENDS
#add_custom_target( template_types_detect DEPENDS
  ${CMAKE_BINARY_DIR}/fastlib/base/basic_types.h 
)
##!set(FASTLIB_SRCS ${FASTLIB_SRCS} ${CMAKE_CURRENT_BINARY_DIR}/../basic_types.h PARENT_SCOPE )
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${CMAKE_BINARY_DIR}/fastlib/base/basic_types.h PARENT_SCOPE )

install(FILES ${CMAKE_BINARY_DIR}/fastlib/base/basic_types.h DESTINATION include/fastlib/base ) #!??

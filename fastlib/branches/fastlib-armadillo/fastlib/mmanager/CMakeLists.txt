cmake_minimum_required(VERSION 2.8)

# Define the files that we need to compile
# Anything not in this list will not be compiled into the library
# Do not include test programs here
set(SOURCES
   memory_manager.h
   memory_manager.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all FASTLIB sources (used at parent scope)
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

#add_library(file ${SOURCES})
# dependencies of file
#target_link_libraries(file
#   base
#   col
#)

# test executable
add_executable(memory_manager_test
   memory_manager_test.cc
)
# dependencies of test executable
target_link_libraries(memory_manager_test
#   base
#   col
#   file
   fastlib
)

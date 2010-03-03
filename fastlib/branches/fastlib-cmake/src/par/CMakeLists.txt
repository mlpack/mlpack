cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   grain.h
   task.h
   thread.h
   thread.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all FASTLIB sources (used at the parent scope)
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_library(par ${SOURCES})
# link dependencies of data
target_link_libraries(par
   base
   col
)

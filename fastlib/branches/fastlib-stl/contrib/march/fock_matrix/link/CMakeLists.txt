cmake_minimum_required(VERSION 2.8)

# This is not compiled into mlpack_contrib by default, since it appears to be custom one-time-use
# code that is not useful anywhere other than this one particular test

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # link_exchange
   link.h
   ../fock_impl/basis_shell.h
   ../fock_impl/shell_pair.h
   link.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(link_exchange
   link_main.cc
)
# link dependencies of executable
target_link_libraries(link_exchange
   fastlib
   mlpack
   mlpack_contrib
)

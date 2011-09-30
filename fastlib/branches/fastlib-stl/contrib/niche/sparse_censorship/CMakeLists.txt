cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
# In this library, these are specified twice, once here, and once for the individual library it belongs to, so make sure
# that you have files in both sections
set(SOURCES
  # for Sparse Censorship
   sparse_censorship.h
   sparse_censorship_impl.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
#set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# The executable for Sparse Censorship

add_executable(censorship
    EXCLUDE_FROM_ALL
    censorship_main.cc
)

target_link_libraries(censorship
#    mlpack
    fastlib
)

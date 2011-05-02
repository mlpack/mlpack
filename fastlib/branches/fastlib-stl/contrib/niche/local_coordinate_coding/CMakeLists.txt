cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
# In this library, these are specified twice, once here, and once for the individual library it belongs to, so make sure
# that you have files in both sections
set(SOURCES
   local_coordinate_coding.h
   local_coordinate_coding_impl.h
   lars.h
   tools.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
#set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# The executable for local coordinate coding
add_executable(lcc
    EXCLUDE_FROM_ALL
    main.cc
)

target_link_libraries(lcc
   fastlib
)

# Test neighbor counts
#add_executable(test_neighbor_counts
#    EXCLUDE_FROM_ALL
#    test_neighbor_counts.cc
#)

#target_link_libraries(test_neighbor_counts
#    fastlib
#)

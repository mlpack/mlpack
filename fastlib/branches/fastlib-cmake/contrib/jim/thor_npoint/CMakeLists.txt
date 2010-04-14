cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # thor_npc
#   thor_npoint.h  # was in build.py but does not exist
   dfs3.h
   dfs3_impl.h
   two_point.h
   three_point.h
   thor_3point.h
   # thor_2pc
   thor_2point.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(thor_2pc
   thor_2pt_main.cc
)
# link dependencies of executable
target_link_libraries(thor_2pc
   fastlib
   mlpack
   mlpack_contrib
)

# Does not compile correctly!
#add_executable(thor_3pc
#   thor_3pt_main.cc
#)
#target_link_libraries(thor_3pc
#   fastlib
#   mlpack
#   mlpack_contrib
#)

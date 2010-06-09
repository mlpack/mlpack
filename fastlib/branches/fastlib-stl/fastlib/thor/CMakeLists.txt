cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   blockdev.h
   cachearray.h
   cachearray_impl.h
   cache.h
   dfs.h
   dfs_impl.h
   rbfs.h
   rbfs_impl.h
   distribcache.h
   gnp.h
   kdtree.h
   kdtree_impl.h
   rpc_base.h
   rpc.h
   rpc_sock.h
   thor.h
   thor_struct.h
   thortree.h
   thortree_impl.h
   thor_utils.h
   thor_utils_impl.h
   sched.h
   sched_impl.h
   distribcache.cc
   blockdev.cc
   sched.cc
   rpc.cc
   rpc_sock.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all FASTLIB sources (used at the parent scope)
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

#add_library(col ${SOURCES})
#add_dependencies(col base) # depends on base library

# test executable
#add_executable(allnn_thor
#   allnn.cc
#)
# link col_test against col and base libraries
#target_link_libraries(allnn_thor
#   fastlib
#)

#add_executable(allnnbfs_thor
#   allnnbfs.cc
#)
#target_link_libraries(allnnbfs_thor
#   fastlib
#)

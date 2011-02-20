cmake_minimum_required(VERSION 2.6)

set(SOURCES
  distributed_dualtree_dfs_dev.h
  distributed_dualtree_dfs.h
  distributed_tripletree_dfs_dev.h
  distributed_tripletree_dfs.h
  dualtree_dfs_dev.h
  dualtree_dfs.h
  dualtree_dfs_iterator_dev.h
  dualtree_trace.h
  triple_distance_sq.h
  triple_range_distance_sq.h
  tripletree_dfs_dev.h
  tripletree_dfs.h
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(CORE_SRCS ${CORE_SRCS} ${DIR_SRCS} PARENT_SCOPE)

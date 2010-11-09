set(SOURCES
#  anmf
#  max_weight_matching
#  auction_max_weight_matching
#  naive_distance_matrix
#  kdtree_distance_matrix
  matching
  kdnode
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_executable(mt_test
   EXCLUDE_FROM_ALL
   mt_test.cpp
)
add_dependencies(mt_test
   mlpack_contrib
)
target_link_libraries(mt_test
  mlpack_contrib
  fastlib
  boost_program_options
)

cmake_minimum_required(VERSION 2.8)

set(SOURCES
  common_types
  value
  variable
  universe
  assignment
  factor_template
  factor_graph
  convergence_measure
  naive_inference
  sum_product_inference
  inference
  logarithm
  gm
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_executable(gm_test_all
   EXCLUDE_FROM_ALL
   gm_test.cpp
)
add_dependencies(gm_test_all
   mlpack_contrib
)
# link dependencies of executable
target_link_libraries(gm_test_all
   mlpack_contrib
   fastlib
)

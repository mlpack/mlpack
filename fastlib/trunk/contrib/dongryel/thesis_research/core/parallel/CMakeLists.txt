cmake_minimum_required(VERSION 2.8)

set(SOURCES
  parallel_sample_sort.h
  sample_distributed_tree_builder.h
  table_exchange.h
  vanilla_distributed_tree_builder.h
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(CORE_SRCS ${CORE_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(distributed_tree-test
               EXCLUDE_FROM_ALL
               distributed_tree.test.cc
              )
# link dependencies of test executable
target_link_libraries(distributed_tree-test
                      core
                      ${Boost_LIBRARIES}
                     )

# test executable
add_executable(parallel_sample_sort-test
               EXCLUDE_FROM_ALL
               parallel_sample_sort.test.cc
              )
# link dependencies of test executable
target_link_libraries(parallel_sample_sort-test
                      core
                      ${Boost_LIBRARIES}
                     )
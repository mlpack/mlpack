cmake_minimum_required(VERSION 2.8)

set(SOURCES
  bounds.h
  general_spacetree.h
  gen_metric_tree.h
  gen_metric_tree_impl.h
  statistic.h
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(CORE_SRCS ${CORE_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(tree-test
               EXCLUDE_FROM_ALL
               tree.test.cc
              )
# link dependencies of test executable
target_link_libraries(tree-test
                      core
                      ${Boost_LIBRARIES}
                     )

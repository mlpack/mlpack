cmake_minimum_required(VERSION 2.8)

set(SOURCES
  distributed_auction.h
  distributed_local_kmeans.h
  parallel_sample_sort.h
  table_exchange.h
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(CORE_SRCS ${CORE_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(distributed_auction-test
               EXCLUDE_FROM_ALL
               distributed_auction.test.cc
              )
# link dependencies of test executable
target_link_libraries(distributed_auction-test
                      core
                      ${Boost_LIBRARIES}
                     )

cmake_minimum_required(VERSION 2.8)

# the test for distributed_allknn
add_executable(distributed_allknn
               EXCLUDE_FROM_ALL
               distributed_allknn.cc
)

# link dependencies of distributed_allknn.cc
target_link_libraries(distributed_allknn
                      core
                      ${TRILINOS_LIBS}
                      ${Boost_LIBRARIES}
                     )

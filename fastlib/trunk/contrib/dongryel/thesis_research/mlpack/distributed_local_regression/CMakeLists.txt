cmake_minimum_required(VERSION 2.6)

# the test for distributed_local_regression
add_executable(distributed_local_regression
               EXCLUDE_FROM_ALL
               distributed_local_regression.cc
)

# link dependencies of distributed_local_regression.cc
target_link_libraries(distributed_local_regression
                      core
                      ${TRILINOS_LIBS}
                      ${Boost_LIBRARIES}
                     )

# the test for distributed_local_regression.test.cc
add_executable(distributed_local_regression-test
               EXCLUDE_FROM_ALL
               distributed_local_regression.test.cc
)

# link dependencies of distributed_local_regression.test.cc
target_link_libraries(distributed_local_regression-test
                      core
                      ${TRILINOS_LIBS}
                      ${Boost_LIBRARIES}
                     )

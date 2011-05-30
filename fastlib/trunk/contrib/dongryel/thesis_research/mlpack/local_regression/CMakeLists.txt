cmake_minimum_required(VERSION 2.6)

# the main driver for local_regression
add_executable(local_regression
               EXCLUDE_FROM_ALL
               local_regression.cc
              )
# link dependencies of local_regression
target_link_libraries(local_regression
                      core
                      ${Boost_LIBRARIES}
                     )
# the test driver for local_regression-test
add_executable(local_regression-test
               EXCLUDE_FROM_ALL
               local_regression.test.cc
              )
# link dependencies of local_regression-test
target_link_libraries(local_regression-test
                      core
                      ${Boost_LIBRARIES}
                     )

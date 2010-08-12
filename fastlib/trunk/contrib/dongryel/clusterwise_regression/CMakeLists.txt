cmake_minimum_required(VERSION 2.8)

# test executable
add_executable(clusterwise_regression
               EXCLUDE_FROM_ALL
               clusterwise_regression.cc
              )
# link dependencies of test executable
target_link_libraries(clusterwise_regression
                      fastlib
                      mlpack
                      ${TRILINOS_LIBS}
                      ${Boost_LIBRARIES}
                     )

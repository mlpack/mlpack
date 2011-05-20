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

cmake_minimum_required(VERSION 2.8)

# the main driver for gp_regression
add_executable(gp_regression
               EXCLUDE_FROM_ALL
               gp_regression.cc
              )
# link dependencies of gp_regression
target_link_libraries(gp_regression
                      core
                      ${Boost_LIBRARIES}
                     )

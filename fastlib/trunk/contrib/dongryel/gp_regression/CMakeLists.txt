cmake_minimum_required(VERSION 2.8)

# the driver: gp_regression
add_executable(gp_regression
               EXCLUDE_FROM_ALL
               gp_regression.cc
              )
# link dependencies of the driver
target_link_libraries(gp_regression
                      fastlib
                      mlpack
                      ${TRILINOS_LIBS}
                      ${Boost_LIBRARIES}
                     )

# test executable: lbfgs_opt_test
add_executable(lbfgs_opt_test
               EXCLUDE_FROM_ALL
               lbfgs_opt_test.cc
              )
# link dependencies of test executable
target_link_libraries(lbfgs_opt_test
                      fastlib
                      mlpack
                      ${TRILINOS_LIBS}
                     )

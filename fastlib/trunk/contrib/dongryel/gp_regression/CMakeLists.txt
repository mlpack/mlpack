cmake_minimum_required(VERSION 2.8)

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

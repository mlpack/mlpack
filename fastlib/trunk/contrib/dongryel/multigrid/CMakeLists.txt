cmake_minimum_required(VERSION 2.8)

# test executable
add_executable(multigrid_test
               EXCLUDE_FROM_ALL
               multigrid_test.cc
              )
# link dependencies of test executable
target_link_libraries(multigrid_test
                      fastlib
                      mlpack
                      ${TRILINOS_LIBS}
                     )

# test executable
#add_executable(ml_preconditioner_test
#               EXCLUDE_FROM_ALL
#               ml_preconditioner_test.cc
#              )
# link dependencies of test executable
#target_link_libraries(ml_preconditioner_test
#                      fastlib
#                      mlpack
#                      ${TRILINOS_LIBS}
#                     )

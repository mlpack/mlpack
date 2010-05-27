cmake_minimum_required(VERSION 2.8)

# test executable
add_executable(bilinear_form_test
               EXCLUDE_FROM_ALL
               bilinear_form_test.cc
              )
# link dependencies of test executable
target_link_libraries(bilinear_form_test
                      fastlib
                      mlpack
                      ${TRILINOS_LIBS}
                     )

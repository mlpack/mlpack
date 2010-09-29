cmake_minimum_required(VERSION 2.8)

# the main driver for parallel_kde
add_executable(parallel_kde
               EXCLUDE_FROM_ALL
               parallel_kde.cc
              )

# link dependencies of parallel_kde
target_link_libraries(parallel_kde
                      core
                      ${TRILINOS_LIBS}
                      ${Boost_LIBRARIES}
                     )

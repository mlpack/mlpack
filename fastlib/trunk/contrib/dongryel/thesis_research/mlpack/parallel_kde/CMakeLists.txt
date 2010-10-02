cmake_minimum_required(VERSION 2.8)

# the test for MPI prefix scan
add_executable(prefix_scan
               EXCLUDE_FROM_ALL
               prefix_scan.cc
)

# link dependencies of prefix_scan
target_link_libraries(prefix_scan
                      core
                      ${TRILINOS_LIBS}
                      ${Boost_LIBRARIES}
                     )

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

cmake_minimum_required(VERSION 2.8)

# the test for distributed_kde
add_executable(distributed_kde
               EXCLUDE_FROM_ALL
               distributed_kde.cc
)

# link dependencies of distributed_kde.cc
target_link_libraries(distributed_kde
                      core
                      ${TRILINOS_LIBS}
                      ${Boost_LIBRARIES}
                     )

# the test for MPI all_reduce
add_executable(all_reduce
               EXCLUDE_FROM_ALL
               all_reduce.cc
)

# link dependencies of all_reduce
target_link_libraries(all_reduce
                      core
                      ${TRILINOS_LIBS}
                      ${Boost_LIBRARIES}
                     )

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

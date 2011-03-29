cmake_minimum_required(VERSION 2.6)

# the test for distributed_kpca
add_executable(distributed_kpca
               EXCLUDE_FROM_ALL
               distributed_kpca.cc
)

# link dependencies of distributed_kpca.cc
target_link_libraries(distributed_kpca
                      core
                      ${TRILINOS_LIBS}
                      ${Boost_LIBRARIES}
                     )

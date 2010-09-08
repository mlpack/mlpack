cmake_minimum_required(VERSION 2.8)

# test executable
add_executable(kde
               EXCLUDE_FROM_ALL
               kde.cc
              )
# link dependencies of test executable
target_link_libraries(kde
                      core
                      ${TRILINOS_LIBS}
                      ${Boost_LIBRARIES}
                     )

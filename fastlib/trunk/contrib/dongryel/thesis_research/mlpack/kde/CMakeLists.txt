cmake_minimum_required(VERSION 2.6)

# the main driver for kde
add_executable(kde
               EXCLUDE_FROM_ALL
               kde.cc
              )
# link dependencies of kde
target_link_libraries(kde
                      core
                      ${Boost_LIBRARIES}
                     )

# the test driver for kde-test
add_executable(kde-test
               EXCLUDE_FROM_ALL
               kde.test.cc
              )
# link dependencies of kde-test
target_link_libraries(kde-test
                      core
                      ${Boost_LIBRARIES}
                     )

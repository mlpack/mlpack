cmake_minimum_required(VERSION 2.8)

# test executable
add_executable(series_expansion-test
               EXCLUDE_FROM_ALL
               series_expansion.test.cc
              )
# link dependencies of test executable
target_link_libraries(series_expansion-test
                      core
                      ${Boost_LIBRARIES}
                     )

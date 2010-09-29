cmake_minimum_required(VERSION 2.8)

# the main driver for nbody_simulator
add_executable(nbody_simulator
               EXCLUDE_FROM_ALL
               nbody_simulator.cc
              )
# link dependencies of nbody_simulator
target_link_libraries(nbody_simulator
                      core
                      ${Boost_LIBRARIES}
                     )

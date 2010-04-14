cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # particle
   particle_tree.h
   two_body_stat.h
   multi_physics_system.h
   force_error.h
   # particle2
   dual_physics_system.h
   # test1
   test_particle_stat.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(particle_simulation
   simulation_driver.cc
)
# link dependencies of executable
target_link_libraries(particle_simulation
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(particle_stat_test
   test_particle_stat.cc
)
target_link_libraries(particle_stat_test
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(particle_dual_simulation
   dual_simulation_driver.cc
)
target_link_libraries(particle_dual_simulation
   fastlib
   mlpack
   mlpack_contrib
)

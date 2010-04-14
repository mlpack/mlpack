cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   atom_tree.h
   lennard_jones.h
#   test_lennard_jones.h  ## exists in build.py but does not actually exist
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all PHYSPACK sources (used at the parent scope)
set(PHYSPACK_SRCS ${PHYSPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
message("mol: ${PHYSPACK_SRCS}")

add_executable(lennard_jones
   LennardJones_main.cc
)
target_link_libraries(lennard_jones
   physpack
   fastlib
)

# Exists in build.py but does not actually exist
#add_executable(lennard_jones_test
#   test_lennard_jones.cc
#)
#target_link_libraries(lennard_jones_test
#   physpack
#   fastlib
#)

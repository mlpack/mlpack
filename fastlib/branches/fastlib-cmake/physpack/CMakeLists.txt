cmake_minimum_required(VERSION 2.8)
project(MLPACK_CONTRIB C CXX Fortran)

# compiler flags
set(CMAKE_CXX_FLAGS "-g -Wall")
add_definitions(-DDISABLE_DISK_MATRIX)

# include directories
include_directories(..)
include_directories(../include)

set(PHYSPACK_SRCS)

# now we need to recurse into each of the contrib methods
set(DIRS
   mol_dynamics
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

# PHYSPACK_SRCS is set in the subdirectories above
add_library(physpack ${PHYSPACK_SRCS})
add_dependencies(physpack fastlib)

# move header files to the right directory
file(GLOB_RECURSE INCLUDE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
add_custom_command(TARGET physpack POST_BUILD
   COMMENT "Moving header files to include/physpack/"
   COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${CMAKE_BINARY_DIR}/include/physpack/)
foreach(incl_file ${INCLUDE_FILES})
   add_custom_command(TARGET mlpack_contrib POST_BUILD
      COMMAND ${CMAKE_COMMAND} ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${incl_file} ${CMAKE_BINARY_DIR}/include/physpack/${incl_file})
endforeach()

# set installation rules for include files
#install(FILES ${CMAKE_BINARY_DIR}/include/physpack/
#   DESTINATION include/physpack)

# set installation rules for mlpack_contrib programs
#install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/
#   DESTINATION bin)

install(TARGETS physpack
   RUNTIME DESTINATION bin
   LIBRARY DESTINATION lib
   ARCHIVE DESTINATION lib
)

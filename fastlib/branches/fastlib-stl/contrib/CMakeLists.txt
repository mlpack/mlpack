cmake_minimum_required(VERSION 2.8)
project(MLPACK_CONTRIB C CXX Fortran)

# compiler flags
set(CMAKE_CXX_FLAGS "-g -Wall")
add_definitions(-DDISABLE_DISK_MATRIX)

# include directories
include_directories(..)
include_directories(../include)

set(MLPACK_STD_LIBS fastlib lapack blas)
set(MLPACK_CONTRIB_SRCS)

# now we need to recurse into each of the contrib methods
set(DIRS
#   archana  # <-- DCT.cc is not usable anywhere else, custom one-time code
   dongryel
   gmravi
   houyang
   huijing
   jim
   march
   niche
   nvasil
   pram
   rriegel
   soyeon
   tqlong
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

# MLPACK_CONTRIB_SRCS is set in the subdirectories above
#message("output: ${MLPACK_CONTRIB_SRCS}")
# don't build mlpack_contrib.a by default, user must specify they want it
add_library(mlpack_contrib EXCLUDE_FROM_ALL ${MLPACK_CONTRIB_SRCS})
add_dependencies(mlpack_contrib mlpack fastlib)
target_link_libraries(mlpack_contrib
  mlpack
  fastlib
  lapack
  blas
)

# move header files to the right directory
file(GLOB_RECURSE INCLUDE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
add_custom_command(TARGET mlpack_contrib POST_BUILD
   COMMENT "Moving header files to include/mlpack_contrib/"
   COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${CMAKE_BINARY_DIR}/include/mlpack_contrib/)
foreach(incl_file ${INCLUDE_FILES})
   add_custom_command(TARGET mlpack_contrib POST_BUILD
      COMMAND ${CMAKE_COMMAND} ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${incl_file} ${CMAKE_BINARY_DIR}/include/mlpack_contrib/${incl_file})
endforeach()

# set installation rules for include files
#install(FILES ${CMAKE_BINARY_DIR}/include/mlpack_contrib/
#   DESTINATION include/mlpack_contrib)

# set installation rules for mlpack_contrib programs
#install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/
#   DESTINATION bin)

install(TARGETS mlpack_contrib 
   RUNTIME DESTINATION bin
   LIBRARY DESTINATION lib
   ARCHIVE DESTINATION lib
   OPTIONAL
) # optional since we don't know if the user chose to build it

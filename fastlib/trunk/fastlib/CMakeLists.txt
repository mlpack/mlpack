cmake_minimum_required(VERSION 2.8)
project(FASTLIB C CXX Fortran)

## External Libraries
#  ls /usr/share/cmake-2.6/Modules/Find* | \
#  perl -ne 's#.*Modules/Find(.*)>cmake#\1#; print'
#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)

#include(CMake/FastlibMacros.cmake)

# distclean option because cmake doesn't support it
#include( CMake/TargetDistclean.cmake OPTIONAL )

## Global compiler flags (dep. on platform TODO)
add_definitions( -DDISABLE_DISK_MATRIX )

set(FASTLIB_SRCS)
## recurse
set(DIRS
   base
   col
   data
   file
   fx
   la
   math
   mmanager  ## not yet
   par
   thor  # requires pthread linking
   tree
#   sparse
)  #!!

option(FASTLIB_WITH_OPTIMIZERS "Include optimization code - requires OPT++." OFF)
if(FASTLIB_WITH_OPTIMIZERS)
   set(DIRS ${DIRS} optimization)
endif(FASTLIB_WITH_OPTIMIZERS)

option(FASTLIB_WITH_SPARSE "Include sparse matrix code - requires Trilinos." ON)
if(FASTLIB_WITH_SPARSE)
   set(DIRS ${DIRS} sparse)
endif(FASTLIB_WITH_SPARSE)

foreach(dir ${DIRS})
  add_subdirectory( ${dir} )
endforeach()

add_library(fastlib ${FASTLIB_SRCS})
target_link_libraries(fastlib
   lapack
   pthread
   ${TRILINOS_LIBS}
)

# move include files to correct directory for mlpack compilation
file(GLOB_RECURSE INCLUDE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h )
add_custom_command(TARGET fastlib POST_BUILD
  COMMENT "Moving header files to include/fastlib/"
  COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${CMAKE_BINARY_DIR}/include/fastlib/)
foreach(incl_file ${INCLUDE_FILES})
  add_custom_command(TARGET fastlib POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${incl_file} ${CMAKE_BINARY_DIR}/include/fastlib/${incl_file} )
endforeach()

add_executable( otrav_test otrav_test.cc )
target_link_libraries( otrav_test fastlib )

# make sure header files are installed correctly
#install(FILES ${CMAKE_BINARY_DIR}/include/fastlib/
#   DESTINATION include/fastlib)

## installation
install(TARGETS fastlib 
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES fastlib.h DESTINATION include/fastlib)

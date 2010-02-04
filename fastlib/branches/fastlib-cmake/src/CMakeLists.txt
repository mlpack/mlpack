cmake_minimum_required(VERSION 2.6)
project(FASTLIB C CXX Fortran)

## External Libraries
#  ls /usr/share/cmake-2.6/Modules/Find* | \
#  perl -ne 's#.*Modules/Find(.*)>cmake#\1#; print'
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

#include(CMake/FastlibMacros.cmake)

## Global compiler flags (dep. on platform TODO)
add_definitions( -DDISABLE_DISK_MATRIX )

## include dirs
#SET(CMAKE_INCLUDE_CURRENT_DIR ON)

set(FASTLIB_INCLUDE_DIRS 
    ${FASTLIB_SOURCE_DIR}
    #${FASTLIB_SOURCE_DIR}/.. 
    #${FASTLIB_SOURCE_DIR}/base/config  # standardize #includes
    #${FASTLIB_BINARY_DIR}/..
    ${FASTLIB_BINARY_DIR}
)
include_directories( ${FASTLIB_INCLUDE_DIRS} )
#TODO link_directories()

set(FASTLIB_SRCS)
## recurse
set(DIRS base col data file fx la math par tree)  #!!

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
  
set_source_files_properties(${CMAKE_BINARY_DIR}/fastlib/base/basic_types.h 
    PROPERTIES GENERATED TRUE)

add_library( fastlib ${FASTLIB_SRCS} )
add_dependencies( fastlib ${DIRS} template_types_detect ) #FIXME
target_link_libraries( fastlib lapack )

add_executable( otrav_test otrav_test.cc )
target_link_libraries( otrav_test fastlib )

## installation
install(TARGETS fastlib 
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES fastlib.h DESTINATION include/fastlib)

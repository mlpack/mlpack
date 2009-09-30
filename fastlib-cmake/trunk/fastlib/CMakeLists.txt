cmake_minimum_required(VERSION 2.6)
project(FASTLIB C CXX Fortran)
#project(FASTLIB C CXX )

## External Libraries
#  ls /usr/share/cmake-2.6/Modules/Find* | \
#  perl -ne 's#.*Modules/Find(.*)>cmake#\1#; print'
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

#include(CMake/FastlibMacros.cmake)

## Global compiler flags (dep. on platform TODO)
set(CMAKE_CXX_FLAGS "-g -Wall -O0")
set(CMAKE_C_FLAGS "-g -Wall -O0")
#set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -O0")
#set(CMAKE_CXX_FLAGS_RELEASE "-g -Wall -O0")
add_definitions( -DDISABLE_DISK_MATRIX )

## include dirs
#SET(CMAKE_INCLUDE_CURRENT_DIR ON)

set(FASTLIB_INCLUDE_DIRS 
    #${FASTLIB_SOURCE_DIR}
    ${FASTLIB_SOURCE_DIR}/.. 
    #${FASTLIB_SOURCE_DIR}/base/config  # standardize #includes
    #${FASTLIB_BINARY_DIR}/..
    ${FASTLIB_BINARY_DIR}
)
include_directories( ${FASTLIB_INCLUDE_DIRS} )
#TODO link_directories()
message("include: ${FASTLIB_INCLUDE_DIRS}")

set(FASTLIB_SRCS)
## recurse
set(DIRS base col data file fx la math par tree)

foreach(dir ${DIRS})
  add_subdirectory( ${dir} fastlib/${dir} )
endforeach()
  
#! add_library( fastlib_int fastlib.h fastlib_int.h)
#! set_target_properties( fastlib_int PROPERTIES LINKER_LANGUAGE CXX)
#! add_dependencies( fastlib_int ${DIRS} ) #FIXME
#! # add_dependencies( fastlib_int
#! #     la base fx file col data math tree par
#! # )
#! target_link_libraries( fastlib_int ${DIRS} )
set_source_files_properties(${CMAKE_BINARY_DIR}/fastlib/base/basic_types.h 
    PROPERTIES GENERATED TRUE)

#add_library( fastlib_int ${FASTLIB_SRCS} )
#add_dependencies( fastlib_int ${DIRS} template_types_detect ) #FIXME
add_library( fastlib ${FASTLIB_SRCS} )
add_dependencies( fastlib ${DIRS} template_types_detect ) #FIXME

add_executable( otrav_test otrav_test.cc )
#target_link_libraries( otrav_test fastlib_int )
target_link_libraries( otrav_test fastlib )

## installation
#install(TARGETS fastlib_int 
install(TARGETS fastlib 
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES fastlib.h DESTINATION include/fastlib)

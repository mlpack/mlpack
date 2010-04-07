cmake_minimum_required(VERSION 2.8)
project(FASTLIB C CXX Fortran)

## External Libraries
#  ls /usr/share/cmake-2.6/Modules/Find* | \
#  perl -ne 's#.*Modules/Find(.*)>cmake#\1#; print'
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

#include(CMake/FastlibMacros.cmake)

# distclean option because cmake doesn't support it
include( CMake/TargetDistclean.cmake OPTIONAL )

## Global compiler flags (dep. on platform TODO)
add_definitions( -DDISABLE_DISK_MATRIX )

#include_directories( ${FASTLIB_INCLUDE_DIRS} )
include_directories(${CMAKE_BINARY_DIR}/include/)
#TODO link_directories()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)

## recurse
add_subdirectory(src) # fastlib
add_subdirectory(mlpack) # mlpack
  
#set_source_files_properties(${CMAKE_BINARY_DIR}/fastlib/base/basic_types.h 
#    PROPERTIES GENERATED TRUE)

#add_library( fastlib ${FASTLIB_SRCS} )
#add_dependencies( fastlib ${DIRS} template_types_detect ) #FIXME
#target_link_libraries( fastlib lapack )

#add_executable( otrav_test otrav_test.cc )
#target_link_libraries( otrav_test fastlib )

## installation
#install(TARGETS fastlib 
#    RUNTIME DESTINATION bin
#    LIBRARY DESTINATION lib
#    ARCHIVE DESTINATION lib
#)

#install(FILES fastlib.h DESTINATION include/fastlib)

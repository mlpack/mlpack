cmake_minimum_required(VERSION 2.8)
project(MLPACK C CXX Fortran)

#find_package(BLAS REQUIRED)
#find_package(LAPACK REQUIRED)

## Global compiler flags (dep. on platform TODO)
set(CMAKE_CXX_FLAGS "-g -Wall")
add_definitions( -DDISABLE_DISK_MATRIX )

## Location of fastlib headers & lib
set(FASTLIB_BASE_DIR /usr/local CACHE PATH
  "Directory where fastlib is installed." )
set(FASTLIB_LIB_DIR ${FASTLIB_BASE_DIR}/lib )
set(FASTLIB_INCLUDE_DIR ${FASTLIB_BASE_DIR}/include )
# set(FASTLIB_LIB_DIR ${FASTLIB_BASE_DIR}/lib CACHE PATH
#   "Directory where the fastlib library is installed." )
# set(FASTLIB_INCLUDE_DIR ${FASTLIB_BASE_DIR}/include CACHE PATH
#   "Directory where the fastlib headers are installed." )
#     
# mark_as_advanced(FASTLIB_LIB_DIR FASTLIB_INCLUDE_DIR) 

##? add_library ?
find_library(FASTLIB_LIB fastlib ${FASTLIB_LIB_DIR})

## include dirs
include_directories( ${FASTLIB_INCLUDE_DIR} )
include_directories( .. )  # mlpack
include_directories( ../include/ )

## link directories
link_directories( ${FASTLIB_LIB_DIR} )

## standard libs
set(MLPACK_STD_LIBS fastlib lapack blas )  ##! this ordering matters :(

## recurse
set( DIRS
    allkfn
    allknn allnn
#    disk_allnn  # does not compile, perhaps due to an API change
    emst
    fastica
    hmm
    infomax_ica
#    isonmf  ## nothing here yet
    kalman
    kde
#    kernel_pca  # requires sparse 
    #mog # requires optimization
#    mlpackdemo  ## matlab routines
    #mvu # requires optimization
    naive_bayes
    nnsvm
    quicsvd
    # range_search  ##! uses contrib code
    regression
    series_expansion
    svm
)

if(WITH_SPARSE)
#   Don't compile this because sparse support in this branch is not really
#   existent
#   set(DIRS ${DIRS} kernel_pca)
endif()

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

# MLPACK_SRCS is set in the subdirectories
add_library(mlpack ${MLPACK_SRCS})
add_dependencies(mlpack fastlib)
target_link_libraries(mlpack
  fastlib
  lapack
  blas
)

if(WITH_SPARSE)
   target_link_libraries(mlpack
      ${TRILINOS_LIBS}
   )
endif()

# move header files to the right directory
file(GLOB_RECURSE INCLUDE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
add_custom_command(TARGET mlpack POST_BUILD
   COMMENT "Moving header files to include/mlpack/"
   COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${CMAKE_BINARY_DIR}/include/mlpack/)
foreach(incl_file ${INCLUDE_FILES})
   add_custom_command(TARGET mlpack POST_BUILD
      COMMAND ${CMAKE_COMMAND} ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${incl_file} ${CMAKE_BINARY_DIR}/include/mlpack/${incl_file} )
endforeach()

# set installation rules for include files
#install(FILES ${CMAKE_BINARY_DIR}/include/mlpack/
#   DESTINATION include/mlpack)

# set installation rules for mlpack programs
#install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/
#   DESTINATION bin)

install(TARGETS mlpack
   RUNTIME DESTINATION bin
   LIBRARY DESTINATION lib
   ARCHIVE DESTINATION lib)

## installation

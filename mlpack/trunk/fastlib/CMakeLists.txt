cmake_minimum_required(VERSION 2.8)
project(FASTLIB C CXX)

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
   math
   tree
)

# This needs to be always included in the future.
if(WITH_OPTIMIZERS)
   set(DIRS ${DIRS} optimization)
endif(WITH_OPTIMIZERS)

foreach(dir ${DIRS})
  add_subdirectory( ${dir} )
endforeach()

add_library(fastlib ${FASTLIB_SRCS})
target_link_libraries(fastlib
  armadillo
  lapack
  blas
  pthread
  boost_program_options
)

# move include files to correct directory for mlpack compilation
file(GLOB_RECURSE INCLUDE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h *.hpp )
add_custom_command(TARGET fastlib POST_BUILD
  COMMENT "Moving header files to include/fastlib/"
  COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${CMAKE_BINARY_DIR}/include/fastlib/)
foreach(incl_file ${INCLUDE_FILES})
  add_custom_command(TARGET fastlib POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${incl_file} ${CMAKE_BINARY_DIR}/include/fastlib/${incl_file} )
endforeach()

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

# Now we have to add the profile-guided optimization targets.
# This is commented out for now, with the intention of adding it back in later.
# So, this commented code serves as a guide.
#if (NOT DEBUG)
#  if (PGO)
    # Use -fprofile-generate in compiling fastlib the first time (which we will
    # call fastlib_unprofiled).
#    add_library(fastlib_unprofiled ${FASTLIB_SRCS})
#    target_link_libraries(fastlib_unprofiled
#        armadillo
#        lapack
#        blas
#        pthread
#        boost_program_options
#    )
#    set_target_properties(fastlib_unprofiled
#        PROPERTIES COMPILE_FLAGS "-fprofile-generate")

    # Add the test target.  This target just runs all the tests, in the hopes
    # that that will give us okay profiling information.  We also remove all the
    # old profiling information.
#    add_custom_target(fastlib_profiling_test_run
#        find ${CMAKE_BINARY_DIR} -iname '*.gcda' -delete
#        COMMAND ${CMAKE_BINARY_DIR}/bin/fastlib_profiling_test
#    )
#    add_dependencies(fastlib_profiling_test_run
#        fastlib_unprofiled
#        fastlib_profiling_test
#    )

    # Now add a profiled target.
#    set_target_properties(fastlib
#        PROPERTIES COMPILE_FLAGS "-fprofile-use")
#    add_dependencies(fastlib
#        fastlib_profiling_test_run
#        fastlib_profiling_test
#        fastlib_unprofiled
#    )

#  endif ()
#endif ()

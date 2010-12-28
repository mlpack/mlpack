cmake_minimum_required(VERSION 2.8)
project(FASTLIB C CXX Fortran)

set(CORE_SRCS)

add_subdirectory(csv_parser)
add_subdirectory(gnp)
add_subdirectory(math)
add_subdirectory(metric_kernels)
add_subdirectory(monte_carlo)
add_subdirectory(optimization)
add_subdirectory(parallel)
add_subdirectory(table)
add_subdirectory(tree)
add_subdirectory(util)

add_library(core ${CORE_SRCS})
target_link_libraries(core
  lapack
  blas
  pthread
  ${TRILINOS_LIBS}
)

# move include files to correct directory for mlpack compilation
file(GLOB_RECURSE INCLUDE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h )
add_custom_command(TARGET core POST_BUILD
  COMMENT "Moving header files to include/core/"
  COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${CMAKE_BINARY_DIR}/include/core/)
foreach(incl_file ${INCLUDE_FILES})
  add_custom_command(TARGET core POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${incl_file} ${CMAKE_BINARY_DIR}/include/core/${incl_file} )
endforeach()

# make sure header files are installed correctly
install(FILES ${INCLUDE_FILES}
   DESTINATION include/core)

## installation
install(TARGETS core
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

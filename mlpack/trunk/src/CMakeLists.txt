## include dirs
include_directories(..)  # mlpack
include_directories(../include/)

## recurse
set(DIRS
  core
  methods
)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

# MLPACK_SRCS is set in the subdirectories
add_library(mlpack ${MLPACK_SRCS})
target_link_libraries(mlpack
  armadillo
  lapack
  blas
)

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

include_directories(..) # <mlpack/[whatever]>

# Add core.hpp to list of sources.
set(${MLPACK_SRCS} ${MLPACK_SRCS} "core.hpp")

## Recurse into both core/ and methods/.
set(DIRS
  core
  methods
  tests
)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

# MLPACK_SRCS is set in the subdirectories.
add_library(mlpack SHARED ${MLPACK_SRCS})
target_link_libraries(mlpack
  armadillo
  ${BOOST_LIBS}
  xml2
)
set_target_properties(mlpack
  PROPERTIES
  VERSION 1.0.0
  SOVERSION 1
)

# Collect all header files in the library.
file(GLOB_RECURSE INCLUDE_H_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
file(GLOB_RECURSE INCLUDE_HPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.hpp)
set(INCLUDE_FILES ${INCLUDE_H_FILES} ${INCLUDE_HPP_FILES})

# Move all of these header files to <builddir>/include/mlpack/ after the library
# is built.  First we have to create that directory though.
add_custom_command(TARGET mlpack POST_BUILD
  COMMENT "Moving header files to include/mlpack/"
  COMMAND ${CMAKE_COMMAND} ARGS -E
    make_directory ${CMAKE_BINARY_DIR}/include/mlpack/)

# Then copy each of the header files over to that directory.
foreach(incl_file ${INCLUDE_FILES})
  add_custom_command(TARGET mlpack POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E
      copy ${CMAKE_CURRENT_SOURCE_DIR}/${incl_file}
           ${CMAKE_BINARY_DIR}/include/mlpack/${incl_file})
endforeach()

# At install time, we simply install that directory of header files we
# collected to include/.
install(DIRECTORY ${CMAKE_BINARY_DIR}/include/mlpack DESTINATION include)

install(TARGETS mlpack
   RUNTIME DESTINATION bin
   LIBRARY DESTINATION lib
   ARCHIVE DESTINATION lib)

# Set generated executables to be installed.
# This will have some odd effects if the user has placed anything else in bin/,
# but I think we can just hope that they haven't.  At least, an auto-builder
# won't do that...
install(DIRECTORY ${CMAKE_BINARY_DIR}/bin/
  DESTINATION bin
  USE_SOURCE_PERMISSIONS
  COMPONENT mlpack
  FILES_MATCHING
  PATTERN *[^_test])

# For 'make test'.
add_custom_target(test
  ${CMAKE_BINARY_DIR}/bin/mlpack_test
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/ # This is where test files are put.
  COMMENT "Running MLPACK test"
  DEPENDS mlpack_test)

include_directories(..) # <mlpack/[whatever]>

# Add core.hpp to list of sources.
set(MLPACK_SRCS ${MLPACK_SRCS} "${CMAKE_CURRENT_SOURCE_DIR}/core.hpp")

## Recurse into both core/ and methods/.
set(DIRS
  bindings
  core
  methods
  tests
)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

# MLPACK_SRCS is set in the subdirectories.
# We don't use a DLL (shared) on Windows because it's a nightmare.  We can't
# easily generate the .def file and we won't put __declspec(dllexport) next to
# every function signature.
if (WIN32)
  add_library(mlpack ${MLPACK_SRCS})
else (WIN32)
  add_library(mlpack SHARED ${MLPACK_SRCS})
endif (WIN32)
target_link_libraries(mlpack
  ${ARMADILLO_LIBRARIES}
  ${Boost_LIBRARIES}
  ${LIBXML2_LIBRARIES}
)
set_target_properties(mlpack
  PROPERTIES
  VERSION 1.0
  SOVERSION 1
)

# Make sure the linker can find the needed library.
# rt: clock_gettime()
if(UNIX AND NOT APPLE)
    target_link_libraries(mlpack rt)
endif(UNIX AND NOT APPLE)

# Log::Assert may require linking against whatever provides backtrace
# functionality.
if(Backtrace_FOUND)
  target_link_libraries(mlpack ${Backtrace_LIBRARIES})
endif(Backtrace_FOUND)

# Collect all header files in the library.
file(GLOB_RECURSE INCLUDE_H_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
file(GLOB_RECURSE INCLUDE_HPP_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.hpp)
set(INCLUDE_FILES ${INCLUDE_H_FILES} ${INCLUDE_HPP_FILES})

# Move all of these header files to <builddir>/include/mlpack/ after the library
# is built.  First we have to create that directory though.
add_custom_target(mlpack_headers)
add_custom_command(TARGET mlpack_headers POST_BUILD
  COMMENT "Moving header files to include/mlpack/"
  COMMAND ${CMAKE_COMMAND} ARGS -E
    make_directory ${CMAKE_BINARY_DIR}/include/mlpack/)

# Then copy each of the header files over to that directory.
foreach(incl_file ${INCLUDE_FILES})
  add_custom_command(TARGET mlpack_headers POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E
      copy ${CMAKE_CURRENT_SOURCE_DIR}/${incl_file}
           ${CMAKE_BINARY_DIR}/include/mlpack/${incl_file})
endforeach()

# At install time, we simply install that directory of header files we
# collected to include/.
install(DIRECTORY ${CMAKE_BINARY_DIR}/include/mlpack DESTINATION include)

# Set generated executables to be installed.  Unfortunately they must manually
# be entered...
install(TARGETS mlpack
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

add_dependencies(mlpack mlpack_headers)

# For 'make test'.
add_custom_target(test
  ${CMAKE_BINARY_DIR}/bin/mlpack_test "--log_level=test_suite" # Set UTF runtime param
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/ # This is where test files are put.
  COMMENT "Running MLPACK test"
  DEPENDS mlpack_test)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/..) # mlpack/mlpack_export.hpp

# Add core.hpp to list of sources.
set(MLPACK_SRCS ${MLPACK_SRCS}
  "${CMAKE_CURRENT_SOURCE_DIR}/base.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/core.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/prereqs.hpp"
)

## Recurse into both core/ and methods/.
set(DIRS
  bindings
  core
  methods
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif ()

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
    make_directory ${CMAKE_BINARY_DIR}/include/mlpack/
  COMMAND ${CMAKE_COMMAND} ARGS -E
    copy ${CMAKE_CURRENT_BINARY_DIR}/mlpack_export.hpp
         ${CMAKE_BINARY_DIR}/include/mlpack)

# Then copy each of the header files over to that directory.
foreach(incl_file ${INCLUDE_FILES})
  add_custom_command(TARGET mlpack_headers POST_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E
      copy ${CMAKE_CURRENT_SOURCE_DIR}/${incl_file}
           ${CMAKE_BINARY_DIR}/include/mlpack/${incl_file}
      BYPRODUCTS ${CMAKE_BINARY_DIR}/include/mlpack/${incl_file})
endforeach()

# At install time, we simply install that directory of header files we
# collected to include/.
install(DIRECTORY "${CMAKE_BINARY_DIR}/include/mlpack" DESTINATION
    "${CMAKE_INSTALL_INCLUDEDIR}")

# Extract the version number.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/core/util/version.hpp"
  VERSION_HPP_CONTENTS)
string(REGEX REPLACE ".*#define MLPACK_VERSION_MAJOR ([0-9]+).*" "\\1"
  MLPACK_VERSION_MAJOR "${VERSION_HPP_CONTENTS}")
string(REGEX REPLACE ".*#define MLPACK_VERSION_MINOR ([0-9]+).*" "\\1"
  MLPACK_VERSION_MINOR "${VERSION_HPP_CONTENTS}")
string(REGEX REPLACE ".*#define MLPACK_VERSION_PATCH [\"]?([0-9x]+)[\"]?.*"
  "\\1" MLPACK_VERSION_PATCH "${VERSION_HPP_CONTENTS}")

set(PACKAGE_VERSION
  "${MLPACK_VERSION_MAJOR}.${MLPACK_VERSION_MINOR}.${MLPACK_VERSION_PATCH}")

post_python_bindings()
post_julia_setup()
post_go_setup()

# If we are building R-bindings, we have to run some setup after we
# recurse into methods/.  If not, this macro is empty.
post_r_setup()

# If we are building Markdown documentation, we have to run some setup after we
# recurse into methods/.  If not, this function is empty.
post_markdown_setup()

# Recurse into the bindings to do any necessary configuration there.
add_subdirectory(bindings)
# Recurse into methods/ to get the definitions of any bindings.
add_subdirectory(methods)

# If necessary, configure the tests.
if (BUILD_TESTS)
  add_subdirectory(tests)
else ()
  # Add convenience target to tell the user they need BUILD_TESTS if they try to
  # build mlpack_test.
  add_custom_target(mlpack_test
      ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/CMake/TestError.cmake)
endif ()

# At install time, we simply install the src/ directory to include/ (though we
# omit bindings/ and tests/).
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/../mlpack.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/base.hpp"
    # Note: config.hpp does not get installed from the source directory!
    # The configured version gets installed.
    "${CMAKE_CURRENT_SOURCE_DIR}/core.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/namespace_compat.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/prereqs.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mlpack/")
install(FILES "${CMAKE_BINARY_DIR}/include/mlpack/config-local.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mlpack"
    RENAME "config.hpp")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/core" DESTINATION
    "${CMAKE_INSTALL_INCLUDEDIR}/mlpack")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/methods" DESTINATION
    "${CMAKE_INSTALL_INCLUDEDIR}/mlpack")

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

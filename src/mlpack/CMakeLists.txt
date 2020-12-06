include_directories(${CMAKE_CURRENT_BINARY_DIR}/..) # mlpack/mlpack_export.hpp

# Add core.hpp to list of sources.
set(MLPACK_SRCS ${MLPACK_SRCS}
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

# MLPACK_SRCS is set in the subdirectories.  The dependencies (MLPACK_LIBRARIES)
# are set in the root CMakeLists.txt.
add_library(mlpack ${MLPACK_SRCS})

# If we are not forcing C++11 support, check that the compiler supports C++11
# and enable it.
if (NOT FORCE_CXX11)
  target_compile_features(mlpack PUBLIC
      cxx_decltype
      cxx_alias_templates
      cxx_auto_type
      cxx_lambdas
      cxx_constexpr
      cxx_rvalue_references
      cxx_static_assert
      cxx_template_template_parameters
      cxx_delegating_constructors
      cxx_variadic_templates
      cxx_nullptr
      cxx_noexcept
  )
endif ()

# Generate export symbols for Windows, instead of adding __declspec(dllimport)
# and __declspec(dllexport) everywhere.  However, those modifiers are still
# necessary for global variables (of which there are a few in mlpack).
set_target_properties(mlpack PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
include(GenerateExportHeader)
generate_export_header(mlpack EXPORT_FILE_NAME mlpack_export.hpp)
if (NOT BUILD_SHARED_LIBS)
  add_definitions(-DMLPACK_STATIC_DEFINE)
endif ()

target_link_libraries(mlpack ${MLPACK_LIBRARIES})

set_target_properties(mlpack
  PROPERTIES
  VERSION 3.4
  SOVERSION 3
)

# Backtrace for Linux need those libs.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(LIBBFD_FOUND AND LIBDL_FOUND AND DEBUG)
    target_link_libraries(mlpack ${LIBBFD_LIBRARIES})
    target_link_libraries(mlpack ${LIBDL_LIBRARIES})
  endif()
endif()

set_target_properties(mlpack PROPERTIES COTIRE_CXX_PREFIX_HEADER_INIT "prereqs.hpp")
cotire(mlpack)

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

# Set generated executables to be installed.  Unfortunately they must manually
# be entered...
install(TARGETS mlpack
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

add_dependencies(mlpack mlpack_headers)

# Extract the version number.
file(READ "${CMAKE_SOURCE_DIR}/src/mlpack/core/util/version.hpp"
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

# Installing CMake Helper Files for mlpack-config.cmake

include(${PROJECT_SOURCE_DIR}/CMake/CMakeExtras.cmake)

include(CMakePackageConfigHelpers)

set_verbose(MLPACK_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/mlpack CACHE STRING
          "Installation directory for cmake files, relative to "
          "${CMAKE_INSTALL_PREFIX}.")
set(VERSION_CONFIG ${PROJECT_BINARY_DIR}/mlpack-config-version.cmake)
set(PROJECT_CONFIG ${PROJECT_BINARY_DIR}/mlpack-config.cmake)
set(TARGETS_EXPORT_NAME mlpack-targets)

set(INSTALL_TARGETS mlpack)

set_verbose(MLPACK_LIBS_DIR ${CMAKE_INSTALL_LIBDIR} CACHE STRING
          "Installation directory for libraries, relative to "
          "${CMAKE_INSTALL_PREFIX}.")

set_verbose(MLPACK_INCS_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE STRING
          "Installation directory for include files, relative to "
          "${CMAKE_INSTALL_PREFIX}.")

# Getting the absolute path, for some reason CMAKE gets confused,
# even with CMAKE_INSTALL_INCLUDEDIR
if (UNIX)
  GNUInstallDirs_get_absolute_install_dir(MLPACK_LIBS_DIR CMAKE_INSTALL_LIBDIR)
  GNUInstallDirs_get_absolute_install_dir(MLPACK_INCS_DIR CMAKE_INSTALL_INCLUDEDIR)
endif()

# Generate the version, config and target files into the build directory.
write_basic_package_version_file(
    ${VERSION_CONFIG}
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY SameMajorVersion)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/CMake/mlpack-config.cmake.in
    ${PROJECT_CONFIG}
    INSTALL_DESTINATION ${MLPACK_CMAKE_DIR}
    PATH_VARS MLPACK_INCS_DIR)
# Use a namespace because CMake provides better diagnostics for namespaced
# imported targets.
export(TARGETS ${INSTALL_TARGETS} NAMESPACE mlpack::
    FILE ${PROJECT_BINARY_DIR}/${TARGETS_EXPORT_NAME}.cmake)

# Install version, config and target files.
install(FILES ${PROJECT_CONFIG} ${VERSION_CONFIG}
    DESTINATION ${MLPACK_CMAKE_DIR})

install(EXPORT ${TARGETS_EXPORT_NAME}
    DESTINATION ${MLPACK_CMAKE_DIR} NAMESPACE mlpack::)

# Install the library and headers.
install(TARGETS ${INSTALL_TARGETS} EXPORT ${TARGETS_EXPORT_NAME}
    LIBRARY DESTINATION ${MLPACK_LIBS_DIR}
    ARCHIVE DESTINATION ${MLPACK_LIBS_DIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

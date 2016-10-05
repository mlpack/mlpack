include_directories(..) # <mlpack/[whatever]>
include_directories(${CMAKE_CURRENT_BINARY_DIR}/..) # mlpack/mlpack_export.hpp

# Add core.hpp to list of sources.
set(MLPACK_SRCS ${MLPACK_SRCS} "${CMAKE_CURRENT_SOURCE_DIR}/core.hpp")

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

# MLPACK_SRCS is set in the subdirectories.  The dependencies (MLPACK_LIBRARIES)
# are set in the root CMakeLists.txt.
add_library(mlpack ${MLPACK_SRCS})

# If we are not forcing C++11 support, check that the compiler supports C++11
# and enable it.
if ((NOT FORCE_CXX11) AND
    (NOT (${CMAKE_MAJOR_VERSION} LESS 3 OR
         (${CMAKE_MAJOR_VERSION} EQUAL 3 AND ${CMAKE_MINOR_VERSION} LESS 1))))
  # Use the newer C++11 checks.
  include(../../CMake/NewCXX11.cmake)
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

target_link_libraries(mlpack
  ${MLPACK_LIBRARIES})

set_target_properties(mlpack
  PROPERTIES
  VERSION 2.0
  SOVERSION 2
)

# Backtrace for Linux need those libs.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(LIBBFD_FOUND AND LIBDL_FOUND AND DEBUG)
    target_link_libraries(mlpack ${LIBBFD_LIBRARIES})
    target_link_libraries(mlpack ${LIBDL_LIBRARIES})
  endif()
endif()

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

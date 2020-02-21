macro (not_found_return message)
  message(STATUS "${message}")
  macro (add_java_binding name)
    # Do nothing.
  endmacro ()

  return()
endmacro ()

# If we are not supposed to make Java bindings, define the macro so it does
# nothing and leave this file.
if (NOT BUILD_JAVA_BINDINGS)
  not_found_return("Not building Java bindings.")
endif ()

# Check that JDK is installed
set(JAVA_VERSION 1.7)
find_package(Java ${JAVA_VERSION} REQUIRED COMPONENTS Development)
if (Java_FOUND)
  include(UseJava)
else()
  not_found_return("Java components weren't found; Java bindings will not be built.")
endif()

# Check that JNI exists
find_package(JNI REQUIRED)
if (NOT JNI_FOUND)
  not_found_return("JNI wasn't found; Java bindings will not be built.")
endif()

# Check if target platform is specified (TODO: test on different platforms)
if (NOT MLPACK_JAVA_TARGET_PLATFORM)
  not_found_return("[JAVA BINDINGS] Target platform should be specified (e.g. linux-x86_64)")
endif()

# Check that maven is installed
if (WIN32)
  find_program(MAVEN_BIN mvn.bat REQUIRED)
else()
  find_program(MAVEN_BIN mvn REQUIRED)
endif()

if (MAVEN_BIN)
  message(STATUS "Found maven: ${MAVEN_BIN}")
else()
  not_found_return("Maven not found; Java bindings will not be built.")
endif()

set(MAVEN ${MAVEN_BIN} CACHE INTERNAL "")

add_custom_target(java_configure ALL)
add_custom_target(java ALL DEPENDS java_configure)

# Create directory structure for maven project
set(MAVEN_BUILD_DIR ${CMAKE_BINARY_DIR}/src/mlpack/bindings/java/maven CACHE INTERNAL "")
add_custom_command(TARGET java_configure PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${MAVEN_BUILD_DIR}/src/main/java/org/mlpack)

# Create directory for bindings generators
set(BUILD_BIN_DIR ${CMAKE_BINARY_DIR}/src/mlpack/bindings/java/build/bin CACHE INTERNAL "")
add_custom_command(TARGET java_configure PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_BIN_DIR})

# Create directory for sources for bindings generators
set(BUILD_SRC_DIR ${CMAKE_BINARY_DIR}/src/mlpack/bindings/java/build/src CACHE INTERNAL "")
add_custom_command(TARGET java_configure PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_SRC_DIR})

# Configure pom.xml
set(MLPACK_JAVA_INCLUDE_PATH ${MLPACK_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
set(MLPACK_JAVA_LINK_PATH ${MLPACK_LIBRARY_DIRS} ${CMAKE_BINARY_DIR}/lib)

# TODO: make sure that all flags are being passed here (-D flags are not ATM)
string(REPLACE " " ";" MLPACK_JAVA_COMPILER_OPTIONS_RAW "${CMAKE_CXX_FLAGS}")
set(MLPACK_JAVA_COMPILER_OPTIONS "<param>-Wno-unused-parameter</param><param>-DBINDING_TYPE=BINDING_TYPE_JAVA</param>")
foreach (flag ${MLPACK_JAVA_COMPILER_OPTIONS_RAW})
  set(MLPACK_JAVA_COMPILER_OPTIONS "${MLPACK_JAVA_COMPILER_OPTIONS}<param>${flag}</param>")
endforeach()

if (NOT WIN32)
  string(REPLACE ";" ":" MLPACK_JAVA_INCLUDE_PATH "${MLPACK_JAVA_INCLUDE_PATH}")
  string(REPLACE ";" ":" MLPACK_JAVA_LINK_PATH "${MLPACK_JAVA_LINK_PATH}")
endif()

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

configure_file(maven/pom.xml.in ${MAVEN_BUILD_DIR}/pom.xml @ONLY)

# Compile cli_util library for java
add_custom_target(java_cli_util ALL
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/maven/CLI.java
        ${MAVEN_BUILD_DIR}/src/main/java/org/mlpack/CLI.java
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/maven/MatrixWithInfo.java
        ${MAVEN_BUILD_DIR}/src/main/java/org/mlpack/MatrixWithInfo.java
    DEPENDS java_configure)

# Group some commonly used files into a single library
add_library(java_generate_parts STATIC
    ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/strip_type.hpp
    ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/print_param_defn.hpp
    ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/print_output_param.hpp
    ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/print_java.cpp
    ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/print_java.hpp
    ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/print_input_param.hpp
    ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/print_doc_functions.hpp
    ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/java_option.hpp
    ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/get_java_type.hpp
    ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/util.hpp
    ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/util.cpp)

# Create *.jar file when bindings are generated
add_custom_target(java_package ALL
    COMMAND ${MAVEN} package
    WORKING_DIRECTORY ${MAVEN_BUILD_DIR}
    DEPENDS java_cli_util)

add_dependencies(java java_package)

macro(add_java_binding method_name)
  # Generate sources for bindings generator
  add_custom_command(OUTPUT ${BUILD_SRC_DIR}/generate_java_${method_name}.cpp
      COMMAND ${CMAKE_COMMAND}
          -DMETHOD_NAME=${method_name}
          -DMETHOD_PATH=${CMAKE_CURRENT_SOURCE_DIR}
          -DGENERATE_CPP_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/generate_java.cpp.in
          -DGENERATE_CPP_OUT=${BUILD_SRC_DIR}/generate_java_${method_name}.cpp
          -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${method_name}_main.cpp
          -P ${CMAKE_SOURCE_DIR}/CMake/ConfigureGenerate.cmake
      DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/generate_java.cpp.in
              ${CMAKE_SOURCE_DIR}/CMake/ConfigureGenerate.cmake)

  # Compile bindings generator
  add_executable(generate_java_${method_name}
      ${BUILD_SRC_DIR}/generate_java_${method_name}.cpp)

  add_dependencies(generate_java_${method_name} java_configure)

  target_link_libraries(generate_java_${method_name}
      mlpack
      java_generate_parts
      ${MLPACK_LIBRARIES})

  set_target_properties(generate_java_${method_name} PROPERTIES
      COMPILE_FLAGS -DBINDING_TYPE=BINDING_TYPE_JAVA
      RUNTIME_OUTPUT_DIRECTORY "${BUILD_BIN_DIR}")

  # Run generator to generate bindings (*.java files)
  add_custom_command(TARGET generate_java_${method_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
          -DGENERATE_BINDING_PROGRAM="${BUILD_BIN_DIR}/generate_java_${method_name}"
          -P ${CMAKE_SOURCE_DIR}/CMake/GenerateBinding.cmake
      WORKING_DIRECTORY ${MAVEN_BUILD_DIR}/src/main/java/org/mlpack/)

  add_dependencies(java_package generate_java_${method_name})
endmacro()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()

macro (not_found_return message)
  message(STATUS "${message}")
  macro (add_java_binding name java_name)
    # Do nothing.
  endmacro ()

  return()
endmacro ()

# If we are not supposed to make Java bindings, define the macro so it does
# nothing and leave this file.
if (NOT BUILD_JAVA_BINDINGS)
  not_found_return("Not building Java bindings.")
endif ()

set(JAVA_VERSION 1.7)
find_package(Java ${JAVA_VERSION} REQUIRED COMPONENTS Development)
if (Java_FOUND)
  include(UseJava)
else()
  not_found_return("Java components weren't found")
endif()

find_package(JNI REQUIRED)
if (NOT JNI_FOUND)
  not_found_return("JNI wasn't found")
endif()

set(JNI_INCLUDE ${JNI_INCLUDE_DIRS} CACHE INTERNAL "")

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  find_program(MAVEN_BIN mvn.bat REQUIRED)
else()
  find_program(MAVEN_BIN mvn REQUIRED)
endif()

if (MAVEN_BIN)
  message(STATUS "Found maven: ${MAVEN}")
else()
  not_found_return("Maven not found")
endif()

set(MAVEN ${MAVEN_BIN} CACHE INTERNAL "")
set(JAVA ${Java_JAVA_EXECUTABLE} CACHE INTERNAL "")

add_custom_target(java_configure ALL)
add_custom_target(java ALL DEPENDS java_configure)

# Create directory structure for maven project
set(MAVEN_BUILD_DIR ${CMAKE_BINARY_DIR}/src/mlpack/bindings/java/maven CACHE INTERNAL "")
add_custom_command(TARGET java_configure PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${MAVEN_BUILD_DIR}/src/main/java/org/mlpack)

# Create directory to output native libraries to
set(NATIVE_OUT_DIR ${MAVEN_BUILD_DIR}/native CACHE INTERNAL "")
add_custom_command(TARGET java_configure PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${NATIVE_OUT_DIR})

# Create directory to generate JNI files to
set(JNI_GEN_DIR ${CMAKE_BINARY_DIR}/src/mlpack/bindings/java/build/gen CACHE INTERNAL "")
add_custom_command(TARGET java_configure PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${JNI_GEN_DIR})

set(BUILD_SRC_DIR ${CMAKE_BINARY_DIR}/src/mlpack/bindings/java/build/src CACHE INTERNAL "")
add_custom_command(TARGET java_configure PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_SRC_DIR})

# Copy maven pom.xml
set(JAVACPP_VERSION 1.5.1)

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

# Make maven resolve dependencies into lib directory (see pom.xml)
add_custom_command(TARGET java_configure PRE_BUILD
    COMMAND ${MAVEN} dependency:copy-dependencies
    WORKING_DIRECTORY ${MAVEN_BUILD_DIR})

# Javacpp was pulled from maven
set(JAVACPP ${MAVEN_BUILD_DIR}/lib/javacpp-${JAVACPP_VERSION}.jar CACHE INTERNAL "")

# Compile cli_util library for java
add_custom_target(jniCLI_configure ALL
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/maven/CLI.java
        ${MAVEN_BUILD_DIR}/src/main/java/org/mlpack/CLI.java
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/maven/MatrixWithInfo.java
        ${MAVEN_BUILD_DIR}/src/main/java/org/mlpack/MatrixWithInfo.java
    DEPENDS java_configure)

add_custom_command(
    OUTPUT
        ${BUILD_SRC_DIR}/cli_util.hpp
        ${BUILD_SRC_DIR}/deleter.hpp
        ${BUILD_SRC_DIR}/cli_util.cpp
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/cli_util.hpp
        ${BUILD_SRC_DIR}/cli_util.hpp
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/deleter.hpp
        ${BUILD_SRC_DIR}/deleter.hpp
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/cli_util.cpp
        ${BUILD_SRC_DIR}/cli_util.cpp)

# Compile java sources with maven
add_custom_command(TARGET jniCLI_configure PRE_BUILD
    COMMAND ${MAVEN} compile -Dchanged.classes=org/mlpack/CLI.java
    WORKING_DIRECTORY ${MAVEN_BUILD_DIR})

# Generate JNI wrappers from compiled java bytecode
add_custom_command(OUTPUT ${JNI_GEN_DIR}/jnijavacpp.cpp ${JNI_GEN_DIR}/jniCLI.cpp
    COMMAND ${Java_JAVA_EXECUTABLE} -jar ${JAVACPP} -classpath .:${MAVEN_BUILD_DIR}/lib/* -nocompile -d ${JNI_GEN_DIR} org/mlpack/CLI.class
    WORKING_DIRECTORY ${MAVEN_BUILD_DIR}/target/classes
    DEPENDS jniCLI_configure)

add_library(jnijavacpp STATIC
    ${JNI_GEN_DIR}/jnijavacpp.cpp)

target_include_directories(jnijavacpp PUBLIC ${JNI_INCLUDE})
set_target_properties(jnijavacpp PROPERTIES
    COMPILE_FLAGS -fPIC)

# Compile shared object
add_library(jniCLI SHARED
    ${JNI_GEN_DIR}/jniCLI.cpp
    ${BUILD_SRC_DIR}/cli_util.hpp
    ${BUILD_SRC_DIR}/cli_util.cpp
    ${BUILD_SRC_DIR}/deleter.hpp)

target_include_directories(jniCLI PUBLIC ${BUILD_SRC_DIR} ${JNI_GEN_DIR} ${JNI_INCLUDE})
target_link_libraries(jniCLI mlpack jnijavacpp)
set_target_properties(jniCLI PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${NATIVE_OUT_DIR}")
add_dependencies(jniCLI jniCLI_configure)
add_dependencies(java jniCLI)

add_library(java_generate_parts STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/strip_type.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/print_param_defn.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/print_output_param.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/print_java.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/print_java.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/print_input_param.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/print_doc_functions.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/java_option.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/get_java_type.hpp)

macro(add_java_binding file_name class_name)
  foreach(model_name ${ARGN})
    configure_file(
        ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/model.java.in
        ${MAVEN_BUILD_DIR}/src/main/java/org/mlpack/${model_name}Type.java)
  endforeach()

  add_custom_command(OUTPUT ${BUILD_SRC_DIR}/generate_java_${class_name}.cpp
      COMMAND ${CMAKE_COMMAND}
          -DFILE_NAME=${file_name}
          -DCLASS_NAME=${class_name}
          -DGENERATE_CPP_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/generate_java.cpp.in
          -DGENERATE_CPP_OUT=${BUILD_SRC_DIR}/generate_java_${class_name}.cpp
          -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${file_name}_main.cpp
          -P ${CMAKE_SOURCE_DIR}/CMake/ConfigureGenerate.cmake
      DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java/generate_java.cpp.in
              ${CMAKE_SOURCE_DIR}/CMake/ConfigureGenerate.cmake)

  add_executable(generate_java_${class_name}
      ${BUILD_SRC_DIR}/generate_java_${class_name}.cpp)

  target_link_libraries(generate_java_${class_name}
      mlpack
      java_generate_parts
      ${MLPACK_LIBRARIES})

  set_target_properties(generate_java_${class_name} PROPERTIES
      COMPILE_FLAGS -DBINDING_TYPE=BINDING_TYPE_JAVA
      RUNTIME_OUTPUT_DIRECTORY "${BUILD_SRC_DIR}")

  add_custom_command(TARGET generate_java_${class_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
          -DGENERATE_BINDING_PROGRAM="${BUILD_SRC_DIR}/generate_java_${class_name}"
          -DBINDING_OUTPUT_FILE=${MAVEN_BUILD_DIR}/src/main/java/org/mlpack/${class_name}.java
          -P ${CMAKE_SOURCE_DIR}/CMake/GenerateBinding.cmake)

  add_custom_target(jni${class_name}_configure ALL
      COMMAND ${MAVEN} compile -Dchanged.classes=org/mlpack/${class_name}.java
      WORKING_DIRECTORY ${MAVEN_BUILD_DIR}
      DEPENDS generate_java_${class_name} jniCLI)

  add_custom_command(OUTPUT ${JNI_GEN_DIR}/jni${class_name}.cpp
      COMMAND ${JAVA} -jar ${JAVACPP} -classpath .:${MAVEN_BUILD_DIR}/lib/* -nocompile -d ${JNI_GEN_DIR} org/mlpack/${class_name}.class
      WORKING_DIRECTORY ${MAVEN_BUILD_DIR}/target/classes)

  add_library(jni${class_name} SHARED
      ${JNI_GEN_DIR}/jni${class_name}.cpp)

  target_include_directories(jni${class_name} PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${BUILD_SRC_DIR}
      ${JNI_GEN_DIR}
      ${JNI_INCLUDE}
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/java)

  target_link_libraries(jni${class_name} mlpack ${MLPACK_LIBRARIES} jnijavacpp)

  set_target_properties(jni${class_name} PROPERTIES
      COMPILE_FLAGS -DBINDING_TYPE=BINDING_TYPE_JAVA
      LIBRARY_OUTPUT_DIRECTORY "${NATIVE_OUT_DIR}")

  add_dependencies(jni${class_name} jni${class_name}_configure)
  add_dependencies(java jni${class_name})
endmacro()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()

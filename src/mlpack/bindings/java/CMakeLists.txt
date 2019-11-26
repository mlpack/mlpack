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

find_package(Java REQUIRED COMPONENTS Development)
if (Java_FOUND)
  include(UseJava)
else()
  not_found_return("Java components weren't found")
endif()

find_package(JNI REQUIRED)
if (NOT JNI_FOUND)
  not_found_return("JNI wasn't found")
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  find_program(MAVEN mvn.bat REQUIRED)
else()
  find_program(MAVEN mvn REQUIRED)
endif()

if (MAVEN)
  message(STATUS "Found maven: ${MAVEN}")
else()
  not_found_return("Maven not found")
endif()

add_custom_target(java_configure ALL)
add_custom_target(java ALL DEPENDS java_configure)

# Create directory structure for maven project
set(MAVEN_BUILD_DIR ${CMAKE_BINARY_DIR}/src/mlpack/bindings/java/maven)
add_custom_command(TARGET java_configure PRE_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${MAVEN_BUILD_DIR}/src/main/java/org/mlpack)

# Create directory to output native libraries to
set(NATIVE_OUT_DIR ${MAVEN_BUILD_DIR}/native)
add_custom_command(TARGET java_configure PRE_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${NATIVE_OUT_DIR})

# Create directory to generate JNI files to
set(JNI_GEN_DIR ${CMAKE_BINARY_DIR}/src/mlpack/bindings/java/build/gen)
add_custom_command(TARGET java_configure PRE_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${JNI_GEN_DIR})

set(BUILD_SRC_DIR ${CMAKE_BINARY_DIR}/src/mlpack/bindings/java/build/src)
add_custom_command(TARGET java_configure PRE_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E make_directory ${BUILD_SRC_DIR}) 

# Copy maven pom.xml
add_custom_command(TARGET java_configure PRE_BUILD
    COMMAND ${CMAKE_COMMAND} ARGS -E copy 
        ${CMAKE_CURRENT_SOURCE_DIR}/maven/pom.xml 
        ${MAVEN_BUILD_DIR}/pom.xml)

# Make maven resolve dependencies into lib directory (see pom.xml)
add_custom_command(TARGET java_configure PRE_BUILD
    COMMAND ${MAVEN} ARGS dependency:copy-dependencies
    WORKING_DIRECTORY ${MAVEN_BUILD_DIR})

# Javacpp was pulled from maven
set(JAVACPP ${MAVEN_BUILD_DIR}/lib/javacpp-1.5.2.jar)

# Compile cli_util library for java
add_custom_target(jniCLI_configure ALL DEPENDS java_configure)

# Copy source files for java util library to build directory
add_custom_command(TARGET jniCLI_configure PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/maven/src/main/java/org/mlpack/CLI.java
        ${MAVEN_BUILD_DIR}/src/main/java/org/mlpack/CLI.java
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/maven/src/main/java/org/mlpack/MatrixWithInfo.java
        ${MAVEN_BUILD_DIR}/src/main/java/org/mlpack/MatrixWithInfo.java)

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
    COMMAND ${MAVEN} ARGS compile -Dchanged.classes=org/mlpack/CLI.java
    WORKING_DIRECTORY ${MAVEN_BUILD_DIR}) 

# Generate JNI wrappers from compiled java bytecode
add_custom_command(OUTPUT ${JNI_GEN_DIR}/jnijavacpp.cpp ${JNI_GEN_DIR}/jniCLI.cpp
    COMMAND ${Java_JAVA_EXECUTABLE} -jar ${JAVACPP} -classpath .:${MAVEN_BUILD_DIR}/lib/* -nocompile -d ${JNI_GEN_DIR} org/mlpack/CLI.class
    WORKING_DIRECTORY ${MAVEN_BUILD_DIR}/target/classes)

# Compile shared object
add_library(jniCLI SHARED
    ${JNI_GEN_DIR}/jnijavacpp.cpp 
    ${JNI_GEN_DIR}/jniCLI.cpp
    ${BUILD_SRC_DIR}/cli_util.hpp
    ${BUILD_SRC_DIR}/cli_util.cpp
    ${BUILD_SRC_DIR}/deleter.hpp)

target_include_directories(jniCLI PUBLIC ${BUILD_SRC_DIR} ${JNI_GEN_DIR} ${JNI_INCLUDE_DIRS})
target_link_libraries(jniCLI mlpack)
set_target_properties(jniCLI PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${NATIVE_OUT_DIR}")
add_dependencies(jniCLI jniCLI_configure)
add_dependencies(java jniCLI)

# Make sure we have no *.class files left
#add_custom_command(TARGET jniCLI POST_BUILD
 #   COMMAND ${MAVEN} ARGS clean
  #  WORKING_DIRECTORY ${MAVEN_BUILD_DIR}) 

macro(add_java_binding name java_name)
  
endmacro(add_java_binding name java_name)

add_subdirectory(tests)

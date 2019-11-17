add_custom_target(jniBindingsTest_configure ALL 
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/BindingsTest.java
        ${MAVEN_BUILD_DIR}/src/main/java/org/mlpack/BindingsTest.java
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/java_bindings_test_main.cpp
        ${BUILD_SRC_DIR}/java_bindings_test_main.cpp
    COMMAND ${MAVEN} compile -Dchanged.classes=org/mlpack/BindingsTest.java
    WORKING_DIRECTORY ${MAVEN_BUILD_DIR}
    DEPENDS java_configure jniCLI)

add_custom_command(OUTPUT ${JNI_GEN_DIR}/jnijavacpp.cpp ${JNI_GEN_DIR}/jniBindingsTest.cpp
    COMMAND ${Java_JAVA_EXECUTABLE} -jar ${JAVACPP} -classpath .:${MAVEN_BUILD_DIR}/lib/* -nocompile -d ${JNI_GEN_DIR} org/mlpack/BindingsTest.class
    WORKING_DIRECTORY ${MAVEN_BUILD_DIR}/target/classes)

add_library(jniBindingsTest SHARED
    ${JNI_GEN_DIR}/jnijavacpp.cpp 
    ${JNI_GEN_DIR}/jniBindingsTest.cpp)

target_include_directories(jniBindingsTest PUBLIC
    ${BUILD_SRC_DIR}
    ${JNI_GEN_DIR}
    ${JNI_INCLUDE_DIRS})

target_link_libraries(jniBindingsTest mlpack)

set_target_properties(jniBindingsTest PROPERTIES
    COMPILE_FLAGS -DBINDING_TYPE=BINDING_TYPE_JAVA
    LIBRARY_OUTPUT_DIRECTORY "${NATIVE_OUT_DIR}")

add_dependencies(jniBindingsTest jniBindingsTest_configure)
add_dependencies(java jniBindingsTest)

add_custom_command(TARGET jniBindingsTest POST_BUILD
    COMMAND ${MAVEN} clean
    COMMAND ${CMAKE_COMMAND} -E make_directory ${MAVEN_BUILD_DIR}/src/test/java/org/mlpack
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/BindingsJUnitTest.java
        ${MAVEN_BUILD_DIR}/src/test/java/org/mlpack/BindingsJUnitTest.java
    WORKING_DIRECTORY ${MAVEN_BUILD_DIR})

add_test(NAME test_java_bindings
    COMMAND ${MAVEN} test
    WORKING_DIRECTORY ${MAVEN_BUILD_DIR})

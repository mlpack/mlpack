add_java_binding(java_bindings_test BindingsTest GaussianKernel)

add_custom_command(TARGET jniBindingsTest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${MAVEN_BUILD_DIR}/src/test/java/org/mlpack
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/BindingsJUnitTest.java
        ${MAVEN_BUILD_DIR}/src/test/java/org/mlpack/BindingsJUnitTest.java
    WORKING_DIRECTORY ${MAVEN_BUILD_DIR})

add_test(NAME test_java_bindings
    COMMAND ${MAVEN} test
    WORKING_DIRECTORY ${MAVEN_BUILD_DIR})

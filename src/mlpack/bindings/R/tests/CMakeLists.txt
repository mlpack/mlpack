# Add the R binding test.
add_r_binding(test_r_binding)

add_test(NAME r_binding_test
    COMMAND R CMD check --no-manual --as-cran mlpack_${PACKAGE_VERSION}.tar.gz
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/R/)

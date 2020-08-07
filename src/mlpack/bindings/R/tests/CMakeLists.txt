# Add the R binding test.
add_r_binding(test_r_binding)

add_test(NAME r_binding_test
    COMMAND ${RSCRIPT_EXECUTABLE} -e "pkgload::load_all('.'); testthat::test_dir('tests/')"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/R/mlpack)

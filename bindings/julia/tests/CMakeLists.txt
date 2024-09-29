# Add the Julia binding test.
add_julia_binding("" test_julia_binding)

add_test(NAME julia_binding_test
    COMMAND ${JULIA_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/runtests.jl
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/)

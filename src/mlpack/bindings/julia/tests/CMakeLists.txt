# Add the Julia binding test.
add_julia_binding("" test_julia_binding)

# Julia expects the runtests.jl file to be in test/, not tests/.
file(COPY runtests.jl DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../mlpack/test/)

add_test(NAME julia_binding_test
    COMMAND ${JULIA_EXECUTABLE} test/runtests.jl
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/)

# Add the Julia binding test.
add_julia_binding("" test_julia_binding "false")

# Add the grouped Julia binding test.
group_bindings("" test_group_julia_binding "train;predict")
add_julia_class("" test_group_julia_binding)

# Make sure the list of grouped bindings makes it to the parent scope.
set(JULIA_GROUP_BINDINGS ${JULIA_GROUP_BINDINGS} PARENT_SCOPE)

# Julia expects the runtests.jl file to be in test/, not tests/.
file(COPY runtests.jl DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../mlpack/test/)

add_test(NAME julia_binding_test
    COMMAND ${JULIA_EXECUTABLE} test/runtests.jl
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/julia/mlpack/)

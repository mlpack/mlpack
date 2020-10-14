# Add the Go binding test.
add_go_binding(test_go_binding)

if (BUILD_GO_BINDINGS)
add_test(NAME go_binding_test
    COMMAND ${GO_EXECUTABLE} test -v  ${CMAKE_CURRENT_SOURCE_DIR}/go_binding_test.go
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/)
set_tests_properties(go_binding_test
      PROPERTIES ENVIRONMENT "GOPATH=$ENV{GOPATH}:${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/;
      LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/src/mlpack.org/v1/mlpack/")
endif()

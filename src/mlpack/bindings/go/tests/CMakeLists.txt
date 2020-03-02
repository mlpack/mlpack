# Add the Go binding test.
add_go_binding(test_go_binding)

if (BUILD_GO_BINDINGS)
add_test(NAME go_binding_test
    COMMAND go test -v  ${CMAKE_CURRENT_SOURCE_DIR}/go_binding_test.go
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/)
set_tests_properties(go_binding_test
      PROPERTIES ENVIRONMENT "GOPATH=$ENV{GOPATH}:${CMAKE_BINARY_DIR};LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/")
endif()

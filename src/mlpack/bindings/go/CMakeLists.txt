if (BUILD_GO_BINDINGS)
  # We need to check here if Go is even available, since the tests will need it.
  find_package(Go 1.11.0)

  if (NOT GO_FOUND)
    # We can't build anything, so define the macro to do nothing.
    macro (add_go_binding name)
      # Do nothing.
    endmacro ()

    return ()
  endif ()

  find_package(Gonum)
  if (NOT GONUM_FOUND)
    # We can't build anything, so define the macro to do nothing.
    macro (add_go_binding name)
      # Do nothing.
    endmacro ()
  endif ()


  # Nothing in this directory will be compiled into mlpack.
  set(BINDING_SOURCES
    get_type.hpp
    get_param.hpp
    get_printable_param.hpp
    go_option.hpp
    mlpack/arma_util.h
    mlpack/arma_util.hpp
    mlpack/cli_util.h
    mlpack/cli_util.hpp
    print_class_defn.hpp
    print_cpp.cpp
    print_cpp.hpp
    print_defn_input.hpp
    print_defn_output.hpp
    print_doc.hpp
    print_doc_functions.hpp
    print_doc_functions_impl.hpp
    print_go.hpp
    print_go.cpp
    print_h.hpp
    print_h.cpp
    print_import_decl.hpp
    print_input_processing.hpp
    print_method_config.hpp
    print_method_init.hpp
    print_output_processing.hpp
    camel_case.hpp
    strip_type.hpp
  )

  # These are all the files we need to compile Go bindings for mlpack that are
  # not a part of mlpack itself.
  set(CGO_SOURCES
    mlpack/arma_util.go
    mlpack/cli_util.go
    mlpack/doc.go
  )

  # These are all the files we need to compile Go bindings for mlpack that are
  # not a part of mlpack itself.
  set(CAPI_SOURCES
    mlpack/capi/arma_util.cpp
    mlpack/capi/arma_util.h
    mlpack/capi/arma_util.hpp
    mlpack/capi/cli_util.cpp
    mlpack/capi/cli_util.h
    mlpack/capi/cli_util.hpp
  )

  # These are all the files we need to compile Go bindings for mlpack that are
  # not a part of mlpack itself.
  set(UTIL_SOURCES
    mlpack/capi/arma_util.cpp
    mlpack/capi/cli_util.cpp
  )

  set(TEST_SOURCES
    tests/go_binding_test.go
  )
  add_custom_target(go ALL DEPENDS mlpack)
  add_custom_target(go_copy ALL DEPENDS mlpack)

  if (BUILD_TESTS)
    foreach(test_file ${TEST_SOURCES})
      add_custom_command(TARGET go_copy PRE_BUILD
          COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different
              ${CMAKE_CURRENT_SOURCE_DIR}/${test_file}
              ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/tests/)
    endforeach ()
  endif ()

  # Copy necessary files after making the mlpack/ directory.
  add_custom_command(TARGET go_copy PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/
      COMMAND ${CMAKE_COMMAND} -E make_directory
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/capi/)
  foreach(go_file ${CAPI_SOURCES})
  add_custom_command(TARGET go_copy PRE_BUILD
      COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/${go_file}
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/capi/)
  endforeach()

  add_custom_command(TARGET go_copy PRE_BUILD
      COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different
          $<TARGET_FILE:mlpack>
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/)

  foreach(cgo_file ${CGO_SOURCES})
  add_custom_command(TARGET go_copy PRE_BUILD
      COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/${cgo_file}
          ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/)
  endforeach()

  add_library(go_util SHARED
          ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/mlpack/capi/arma_util.cpp
          ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/mlpack/capi/cli_util.cpp)
  target_link_libraries(go_util mlpack ${MLPACK_LIBRARIES})
  target_compile_definitions(go_util PUBLIC "BINDING_TYPE=BINDING_TYPE_GO")
  set_target_properties(go_util PROPERTIES
     LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/)

  # Set the include directories correctly.
  get_property(GO_INCLUDE_DIRECTORIES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
     PROPERTY INCLUDE_DIRECTORIES)
  set (GO_INCLDIRS "${GO_INCLUDE_DIRECTORIES}")
endif()

# Add a macro to build a go binding.
macro (add_go_binding name)
if (BUILD_GO_BINDINGS)
  # Append sources (with directory name) to list of all mlpack sources (used at
  # the parent scope).
  set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

  # Create .cpp file for C API, e.g. pca.cpp.
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_cpp_${name}.cpp
     COMMAND ${CMAKE_COMMAND}
         -DGENERATE_BINDING_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/generate_cpp.cpp.in
         -DGENERATE_BINDING_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_cpp_${name}.cpp
         -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp
         -DPROGRAM_NAME=${name}
         -P ${CMAKE_SOURCE_DIR}/CMake/ConfigureGoHCPP.cmake
     DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/generate_cpp.cpp.in)

  add_executable(generate_cpp_${name}
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_cpp_${name}.cpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/print_cpp.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/print_cpp.cpp)
  target_link_libraries(generate_cpp_${name} mlpack ${MLPACK_LIBRARIES})
  set_target_properties(generate_cpp_${name} PROPERTIES COMPILE_FLAGS
      -DBINDING_TYPE=BINDING_TYPE_GO)
  add_custom_command(TARGET generate_cpp_${name} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
          -DGENERATE_BINDING_PROGRAM=${CMAKE_BINARY_DIR}/bin/generate_cpp_${name}
          -DBINDING_OUTPUT_FILE=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/capi/${name}.cpp
          -P ${CMAKE_SOURCE_DIR}/CMake/GenerateGoBinding.cmake)

  add_dependencies(generate_cpp_${name} go_copy)

  # Create .h file for C API, e.g. pca.h.
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_h_${name}.cpp
     COMMAND ${CMAKE_COMMAND}
         -DGENERATE_BINDING_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/generate_h.cpp.in
         -DGENERATE_BINDING_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_h_${name}.cpp
         -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp
         -DPROGRAM_NAME=${name}
         -P ${CMAKE_SOURCE_DIR}/CMake/ConfigureGoHCPP.cmake
     DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/generate_h.cpp.in)

  add_executable(generate_h_${name}
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_h_${name}.cpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/print_h.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/print_h.cpp)
  target_link_libraries(generate_h_${name} mlpack ${MLPACK_LIBRARIES})
  set_target_properties(generate_h_${name} PROPERTIES COMPILE_FLAGS
      -DBINDING_TYPE=BINDING_TYPE_GO)
  add_custom_command(TARGET generate_h_${name} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
          -DGENERATE_BINDING_PROGRAM=${CMAKE_BINARY_DIR}/bin/generate_h_${name}
          -DBINDING_OUTPUT_FILE=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/capi/${name}.h
          -P ${CMAKE_SOURCE_DIR}/CMake/GenerateGoBinding.cmake)

  # Create .go file, pca.go.
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_go_${name}.cpp
     COMMAND ${CMAKE_COMMAND}
         -DGENERATE_BINDING_IN=${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/generate_go.cpp.in
         -DGENERATE_BINDING_OUT=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_go_${name}.cpp
         -DPROGRAM_MAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/${name}_main.cpp
         -DPROGRAM_NAME=${name}
         -P ${CMAKE_SOURCE_DIR}/CMake/ConfigureGoHCPP.cmake
     DEPENDS ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/generate_go.cpp.in)

  add_executable(generate_go_${name}
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/generate_go_${name}.cpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/print_go.hpp
      ${CMAKE_SOURCE_DIR}/src/mlpack/bindings/go/print_go.cpp)
  target_link_libraries(generate_go_${name} mlpack ${MLPACK_LIBRARIES})
  set_target_properties(generate_go_${name} PROPERTIES COMPILE_FLAGS
      -DBINDING_TYPE=BINDING_TYPE_GO)
  add_custom_command(TARGET generate_go_${name} POST_BUILD
      COMMAND ${CMAKE_COMMAND}
          -DGENERATE_BINDING_PROGRAM=${CMAKE_BINARY_DIR}/bin/generate_go_${name}
          -DBINDING_OUTPUT_FILE=${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/${name}.go
          -P ${CMAKE_SOURCE_DIR}/CMake/GenerateGoBinding.cmake)

  add_dependencies(generate_cpp_${name} generate_go_${name})
  add_dependencies(generate_cpp_${name} generate_h_${name})

  # Create, e.g., libmlpack_go_pca.so.
  add_library(mlpack_go_${name} SHARED
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/capi/${name}.cpp
      ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/capi/${name}.h)
  set_source_files_properties(${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/capi/${name}.cpp
      PROPERTIES GENERATED TRUE)
  set_source_files_properties(${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/capi/${name}.h
      PROPERTIES GENERATED TRUE)
  target_link_libraries(mlpack_go_${name} mlpack ${MLPACK_LIBRARIES})
  target_compile_definitions(mlpack_go_${name} PUBLIC "BINDING_TYPE=BINDING_TYPE_GO")
  set_target_properties(mlpack_go_${name} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src/mlpack/bindings/go/mlpack/)

  install(TARGETS mlpack_go_${name}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  add_dependencies(mlpack_go_${name} generate_cpp_${name})
  add_dependencies(mlpack_go_${name} go_util)
  add_dependencies(go mlpack_go_${name})
endif ()
endmacro ()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif ()
